#include "models/glm5next/startup.h"

#include "core/platform.h"
#include "models/glm5next/program/graph_inputs.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace ninfer::models::glm5next {

KvCacheVectorFormat index_format_from_environment() {
    const auto* value = std::getenv("NINFER_KPOOL_INDEXER_BF16");
    if (!value || std::string_view(value) == "0")
        return KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    if (std::string_view(value) == "1") return KvCacheVectorFormat::BFloat16;
    throw std::invalid_argument("NINFER_KPOOL_INDEXER_BF16 must be 0 or 1");
}

bool can_narrow_dflash_targets(const EngineOptions& options) noexcept {
    return options.use_cuda_graph && options.purpose == EnginePurpose::Generation &&
           options.speculative.length_policy == SpeculativeLengthPolicy::Adaptive &&
           options.speculative.backend == SpeculativeBackend::DFlash2 &&
           options.speculative.draft_tokens >= 5;
}

RankPreparation::RankPreparation(const artifact::Reader& reader, RankLoadRequest request,
                                 DeviceContext& device, ContextCommunication peer,
                                 std::uint32_t global_node,
                                 const distributed::GroupDescription* experts)
    : device_(device), options_(std::move(request.options)), global_node_(global_node),
      peer_(std::move(peer)), control_owner_(request.control_owner) {
    if (request.artifact != reader.artifact_id())
        throw artifact::ArtifactError("GLM5Next worker artifact differs from coordinator");
    auto definition = read_definition(reader, load_options(options_));
    load_.emplace(
        resolve_rank_load(reader, std::move(definition), std::move(request.rank), device.device()));
    memory_      = {load_->materialization().device_capacity_bytes,
                    load_->materialization().host_capacity_bytes, request.host_reservation_bytes};
    auto profile = execution::describe_model(load_->definition(), load_->weight_data(),
                                             load_->expert_data(), load_->rank_plan());
#if NINFER_ENABLE_DISTRIBUTED
    if (experts) {
        const auto& rp  = load_->rank_plan();
        const auto peer = experts->index(rp.placement.global_rank);
        const auto& c   = profile->config.text;
        auto route = std::make_shared<moe::ResidentExchangePlan>(rp.expert_domain, *experts, peer,
                                                                 c.hidden, c.moe.top_k);
        const auto bind = [&](execution::MoeProfile& layer) {
            layer.exchange_workspace =
                route->stage(layer.bank).sigmoid_workspace_bytes(peer, layer.math);
        };
        for (auto& layer : profile->text.layers)
            if (auto* m = std::get_if<execution::MoeProfile>(&layer.ffn)) bind(*m);
        if (profile->mtp) bind(profile->mtp->moe);
        profile->exchange_bytes = route->bytes();
        profile->exchange       = std::move(route);
        expert_requirements_ =
            moe::expert_exchange_requirements(*experts, *rp.expert_domain, peer, c.hidden,
                                              c.moe.top_k, moe::expert_exchange_block_columns);
    }
#endif
    plan_        = plan_execution(
        std::move(profile), options_, request.index_format, device.multiprocessor_count(),
        device_memory_snapshot(device.device()).shared_physical_memory, std::move(request.batches));
    if (request.narrow_dflash_targets) select_narrow_targets(true);
    capacity_ = describe_execution_capacity(plan_);
}

RankPreparation::~RankPreparation() = default;

runtime::PreparationMemory RankPreparation::memory() const { return memory_; }

std::unique_ptr<moe::ResidentExpertExchange> RankPreparation::prepare_experts() const {
#if NINFER_ENABLE_DISTRIBUTED
    if (expert_group_) return plan_.profile->exchange->prepare(*expert_group_);
#endif
    return {};
}

void RankPreparation::materialize() {
    if (!load_) throw std::logic_error("GLM5Next rank weights are already materialized");
    instance_ = std::make_unique<Instance>(
        materialize_model(std::move(*load_), device_, &options_.startup_observer), plan_.profile);
    load_.reset();
}

runtime::RankPreparationReport RankPreparation::report(std::size_t available) const {
    if (!instance_) throw std::logic_error("GLM5Next parameters have not been materialized");
    const auto& model  = *instance_->model;
    const auto& s      = model.storage_stats();
    const auto reserve = graph_memory_.rebuild_estimate_bytes;
    return {.capacity             = {capacity_, available > reserve ? available - reserve : 0},
            .weight_bytes         = s.device_capacity_bytes + model.metadata_bytes(),
            .host_weight_bytes    = s.host_capacity_bytes,
            .staging_bytes        = s.peak_staging_bytes,
            .read_bytes           = s.read_bytes,
            .upload_bytes         = s.h2d_bytes,
            .load_seconds         = s.upload_seconds,
            .device_objects       = s.device_object_count,
            .host_objects         = s.host_object_count,
            .preparation_required = plan_.use_cuda_graph,
            .control_owner        = control_owner_,
            .graph_groups         = plan_.use_cuda_graph && expert_requirements_
                                        ? std::vector<distributed::GroupId>{expert_requirements_->group}
                                        : std::vector<distributed::GroupId>{}};
}

void RankPreparation::prepare_probe() {
    if (!plan_.use_cuda_graph) return;
    create_probe(detail::graph_context_bytes(plan_));
}

void RankPreparation::prepare_rebuild_probe() {
    if (!plan_.use_cuda_graph) return;
    create_probe(capacity_.minimum_capacity().device_bytes);
}

void RankPreparation::create_probe(std::size_t context_bytes) {
    auto plan                               = plan_;
    plan.capacity                           = {};
    plan.capacity.device_bytes              = context_bytes;
    plan.capacity.runtime_reservation_bytes = plan.reservation_bytes();
    auto resources                          = std::make_unique<Resources>(std::move(plan), device_);
    probe_ = std::make_unique<RankProgram>(instance_->parameters, std::move(resources), device_,
                                           binding_, prepare_experts());
}

void RankPreparation::calibrate() {
    if (!probe_) return;
    graph_memory_ = begin_graph_calibration(
        device_, [&](GraphPreparationMode mode) { return probe_->prepare_execution(mode); },
        [&] { probe_->release_execution(); });
    probe_.reset();
    device_.synchronize();
}

void RankPreparation::calibrate_rebuild() {
    if (!probe_) return;
    complete_graph_calibration(
        graph_memory_, device_,
        [&](GraphPreparationMode mode) { return probe_->prepare_execution(mode); },
        [&] { probe_->release_execution(); });
    probe_.reset();
    device_.synchronize();
}

void RankPreparation::prepare_execution() {
    instance_->program->prepare_execution(GraphPreparationMode::Capture);
}

void RankPreparation::prepare(runtime::ContextCapacity capacity, std::size_t available,
                              std::size_t available_host) {
    if (!instance_ || instance_->program)
        throw std::logic_error(
            "GLM5Next resource allocation requires weights and an empty resource slot");
    const bool shared_memory = device_memory_snapshot(device_.device()).shared_physical_memory;
    const auto reserve       = graph_memory_.rebuild_estimate_bytes;
    available                = available > reserve ? available - reserve : 0;
    const auto reserved      = capacity_.reservation_bytes(capacity);
    const auto control       = control_owner_ && shared_memory ? capacity_.control_bytes(capacity) : 0;
    if (reserved > available || control > available - reserved) {
        const auto required = reserved + control;
        throw std::runtime_error("GLM5Next rank allocation budget exceeded: required " +
                                 std::to_string(required) + " bytes, available " +
                                 std::to_string(available) + " bytes, shortfall " +
                                 std::to_string(required - available) + " bytes");
    }
    auto plan     = plan_;
    plan.capacity = runtime::resolve_context_capacity(
        ContextCapacityPolicy::explicit_capacity(capacity.device_bytes), capacity_, available,
        available_host, shared_memory, control_owner_);
    auto resources     = std::make_unique<Resources>(std::move(plan), device_, peer_);
    instance_->program = std::make_unique<RankProgram>(instance_->parameters, std::move(resources),
                                                       device_, binding_, prepare_experts());
}

void RankPreparation::release_resources() {
    const bool active = probe_ || (instance_ && instance_->program);
    probe_.reset();
    if (instance_ && instance_->program) {
        instance_->program->drain();
        if (const auto graph = instance_->program->resource_budget().graphs)
            graph_memory_.rebuild_estimate_bytes =
                std::max(graph_memory_.rebuild_estimate_bytes, graph->preparation_peak_bytes);
        instance_->program.reset();
    }
    if (active) device_.synchronize();
}

void RankPreparation::select_narrow_targets(bool enabled, GraphMemorySummary measured) {
    if ((enabled && !can_narrow_dflash_targets(options_)) || (instance_ && instance_->program))
        throw std::logic_error("GLM5Next cannot change the decode graph catalog");
    probe_.reset();
    plan_.shape.narrow_dflash_targets = enabled;
    plan_.decode_routes               = make_decode_routes(plan_.shape, *plan_.batches);
    graph_memory_                     = measured;
}

RankResourceDescription RankPreparation::describe() const {
    if (!instance_ || !instance_->program)
        throw std::logic_error("GLM5Next resources have not been prepared");
    const auto& p = instance_->program->plan();
    const auto& c = p.capacity;
    RankResourceDescription out{.rank    = p.profile->replica_rank(),
                                .shape   = p.shape,
                                .batches = p.batches->spec(),
                                .state   = describe_state(p.state),
                                .context = describe_context(p)};
    out.observation.rank     = instance_->model->rank_plan().placement.global_rank;
    out.observation.node     = global_node_;
    out.observation.hardware = device_.properties().name;
    out.observation.memory   = instance_->program->observe_resources();

    out.budget = {.runtime_reservation_bytes         = c.runtime_reservation_bytes,
                  .minimum_runtime_reservation_bytes = c.minimum_runtime_reservation_bytes,
                  .context_capacity_bytes            = c.device_bytes,
                  .minimum_context_capacity_bytes    = c.minimum_device_bytes,
                  .context_control_budget_bytes      = c.control_reservation_bytes,
                  .available_after_weights_bytes     = c.available_after_weights_bytes,
                  .planned_slack_bytes               = c.planned_slack_bytes};
    if (plan_.use_cuda_graph) {
        auto observed     = instance_->program->resource_budget().graphs;
        out.budget.graphs = graph_memory_;
        if (observed) {
            out.budget.graphs->preparation_increment_bytes = observed->preparation_increment_bytes;
            out.budget.graphs->preparation_peak_bytes      = observed->preparation_peak_bytes;
        }
    }
    out.caches.push_back(p.main.cache.page.geometry);
    if (p.mtp) out.caches.push_back(p.mtp->cache.page.geometry);
    if (p.workspace.vision) out.vision_max_features = p.workspace.vision->max_features;
    return out;
}

std::unique_ptr<RankPreparation::Instance> RankPreparation::finish() && {
    if (!instance_ || !instance_->program)
        throw std::logic_error("GLM5Next execution resources have not been prepared");
    return std::move(instance_);
}

Frontend create_frontend(const ModelDefinition& definition, const EngineOptions& o) {
    return make_frontend(definition.resources(),
                         {o.chat_template_path, o.grammar_cache_bytes,
                          definition.config().vision.has_value(), o.max_context,
                          o.media_cache_bytes, o.media_live_bytes, o.media_preprocess_threads});
}

Startup::Startup(const artifact::Reader& reader, const artifact::Reader& rank,
                 const EngineOptions& options, Device)
    : definition_(read_definition(reader, load_options(options))),
      index_format_(index_format_from_environment()) {
    if (reader.artifact_id() != rank.artifact_id())
        throw artifact::ArtifactError("GLM5Next local artifact differs from coordinator");
    rank_ = options.deployment ? plan_parallel(*definition_, *options.deployment).ranks.at(0)
                               : plan_single_rank(*definition_);
    if (rank_.placement.tp.size != 1 ||
        (options.deployment && options.deployment->ranks.size() != 1))
        throw std::invalid_argument("GLM5Next local startup requires one rank");
}

Frontend Startup::frontend(const EngineOptions& options) const {
    return create_frontend(*definition_, options);
}

bool Startup::narrow_targets(const EngineOptions& options) const noexcept {
    return can_narrow_dflash_targets(options);
}

std::unique_ptr<RankPreparation> Startup::preparation(const artifact::Reader& reader,
                                                      const EngineOptions& options,
                                                      DeviceContext& device) const {
    return std::make_unique<RankPreparation>(
        reader,
        RankLoadRequest{.artifact = definition_->artifact_id(),
                        .rank     = rank_,
                        .options  = options,
                        .batches  = make_batch_execution_spec(
                            options.max_concurrency, 1,
                            options.use_cuda_graph && options.purpose == EnginePurpose::Generation),
                        .index_format = index_format_,
                        .host_reservation_bytes =
                            options.enable_vision ? options.media_live_bytes : 0,
                        .control_owner = true},
        device);
}

PreparedSession Startup::session(const RankResourceDescription& rank,
                                 std::uint32_t interval) const {
    return prepare_session(definition_, prepare_replica_context({&rank_, 1}, {&rank, 1}), interval);
}

std::string prefill_signature(const ModelDefinition& d, std::uint32_t chunk, std::uint32_t ranks,
                              std::uint32_t ep_size) {
    const auto& c = d.config().text;
    return "glm5next:h=" + std::to_string(c.hidden) + ";layers=" + std::to_string(c.layers.size()) +
           ";chunk=" + std::to_string(chunk) + ";tp=" + std::to_string(ranks) +
           ";ep=" + std::to_string(ep_size) + ";adp=" + std::to_string(ranks);
}

std::string Startup::prefill_signature(std::uint32_t chunk) const {
    return glm5next::prefill_signature(*definition_, chunk, 1, 1);
}

} // namespace ninfer::models::glm5next
