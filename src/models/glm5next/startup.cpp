#include "models/glm5next/startup.h"
#include "core/platform.h"

namespace ninfer::models::glm5next {

RankPreparation::RankPreparation(const artifact::Reader& reader, RankLoadRequest request,
                                 DeviceContext& device, ContextCommunication peer)
    : device_(device), options_(std::move(request.options)), peer_(peer),
      control_owner_(request.control_owner) {
    if (request.artifact != reader.artifact_id())
        throw artifact::ArtifactError("GLM5Next worker artifact differs from coordinator");
    if (load_options(options_).mtp())
        throw std::invalid_argument("GLM5Next MTP execution is not implemented");
    if (options_.use_cuda_graph)
        throw std::invalid_argument("GLM5Next rank preparation requires eager execution");
    auto definition = read_definition(reader, load_options(options_));
    load_.emplace(
        resolve_rank_load(reader, std::move(definition), std::move(request.rank), device.device()));
    memory_ = {load_->materialization().device_capacity_bytes,
               load_->materialization().host_capacity_bytes,
               options_.context_capacity.automatic_headroom_bytes, request.host_reservation_bytes};
    auto profile =
        execution::describe_model(load_->definition(), load_->weight_data(), load_->rank_plan());
    plan_     = plan_execution(std::move(profile), options_, device.multiprocessor_count());
    capacity_ = describe_execution_capacity(plan_);
}

RankPreparation::~RankPreparation() = default;

runtime::PreparationMemory RankPreparation::memory() const { return memory_; }

void RankPreparation::materialize() {
    if (!load_) throw std::logic_error("GLM5Next rank weights are already materialized");
    instance_ = std::make_unique<Instance>(
        materialize_model(std::move(*load_), device_, &options_.startup_observer), plan_.profile);
    load_.reset();
}

runtime::RankPreparationReport RankPreparation::report(std::size_t available) const {
    if (!instance_) throw std::logic_error("GLM5Next parameters have not been materialized");
    const auto& model = *instance_->model;
    const auto& s     = model.storage_stats();
    return {.capacity          = {capacity_, available},
            .weight_bytes      = s.device_capacity_bytes + model.metadata_bytes(),
            .host_weight_bytes = s.host_capacity_bytes,
            .staging_bytes     = s.peak_staging_bytes,
            .read_bytes        = s.read_bytes,
            .upload_bytes      = s.h2d_bytes,
            .load_seconds      = s.upload_seconds,
            .device_objects    = s.device_object_count,
            .host_objects      = s.host_object_count,
            .control_owner     = control_owner_};
}

void RankPreparation::prepare(runtime::ContextCapacity capacity, std::size_t available,
                              std::size_t available_host) {
    if (!instance_ || instance_->program)
        throw std::logic_error(
            "GLM5Next resource allocation requires weights and an empty resource slot");
    const auto reserved = capacity_.reservation_bytes(capacity);
    const auto control =
        control_owner_ && NINFER_SHARED_PHYSICAL_MEMORY ? capacity_.control_bytes(capacity) : 0;
    if (reserved > available || control > available - reserved)
        throw DeviceAllocationError("GLM5Next rank runtime capacity exceeds available memory",
                                    reserved);
    auto plan     = plan_;
    plan.capacity = runtime::resolve_context_capacity(
        ContextCapacityPolicy::explicit_capacity(capacity.device_bytes), capacity_, available,
        available_host, NINFER_SHARED_PHYSICAL_MEMORY != 0, control_owner_);
    auto resources     = std::make_unique<Resources>(std::move(plan), device_, peer_);
    instance_->program = std::make_unique<RankProgram>(instance_->parameters, std::move(resources),
                                                       device_, binding_);
}

void RankPreparation::release_resources() {
    if (instance_ && instance_->program) {
        instance_->program->drain();
        instance_->program.reset();
    }
}

RankResourceDescription RankPreparation::describe() const {
    if (!instance_ || !instance_->program)
        throw std::logic_error("GLM5Next resources have not been prepared");
    const auto& p = instance_->program->plan();
    const auto& c = p.capacity;
    RankResourceDescription out{.rank    = p.profile->local.rank,
                                .shape   = p.shape,
                                .state   = describe_state(p.state),
                                .context = describe_context(p)};
    out.observation.rank     = out.rank;
    out.observation.hardware = device_.properties().name;
    out.observation.memory   = instance_->program->observe_resources();

    out.budget = {.runtime_reservation_bytes         = c.runtime_reservation_bytes,
                  .minimum_runtime_reservation_bytes = c.minimum_runtime_reservation_bytes,
                  .context_capacity_bytes            = c.device_bytes,
                  .minimum_context_capacity_bytes    = c.minimum_device_bytes,
                  .context_control_budget_bytes      = c.control_reservation_bytes,
                  .available_after_weights_bytes     = c.available_after_weights_bytes,
                  .headroom_bytes                    = c.automatic_headroom_bytes,
                  .planned_slack_bytes               = c.planned_slack_bytes};
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
    : definition_(read_definition(reader, load_options(options))) {
    if (reader.artifact_id() != rank.artifact_id())
        throw artifact::ArtifactError("GLM5Next local artifact differs from coordinator");
    rank_ = options.deployment ? plan_parallel(*definition_, *options.deployment).ranks.at(0)
                               : plan_single_rank(*definition_);
    if (rank_.size != 1) throw std::invalid_argument("GLM5Next local startup requires one rank");
}

Frontend Startup::frontend(const EngineOptions& options) const {
    return create_frontend(*definition_, options);
}

std::unique_ptr<RankPreparation> Startup::preparation(const artifact::Reader& reader,
                                                      const EngineOptions& options,
                                                      DeviceContext& device) const {
    return std::make_unique<RankPreparation>(
        reader,
        RankLoadRequest{definition_->artifact_id(), rank_, options,
                        options.enable_vision ? options.media_live_bytes : 0, true},
        device);
}

PreparedSession Startup::session(const RankResourceDescription& rank,
                                 std::uint32_t interval) const {
    ParallelPlan group;
    group.ranks.push_back(rank_);
    return prepare_session(definition_, prepare_context(group, {&rank, 1}), interval);
}

std::string prefill_signature(const ModelDefinition& d, std::uint32_t chunk, std::uint32_t ranks) {
    const auto& c = d.config().text;
    return "glm5next:h=" + std::to_string(c.hidden) + ";layers=" + std::to_string(c.layers.size()) +
           ";chunk=" + std::to_string(chunk) + ";tp=" + std::to_string(ranks) +
           ";ep=" + std::to_string(ranks) + ";adp=" + std::to_string(ranks);
}

std::string Startup::prefill_signature(std::uint32_t chunk) const {
    return glm5next::prefill_signature(*definition_, chunk, 1);
}

} // namespace ninfer::models::glm5next
