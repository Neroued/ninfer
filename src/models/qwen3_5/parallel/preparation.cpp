#include "models/qwen3_5/parallel/preparation.h"

#include "core/platform.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/parallel/codec.h"
#include "models/qwen3_5/parallel/wire.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5 {
namespace {

RankLoadRequest decode_and_verify_request(const artifact::Reader& reader,
                                          std::span<const std::byte> payload,
                                          const distributed::RankIdentity& identity,
                                          const StartupObserver& observer) {
    distributed::wire::Decoder in(payload);
    auto r = decode_rank_load(in);
    in.finish();
    if (r.artifact != reader.artifact_id()) {
        throw artifact::ArtifactError("Qwen3.5 worker artifact differs from coordinator");
    }
    if (r.rank.rank != 0 || r.rank.size != 1 || r.rank.ep_size != 1 || identity.tp.size != 1 ||
        identity.ep.size != 1 || identity.pp.size != 1 || identity.adp.size != 1 ||
        identity.dp.size < 1) {
        throw std::invalid_argument("Qwen3.5 Stage 1 worker requires pure DP (TP=EP=PP=ADP=1)");
    }
    if (compute_rank_shard_digest(reader.artifact_id(), models::load_options(r.options), r.rank) !=
        identity.shard_digest) {
        throw std::invalid_argument("Qwen3.5 rank shard digest differs from topology plan");
    }
    r.options.startup_observer = observer;
    return r;
}

} // namespace

runtime::WorkPayloadCapacity work_payload_capacity(const execution::ModelProfile& profile,
                                                   const EngineOptions& options) {
    const WorkContract contract{
        .max_context     = options.max_context,
        .max_concurrency = options.max_concurrency,
        .prefill_chunk =
            std::min(options.max_context, options.prefill_chunk ? options.prefill_chunk : 1024U),
        .draft_window        = options.speculative.draft_tokens,
        .backend             = options.speculative.backend,
        .public_tokens       = profile.public_tokens,
        .control_token_limit = profile.control_token_limit,
        .kv_pool_count =
            1U + (profile.options.mtp() ||
                  (profile.options.masked_draft() && profile.config.draft->full_layer_count() > 0)),
    };
    return contract.payload_capacity();
}

std::vector<std::uint32_t> narrow_target_widths_for(const EngineOptions& options) {
    if (!options.use_cuda_graph || options.purpose != EnginePurpose::Generation ||
        options.speculative.length_policy != SpeculativeLengthPolicy::Adaptive ||
        !is_masked_draft_backend(options.speculative.backend) ||
        options.speculative.draft_tokens < 5) {
        return {};
    }
    return {4U, options.speculative.draft_tokens + 1U};
}

SessionPlan RankExecutionDescription::to_session_plan(
    std::shared_ptr<const execution::ModelProfile> profile) const {
    SessionPlan out;
    out.batch_execution          = std::make_shared<const BatchExecutionPlan>(batches);
    out.decode_routes            = std::make_shared<const speculative::DecodeGraphRoutes>(
        *out.batch_execution, speculative_backend, draft_window, target_widths);
    out.decode_path              = decode_path;
    out.split_proposal_narrowing = split_proposal_narrowing;
    out.profile                  = std::move(profile);
    out.context                  = context;
    out.kv_pools                 = kv_pools;
    out.capacity                 = capacity;
    out.max_concurrency          = max_concurrency;
    out.prefill_chunk            = prefill_chunk;
    out.draft_window             = draft_window;
    out.speculative_backend      = speculative_backend;
    out.length_policy            = length_policy;
    out.hybrid_max_tokens        = hybrid_max_tokens;
    out.context_cache            = context_cache;
    out.kv_storage               = kv_storage;
    out.linear_state_dtype       = linear_state_dtype;
    out.proposal_head            = proposal_head;
    out.state_transfer           = state_transfer;
    out.dflash_local_group       = dflash_local_group;
    out.dflash_local_transfer    = dflash_local_transfer;
    out.vision_max_features      = vision_max_features;
    out.causal_scoring           = causal_scoring;
    return out;
}

RankPreparation::RankPreparation(const artifact::Reader& reader, RankLoadRequest request,
                                 DeviceContext& device, std::uint32_t global_rank,
                                 std::uint32_t global_node)
    : device_(device), options_(std::move(request.options)), global_rank_(global_rank),
      global_node_(global_node), control_owner_(request.control_owner) {
    if (request.artifact != reader.artifact_id()) {
        throw artifact::ArtifactError("Qwen3.5 worker artifact differs from coordinator");
    }
    options_.context_cache.host_capacity_bytes = 0;
    options_.prefill_chunk                     = std::min(
        options_.max_context, options_.prefill_chunk ? options_.prefill_chunk : 1024U);
    if (options_.purpose == EnginePurpose::CausalScoring) {
        options_.use_cuda_graph = false;
    }
    definition_ = read_definition(reader, load_options(options_));
    load_.emplace(resolve_rank_load(reader, definition_, request.rank, device_.device()));
    profile_ = execution::describe_model(*definition_, load_->expert_data());
    memory_  = {load_->materialization().device_capacity_bytes,
                load_->materialization().host_capacity_bytes, 0};
    if (request.narrow_targets) {
        target_widths_ = narrow_target_widths_for(options_);
    }
    auto planner = make_sequence_planner(profile_, device_, options_, nullptr, target_widths_);
    capacity_    = planner.capacity_plan();

    memory_.host_reservation_bytes = runtime::protocol::work_payload_memory(
        work_payload_capacity(*profile_, options_), RankProgram::capacity(), 1, false);
}

RankPreparation::RankPreparation(const artifact::Reader& reader, std::span<const std::byte> payload,
                                 const distributed::RankIdentity& identity, DeviceContext& device,
                                 const distributed::ExecutionTopology&,
                                 distributed::CommunicationContext&,
                                 const StartupObserver& observer)
    : RankPreparation(reader, decode_and_verify_request(reader, payload, identity, observer),
                      device, identity.rank, identity.node) {}

RankPreparation::~RankPreparation() = default;

void RankPreparation::materialize() {
    if (!load_) throw std::logic_error("Qwen3.5 rank weights are already materialized");
    instance_ = std::make_unique<Instance>(
        materialize_model(std::move(*load_), device_, &options_.startup_observer), profile_);
    load_.reset();
}

std::vector<distributed::GroupRequirements> RankPreparation::communication_requirements() const {
    return {};
}

runtime::RankPreparationReport RankPreparation::report(std::size_t available) const {
    if (!instance_) throw std::logic_error("Qwen3.5 parameters have not been materialized");
    const auto& model  = *instance_->model;
    const auto& s      = model.storage_stats();
    const auto reserve = graph_memory_.rebuild_estimate_bytes;
    return {.capacity             = {capacity_, available > reserve ? available - reserve : 0},
            .weight_bytes         = s.device_capacity_bytes,
            .host_weight_bytes    = s.host_capacity_bytes,
            .staging_bytes        = s.peak_staging_bytes,
            .read_bytes           = s.read_bytes,
            .upload_bytes         = s.h2d_bytes,
            .load_seconds         = s.upload_seconds,
            .device_objects       = s.device_object_count,
            .host_objects         = s.host_object_count,
            .preparation_required = options_.use_cuda_graph,
            .control_owner        = control_owner_};
}

void RankPreparation::create_probe(GraphProbeBacking backing) {
    if (!options_.use_cuda_graph) return;
    auto plan = make_graph_probe_sequence_plan(profile_, device_, options_, nullptr,
                                               target_widths_, backing);
    probe_ = create_uninitialized_rank_program(instance_->parameters, std::move(plan), device_,
                                               StartupObserver{});
}

void RankPreparation::prepare_probe() { create_probe(GraphProbeBacking::Inputs); }

void RankPreparation::prepare_rebuild_probe() {
    create_probe(GraphProbeBacking::StartupWorkingSet);
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
    if (options_.use_cuda_graph) {
        (void)instance_->program->prepare_execution(GraphPreparationMode::Capture);
    }
}

void RankPreparation::prepare(runtime::ContextCapacity capacity, std::size_t available,
                              std::size_t available_host) {
    if (!instance_ || instance_->program) {
        throw std::logic_error(
            "Qwen3.5 resource allocation requires weights and an empty resource slot");
    }
    const bool shared_memory = device_memory_snapshot(device_.device()).shared_physical_memory;
    const auto reserve       = graph_memory_.rebuild_estimate_bytes;
    available                = available > reserve ? available - reserve : 0;
    const auto reserved      = capacity_.reservation_bytes(capacity);
    const auto control =
        control_owner_ && shared_memory ? capacity_.control_bytes(capacity) : 0;
    if (reserved > available || control > available - reserved) {
        const auto required = reserved + control;
        throw std::runtime_error("Qwen3.5 rank allocation budget exceeded: required " +
                                 std::to_string(required) + " bytes, available " +
                                 std::to_string(available) + " bytes, shortfall " +
                                 std::to_string(required - available) + " bytes");
    }
    auto planner    = make_sequence_planner(profile_, device_, options_, nullptr, target_widths_);
    auto resolution = runtime::resolve_context_capacity(
        ContextCapacityPolicy::explicit_capacity(capacity.device_bytes), planner.capacity_plan(),
        available, available_host, shared_memory, control_owner_);
    auto seq      = std::move(planner).finalize(resolution.capacity());
    session_plan_ = plan_session(seq);
    resolution_   = resolution;
    instance_->program = create_uninitialized_rank_program(instance_->parameters, std::move(seq),
                                                           device_, options_.startup_observer);
}

void RankPreparation::release_resources() {
    probe_.reset();
    if (instance_ && instance_->program) {
        instance_->program->drain();
        if (const auto graph = instance_->program->graph_memory_observation()) {
            graph_memory_.rebuild_estimate_bytes =
                std::max(graph_memory_.rebuild_estimate_bytes, graph->preparation_peak_bytes);
        }
        instance_->program.reset();
    }
    session_plan_.reset();
    resolution_.reset();
}

void RankPreparation::select_narrow_targets(bool enabled, GraphMemorySummary measured) {
    if ((enabled && narrow_target_widths_for(options_).empty()) ||
        (instance_ && instance_->program)) {
        throw std::logic_error("Qwen3.5 cannot change the decode graph catalog");
    }
    probe_.reset();
    target_widths_ = enabled ? narrow_target_widths_for(options_) : std::vector<std::uint32_t>{};
    capacity_ =
        make_sequence_planner(profile_, device_, options_, nullptr, target_widths_).capacity_plan();
    graph_memory_ = measured;
}

RankExecutionDescription RankPreparation::describe() const {
    if (!instance_ || !instance_->program || !session_plan_ || !resolution_) {
        throw std::logic_error("Qwen3.5 resources have not been prepared");
    }
    const auto& s = *session_plan_;
    const auto& c = *resolution_;
    RankExecutionDescription out{
        .rank                     = global_rank_,
        .batches                  = s.batch_execution->spec(),
        .target_widths            = target_widths_,
        .decode_path              = s.decode_path,
        .split_proposal_narrowing = s.split_proposal_narrowing,
        .context                  = s.context,
        .kv_pools                 = s.kv_pools,
        .capacity                 = s.capacity,
        .max_concurrency          = s.max_concurrency,
        .prefill_chunk            = s.prefill_chunk,
        .draft_window             = s.draft_window,
        .speculative_backend      = s.speculative_backend,
        .length_policy            = s.length_policy,
        .hybrid_max_tokens        = s.hybrid_max_tokens,
        .context_cache            = s.context_cache,
        .kv_storage               = s.kv_storage,
        .linear_state_dtype       = s.linear_state_dtype,
        .proposal_head            = s.proposal_head,
        .state_transfer           = s.state_transfer,
        .dflash_local_group       = s.dflash_local_group,
        .dflash_local_transfer    = s.dflash_local_transfer,
        .vision_max_features      = s.vision_max_features,
        .causal_scoring           = s.causal_scoring,
    };
    out.observation.rank     = global_rank_;
    out.observation.node     = global_node_;
    out.observation.hardware = device_.properties().name;
    out.observation.memory   = instance_->program->observe_resources();

    out.budget = {
        .runtime_reservation_bytes         = c.runtime_reservation_bytes,
        .minimum_runtime_reservation_bytes = c.minimum_runtime_reservation_bytes,
        .context_capacity_bytes            = c.device_bytes,
        .minimum_context_capacity_bytes    = c.minimum_device_bytes,
        .context_control_budget_bytes      = c.control_reservation_bytes,
        .available_after_weights_bytes     = c.available_after_weights_bytes,
        .planned_slack_bytes               = c.planned_slack_bytes,
    };
    if (options_.use_cuda_graph) {
        auto observed     = instance_->program->graph_memory_observation();
        out.budget.graphs = graph_memory_;
        if (observed) {
            out.budget.graphs->preparation_increment_bytes = observed->preparation_increment_bytes;
            out.budget.graphs->preparation_peak_bytes      = observed->preparation_peak_bytes;
        }
    }
    return out;
}

ExecutionCodec RankPreparation::codec(const distributed::ExecutionTopology& topology) const {
    return ExecutionCodec(*definition_, describe(), topology);
}

std::unique_ptr<RankPreparation::Instance> RankPreparation::finish() && {
    if (!instance_ || !instance_->program) {
        throw std::logic_error("Qwen3.5 execution resources have not been prepared");
    }
    return std::move(instance_);
}

} // namespace ninfer::models::qwen3_5
