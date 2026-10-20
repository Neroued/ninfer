#include "models/qwen4_exp/parallel/preparation.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

RankPreparation::RankPreparation(const artifact::Reader& reader, std::span<const std::byte> payload,
                                 const distributed::RankIdentity& identity, DeviceContext& device,
                                 const distributed::ExecutionTopology& topology,
                                 distributed::CommunicationContext& communication)
    : device_(device) {
    const auto role = std::find_if(topology.roles.begin(), topology.roles.end(),
                                   [](const auto& r) { return r.role == "tensor"; });
    if (role == topology.roles.end())
        throw std::invalid_argument("Qwen4Exp tensor group is absent");
    const auto& group = topology.group(role->group);
    binding_          = {&communication.group(group.id), group.index(topology.output_rank)};
    distributed::wire::Decoder in(payload);
    auto request = decode_rank_load(in);
    in.finish();
    if (request.rank.rank != identity.rank || request.rank.size != identity.tp.size ||
        identity.tp.rank != identity.rank || identity.ep.size != request.rank.size ||
        identity.ep.rank != identity.rank || identity.dp.size != 1 || identity.pp.size != 1)
        throw std::invalid_argument("Qwen4Exp rank plan differs from worker identity");
    if (request.artifact != reader.artifact_id())
        throw artifact::ArtifactError("Qwen4Exp worker artifact differs from coordinator");
    options_                = std::move(request.options);
    host_reservation_bytes_ = request.host_reservation_bytes;
    control_owner_          = request.control_owner;
    auto definition         = read_definition(reader, models::load_options(options_));
    load_.emplace(resolve_rank_load(reader, std::move(definition), std::move(request.rank),
                                    device_.device()));
    profile_ =
        execution::describe_model(load_->definition(), load_->weight_data(), load_->rank_plan());
    curve_ = describe_execution_capacity(profile_, options_, device_, control_owner_);
}

runtime::PreparationMemory RankPreparation::memory() const {
    return {load_->materialization().device_capacity_bytes,
            options_.kv_capacity.automatic_headroom_bytes, host_reservation_bytes_};
}

void RankPreparation::materialize() {
    if (!load_) throw std::logic_error("Qwen4Exp rank weights are already materialized");
    instance_ = std::make_unique<Instance>(
        materialize_model(std::move(*load_), device_, &options_.startup_observer), profile_);
    load_.reset();
}

runtime::RankPreparationReport RankPreparation::report(std::size_t available) const {
    const auto& model  = *instance_->model;
    const auto& s      = model.storage_stats();
    const auto reserve = graph_memory_.rebuild_estimate_bytes;
    return {.capacity             = {curve_, available > reserve ? available - reserve : 0},
            .weight_bytes         = s.device_capacity_bytes + model.metadata_bytes(),
            .host_weight_bytes    = s.host_capacity_bytes,
            .staging_bytes        = s.peak_staging_bytes,
            .read_bytes           = s.read_bytes,
            .upload_bytes         = s.h2d_bytes,
            .load_seconds         = s.upload_seconds,
            .device_objects       = s.device_object_count,
            .host_objects         = s.host_object_count,
            .preparation_required = options_.use_cuda_graph};
}

void RankPreparation::prepare_probe() {
    if (!options_.use_cuda_graph) return;
    auto plan = plan_graph_probe(profile_, {}, options_, device_, {});
    probe_    = std::make_unique<RankProgram>(instance_->parameters, std::move(plan), device_,
                                              options_.startup_observer, binding_);
}

void RankPreparation::calibrate() {
    if (!probe_) return;
    const auto before = device_memory_snapshot(device_.device());
    probe_->prepare_execution();
    const auto ready = device_memory_snapshot(device_.device());
    probe_->release_execution();
    const auto released                       = device_memory_snapshot(device_.device());
    graph_memory_.calibration_increment_bytes = device_memory_growth(before, ready);
    graph_memory_.calibration_retained_bytes  = device_memory_growth(before, released);
    graph_memory_.rebuild_estimate_bytes =
        std::max<std::int64_t>(0, device_memory_growth(released, ready));
    probe_.reset();
    device_.synchronize();
}

void RankPreparation::prepare_execution() { instance_->program->prepare_execution(); }

void RankPreparation::prepare(runtime::ContextCapacity capacity, std::size_t available) {
    if (instance_->program) throw std::logic_error("Qwen4Exp rank Program is already prepared");
    auto options = options_;
    options.kv_capacity =
        KvCapacityPolicy::explicit_capacity(curve_.resolved_tokens(capacity.primary_page_groups));
    options.context_cache.extra_state_slots = capacity.device_state_slots - options.max_concurrency;
    const auto reserve                      = graph_memory_.rebuild_estimate_bytes;
    auto plan                               = plan_execution(profile_, {}, options, device_,
                               available > reserve ? available - reserve : 0, {}, control_owner_);
    if (!plan) throw DeviceAllocationError("rank runtime capacity exceeds available memory");
    instance_->program = std::make_unique<RankProgram>(
        instance_->parameters, std::move(*plan), device_, options_.startup_observer, binding_);
}

void RankPreparation::release_program() {
    instance_->program.reset();
    device_.synchronize();
}

RankExecutionDescription RankPreparation::describe() const {
    const auto& program = *instance_->program;
    RankExecutionDescription result{program.session_plan(), program.capacity(),
                                    program.resource_budget()};
    result.observation.rank     = profile_->local.rank;
    result.observation.memory   = program.observe_resources();
    result.observation.hardware = device_.properties().name;
    result.observation.physical_memory.emplace();
    if (result.budget.graphs) {
        const auto increment = result.budget.graphs->preparation_increment_bytes;
        result.budget.graphs = graph_memory_;
        result.budget.graphs->preparation_increment_bytes = increment;
    }
    return result;
}

ExecutionCodec RankPreparation::codec(const distributed::ExecutionTopology& topology) const {
    return ExecutionCodec(instance_->model->definition(), describe(), topology);
}

std::unique_ptr<RankPreparation::Instance> RankPreparation::finish() && {
    if (!instance_ || !instance_->program)
        throw std::logic_error("Qwen4Exp rank Program has not been prepared");
    return std::move(instance_);
}

} // namespace ninfer::models::qwen4_exp
