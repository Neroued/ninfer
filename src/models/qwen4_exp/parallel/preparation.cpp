#include "models/qwen4_exp/parallel/preparation.h"
#include "models/common/vocabulary.h"

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
        identity.ep.rank != identity.rank || identity.dp.size != 1 || identity.pp.size != 1 ||
        identity.adp.size != 1)
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
    capacity_plan_ = describe_execution_capacity(profile_, options_, device_);
    const WorkContract contract{describe_execution_shape(*profile_, options_),
                                profile_->config.text.vocabulary,
                                profile_->public_tokens,
                                bool(profile_->mtp),
                                bool(profile_->vision),
                                profile_->control_token_limit};
    host_reservation_bytes_ +=
        runtime::protocol::work_payload_memory(contract.payload_capacity(topology.ranks.size()),
                                               {1, 1}, topology.ranks.size(), control_owner_);
}

runtime::PreparationMemory RankPreparation::memory() const {
    return {load_->materialization().device_capacity_bytes,
            load_->materialization().host_capacity_bytes,
            options_.context_capacity.automatic_headroom_bytes, host_reservation_bytes_};
}

void RankPreparation::materialize() {
    if (!load_) throw std::logic_error("Qwen4Exp rank weights are already materialized");
    instance_ = std::make_unique<Instance>(
        materialize_model(std::move(*load_), device_, &options_.startup_observer), profile_);
    load_.reset();
}

void RankPreparation::prepare_communication() {
    using distributed::Collective;
    using distributed::Reduction;
    const auto plan   = plan_graph_probe(profile_, {}, options_, device_, {});
    const auto& shape = plan.shape;
    const auto hidden = std::size_t(profile_->config.text.hidden);
    const auto batch  = std::size_t(shape.max_concurrency);
    const auto decode = batch * (shape.draft_tokens + 1);
    std::vector<distributed::CollectivePreparation> work;
    for (const auto columns : {std::size_t(1), batch, decode, std::size_t(shape.prefill_chunk)}) {
        work.push_back({Collective::AllReduce, DType::BF16, hidden * columns});
        if (plan.persistent.buffers.ple_embedding)
            work.push_back({Collective::AllReduce, DType::U32,
                            profile_->config.text.ple.embedding * columns *
                                dtype_size(DType::BF16) / dtype_size(DType::U32)});
    }
    if (shape.purpose == EnginePurpose::Generation) {
        for (const auto columns : {std::size_t(1), batch, decode})
            work.push_back(
                {Collective::AllGather, DType::I64, DistributedVocabulary::candidates * columns});
        if (profile_->mtp)
            work.push_back({Collective::AllReduce, DType::U64, batch, Reduction::Max});
    } else {
        work.push_back({Collective::AllGather, DType::FP32,
                        3 * std::size_t(std::min(shape.prefill_chunk, 1024U))});
    }
    if (plan.workspace.vision)
        for (std::uint32_t root = 0; root < binding_.group->size(); ++root)
            work.push_back({Collective::Broadcast, DType::BF16,
                            plan.workspace.vision->output_bytes / dtype_size(DType::BF16),
                            Reduction::Sum, root});
    binding_.group->prepare(work, {}, device_.execution_view());
}

runtime::RankPreparationReport RankPreparation::report(std::size_t available) const {
    const auto& model  = *instance_->model;
    const auto& s      = model.storage_stats();
    const auto reserve = graph_memory_.rebuild_estimate_bytes;
    return {.capacity             = {capacity_plan_, available > reserve ? available - reserve : 0},
            .weight_bytes         = s.device_capacity_bytes + model.metadata_bytes(),
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

void RankPreparation::prepare_probe() {
    if (!options_.use_cuda_graph) return;
    create_probe(GraphProbeBacking::Inputs);
}

void RankPreparation::prepare_rebuild_probe() {
    if (!options_.use_cuda_graph) return;
    create_probe(GraphProbeBacking::StartupWorkingSet);
}

void RankPreparation::create_probe(GraphProbeBacking backing) {
    auto plan = plan_graph_probe(profile_, {}, options_, device_, {}, backing);
    probe_    = std::make_unique<RankProgram>(instance_->parameters, std::move(plan), device_,
                                              options_.startup_observer, binding_);
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
                              std::size_t available_host_bytes) {
    if (instance_->program) throw std::logic_error("Qwen4Exp rank Program is already prepared");
    auto options             = options_;
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(capacity.device_bytes);
    const auto reserve       = graph_memory_.rebuild_estimate_bytes;
    auto plan                = plan_execution(profile_, {}, options, device_,
                               available > reserve ? available - reserve : 0, {}, control_owner_,
                                              available_host_bytes);
    if (!plan) throw DeviceAllocationError("rank runtime capacity exceeds available memory");
    instance_->program = std::make_unique<RankProgram>(
        instance_->parameters, std::move(*plan), device_, options_.startup_observer, binding_);
}

void RankPreparation::release_resources() {
    if (instance_->program)
        if (const auto graph = instance_->program->graph_memory_observation())
            graph_memory_.rebuild_estimate_bytes =
                std::max(graph_memory_.rebuild_estimate_bytes, graph->preparation_peak_bytes);
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
        const auto peak      = result.budget.graphs->preparation_peak_bytes;
        result.budget.graphs = graph_memory_;
        result.budget.graphs->preparation_increment_bytes = increment;
        result.budget.graphs->preparation_peak_bytes      = peak;
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
