#include "models/qwen4_exp/parallel/preparation.h"
#include "models/common/vocabulary.h"
#include "core/platform.h"
#include "distributed/pipeline_link.h"
#include "models/common/moe/exchange.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

RankPreparation::RankPreparation(const artifact::Reader& reader, std::span<const std::byte> payload,
                                 const distributed::RankIdentity& identity, DeviceContext& device,
                                 const distributed::ExecutionTopology& topology,
                                 distributed::CommunicationContext& communication,
                                 const StartupObserver& observer)
    : device_(device), reader_(reader) {
    distributed::wire::Decoder in(payload);
    auto request = decode_rank_load(in, identity, reader.artifact_id());
    in.finish();
    if (identity.tp.size > 1) {
        const auto& group    = topology.role_group("tensor", identity.rank);
        const auto result_id =
            topology.replica_for_rank(identity.rank).stage_ranks.at(identity.pp.rank).front();
        binding_             = {&communication.group(group.id), group.index(result_id)};
    }
    batches_ = std::make_shared<const BatchExecutionPlan>(std::move(request.batches));
    options_ = std::move(request.options);
    options_.startup_observer = observer;
    control_owner_            = request.control_owner;
    auto definition           = read_definition(reader, models::load_options(options_));
    const distributed::GroupDescription* expert_group = nullptr;
    if (identity.ep.size > identity.tp.size) {
        if (!request.rank.expert_domain)
            throw std::invalid_argument("missing expert exchange placement");
        const auto& group    = topology.role_group("expert", identity.rank);
        expert_group         = &group;
        expert_group_        = &communication.group(group.id);
    }
    placements_ = request.rank.ple_candidates;
    if (request.rank.ple_shards !=
        (request.rank.owns_ple ? ple::shard_range(request.rank.ple_placement,
                                                  definition->config().text.ple.shards, identity.tp)
                               : IndexRange{}))
        throw std::invalid_argument(
            "Qwen4Exp PLE shard range differs from its consumer coordinate");
    if (request.rank.owns_ple &&
        std::ranges::find(placements_, ple::Placement::Sharded) != placements_.end()) {
        const auto& consumers = topology.role_group("ple", identity.rank);
        ple_group_            = &communication.group(consumers.id);
        if (consumers.index(identity.rank) != identity.tp.rank ||
            consumers.members !=
                topology.replica_for_rank(identity.rank).stage_ranks.at(identity.pp.rank))
            throw std::invalid_argument(
                "Qwen4Exp PLE shards must cover consumers of the same windows in order");
    }
    load_.emplace(resolve_rank_load(reader, std::move(definition), std::move(request.rank),
                                    device_.device()));
    auto profile = execution::describe_model(load_->definition(), load_->weight_data(), load_->expert_data(),
                                            load_->rank_plan());
    if (expert_group) {
        const auto peer = expert_group->index(identity.rank);
        auto plan = std::make_shared<moe::ResidentExchangePlan>(load_->rank_plan().expert_domain, *expert_group, peer,
                                profile->config.text.hidden, profile->config.text.moe.top_k);
        const auto bind = [&](execution::MoeProfile& layer) {
            layer.exchange_workspace_bytes = plan->stage(layer.bank).softmax_workspace_bytes(peer, layer.math);
        };
        for (auto& layer : profile->text.layers) bind(layer.moe);
        if (profile->mtp) bind(profile->mtp->layer.moe);
        expert_requirements_    = plan->requirements();
        profile->exchange_bytes = plan->bytes();
        profile->exchange = std::move(plan);
    }
    profile_ = std::move(profile);
    const auto shape = describe_execution_shape(*profile_, options_);
    const auto bind_link =
        [&](const distributed::GroupDescription* edge) -> const distributed::Group* {
        if (!edge) return nullptr;
        pipeline_requirements_.push_back(distributed::PpStageLink::requirements(
            *edge, identity.rank,
            pipeline_packet_bytes(*profile_,
                                  std::max(shape.prefill_chunk, shape.max_concurrency))));
        return &communication.group(edge->id);
    };
    pipeline_ = {bind_link(topology.pipeline_input(identity.rank)),
                 bind_link(topology.pipeline_output(identity.rank))};
    const WorkContract contract{describe_execution_shape(*profile_, options_),
                                profile_->config.text.vocabulary,
                                profile_->public_tokens,
                                bool(profile_->mtp),
                                bool(profile_->vision),
                                profile_->control_token_limit,
                                batches_};
    host_reservation_bytes_ = runtime::protocol::work_payload_memory(
        contract.payload_capacity(profile_->replica_size()),
        runtime::ExecutionCapacity::pipeline(identity.pp.size, shape.max_concurrency,
                                             shape.purpose == EnginePurpose::Generation),
        profile_->replica_size(), false);
}

runtime::PreparationMemory RankPreparation::memory() const {
    return {load_->materialization().device_capacity_bytes,
            load_->materialization().host_capacity_bytes, host_reservation_bytes_};
}

void RankPreparation::materialize() {
    if (!load_) throw std::logic_error("Qwen4Exp rank weights are already materialized");
    instance_ = std::make_unique<Instance>(
        materialize_weights(std::move(*load_), device_, &options_.startup_observer), profile_);
    load_.reset();
}

runtime::PlacementOffer RankPreparation::offer_placement(std::size_t id) const {
    const auto& model = *instance_->model;
    const auto layout = placements_.at(id);
    if (!profile_->owns_ple)
        return {describe_execution_capacity(profile_, options_, batches_, device_, {}), 0, 0};
    auto table = plan_ple_table(reader_, model.definition(), layout, model.rank_plan().placement.tp,
                                device_.device());
    auto profile           = std::make_shared<execution::ModelProfile>(*profile_);
    profile->ple_placement = layout;
    auto plan = describe_execution_capacity(profile, options_, batches_, device_, table.geometry());
    plan.fixed_device_bytes += table.device_bytes();
    plan.fixed_host_bytes += table.host_bytes();
    if (device_memory_snapshot(device_.device()).shared_physical_memory)
        plan.fixed_device_bytes += table.host_bytes();
    return {std::move(plan), table.device_bytes(), table.host_bytes()};
}

void RankPreparation::materialize_placement(std::size_t id) {
    auto& model       = *instance_->model;
    const auto layout = placements_.at(id);
    if (!profile_->owns_ple) {
        capacity_plan_ = describe_execution_capacity(profile_, options_, batches_, device_, {});
        return;
    }
    auto table = plan_ple_table(reader_, model.definition(), layout, model.rank_plan().placement.tp,
                                device_.device());
    ple_geometry_           = table.geometry();
    profile_->ple_placement = layout;
    capacity_plan_ =
        describe_execution_capacity(profile_, options_, batches_, device_, ple_geometry_);
    model.prepare_ple(std::move(table), device_, &options_.startup_observer);
}

void RankPreparation::release_placement() {
    release_resources();
    instance_->model->release_ple();
    graph_memory_  = {};
    capacity_plan_ = {};
}

std::vector<distributed::GroupRequirements> RankPreparation::communication_requirements() const {
    auto requirements = pipeline_requirements_;
    if (expert_requirements_) requirements.push_back(*expert_requirements_);
    if (!binding_.group) return requirements;
    using distributed::Collective;
    using distributed::Reduction;
    const auto plan   = plan_graph_probe(profile_, ple_geometry_, options_, batches_, device_, {});
    const auto& shape = plan.shape;
    const auto hidden = std::size_t(profile_->config.text.hidden);
    const auto batch  = std::size_t(shape.max_concurrency);
    const auto decode = batch * (shape.draft_tokens + 1);
    distributed::GroupRequirements result{binding_.group->id(), {}, {}};
    auto& work         = result.collectives;
    const auto columns = std::max({batch, decode, std::size_t(shape.prefill_chunk)});
    work.push_back({Collective::AllReduce, DType::BF16, hidden, hidden * columns, hidden});
    const auto sequence = execution::sequence_parallel_policy(profile_->local.size);
    if (pipeline_.input) {
        const auto elements = StageBoundaryField::transfer_elements(
            profile_->config.text.residual_width(), TpLayout::Replicated, profile_->local.size);
        work.push_back({Collective::AllGather, DType::BF16, 1, elements * columns, 1});
    }
    if (sequence.minimum_columns &&
        shape.prefill_chunk >= std::uint32_t(sequence.minimum_columns)) {
        const auto local = sequence.partition(dimension(shape.prefill_chunk), 0).segment;
        work.push_back({Collective::ReduceScatter, DType::BF16, hidden, hidden * local, hidden});
        work.push_back({Collective::AllGather, DType::BF16, hidden, hidden * local, hidden});
        if (profile_->mtp) {
            const auto conditioning = std::size_t(profile_->config.text.residual_width());
            for (std::uint32_t owner = 0; owner < profile_->local.size; ++owner)
                work.push_back({Collective::Broadcast, DType::BF16, conditioning,
                                conditioning * batch, 1, Reduction::Sum, owner});
        }
    }
    if (profile_->owns_ple && profile_->ple_placement == ple::Placement::Sharded) {
        const auto words =
            profile_->config.text.ple.embedding * dtype_size(DType::BF16) / dtype_size(DType::U32);
        work.push_back({Collective::AllReduce, DType::U32, words, words * columns, words});
    }
    if (profile_->output_stage() && shape.purpose == EnginePurpose::Generation) {
        constexpr auto candidates = std::size_t(DistributedVocabulary::candidates);
        work.push_back(
            {Collective::AllGather, DType::I64, candidates, candidates * decode, candidates});
        if (profile_->mtp) {
            constexpr auto records = std::size_t(DistributedVocabulary::candidates_with_lse);
            work.push_back({Collective::AllGather, DType::U64, records, records * batch, records});
            work.push_back({Collective::AllGather, DType::U64, 2, 2 * batch, 2});
        }
    } else if (profile_->output_stage()) {
        work.push_back({Collective::AllGather, DType::FP32, 3,
                        3 * std::size_t(std::min(shape.prefill_chunk, 1024U)), 3});
    }
    if (plan.workspace.vision)
        for (std::uint32_t root = 0; root < binding_.group->size(); ++root)
            work.push_back({Collective::Broadcast, DType::BF16, hidden,
                            plan.workspace.vision->output_bytes / dtype_size(DType::BF16), hidden,
                            Reduction::Sum, root});
    if (!requirements.empty() && requirements.front().group == result.group)
        requirements.front().collectives = std::move(result.collectives);
    else
        requirements.push_back(std::move(result));
    return requirements;
}

runtime::RankPreparationReport RankPreparation::report(std::size_t available) const {
    const auto& model  = *instance_->model;
    const auto& s      = model.storage_stats();
    const auto reserve = graph_memory_.rebuild_estimate_bytes;
    return {.capacity             = {capacity_plan_, available > reserve ? available - reserve : 0},
            .weight_bytes         = s.device_capacity_bytes + model.metadata_bytes(),
            .host_weight_bytes    = s.host_capacity_bytes + model.ple_host_bytes(),
            .staging_bytes        = s.peak_staging_bytes,
            .read_bytes           = s.read_bytes,
            .upload_bytes         = s.h2d_bytes,
            .load_seconds         = s.upload_seconds,
            .device_objects       = s.device_object_count,
            .host_objects         = s.host_object_count,
            .preparation_required = options_.use_cuda_graph,
            .control_owner        = control_owner_,
            .graph_groups         = options_.use_cuda_graph && expert_requirements_
                                        ? std::vector<distributed::GroupId>{expert_requirements_->group}
                                        : std::vector<distributed::GroupId>{}};
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
    auto plan = plan_graph_probe(profile_, ple_geometry_, options_, batches_, device_, {}, backing);
    probe_    = std::make_unique<RankProgram>(
        instance_->parameters, std::move(plan), device_, options_.startup_observer, binding_,
        expert_group_ ? profile_->exchange->prepare(*expert_group_) : nullptr,
        profile_->owns_ple && profile_->ple_placement == ple::Placement::Sharded ? ple_group_
                                                                                    : nullptr,
        pipeline_);
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
    auto plan                = plan_execution(profile_, ple_geometry_, options, batches_, device_,
                               available > reserve ? available - reserve : 0, {}, control_owner_,
                                              available_host_bytes);
    if (!plan) {
        const bool shared_memory = device_memory_snapshot(device_.device()).shared_physical_memory;
        const auto control =
            control_owner_ && shared_memory ? capacity_plan_.control_bytes(capacity) : 0;
        const auto required  = capacity_plan_.reservation_bytes(capacity) + control;
        const auto remaining = available > reserve ? available - reserve : 0;
        throw runtime::ContextBudgetError(
            "Qwen4Exp rank allocation budget exceeded: required " + std::to_string(required) +
            " bytes, available " + std::to_string(remaining) + " bytes, shortfall " +
            std::to_string(required - std::min(required, remaining)) + " bytes");
    }
    instance_->program = std::make_unique<RankProgram>(
        instance_->parameters, std::move(*plan), device_, options_.startup_observer, binding_,
        expert_group_ ? profile_->exchange->prepare(*expert_group_) : nullptr,
        profile_->owns_ple && profile_->ple_placement == ple::Placement::Sharded ? ple_group_
                                                                                 : nullptr,
        pipeline_);
}

void RankPreparation::release_resources() {
    const bool active = probe_ || (instance_ && instance_->program);
    probe_.reset();
    if (instance_ && instance_->program)
        if (const auto graph = instance_->program->graph_memory_observation())
            graph_memory_.rebuild_estimate_bytes =
                std::max(graph_memory_.rebuild_estimate_bytes, graph->preparation_peak_bytes);
    if (instance_) instance_->program.reset();
    if (active) device_.synchronize();
}

RankExecutionDescription RankPreparation::describe() const {
    const auto& program = *instance_->program;
    RankExecutionDescription result{program.session_plan(), program.capacity(),
                                    program.resource_budget()};
    result.observation.rank     = instance_->model->rank_plan().placement.global_rank;
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
