#include "models/qwen4_exp/program/internal.h"
#include "core/startup.h"

#include <algorithm>
#include <type_traits>

namespace ninfer::models::qwen4_exp::detail {
namespace {
bool same_member(const runtime::ExecutionMember& a, const runtime::ExecutionMember& b) {
    return a.sequence == b.sequence && a.residency == b.residency;
}

void check_selectors(context::StateSelectors offsets, const StateImageStorage& storage) {
    (void)storage.image(offsets.source);
    (void)storage.image(offsets.destination);
}

void check_table(context::TableRow row, std::uint32_t pool, const QsaCache& cache) {
    if (!row.valid() || row.pool_id != pool ||
        row.index >= static_cast<std::uint32_t>(cache.tables().row_count()))
        throw std::invalid_argument("Qwen4Exp table row exceeds its local pool");
}
} // namespace

RankProgramImpl::RankProgramImpl(const execution::Parameters& p, ExecutionPlan plan,
                                 DeviceContext& d, const StartupObserver& observer,
                                 ParallelBinding binding,
                                 std::unique_ptr<moe::ResidentExpertExchange> expert_exchange,
                                 const distributed::Group* ple_group, PipelineBinding pipeline)
    : parameters(p), device(d), cuda(d.execution_view()), exchange(std::move(expert_exchange)),
      resources(p, std::move(plan), d, observer), ple_inputs(d), ple_ready(d), draft_copied(d),
      draft_output(resources.plan.pinned_draft_bytes), shape(resources.plan.shape),
      prefill_timer(d), gpu_timer(d, shape.gpu_timing), compute_done(d), context_done(d),
      bindings(shape.max_concurrency), input_rows(shape.max_concurrency),
      mtp_rows(shape.max_concurrency), valid_rows(shape.max_concurrency) {
#if NINFER_ENABLE_DISTRIBUTED
    const auto& persistent = resources.plan.persistent;
    if (pipeline.input)
        pipeline_input.emplace(*pipeline.input, d,
                               persistent.pipeline_input.bind(resources.persistent()));
    if (pipeline.output)
        pipeline_output.emplace(*pipeline.output, d,
                                persistent.pipeline_output.bind(resources.persistent()));
    if (bool(pipeline_input) == p.profile->input_stage() ||
        bool(pipeline_output) == p.profile->output_stage())
        throw std::logic_error("Qwen4Exp pipeline endpoints differ from the layer assignment");
#endif
    retained.resize(runtime::ExecutionCapacity::pipeline(p.profile->stage.size,
                                                         shape.max_concurrency,
                                                         shape.purpose == EnginePurpose::Generation)
                        .compute_frames);
    if (binding.group) parallel.emplace(binding);
    if (ple_group) ple_merge.emplace(ParallelBinding{ple_group});
    if (parallel) parallel->track_graph_group(ple_group);
    if (exchange && shape.cuda_graph) {
        service_graphs.emplace(exchange->group());
        if (parallel) parallel->track_graph_group(&exchange->group());
    }
    if (bool(ple_merge) != (parameters.profile->owns_ple &&
                            parameters.profile->ple_placement == ple::Placement::Sharded))
        throw std::logic_error("Qwen4Exp PLE merge group differs from table placement");
    if (static_cast<bool>(exchange) != static_cast<bool>(parameters.profile->exchange))
        throw std::logic_error("Qwen4Exp Program is missing its prepared expert route");
    if (parameters.profile->local.size != (parallel ? parallel->size() : 1))
        throw std::logic_error("Qwen4Exp Program communication differs from its weight partition");
    if (resources.ple_prepare) host_ple = std::make_unique<PlePipeline>(resources, d);
}

RankProgramImpl::~RankProgramImpl() {
    try {
        drain();
    } catch (const std::logic_error&) { std::terminate(); } catch (...) {
    }
    graphs.reset();
    for (auto& resident : bindings)
        if (resident) retire_vision(*resident);
}

ResidentBinding& RankProgramImpl::binding(runtime::ResidencyId residency) {
    if (residency.lane.value >= shape.max_concurrency)
        throw std::logic_error("Qwen4Exp residency lane is not available");
    auto& resident = bindings[residency.lane.value];
    if (!resident || resident->member.residency != residency)
        throw std::logic_error("stale Qwen4Exp residency");
    return *resident;
}

void RankProgramImpl::validate_work(const Work& work, runtime::FrameKind kind) const {
    WorkContract{shape,
                 parameters.profile->config.text.vocabulary,
                 parameters.profile->public_tokens,
                 bool(parameters.mtp),
                 bool(parameters.vision),
                 parameters.profile->control_token_limit,
                 resources.plan.batches,
                 resources.plan.decode_routes}
        .validate(work, kind);
    if (kind == runtime::FrameKind::Compute) {
        const bool needs_step =
            exchange && (std::holds_alternative<moe::ExpertServiceWork>(work.operation) ||
                         plan_expert_step(work, shape.prefill_chunk).has_value());
        if (needs_step != work.exchange.has_value() ||
            (work.exchange && work.exchange->id != last_expert_step + 1))
            throw std::invalid_argument(
                "Qwen4Exp expert step differs from the prepared communication order");
        if (std::holds_alternative<moe::ExpertServiceWork>(work.operation)) return;
    }
    const auto check_resident = [&](const runtime::ExecutionMember& member) {
        const auto& resident = bindings[member.residency.lane.value];
        if (!resident || !same_member(resident->member, member))
            throw std::logic_error("Qwen4Exp work refers to a stale resident");
    };
    const auto busy = [&](runtime::ResidencyId id) {
        const auto contains = [&](const auto& request) {
            return std::ranges::any_of(work_rows(request.work),
                                       [&](const auto& row) { return row.member.residency == id; });
        };
        return (compute && contains(compute->request)) ||
               std::ranges::any_of(
                   retained, [&](const auto& slot) { return slot && contains(slot->request); }) ||
               std::ranges::any_of(waiting, contains);
    };
    for (const auto& row : work_rows(work)) {
        if (busy(row.member.residency))
            throw std::logic_error("Qwen4Exp residency has live compute");
        check_resident(row.member);
        check_selectors(row.states.at(parameters.profile->replica_rank()),
                        resources.states->storage());
        if (resources.main->has_pages()) check_table(row.main_table, 0, *resources.main);
        if (parameters.mtp) check_table(row.mtp_table, 1, *resources.mtp);
    }
    if (const auto* forward = std::get_if<ForwardWork>(&work.operation);
        forward && forward->prefetch)
        check_resident(forward->prefetch->member);
    if (const auto* context = std::get_if<ContextWork>(&work.operation)) {
        std::vector<std::uint32_t> released;
        released.reserve(context->releases.size());
        for (const auto residency : context->releases) {
            const auto lane = residency.lane.value;
            if (!bindings[lane] || bindings[lane]->member.residency != residency)
                throw std::logic_error("Qwen4Exp release refers to a stale resident");
            released.push_back(lane);
            if (busy(residency)) throw std::logic_error("Qwen4Exp resident still has live compute");
        }
        std::sort(released.begin(), released.end());
        const auto is_released = [&](std::uint32_t lane) {
            return std::binary_search(released.begin(), released.end(), lane);
        };
        for (const auto& resident : context->bindings) {
            const auto lane = resident.member.residency.lane.value;
            if ((bindings[lane] && !is_released(lane)) ||
                (bindings[lane] && bindings[lane]->member.residency == resident.member.residency))
                throw std::logic_error("Qwen4Exp residency is already installed");
            for (std::size_t i = 0; i < bindings.size(); ++i)
                if (bindings[i] && !is_released(i) &&
                    bindings[i]->member.sequence == resident.member.sequence)
                    throw std::logic_error("Qwen4Exp sequence is already resident");
        }
    }
    const auto& actions =
        kind == runtime::FrameKind::Context ? resources.context_actions : resources.compute_actions;
    actions->validate(local_actions(work.actions, parameters.profile->replica_rank()));
}

void RankProgramImpl::validate_decision(runtime::ExecutionTicket ticket,
                                        const runtime::DecisionInput& input) const {
    if (!compute || compute->request.ticket != ticket || !compute->decision_requested ||
        compute->decision || input.stage != 0 || compute->phase == ComputePhase::PendingCommit ||
        compute->phase == ComputePhase::Retiring)
        throw std::logic_error("Qwen4Exp decision does not match the active execution");
    const auto& frame = *compute;
    std::vector<std::uint32_t> positions(frame.rows.size());
    for (std::size_t b = 0; b < frame.rows.size(); ++b)
        if (frame.rows[b].constrained)
            positions[b] = frame.verify ? frame.verify->extents[b] + 1 : 1;
    validate_decision_rows(input, 0, parameters.profile->public_tokens,
                           {positions.data(), frame.rows.size()});
}

void RankProgramImpl::supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             runtime::DecisionInput&& input) {
    if (!compute || compute->request.ticket != ticket)
        throw std::logic_error("Qwen4Exp decision refers to an inactive execution");
    compute->decision   = std::move(input);
    compute->retirement = command;
}

void RankProgramImpl::prepare_commit(runtime::ExecutionTicket ticket, const Commit& commit) const {
    const auto* active = find_frame(ticket);
    if (!active || active->deferred_commit || active->deferred_discard ||
        active->phase != ComputePhase::PendingCommit)
        throw std::logic_error("Qwen4Exp commit has no pending result");
    const auto& frame = *active;
    if (!parameters.profile->output_stage()) {
        const auto rows = work_rows(frame.request.work);
        if (rows.size() != commit.rows.size())
            throw std::logic_error("pipeline commit changed membership");
        for (std::size_t i = 0; i < rows.size(); ++i)
            if (!same_member(rows[i].member, commit.rows[i].member))
                throw std::logic_error("pipeline commit changed residency");
    } else
        WorkContract{shape,
                     parameters.profile->config.text.vocabulary,
                     parameters.profile->public_tokens,
                     bool(parameters.mtp),
                     bool(parameters.vision),
                     parameters.profile->control_token_limit,
                     resources.plan.batches,
                     resources.plan.decode_routes}
            .validate_commit(frame.request.work, frame.result.rows, commit);
    if (std::holds_alternative<moe::ExpertServiceWork>(frame.request.work.operation)) return;
    for (const auto& row : commit.rows)
        check_selectors(row.states.at(parameters.profile->replica_rank()),
                        resources.states->storage());
    resources.compute_actions->validate(
        local_actions(commit.actions, parameters.profile->replica_rank()));
}

bool RankProgramImpl::can_discard(runtime::ExecutionTicket ticket) const {
    const auto* f = find_frame(ticket);
    if (!f || f->deferred_commit || f->deferred_discard) return false;
    return f->phase == ComputePhase::PendingCommit ||
           (f->phase == ComputePhase::AwaitResult && parameters.profile->stage.size == 1 &&
            !f->request.work.exchange) ||
           (f->phase == ComputePhase::Decision && !f->decision);
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {
RankProgram::RankProgram(const execution::Parameters& parameters, ExecutionPlan plan,
                         DeviceContext& device, const StartupObserver& observer,
                         ParallelBinding binding,
                         std::unique_ptr<moe::ResidentExpertExchange> expert_exchange,
                         const distributed::Group* ple_group, PipelineBinding pipeline)
    : impl_(std::make_unique<detail::RankProgramImpl>(parameters, std::move(plan), device, observer,
                                                      binding, std::move(expert_exchange),
                                                      ple_group, pipeline)) {}

GraphPreparationSample RankProgram::prepare_execution(GraphPreparationMode mode) {
    return impl_->initialize_graphs(mode);
}

void RankProgram::release_execution() { impl_->release_graphs(); }

runtime::ExecutionCapacity RankProgram::capacity() const noexcept {
    return {static_cast<std::uint32_t>(impl_->retained.size()), 1};
}

RankProgram::~RankProgram() = default;

SessionPlan RankProgram::session_plan() const {
    auto plan = plan_session(impl_->resources.plan);
    return plan;
}

void RankProgram::validate_work(const Work& work, runtime::FrameKind kind) const {
    impl_->validate_work(work, kind);
}

void RankProgram::start(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    impl_->start(std::move(request));
}

void RankProgram::validate_decision(runtime::ExecutionTicket ticket,
                                    const runtime::DecisionInput& input) const {
    impl_->validate_decision(ticket, input);
}

void RankProgram::supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                         runtime::DecisionInput&& input) {
    impl_->supply(ticket, command, std::move(input));
}

void RankProgram::prepare_commit(runtime::ExecutionTicket ticket, Commit& commit) const {
    impl_->prepare_commit(ticket, commit);
}

void RankProgram::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                         Commit&& commit) {
    impl_->commit(ticket, command, std::move(commit));
}

bool RankProgram::can_discard(runtime::ExecutionTicket ticket) const {
    return impl_->can_discard(ticket);
}

void RankProgram::discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
    impl_->discard(ticket, command);
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgram::poll() { return impl_->poll(); }

void RankProgram::drain() { impl_->drain(); }

std::optional<GraphMemorySummary> RankProgram::graph_memory_observation() const {
    return impl_->graph_memory;
}

RankMemoryBudget RankProgram::resource_budget() const {
    RankMemoryBudget result;
    const auto& plan                         = impl_->resources.plan;
    result.runtime_reservation_bytes         = plan.device_reservation_bytes();
    result.minimum_runtime_reservation_bytes = plan.capacity.minimum_runtime_reservation_bytes;
    result.context_capacity_bytes            = plan.capacity.device_bytes;
    result.minimum_context_capacity_bytes    = plan.capacity.minimum_device_bytes;
    result.context_control_budget_bytes      = plan.capacity.metadata_budget_bytes;
    result.graphs                            = impl_->graph_memory;
    return result;
}

RankMemoryUsage RankProgram::observe_resources() const { return impl_->observe_resources(); }

MemorySpan RankProgram::input_storage() {
    return impl_->resources.vision ? impl_->resources.vision->staging() : MemorySpan{};
}

void RankProgram::reset_measurement_peaks() noexcept { impl_->reset_measurement_peaks(); }

} // namespace ninfer::models::qwen4_exp
