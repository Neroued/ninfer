#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/program/device_graphs.h"
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
                                 ParallelBinding binding)
    : parameters(p), device(d), cuda(d.execution_view()),
      resources(p, std::move(plan), d, observer), ple_inputs(d), ple_ready(d), draft_copied(d),
      draft_output(resources.plan.pinned_draft_bytes), shape(resources.plan.shape),
      prefill_timer(d), gpu_timer(d, shape.gpu_timing), compute_done(d), context_done(d) {
    if (binding.group) parallel.emplace(binding);
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
    device_graphs.reset();
    commit_graphs.reset();
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
                 parameters.profile->control_token_limit}
        .validate(work, kind);
    const auto check_resident = [&](const runtime::ExecutionMember& member) {
        const auto& resident = bindings[member.residency.lane.value];
        if (!resident || !same_member(resident->member, member))
            throw std::logic_error("Qwen4Exp work refers to a stale resident");
    };
    for (const auto& row : work_rows(work)) {
        check_resident(row.member);
        check_selectors(row.states.at(parameters.profile->local.rank), resources.states->storage());
        check_table(row.main_table, 0, *resources.main);
        if (parameters.mtp) check_table(row.mtp_table, 1, *resources.mtp);
    }
    if (const auto* forward = std::get_if<ForwardWork>(&work.operation);
        forward && forward->prefetch)
        check_resident(forward->prefetch->member);
    if (const auto* context = std::get_if<ContextWork>(&work.operation)) {
        std::array<bool, kMaximumConcurrency> released{};
        for (const auto residency : context->releases) {
            const auto lane = residency.lane.value;
            if (!bindings[lane] || bindings[lane]->member.residency != residency)
                throw std::logic_error("Qwen4Exp release refers to a stale resident");
            released[lane] = true;
            if (compute)
                for (const auto& row : work_rows(compute->request.work))
                    if (row.member.residency == residency)
                        throw std::logic_error("Qwen4Exp resident still has live compute");
        }
        for (const auto& resident : context->bindings) {
            const auto lane = resident.member.residency.lane.value;
            if ((bindings[lane] && !released[lane]) ||
                (bindings[lane] && bindings[lane]->member.residency == resident.member.residency))
                throw std::logic_error("Qwen4Exp residency is already installed");
            for (std::size_t i = 0; i < bindings.size(); ++i)
                if (bindings[i] && !released[i] &&
                    bindings[i]->member.sequence == resident.member.sequence)
                    throw std::logic_error("Qwen4Exp sequence is already resident");
        }
    }
    const auto& actions =
        kind == runtime::FrameKind::Context ? resources.context_actions : resources.compute_actions;
    actions->validate(local_actions(work.actions, parameters.profile->local.rank));
}

void RankProgramImpl::validate_decision(runtime::ExecutionTicket ticket,
                                        const runtime::DecisionInput& input) const {
    if (!compute || compute->request.ticket != ticket || !compute->decision_requested ||
        compute->decision || input.stage != 0 || compute->phase == ComputePhase::PendingCommit ||
        compute->phase == ComputePhase::Retiring)
        throw std::logic_error("Qwen4Exp decision does not match the active execution");
    const auto& frame = *compute;
    std::array<std::uint32_t, kMaximumConcurrency> positions{};
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
    if (!compute || compute->request.ticket != ticket ||
        compute->phase != ComputePhase::PendingCommit)
        throw std::logic_error("Qwen4Exp commit does not match the pending execution");
    const auto& frame = *compute;
    WorkContract{shape,
                 parameters.profile->config.text.vocabulary,
                 parameters.profile->public_tokens,
                 bool(parameters.mtp),
                 bool(parameters.vision),
                 parameters.profile->control_token_limit}
        .validate_commit(frame.request.work, frame.result.rows, commit);
    for (const auto& row : commit.rows)
        check_selectors(row.states.at(parameters.profile->local.rank), resources.states->storage());
    resources.compute_actions->validate(
        local_actions(commit.actions, parameters.profile->local.rank));
}

bool RankProgramImpl::can_discard(runtime::ExecutionTicket ticket) const {
    if (!compute || compute->request.ticket != ticket) return false;
    return compute->phase == ComputePhase::PendingCommit ||
           compute->phase == ComputePhase::AwaitResult ||
           (compute->phase == ComputePhase::Decision && !compute->decision);
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {
RankProgram::RankProgram(const execution::Parameters& parameters, ExecutionPlan plan,
                         DeviceContext& device, const StartupObserver& observer,
                         ParallelBinding binding)
    : impl_(std::make_unique<detail::RankProgramImpl>(parameters, std::move(plan), device, observer,
                                                      binding)) {}

void RankProgram::prepare_execution() {
    if (impl_->shape.cuda_graph) { impl_->initialize_graphs(); }
}

void RankProgram::release_execution() { impl_->release_graphs(); }

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
