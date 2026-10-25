#include "models/glm5next/program/internal.h"
#include <algorithm>
#include "core/transfer.h"

namespace ninfer::models::glm5next::detail {
namespace {
bool same(const runtime::ExecutionMember& a, const runtime::ExecutionMember& b) {
    return a.sequence == b.sequence && a.residency == b.residency;
}
} // namespace

RankProgramImpl::RankProgramImpl(const execution::Parameters& p, std::unique_ptr<Resources> r,
                                 DeviceContext& d, ParallelBinding binding)
    : parameters(p), device(d), cuda(d.execution_view()), owner(std::move(r)), resources(*owner),
      shape(resources.plan.shape), timer(d, shape.gpu_timing), compute_done(d), context_done(d) {
    if (p.profile != resources.plan.profile)
        throw std::invalid_argument("GLM5Next resources have another profile");
    if (binding.group) parallel.emplace(binding, d);
    if (p.profile->local.size != (parallel ? parallel->size() : 1))
        throw std::invalid_argument("GLM5Next communication differs from weight partition");
    if (p.mtp) throw std::invalid_argument("GLM5Next eager text requires speculative backend none");
    if (shape.gpu_timing) prefill_timer.emplace(d);
}

RankProgramImpl::~RankProgramImpl() {
    try {
        drain();
    } catch (const std::logic_error&) { std::terminate(); } catch (...) {
    }
}

WorkContract RankProgramImpl::contract() const {
    return {shape, parameters.profile->config.text.vocabulary, parameters.profile->public_tokens,
            parameters.profile->local.size};
}

void RankProgramImpl::validate_work(const Work& work, runtime::FrameKind kind) const {
    contract().validate(work, kind);
    const auto rank   = parameters.profile->local.rank;
    const auto active = [&](const runtime::ExecutionMember& m) {
        const auto& binding = bindings.at(m.residency.lane.value);
        if (!binding || !same(*binding, m)) throw std::logic_error("GLM5Next stale residency");
    };
    for (const auto& row : work_rows(work)) {
        active(row.member);
        const auto offsets = row.states.at(rank);
        const auto& image  = resources.states.image(rank == row.attention_rank);
        (void)image.image(offsets.source);
        (void)image.image(offsets.destination);
        if (row.main_table.index >= std::uint32_t(resources.main.tables().matrix().ne[1]))
            throw std::invalid_argument("GLM5Next table row exceeds rank storage");
    }
    if (const auto* c = std::get_if<ContextWork>(&work.operation)) {
        std::array<bool, kMaximumConcurrency> released{};
        for (auto r : c->releases) {
            if (!bindings[r.lane.value] || bindings[r.lane.value]->residency != r)
                throw std::logic_error("GLM5Next stale residency release");
            if (compute)
                for (const auto& row : work_rows(compute->request.work))
                    if (row.member.residency == r)
                        throw std::logic_error("GLM5Next residency has live compute");
            released[r.lane.value] = true;
        }
        for (const auto& b : c->bindings) {
            const auto lane = b.member.residency.lane.value;
            if (bindings[lane] &&
                (!released[lane] || bindings[lane]->residency == b.member.residency))
                throw std::logic_error("GLM5Next residency already installed");
            for (std::size_t i = 0; i < bindings.size(); ++i)
                if (bindings[i] && !released[i] && bindings[i]->sequence == b.member.sequence)
                    throw std::logic_error("GLM5Next sequence already bound");
        }
    }
    const auto& migration =
        kind == runtime::FrameKind::Compute ? resources.compute_actions : resources.context_actions;
    migration->validate(local_actions(work.actions, rank));
}

void RankProgramImpl::start(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    if (drained) throw std::logic_error("GLM5Next Program is drained");
    const auto rank = parameters.profile->local.rank;
    if (request.ticket.frame.kind == runtime::FrameKind::Context) {
        if (context) throw std::logic_error("GLM5Next context frame is occupied");
        context          = std::make_unique<ContextFrame>();
        context->request = std::move(request);
        resources.context_actions->submit(local_actions(context->request.work.actions, rank));
        advance_context();
    } else {
        if (compute) throw std::logic_error("GLM5Next compute frame is occupied");
        compute             = std::make_unique<ComputeFrame>();
        compute->request    = std::move(request);
        compute->retirement = compute->request.command;
        compute->result.rows.resize(work_rows(compute->request.work).size());
        resources.compute_actions->submit(local_actions(compute->request.work.actions, rank));
        timer.begin();
        request_decision();
        advance_compute();
    }
}

void RankProgramImpl::validate_decision(runtime::ExecutionTicket ticket,
                                        const runtime::DecisionInput& in) const {
    if (!compute || compute->request.ticket != ticket || !compute->requested || compute->decision ||
        compute->phase == ComputePhase::Retiring || compute->phase == ComputePhase::PendingCommit)
        throw std::logic_error("GLM5Next decision has no pending request");
    contract().validate(compute->request.work, *compute->requested, in);
}

void RankProgramImpl::supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             runtime::DecisionInput&& in) {
    validate_decision(ticket, in);
    compute->decision   = std::move(in);
    compute->retirement = command;
}

void RankProgramImpl::prepare_commit(runtime::ExecutionTicket ticket, const Commit& commit) const {
    if (!compute || compute->request.ticket != ticket ||
        compute->phase != ComputePhase::PendingCommit)
        throw std::logic_error("GLM5Next commit has no pending result");
    contract().validate_commit(compute->request.work, compute->result.rows, commit);
    resources.compute_actions->validate(
        local_actions(commit.actions, parameters.profile->local.rank));
}

bool RankProgramImpl::can_discard(runtime::ExecutionTicket ticket) const {
    return compute && compute->request.ticket == ticket && compute->phase != ComputePhase::Retiring;
}

void RankProgramImpl::discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
    if (!can_discard(ticket)) throw std::logic_error("GLM5Next discard has no active execution");
    std::erase_if(events, [&](const auto& e) {
        return std::visit([&](const auto& v) { return v.ticket == ticket; }, e);
    });
    timer.resolve();
    timer.record_retirement();
    compute_done.record(cuda.stream());
    compute->phase      = ComputePhase::Retiring;
    compute->retirement = command;
}

void RankProgramImpl::drain() {
    if (drained) return;
    std::exception_ptr error;
    try {
        resources.synchronize();
    } catch (...) { error = std::current_exception(); }
    try {
        if (parallel) parallel->synchronize();
    } catch (...) {
        if (!error) error = std::current_exception();
    }
    compute.reset();
    context.reset();
    events.clear();
    for (auto& b : bindings) b.reset();
    drained = true;
    if (error) std::rethrow_exception(error);
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgramImpl::poll() {
    advance_context();
    advance_compute();
    if (events.empty()) return {};
    auto out = std::move(events.front());
    events.pop_front();
    return out;
}
} // namespace ninfer::models::glm5next::detail

namespace ninfer::models::glm5next {
RankProgram::RankProgram(const execution::Parameters& p, std::unique_ptr<Resources> r,
                         DeviceContext& d, ParallelBinding b)
    : impl_(std::make_unique<detail::RankProgramImpl>(p, std::move(r), d, b)) {}

RankProgram::~RankProgram() = default;

void RankProgram::validate_work(const Work& w, runtime::FrameKind k) const {
    impl_->validate_work(w, k);
}

void RankProgram::start(runtime::ExecutionRequest<ExecutionTypes>&& r) {
    impl_->start(std::move(r));
}

void RankProgram::validate_decision(runtime::ExecutionTicket t,
                                    const runtime::DecisionInput& d) const {
    impl_->validate_decision(t, d);
}

void RankProgram::supply(runtime::ExecutionTicket t, runtime::CommandSequence c,
                         runtime::DecisionInput&& d) {
    impl_->supply(t, c, std::move(d));
}

void RankProgram::prepare_commit(runtime::ExecutionTicket t, Commit& c) const {
    impl_->prepare_commit(t, c);
}

void RankProgram::commit(runtime::ExecutionTicket t, runtime::CommandSequence s, Commit&& c) {
    impl_->commit(t, s, std::move(c));
}

void RankProgram::discard(runtime::ExecutionTicket t, runtime::CommandSequence c) {
    impl_->discard(t, c);
}

bool RankProgram::can_discard(runtime::ExecutionTicket t) const { return impl_->can_discard(t); }

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgram::poll() { return impl_->poll(); }

void RankProgram::drain() { impl_->drain(); }

const ExecutionPlan& RankProgram::plan() const { return impl_->resources.plan; }

RankMemoryUsage RankProgram::observe_resources() const { return impl_->memory(); }

RankMemoryBudget RankProgram::resource_budget() const { return impl_->budget(); }

void RankProgram::reset_measurement_peaks() noexcept { impl_->reset_peaks(); }

} // namespace ninfer::models::glm5next

namespace ninfer::models::glm5next::detail {

void RankProgramImpl::advance_context() {
    if (!context) return;
    auto& f          = *context;
    const auto& work = std::get<ContextWork>(f.request.work.operation);
    if (!f.submitted) {
        const auto transfer   = device.transfer_view();
        const auto vocabulary = parameters.profile->public_tokens;
        for (auto r : work.releases) bindings[r.lane.value].reset();
        for (std::size_t i = 0; i < work.bindings.size(); ++i) {
            const auto& b   = work.bindings[i];
            const auto lane = b.member.residency.lane.value;
            bindings[lane]  = b.member;
            if (!resources.plan.frame.token_counts) continue;
            auto counts = resources.tensor(*resources.plan.frame.token_counts).slice(1, lane, 1);
            if (b.generated.empty())
                CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), transfer.stream()));
            else {
                auto* host =
                    static_cast<std::int32_t*>(resources.context_counts().data) + i * vocabulary;
                std::fill_n(host, vocabulary, 0);
                for (auto id : b.generated) ++host[id];
                copy_bytes_async({counts.data, counts.bytes(), transfer.device()},
                                 {host, counts.bytes(), Device::cpu(), true}, transfer);
            }
        }
        context_done.record(transfer.stream());
        f.submitted = true;
    }
    if (!context_done.ready()) return;
    const auto transfers = resources.context_actions->observations();
    ContextResult result{{transfers.begin(), transfers.end()}};
    resources.context_actions->finish();
    const auto ticket  = f.request.ticket;
    const auto command = f.request.command;
    context.reset();
    events.emplace_back(runtime::ResultReady<ExecutionTypes>{ticket, Result(std::move(result))});
    events.emplace_back(runtime::DeviceRetired{ticket, command, observation()});
}

RankExecutionObservation RankProgramImpl::observation() const {
    RankExecutionObservation out;
    out.workspace_peak_bytes   = std::max(lifetime_peak, resources.workspace.peak_used());
    out.measurement_peak_bytes = resources.workspace.peak_used();
    return out;
}

RankMemoryUsage RankProgramImpl::memory() const {
    auto out      = resources.observe();
    const auto& w = parameters.model.storage_stats();
    out.allocations.device_weights_bytes =
        w.device_capacity_bytes + parameters.model.metadata_bytes();
    out.allocations.host_weights_bytes = w.host_capacity_bytes;
    out.workspace.required_peak_bytes  = std::max(out.workspace.required_peak_bytes, lifetime_peak);
    return out;
}

RankMemoryBudget RankProgramImpl::budget() const {
    const auto& c = resources.plan.capacity;
    return {.runtime_reservation_bytes         = c.runtime_reservation_bytes,
            .minimum_runtime_reservation_bytes = c.minimum_runtime_reservation_bytes,
            .context_capacity_bytes            = c.device_bytes,
            .minimum_context_capacity_bytes    = c.minimum_device_bytes,
            .context_control_budget_bytes      = c.control_reservation_bytes};
}

void RankProgramImpl::reset_peaks() {
    lifetime_peak = std::max(lifetime_peak, resources.workspace.peak_used());
    resources.workspace.reset_peak();
    if (resources.vision) resources.vision->reset_peaks();
}

} // namespace ninfer::models::glm5next::detail
