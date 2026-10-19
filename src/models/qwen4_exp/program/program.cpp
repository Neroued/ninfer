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

void check_selectors(context::StateSelectors slots, std::int32_t count) {
    if (slots.source < 0 || slots.source >= count || slots.destination < 0 ||
        slots.destination >= count)
        throw std::invalid_argument("Qwen4Exp state selectors exceed local storage");
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
      prefill_timer(d), compute_done(d), context_done(d) {
    if (binding.group) parallel.emplace(binding, d);
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
    WorkContract{shape, parameters.profile->config.text.vocabulary,
                 parameters.profile->public_tokens, bool(parameters.mtp), bool(parameters.vision)}
        .validate(work, kind);
    const auto check_resident = [&](const runtime::ExecutionMember& member) {
        const auto& resident = bindings[member.residency.lane.value];
        if (!resident || !same_member(resident->member, member))
            throw std::logic_error("Qwen4Exp work refers to a stale resident");
    };
    for (const auto& row : work_rows(work)) {
        check_resident(row.member);
        check_selectors(row.slots, resources.states->storage().slot_count());
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
    actions->validate(work.actions);
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
    WorkContract{shape, parameters.profile->config.text.vocabulary,
                 parameters.profile->public_tokens, bool(parameters.mtp), bool(parameters.vision)}
        .validate_commit(frame.request.work, frame.result.rows, commit);
    for (const auto& row : commit.rows)
        check_selectors(row.slots, resources.states->storage().slot_count());
    resources.compute_actions->validate(commit.actions);
}

bool RankProgramImpl::can_discard(runtime::ExecutionTicket ticket) const {
    if (!compute || compute->request.ticket != ticket) return false;
    return compute->phase == ComputePhase::PendingCommit ||
           compute->phase == ComputePhase::AwaitResult ||
           (compute->phase == ComputePhase::Decision && !compute->decision);
}

void RankProgramImpl::reset_memory_peaks() noexcept {
    resources.workspace.reset_peak();
    if (graphs) graphs->reset_workspace_peak();
    if (device_graphs) device_graphs->reset_workspace_peak();
    vision_handoff_peak_bytes = active_vision_bytes();
    vision_workspace_peak_bytes =
        vision_handoff_peak_bytes
            ? resources.plan.workspace.vision->handoff_offset_bytes + vision_handoff_peak_bytes
            : 0;
}

MemorySummary RankProgramImpl::memory_summary() const {
    const auto& plan    = resources.plan;
    const auto& weights = parameters.model.storage_stats();
    MemorySummary out;
    out.cuda_graph    = graph_memory;
    out.device        = device.device().index();
    out.max_context   = shape.max_context;
    out.prefill_chunk = shape.prefill_chunk;
    out.kv_capacity   = plan.capacity.resolved_tokens;
    out.kv_cache      = shape.kv_storage;
    out.weights       = {weights.device_capacity_bytes, weights.device_capacity_bytes,
                         weights.device_capacity_bytes};
    out.sequence      = {plan.persistent.bytes, plan.persistent.bytes, plan.persistent.bytes};
    const auto active_vision = active_vision_bytes();
    const auto active_workspace =
        active_vision ? std::max(resources.workspace.used(),
                                 plan.workspace.vision->handoff_offset_bytes + active_vision)
                      : resources.workspace.used();
    out.workspace = {plan.workspace.bytes, active_workspace,
                     std::max({resources.workspace.peak_used(), vision_workspace_peak_bytes,
                               graphs ? graphs->workspace_peak() : 0,
                               device_graphs ? device_graphs->workspace_peak() : 0})};
    if (plan.workspace.vision) {
        const auto& vision   = *plan.workspace.vision;
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(shape.max_context, qwen::kMaximumPromptVisionTokens)),
            .max_item_tokens        = vision.max_merged_tokens,
            .general_capacity_bytes = vision.general_capacity_bytes,
            .encode_peak_bytes      = vision.encode_peak_bytes,
            .handoff_offset_bytes   = vision.handoff_offset_bytes,
            .handoff_capacity_bytes = vision.handoff_capacity_bytes,
            .handoff_active_bytes   = active_vision,
            .handoff_peak_bytes     = vision_handoff_peak_bytes};
    }
    out.workspace_logical_peak_bytes = out.workspace.peak_used_bytes;
    out.runtime_reservation_bytes    = plan.device_reservation_bytes();
    out.host_context_capacity_bytes  = plan.host_context_bytes;
    for (std::uint32_t i = 0; i < resources.pool_count(); ++i) {
        const auto& pool = resources.pool(i).pages();
        out.kv_payload_bytes += pool.device_copy_work(1).payload_bytes * pool.capacity_pages();
    }
    if (!resources.experts) return out;
    const auto cache        = resources.experts->cache_usage();
    const auto& description = plan.moe->description();
    out.moe.emplace();
    auto& summary                 = *out.moe;
    summary.host_weights_bytes    = weights.host_capacity_bytes;
    summary.registered_host_bytes = weights.registered_host_bytes;
    summary.cache_bytes           = description.cache_bytes;
    summary.fixed_bytes           = description.fixed_bytes;
    summary.window_bytes          = summary.cache_bytes - summary.fixed_bytes;
    summary.window_count          = description.window_count;
    summary.used_bytes            = cache.used_bytes;
    summary.target_bytes          = description.target_bytes;
    summary.ready_bytes           = cache.ready_cache_bytes;
    summary.cpu_workers           = description.cpu_workers;
    summary.bridge_device_bytes   = description.bridge_device_bytes;
    summary.bridge_pinned_bytes   = description.bridge_pinned_bytes;
    summary.cpu_workspace_bytes   = description.cpu_workspace_bytes;
    summary.uploaded_bytes        = cache.data_bytes;
    summary.prefetched_bytes      = cache.prefetched_bytes;
    for (const auto& layer : description.layers) {
        summary.fixed_experts_per_layer.push_back(layer.fixed_experts);
        summary.target_experts_per_layer.push_back(layer.cache_experts);
        summary.ready_experts_per_layer.push_back(
            cache.ready_experts_per_bank.at(layer.bank.index));
    }
    return out;
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

runtime::ExecutionCapacity RankProgram::capacity() const noexcept { return {1, 1}; }

SessionPlan RankProgram::session_plan() const {
    auto plan = plan_session(impl_->resources.plan);
    for (std::uint32_t i = 0; i < impl_->resources.pool_count(); ++i)
        plan.context.pools[i].transfer = impl_->resources.pool(i).pages().transfer_profile();
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

MemorySummary RankProgram::memory_summary() const { return impl_->memory_summary(); }

std::vector<context::KVTransferProfile> RankProgram::context_transfer_profiles() const {
    std::vector<context::KVTransferProfile> profiles;
    profiles.reserve(impl_->resources.pool_count());
    for (std::uint32_t i = 0; i < impl_->resources.pool_count(); ++i)
        profiles.push_back(impl_->resources.pool(i).pages().transfer_profile());
    return profiles;
}

void RankProgram::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

} // namespace ninfer::models::qwen4_exp
