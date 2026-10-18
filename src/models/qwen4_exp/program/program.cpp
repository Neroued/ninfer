#include "models/qwen4_exp/program/internal.h"
#include "core/startup.h"

#include <algorithm>
#include <type_traits>

namespace ninfer::models::qwen4_exp::detail {
namespace {
std::span<const WorkRow> work_rows(const Work& work) {
    return std::visit(
        [](const auto& operation) -> std::span<const WorkRow> {
            using Operation = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<Operation, ForwardWork> ||
                          std::is_same_v<Operation, MtpWork>)
                return operation.rows;
            else if constexpr (std::is_same_v<Operation, ScoreWork>)
                return {&operation.row, 1};
            else
                return {};
        },
        work.operation);
}

bool same_member(const runtime::ExecutionMember& a, const runtime::ExecutionMember& b) {
    return a.sequence == b.sequence && a.residency == b.residency;
}

void check_member(const runtime::ExecutionMember& member, std::uint32_t lanes) {
    if (!member.sequence.value || !member.sequence.generation ||
        member.residency.lane.value >= lanes || !member.residency.generation)
        throw std::invalid_argument("invalid Qwen4Exp execution member");
}

void check_tokens(std::span<const TokenId> ids, std::uint32_t vocabulary) {
    for (const auto id : ids)
        if (id < 0 || static_cast<std::uint32_t>(id) >= vocabulary)
            throw std::invalid_argument("Qwen4Exp input token is outside the model vocabulary");
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
                                 DeviceContext& d, const StartupObserver& observer)
    : parameters(p), device(d), cuda(d.execution_view()),
      resources(p, std::move(plan), d, observer), ple(resources, d), shape(resources.plan.shape),
      prefill_timer(d), compute_done(d), context_done(d) {}

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
    const auto& config        = parameters.profile->config.text;
    const auto state_slots    = resources.states->storage().slot_count();
    const auto check_resident = [&](const runtime::ExecutionMember& member) {
        check_member(member, shape.max_concurrency);
        const auto& resident = bindings[member.residency.lane.value];
        if (!resident || !same_member(resident->member, member))
            throw std::logic_error("Qwen4Exp work refers to a stale resident");
    };
    const auto check_rows = [&](std::span<const WorkRow> rows) {
        if (kind != runtime::FrameKind::Compute || rows.empty() ||
            rows.size() > shape.max_concurrency)
            throw std::invalid_argument("Qwen4Exp compute shape differs from its reservation");
        for (std::size_t b = 0; b < rows.size(); ++b) {
            const auto& row = rows[b];
            check_resident(row.member);
            check_selectors(row.slots, state_slots);
            check_table(row.main_table, 0, *resources.main);
            if (parameters.mtp) {
                check_table(row.mtp_table, 1, *resources.mtp);
                if (row.mtp_frontier != mtp_history_frontier(row.main_frontier))
                    throw std::invalid_argument("Qwen4Exp work has an unaligned MTP history");
            } else if (row.mtp_table.valid() || row.mtp_frontier) {
                throw std::invalid_argument("Qwen4Exp work refers to an absent MTP history");
            }
            if (row.main_frontier > shape.max_context ||
                row.ids.size() > shape.max_context - row.main_frontier ||
                (!row.rope_positions.empty() && row.rope_positions.size() != 3 * row.ids.size()))
                throw std::invalid_argument("Qwen4Exp input window exceeds its shape");
            std::uint32_t previous_boundary = 0;
            for (const auto boundary : row.window_ends) {
                if (boundary <= previous_boundary || boundary >= row.ids.size())
                    throw std::invalid_argument("Qwen4Exp execution window boundaries are invalid");
                previous_boundary = boundary;
            }
            check_tokens(row.ids, config.vocabulary);
            check_tokens(row.count_tokens, parameters.profile->public_tokens);
            check_tokens(row.ple_raw_tail, config.vocabulary);
            for (const auto position : row.rope_positions)
                if (position < 0)
                    throw std::invalid_argument("Qwen4Exp input has a negative RoPE position");
            if (parameters.mtp && row.main_frontier)
                for (const auto position : row.previous_rope)
                    if (position < 0)
                        throw std::invalid_argument("Qwen4Exp previous RoPE position is negative");
            for (std::size_t i = 0; i < b; ++i) {
                const auto& other = rows[i];
                if (row.member.residency.lane == other.member.residency.lane ||
                    row.member.sequence == other.member.sequence ||
                    row.main_table.index == other.main_table.index ||
                    (parameters.mtp && row.mtp_table.index == other.mtp_table.index) ||
                    row.slots.destination == other.slots.destination ||
                    row.slots.destination == other.slots.source ||
                    row.slots.source == other.slots.destination)
                    throw std::invalid_argument("Qwen4Exp batch rows have overlapping writes");
            }
        }
    };
    std::visit(
        [&](const auto& operation) {
            using Operation = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<Operation, ForwardWork>) {
                if (shape.purpose != EnginePurpose::Generation || operation.rows.empty())
                    throw std::invalid_argument("Qwen4Exp generation work is unavailable");
                const auto tokens = operation.rows.front().ids.size();
                if (operation.kind != ForwardKind::Control && tokens > shape.prefill_chunk)
                    throw std::invalid_argument("Qwen4Exp forward window exceeds its frame");
                check_rows(operation.rows);
                for (const auto& row : operation.rows)
                    if (row.ids.size() != tokens)
                        throw std::invalid_argument("Qwen4Exp forward rows have different lengths");
                const bool sampling = operation.sample != SampleKind::None;
                switch (operation.kind) {
                case ForwardKind::Prefill:
                    if (operation.rows.size() != 1 || operation.sample == SampleKind::NextToken ||
                        (!tokens && (operation.sample != SampleKind::FirstToken ||
                                     !operation.rows.front().main_frontier)))
                        throw std::invalid_argument("invalid Qwen4Exp prefill work");
                    break;
                case ForwardKind::Replay:
                    if (operation.rows.size() != 1 || !tokens || sampling)
                        throw std::invalid_argument("invalid Qwen4Exp replay work");
                    break;
                case ForwardKind::Ordinary:
                    if (tokens != 1 || operation.sample != SampleKind::NextToken)
                        throw std::invalid_argument("invalid Qwen4Exp ordinary decode work");
                    break;
                case ForwardKind::Control:
                    if (!tokens || sampling)
                        throw std::invalid_argument("invalid Qwen4Exp control work");
                    break;
                default:
                    throw std::invalid_argument("invalid Qwen4Exp forward kind");
                }
                if (operation.sample != SampleKind::None &&
                    operation.sample != SampleKind::FirstToken &&
                    operation.sample != SampleKind::NextToken)
                    throw std::invalid_argument("invalid Qwen4Exp sampling kind");
                if (operation.prefetch) {
                    const auto& next = *operation.prefetch;
                    check_resident(next.member);
                    if (operation.rows.size() != 1 ||
                        !same_member(next.member, operation.rows.front().member) ||
                        next.frontier != operation.rows.front().main_frontier + tokens ||
                        next.ids.empty() || next.ids.size() > shape.prefill_chunk ||
                        next.frontier > shape.max_context ||
                        next.ids.size() > shape.max_context - next.frontier)
                        throw std::invalid_argument("invalid Qwen4Exp PLE prefetch window");
                    check_tokens(next.ids, config.vocabulary);
                    check_tokens(next.raw_tail, config.vocabulary);
                }
            } else if constexpr (std::is_same_v<Operation, MtpWork>) {
                if (shape.purpose != EnginePurpose::Generation || !parameters.mtp ||
                    !shape.draft_tokens || operation.draft_extents.size() != operation.rows.size())
                    throw std::invalid_argument("Qwen4Exp MTP work is unavailable");
                check_rows(operation.rows);
                bool any = false;
                for (std::size_t b = 0; b < operation.rows.size(); ++b) {
                    const auto& row   = operation.rows[b];
                    const auto extent = operation.draft_extents[b];
                    if (row.ids.size() != 1 || !row.main_frontier || extent > shape.draft_tokens ||
                        extent >= shape.max_context - row.main_frontier)
                        throw std::invalid_argument("invalid Qwen4Exp MTP input extent");
                    any |= extent != 0;
                }
                if (!any) throw std::invalid_argument("Qwen4Exp MTP work has no draft positions");
            } else if constexpr (std::is_same_v<Operation, ScoreWork>) {
                if (shape.purpose != EnginePurpose::CausalScoring || operation.row.ids.empty() ||
                    operation.row.ids.size() > shape.prefill_chunk ||
                    operation.first_target_column > operation.row.ids.size() ||
                    operation.targets.size() >
                        operation.row.ids.size() - operation.first_target_column ||
                    operation.row.constrained)
                    throw std::invalid_argument("invalid Qwen4Exp scoring work");
                check_rows({&operation.row, 1});
                check_tokens(operation.targets, parameters.profile->public_tokens);
            } else {
                if (kind != runtime::FrameKind::Context ||
                    operation.bindings.size() > shape.max_concurrency ||
                    operation.releases.size() > shape.max_concurrency)
                    throw std::invalid_argument("invalid Qwen4Exp context reservation");
                std::array<bool, kMaximumConcurrency> released{}, installed{};
                for (const auto residency : operation.releases) {
                    const auto lane = residency.lane.value;
                    if (lane >= shape.max_concurrency || released[lane] || !bindings[lane] ||
                        bindings[lane]->member.residency != residency)
                        throw std::logic_error("Qwen4Exp release refers to a stale resident");
                    released[lane] = true;
                    if (compute)
                        for (const auto& row : work_rows(compute->request.work))
                            if (row.member.residency == residency)
                                throw std::logic_error("Qwen4Exp resident still has live compute");
                }
                for (const auto& resident : operation.bindings) {
                    check_member(resident.member, shape.max_concurrency);
                    const auto lane = resident.member.residency.lane.value;
                    if (installed[lane] || (bindings[lane] && !released[lane]) ||
                        (bindings[lane] &&
                         bindings[lane]->member.residency == resident.member.residency))
                        throw std::logic_error("Qwen4Exp residency is already installed");
                    for (const auto& other : operation.bindings) {
                        if (&other == &resident) break;
                        if (other.member.sequence == resident.member.sequence)
                            throw std::invalid_argument("duplicate Qwen4Exp sequence binding");
                    }
                    for (std::size_t i = 0; i < bindings.size(); ++i)
                        if (bindings[i] && !released[i] &&
                            bindings[i]->member.sequence == resident.member.sequence)
                            throw std::logic_error("Qwen4Exp sequence is already resident");
                    if (resident.vision && !parameters.vision)
                        throw std::invalid_argument("Qwen4Exp Vision resources were not prepared");
                    check_tokens(resident.generated, parameters.profile->public_tokens);
                    installed[lane] = true;
                }
            }
        },
        work.operation);
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
    const auto words  = (std::size_t(parameters.profile->public_tokens) + 31) / 32;
    std::array<bool, kMaximumConcurrency> seen{};
    for (const auto& row : input.rows) {
        if (row.member >= frame.rows.size() || seen[row.member] ||
            !frame.rows[row.member].constrained)
            throw std::invalid_argument("Qwen4Exp decision has an unexpected member");
        const auto positions = frame.verify ? frame.verify->extents[row.member] + 1 : 1;
        if (row.positions != positions || row.words.size() != positions * words ||
            (positions < 32 && (row.dead_positions >> positions)))
            throw std::invalid_argument("Qwen4Exp decision mask shape is inconsistent");
        seen[row.member] = true;
    }
    for (std::size_t b = 0; b < frame.rows.size(); ++b)
        if (frame.rows[b].constrained != seen[b])
            throw std::invalid_argument("Qwen4Exp decision is missing a constrained member");
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
    const auto& frame   = *compute;
    const auto rows     = work_rows(frame.request.work);
    const auto* forward = std::get_if<ForwardWork>(&frame.request.work.operation);
    const bool scoring  = std::holds_alternative<ScoreWork>(frame.request.work.operation);
    const bool sampled  = frame.verify || (forward && forward->sample != SampleKind::None);
    if (commit.rows.size() != rows.size() || (!scoring && frame.result.rows.size() != rows.size()))
        throw std::logic_error("Qwen4Exp commit rows differ from its execution");
    for (std::size_t b = 0; b < commit.rows.size(); ++b) {
        const auto& row    = commit.rows[b];
        const auto& source = rows[b];
        if (!same_member(row.member, source.member) || row.slots.source != source.slots.source)
            throw std::logic_error("Qwen4Exp commit member differs from its execution");
        if (!frame.verify && row.slots.destination != source.slots.destination)
            throw std::logic_error(
                "Qwen4Exp committed destination differs from its completed write");
        check_selectors(row.slots, resources.states->storage().slot_count());
        if (sampled) {
            const auto& result  = frame.result.rows[b];
            const auto produced = result.tokens.size();
            const auto reached  = produced < 32 ? (std::uint32_t(1) << produced) - 1 : UINT32_MAX;
            const bool failed   = (result.dead_positions & reached) != 0;
            if ((row.cancelled && row.failed) || (row.failed && !failed) ||
                ((row.cancelled || row.failed)
                     ? (!row.terminal || row.accepted != 0)
                     : (!row.accepted || row.accepted > produced ||
                        (!row.terminal && row.accepted != produced) || failed)))
                throw std::invalid_argument("invalid Qwen4Exp generated-prefix commit");
        } else {
            const auto processed = scoring ? source.ids.size() : frame.result.rows[b].processed;
            if (row.failed ||
                (row.cancelled ? (!row.terminal || row.accepted != 0) : row.accepted != processed))
                throw std::invalid_argument("invalid Qwen4Exp processed-input commit");
        }
        for (std::size_t i = 0; i < commit.rows.size(); ++i) {
            if (i == b) continue;
            if (row.slots.destination == commit.rows[i].slots.destination ||
                row.slots.destination == rows[i].slots.source)
                throw std::invalid_argument("Qwen4Exp commit rows have overlapping writes");
        }
    }
    resources.compute_actions->validate(commit.actions);
}

void RankProgramImpl::reset_memory_peaks() noexcept {
    resources.workspace.reset_peak();
    if (graphs) graphs->reset_workspace_peak();
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
                               graphs ? graphs->workspace_peak() : 0})};
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
                         DeviceContext& device, const StartupObserver& observer)
    : impl_(std::make_unique<detail::RankProgramImpl>(parameters, std::move(plan), device,
                                                      observer)) {
    if (impl_->shape.cuda_graph) {
        StartupPhaseScope phase(observer, StartupPhase::CudaGraphPrepare);
        impl_->initialize_graphs();
        phase.complete();
    }
}

RankProgram::~RankProgram() = default;

runtime::ExecutionCapacity RankProgram::capacity() const noexcept { return {1, 1}; }

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
