#include "models/qwen4_exp/work.h"
#include "models/common/context/wire.h"
#include "runtime/executor/protocol.h"

#include <algorithm>
#include <type_traits>

namespace ninfer::models::qwen4_exp {
ExecutionWork ExecutionTypes::describe_work(const Work& work) {
    ExecutionWork result;
    std::visit(
        [&](const auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, ForwardWork>) {
                result.kind = op.kind == ForwardKind::Prefill   ? ExecutionKind::Prefill
                              : op.kind == ForwardKind::Replay  ? ExecutionKind::Replay
                              : op.kind == ForwardKind::Control ? ExecutionKind::Control
                                                                : ExecutionKind::Decode;
                result.rows = op.rows.size();
                for (const auto& row : op.rows) result.input_tokens += row.ids.size();
                result.scheduled_positions = result.input_tokens;
            } else if constexpr (std::is_same_v<T, MtpWork>) {
                result.kind = ExecutionKind::Decode;
                result.rows = result.input_tokens = op.rows.size();
                for (const auto extent : op.draft_extents) result.scheduled_positions += extent + 1;
            } else if constexpr (std::is_same_v<T, ScoreWork>) {
                result.kind         = ExecutionKind::Score;
                result.rows         = 1;
                result.input_tokens = result.scheduled_positions = op.row.ids.size();
            }
        },
        work.operation);
    return result;
}

namespace {
bool same_member(const runtime::ExecutionMember& a, const runtime::ExecutionMember& b) {
    return a.sequence == b.sequence && a.residency == b.residency;
}

void member(const runtime::ExecutionMember& value, std::uint32_t lanes) {
    if (!value.sequence.value || !value.sequence.generation ||
        value.residency.lane.value >= lanes || !value.residency.generation)
        throw std::invalid_argument("invalid Qwen4Exp execution member");
}

void tokens(std::span<const TokenId> values, std::uint32_t vocabulary) {
    for (auto value : values)
        if (value < 0 || static_cast<std::uint32_t>(value) >= vocabulary)
            throw std::invalid_argument("Qwen4Exp token is outside its vocabulary");
}
} // namespace

std::span<const WorkRow> work_rows(const Work& work) {
    return std::visit(
        [](const auto& operation) -> std::span<const WorkRow> {
            using T = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<T, ForwardWork> || std::is_same_v<T, MtpWork>)
                return operation.rows;
            else if constexpr (std::is_same_v<T, ScoreWork>)
                return {&operation.row, 1};
            else
                return {};
        },
        work.operation);
}

const context::ContextActions& local_actions(const ContextActions& actions, std::uint32_t rank) {
    const auto found = actions.find(rank);
    static const context::ContextActions empty;
    return found == actions.end() ? empty : found->second;
}

runtime::WorkPayloadCapacity WorkContract::payload_capacity(std::uint32_t rank_count) const {
    const std::size_t batch = shape.max_concurrency, chunk = shape.prefill_chunk;
    const auto header       = runtime::protocol::envelope_capacity(rank_count);
    const std::size_t pools = shape.draft_tokens ? 2 : 1;
    const auto actions      = [&](std::size_t rows, std::size_t tokens) {
        return context::write_action_capacity(rows, pools, tokens);
    };
    const auto payload = [&](std::size_t rows, std::size_t span, std::size_t token_bytes) {
        return header + actions(rows, span) + 128 + rows * (204 + token_bytes);
    };
    const std::size_t width = shape.draft_tokens + 1;
    const auto work         = std::max({payload(1, chunk, 20 * chunk), payload(batch, width, 16),
                                        payload(batch, control_token_limit, 8ULL * control_token_limit)});
    const auto context      = header + context::action_batch_capacity() + 128 +
                         batch * (88 + TokenCounts::encoded_capacity(public_tokens));
    const auto decision = header + 12 + batch * (20 + 4 * width * ((public_tokens + 31ULL) / 32));
    const auto commit   = header + 64 + batch * (64 + 20 * rank_count) + actions(batch, width);
    const auto context_rows =
        sizeof(Work) + sizeof(ContextWork) +
        batch * (sizeof(ResidencyBinding) + sizeof(runtime::ResidencyId) + 4ULL * public_tokens);
    const auto actions_storage = context::action_batch_storage_capacity() + 128;
    return {work,
            context,
            decision,
            commit,
            context_rows + actions_storage,
            context_rows + rank_count * actions_storage};
}

void WorkContract::validate(const Work& work, runtime::FrameKind kind) const {
    const auto rows     = work_rows(work);
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    const bool control  = forward && forward->kind == ForwardKind::Control;
    if (!std::holds_alternative<ContextWork>(work.operation)) {
        if (kind != runtime::FrameKind::Compute || rows.empty() ||
            rows.size() > shape.max_concurrency)
            throw std::invalid_argument("Qwen4Exp compute shape differs from its reservation");
        for (std::size_t b = 0; b < rows.size(); ++b) {
            const auto& row = rows[b];
            if (row.control_last_token.has_value() != control ||
                (!control && !row.window_ends.empty()) || (control && !row.rope_positions.empty()))
                throw std::invalid_argument("Qwen4Exp control fields outside control work");
            member(row.member, shape.max_concurrency);
            if (row.states.empty() || !row.main_table.valid() || row.main_table.pool_id != 0)
                throw std::invalid_argument("Qwen4Exp work has invalid context selectors");
            if (mtp) {
                if (!row.mtp_table.valid() || row.mtp_table.pool_id != 1 ||
                    row.mtp_frontier != (row.main_frontier ? row.main_frontier - 1 : 0))
                    throw std::invalid_argument("Qwen4Exp work has an unaligned MTP history");
            } else if (row.mtp_table.valid() || row.mtp_frontier)
                throw std::invalid_argument("Qwen4Exp work refers to an absent MTP history");
            if (row.main_frontier > shape.max_context ||
                row.ids.size() > shape.max_context - row.main_frontier ||
                (!row.rope_positions.empty() && row.rope_positions.size() != 3 * row.ids.size()))
                throw std::invalid_argument("Qwen4Exp input window exceeds its shape");
            std::uint32_t previous = 0;
            for (auto boundary : row.window_ends) {
                if (boundary <= previous || boundary >= row.ids.size())
                    throw std::invalid_argument("Qwen4Exp input window boundaries are invalid");
                previous = boundary;
            }
            tokens(row.ids, vocabulary);
            if (control && !row.ids.empty())
                tokens(std::span<const TokenId>(row.ids).subspan(1), public_tokens);
            if (row.control_last_token) tokens({&*row.control_last_token, 1}, public_tokens);
            tokens(row.ple_raw_tail, vocabulary);
            for (auto position : row.rope_positions)
                if (position < 0) throw std::invalid_argument("negative Qwen4Exp RoPE position");
            if (mtp && row.main_frontier)
                for (auto position : row.previous_rope)
                    if (position < 0)
                        throw std::invalid_argument("negative previous Qwen4Exp RoPE position");
            for (std::size_t i = 0; i < b; ++i) {
                const auto& other = rows[i];
                if (row.member.residency.lane == other.member.residency.lane ||
                    row.member.sequence == other.member.sequence ||
                    row.main_table.index == other.main_table.index ||
                    (mtp && row.mtp_table.index == other.mtp_table.index))
                    throw std::invalid_argument("Qwen4Exp batch rows have overlapping writes");
            }
        }
    }
    std::visit(
        [&](const auto& operation) {
            using T = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<T, ForwardWork>) {
                if (shape.purpose != EnginePurpose::Generation)
                    throw std::invalid_argument("generation work is unavailable");
                const auto count = rows.front().ids.size();
                if (operation.kind != ForwardKind::Control && count > shape.prefill_chunk)
                    throw std::invalid_argument("Qwen4Exp forward window exceeds its frame");
                for (const auto& row : rows)
                    if (row.ids.size() != count)
                        throw std::invalid_argument("Qwen4Exp forward rows have different lengths");
                if (operation.visual) {
                    const auto& use = operation.visual->window;
                    if (!vision || rows.size() != 1 || !use.id || !use.plan ||
                        use.owner.sequence != rows.front().member.sequence ||
                        use.owner.residency != rows.front().member.residency ||
                        (operation.kind != ForwardKind::Prefill &&
                         operation.kind != ForwardKind::Replay) ||
                        use.produce != bool(operation.visual->input))
                        throw std::invalid_argument(
                            "Qwen4Exp Vision work differs from its window owner");
                }
                const bool sampling = operation.sample != SampleKind::None;
                switch (operation.kind) {
                case ForwardKind::Prefill:
                    if (rows.size() != 1 || operation.sample == SampleKind::NextToken ||
                        (!count && (operation.sample != SampleKind::FirstToken ||
                                    !rows.front().main_frontier)))
                        throw std::invalid_argument("invalid Qwen4Exp prefill work");
                    break;
                case ForwardKind::Replay:
                    if (rows.size() != 1 || !count || sampling)
                        throw std::invalid_argument("invalid Qwen4Exp replay work");
                    break;
                case ForwardKind::Ordinary:
                    if (count != 1 || operation.sample != SampleKind::NextToken)
                        throw std::invalid_argument("invalid Qwen4Exp ordinary work");
                    break;
                case ForwardKind::Control:
                    if (!count || count > control_token_limit || sampling)
                        throw std::invalid_argument("invalid Qwen4Exp control work");
                    break;
                default:
                    throw std::invalid_argument("invalid Qwen4Exp forward kind");
                }
                if (operation.sample != SampleKind::None &&
                    operation.sample != SampleKind::FirstToken &&
                    operation.sample != SampleKind::NextToken)
                    throw std::invalid_argument("invalid Qwen4Exp sample kind");
                if (operation.prefetch) {
                    const auto& next = *operation.prefetch;
                    member(next.member, shape.max_concurrency);
                    if ((operation.kind != ForwardKind::Prefill &&
                         operation.kind != ForwardKind::Replay) ||
                        rows.size() != 1 || !same_member(next.member, rows.front().member) ||
                        next.frontier != rows.front().main_frontier + count || next.ids.empty() ||
                        next.ids.size() > shape.prefill_chunk ||
                        next.frontier > shape.max_context ||
                        next.ids.size() > shape.max_context - next.frontier)
                        throw std::invalid_argument("invalid Qwen4Exp PLE prefetch window");
                    tokens(next.ids, vocabulary);
                    tokens(next.raw_tail, vocabulary);
                }
            } else if constexpr (std::is_same_v<T, MtpWork>) {
                if (shape.purpose != EnginePurpose::Generation || !mtp || !shape.draft_tokens ||
                    operation.draft_extents.size() != rows.size())
                    throw std::invalid_argument("Qwen4Exp MTP work is unavailable");
                bool any = false;
                for (std::size_t i = 0; i < rows.size(); ++i) {
                    const auto extent = operation.draft_extents[i];
                    if (rows[i].ids.size() != 1 || !rows[i].main_frontier ||
                        extent > shape.draft_tokens ||
                        extent >= shape.max_context - rows[i].main_frontier)
                        throw std::invalid_argument("invalid Qwen4Exp MTP input extent");
                    any |= extent != 0;
                }
                if (!any) throw std::invalid_argument("Qwen4Exp MTP work has no draft positions");
            } else if constexpr (std::is_same_v<T, ScoreWork>) {
                if (shape.purpose != EnginePurpose::CausalScoring || operation.row.ids.empty() ||
                    operation.row.ids.size() > shape.prefill_chunk ||
                    operation.first_target_column > operation.row.ids.size() ||
                    operation.targets.size() >
                        operation.row.ids.size() - operation.first_target_column ||
                    operation.row.constrained)
                    throw std::invalid_argument("invalid Qwen4Exp scoring work");
                tokens(operation.targets, public_tokens);
            } else {
                if (kind != runtime::FrameKind::Context ||
                    operation.bindings.size() > shape.max_concurrency ||
                    operation.releases.size() > shape.max_concurrency)
                    throw std::invalid_argument("invalid Qwen4Exp context reservation");
                std::array<bool, kMaximumConcurrency> installed{}, released{};
                for (auto r : operation.releases) {
                    if (r.lane.value >= shape.max_concurrency || !r.generation ||
                        released[r.lane.value])
                        throw std::invalid_argument("invalid Qwen4Exp residency release");
                    released[r.lane.value] = true;
                }
                for (const auto& binding : operation.bindings) {
                    member(binding.member, shape.max_concurrency);
                    if (installed[binding.member.residency.lane.value])
                        throw std::invalid_argument("duplicate Qwen4Exp binding lane");
                    installed[binding.member.residency.lane.value] = true;
                    for (const auto& previous : operation.bindings) {
                        if (&previous == &binding) break;
                        if (previous.member.sequence == binding.member.sequence)
                            throw std::invalid_argument("duplicate Qwen4Exp sequence binding");
                    }
                    binding.counts.validate(public_tokens, shape.max_context);
                    if (shape.purpose != EnginePurpose::Generation && !binding.counts.empty())
                        throw std::invalid_argument("scoring residency has generation counts");
                }
            }
        },
        work.operation);
}

void validate_decision_rows(const runtime::DecisionInput& input, std::uint32_t stage,
                            std::uint32_t vocabulary, std::span<const std::uint32_t> positions) {
    if (input.stage != stage) throw std::invalid_argument("Qwen4Exp decision stage differs");
    const auto words = (std::size_t(vocabulary) + 31) / 32;
    std::array<bool, kMaximumConcurrency> seen{};
    if (positions.size() > seen.size())
        throw std::invalid_argument("Qwen4Exp decision batch is too large");
    for (const auto& row : input.rows) {
        if (row.member >= positions.size() || seen[row.member] || !positions[row.member] ||
            row.positions != positions[row.member] || row.words.size() != words * row.positions ||
            (row.positions < 32 && (row.dead_positions >> row.positions)))
            throw std::invalid_argument(
                "Qwen4Exp decision mask shape or membership is inconsistent");
        seen[row.member] = true;
    }
    for (std::size_t i = 0; i < positions.size(); ++i)
        if (bool(positions[i]) != seen[i])
            throw std::invalid_argument("Qwen4Exp decision omitted a member");
}

void WorkContract::validate(const Work& work, const runtime::DecisionRequest& request,
                            const runtime::DecisionInput& input) const {
    const auto rows = work_rows(work);
    std::array<std::uint32_t, kMaximumConcurrency> positions{};
    if (rows.size() > positions.size() || request.vocabulary != public_tokens)
        throw std::invalid_argument("Qwen4Exp decision request differs from its work");
    for (const auto& row : request.rows) {
        if (row.member >= rows.size() || !rows[row.member].constrained || positions[row.member])
            throw std::invalid_argument("Qwen4Exp decision request has an unexpected member");
        positions[row.member] = row.drafts.size() + 1;
    }
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (rows[i].constrained != bool(positions[i]))
            throw std::invalid_argument("incomplete Qwen4Exp decision request");
    validate_decision_rows(input, request.stage, public_tokens, {positions.data(), rows.size()});
}

void WorkContract::validate(const Work& work, const Result& result, const Commit& commit) const {
    const auto* computed = std::get_if<ComputeResult>(&result);
    const bool scoring   = std::holds_alternative<ScoreWork>(work.operation);
    if ((scoring && !std::holds_alternative<ScoreResult>(result)) || (!scoring && !computed))
        throw std::logic_error("Qwen4Exp result kind differs from its work");
    validate_commit(
        work, computed ? std::span<const RowResult>(computed->rows) : std::span<const RowResult>{},
        commit);
}

void WorkContract::validate_commit(const Work& work, std::span<const RowResult> results,
                                   const Commit& commit) const {
    const auto rows     = work_rows(work);
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    const bool verify   = std::holds_alternative<MtpWork>(work.operation);
    const bool scoring  = std::holds_alternative<ScoreWork>(work.operation);
    const bool sampled  = verify || (forward && forward->sample != SampleKind::None);
    if (commit.rows.size() != rows.size() || (!scoring && results.size() != rows.size()))
        throw std::logic_error("Qwen4Exp commit rows differ from its result");
    for (std::size_t b = 0; b < rows.size(); ++b) {
        const auto& row    = commit.rows[b];
        const auto& source = rows[b];
        if (!same_member(row.member, source.member) || row.states.size() != source.states.size())
            throw std::logic_error("Qwen4Exp commit differs from its work");
        for (const auto& [rank, offsets] : row.states) {
            const auto previous = source.states.at(rank);
            if (offsets.source != previous.source ||
                (!verify && offsets.destination != previous.destination))
                throw std::logic_error("Qwen4Exp commit changes its prepared State binding");
        }
        if (sampled) {
            const auto& output  = results[b];
            const auto produced = output.tokens.size();
            const auto reached  = produced < 32 ? (std::uint32_t(1) << produced) - 1 : UINT32_MAX;
            const bool failed   = (output.dead_positions & reached) != 0;
            if ((row.cancelled && row.failed) || (row.failed && !failed) ||
                ((row.cancelled || row.failed)
                     ? (!row.terminal || row.accepted != 0)
                     : (!row.accepted || row.accepted > produced ||
                        (!row.terminal && row.accepted != produced) || failed)))
                throw std::invalid_argument("invalid Qwen4Exp generated-prefix commit");
        } else {
            const auto processed = scoring ? source.ids.size() : results[b].processed;
            if (row.failed ||
                (row.cancelled ? (!row.terminal || row.accepted != 0) : row.accepted != processed))
                throw std::invalid_argument("invalid Qwen4Exp processed-input commit");
        }
        for (const auto& [rank, offsets] : row.states)
            for (std::size_t i = 0; i < rows.size(); ++i)
                if (i != b && (offsets.destination == commit.rows[i].states.at(rank).destination ||
                               offsets.destination == rows[i].states.at(rank).source))
                    throw std::invalid_argument("Qwen4Exp commit rows have overlapping writes");
    }
}

} // namespace ninfer::models::qwen4_exp
