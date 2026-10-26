#include "models/glm5next/work.h"
#include "models/common/context/wire.h"
#include "runtime/executor/protocol.h"

#include "models/glm5next/state/continuation.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(std::string("GLM5Next work: ") + message);
}

bool same_member(const runtime::ExecutionMember& a, const runtime::ExecutionMember& b) {
    return a.sequence == b.sequence && a.residency == b.residency;
}

void member(const runtime::ExecutionMember& m, std::uint32_t lanes) {
    require(m.sequence.value && m.sequence.generation && m.residency.generation &&
                m.residency.lane.value < lanes,
            "invalid execution member");
}

void tokens(std::span<const TokenId> values, std::uint32_t vocabulary) {
    for (const auto token : values)
        require(token >= 0 && std::uint32_t(token) < vocabulary, "token outside vocabulary");
}
} // namespace

std::span<const WorkRow> work_rows(const Work& work) {
    if (const auto* f = std::get_if<ForwardWork>(&work.operation)) return f->rows;
    if (const auto* m = std::get_if<MtpWork>(&work.operation)) return m->rows;
    if (const auto* s = std::get_if<ScoreWork>(&work.operation)) return {&s->row, 1};
    return {};
}

BatchPlan plan_batch(std::span<const WorkRow> rows, std::uint32_t ranks) {
    require(ranks > 0, "empty rank group");
    BatchPlan result;
    for (std::uint32_t rank = 0; rank < ranks; ++rank) {
        const auto begin = static_cast<std::uint32_t>(result.members.size());
        for (std::uint32_t row = 0; row < rows.size(); ++row)
            if (rows[row].attention_rank == rank) result.members.push_back(row);
        result.owners.push_back({begin, static_cast<std::uint32_t>(result.members.size()) - begin});
    }
    require(result.members.size() == rows.size(), "attention owner outside rank group");
    return result;
}

const context::ContextActions& local_actions(const ContextActions& actions, std::uint32_t rank) {
    static const context::ContextActions empty;
    const auto found = actions.find(rank);
    return found == actions.end() ? empty : found->second;
}

ExecutionWork ExecutionTypes::describe_work(const Work& work) {
    ExecutionWork out;
    if (const auto* f = std::get_if<ForwardWork>(&work.operation)) {
        out.kind = f->kind == ForwardKind::Prefill   ? ExecutionKind::Prefill
                   : f->kind == ForwardKind::Replay  ? ExecutionKind::Replay
                   : f->kind == ForwardKind::Control ? ExecutionKind::Control
                                                     : ExecutionKind::Decode;
    } else if (std::holds_alternative<ScoreWork>(work.operation))
        out.kind = ExecutionKind::Score;
    const auto rows = work_rows(work);
    out.rows        = rows.size();
    for (const auto& row : rows) out.input_tokens += row.ids.size();
    out.scheduled_positions = out.input_tokens;
    if (const auto* m = std::get_if<MtpWork>(&work.operation)) {
        out.kind = ExecutionKind::Decode;
        for (auto n : m->draft_extents) out.scheduled_positions += n;
    }
    return out;
}

runtime::WorkPayloadCapacity WorkContract::payload_capacity(std::uint32_t rank_count) const {
    const std::size_t batch = shape.max_concurrency, chunk = shape.prefill_chunk;
    const auto header       = runtime::protocol::envelope_capacity(rank_count);
    const std::size_t pools = shape.draft_tokens ? 2 : 1;
    const auto actions      = [&](std::size_t rows, std::size_t tokens) {
        return context::write_action_capacity(rows, pools, tokens);
    };
    const auto payload = [&](std::size_t rows, std::size_t span, std::size_t token_bytes) {
        return header + actions(rows, span) + 128 + rows * (192 + 20 * rank_count + token_bytes);
    };
    const std::size_t width = shape.draft_tokens + 1;
    const auto work =
        shape.purpose == EnginePurpose::CausalScoring
            ? payload(1, chunk, 8 * chunk)
            : std::max({payload(1, chunk, 4 * chunk), payload(batch, width, 4),
                        payload(batch, control_token_limit, 8ULL * control_token_limit)});
    const auto context = header + context::action_batch_capacity() + 128 +
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
    if (const auto* c = std::get_if<ContextWork>(&work.operation)) {
        require(kind == runtime::FrameKind::Context &&
                    c->bindings.size() <= shape.max_concurrency &&
                    c->releases.size() <= shape.max_concurrency,
                "invalid context frame");
        std::array<bool, kMaximumConcurrency> bound{}, released{};
        for (const auto& r : c->releases) {
            require(r.lane.value < shape.max_concurrency && r.generation && !released[r.lane.value],
                    "invalid residency release");
            released[r.lane.value] = true;
        }
        for (const auto& b : c->bindings) {
            member(b.member, shape.max_concurrency);
            require(!bound[b.member.residency.lane.value], "duplicate residency binding");
            bound[b.member.residency.lane.value] = true;
            b.counts.validate(public_tokens, shape.max_context);
            require(shape.purpose == EnginePurpose::Generation || b.counts.empty(),
                    "scoring residency has generation counts");
        }
        require(work.batch.members.empty() && work.batch.owners.empty(),
                "context has compute rows");
        return;
    }
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    const bool control  = forward && forward->kind == ForwardKind::Control;
    const auto rows     = work_rows(work);
    require(kind == runtime::FrameKind::Compute && !rows.empty() &&
                rows.size() <= shape.max_concurrency,
            "compute frame extent");
    require(work.batch == plan_batch(rows, ranks), "physical row plan differs from membership");
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        require(r.control_last_token.has_value() == control && (control || r.window_ends.empty()),
                "control fields outside control work");
        member(r.member, shape.max_concurrency);
        require(r.states.size() == ranks && r.main_table.valid() && r.main_table.pool_id == 0,
                "incomplete context binding");
        require(shape.draft_tokens ? (r.mtp_table.valid() && r.mtp_table.pool_id == 1 &&
                                      r.mtp_frontier == mtp_history_frontier(r.main_frontier))
                                   : !r.mtp_table.valid(),
                "MTP history binding");
        for (std::uint32_t rank = 0; rank < ranks; ++rank) {
            const auto found = r.states.find(rank);
            require(found != r.states.end() && found->second.source >= 0 &&
                        found->second.destination >= 0,
                    "invalid State selector");
        }
        require(r.main_frontier <= shape.max_context &&
                    r.ids.size() <= shape.max_context - r.main_frontier,
                "input exceeds context");
        tokens(r.ids, vocabulary);
        if (control && !r.ids.empty())
            tokens(std::span<const TokenId>(r.ids).subspan(1), public_tokens);
        if (r.control_last_token) tokens({&*r.control_last_token, 1}, public_tokens);
        std::uint32_t previous = 0;
        for (auto end : r.window_ends) {
            require(end > previous && end < r.ids.size(), "invalid window boundary");
            previous = end;
        }
        for (std::size_t j = 0; j < i; ++j) {
            require(r.member.sequence != rows[j].member.sequence &&
                        r.member.residency.lane != rows[j].member.residency.lane &&
                        r.main_table.index != rows[j].main_table.index &&
                        (!shape.draft_tokens || r.mtp_table.index != rows[j].mtp_table.index),
                    "overlapping batch owners");
            for (const auto& [rank, offsets] : r.states) {
                const auto other = rows[j].states.at(rank);
                require(offsets.destination != other.destination &&
                            offsets.destination != other.source &&
                            other.destination != offsets.source,
                        "overlapping State writes");
            }
        }
    }
    if (const auto* f = std::get_if<ForwardWork>(&work.operation)) {
        require(shape.purpose == EnginePurpose::Generation, "generation on scoring instance");
        const auto n = rows.front().ids.size();
        if (f->visual) {
            const auto& use = f->visual->window;
            require(vision_max_features && rows.size() == 1 && n && use.id && use.plan &&
                        same_member(use.owner, rows.front().member) &&
                        (f->kind == ForwardKind::Prefill || f->kind == ForwardKind::Replay) &&
                        use.produce == bool(f->visual->input),
                    "Vision work differs from its prepared window or owner");
            require(use.plan->features && use.plan->features <= vision_max_features &&
                        use.plan->consume_end <= shape.max_context,
                    "Vision window exceeds prepared capacity");
        }
        for (const auto& r : rows) require(r.ids.size() == n, "unequal forward widths");
        require(f->kind == ForwardKind::Control || n <= shape.prefill_chunk,
                "window exceeds frame");
        switch (f->kind) {
        case ForwardKind::Prefill:
            require(rows.size() == 1 &&
                        (f->sample == SampleKind::None || f->sample == SampleKind::FirstToken) &&
                        (n || (f->sample == SampleKind::FirstToken && rows[0].main_frontier)),
                    "prefill extent");
            break;
        case ForwardKind::Replay:
            require(rows.size() == 1 && n && f->sample == SampleKind::None, "replay extent");
            break;
        case ForwardKind::Ordinary:
            require(n == 1 && f->sample == SampleKind::NextToken, "ordinary extent");
            break;
        case ForwardKind::Control:
            require(n && n <= control_token_limit && f->sample == SampleKind::None,
                    "control extent");
            break;
        default:
            require(false, "unknown forward kind");
        }
    } else if (const auto* m = std::get_if<MtpWork>(&work.operation)) {
        require(shape.purpose == EnginePurpose::Generation && shape.draft_tokens &&
                    m->draft_extents.size() == rows.size(),
                "MTP extent");
        bool live = false;
        for (std::size_t b = 0; b < rows.size(); ++b) {
            const auto k = m->draft_extents[b];
            require(rows[b].ids.size() == 1 && rows[b].main_frontier && k <= shape.draft_tokens &&
                        k < shape.max_context - rows[b].main_frontier,
                    "MTP verify window");
            live |= k != 0;
        }
        require(live, "MTP round has no proposals");
    } else {
        const auto& s = std::get<ScoreWork>(work.operation);
        require(shape.purpose == EnginePurpose::CausalScoring && !s.row.constrained &&
                    !s.row.ids.empty() && s.row.ids.size() <= shape.prefill_chunk &&
                    s.first_target_column <= s.row.ids.size() &&
                    s.targets.size() <= s.row.ids.size() - s.first_target_column,
                "scoring extent");
        tokens(s.targets, public_tokens);
    }
}

void WorkContract::validate(const Work& work, const runtime::DecisionRequest& request,
                            const runtime::DecisionInput& input) const {
    const auto rows = work_rows(work);
    require(request.vocabulary == public_tokens && request.stage == input.stage,
            "decision identity");
    std::array<bool, kMaximumConcurrency> requested{}, seen{};
    for (const auto& r : request.rows) {
        require(r.member < rows.size() && rows[r.member].constrained && !requested[r.member] &&
                    r.drafts.size() ==
                        (std::holds_alternative<MtpWork>(work.operation)
                             ? std::get<MtpWork>(work.operation).draft_extents.at(r.member)
                             : 0),
                "decision member");
        tokens(r.drafts, public_tokens);
        requested[r.member] = true;
    }
    for (const auto& r : input.rows) {
        const auto positions =
            std::holds_alternative<MtpWork>(work.operation) && r.member < rows.size()
                ? std::get<MtpWork>(work.operation).draft_extents[r.member] + 1
                : 1;
        require(r.member < rows.size() && requested[r.member] && !seen[r.member] &&
                    r.positions == positions &&
                    r.words.size() == positions * ((std::size_t(public_tokens) + 31) / 32) &&
                    (std::uint64_t(r.dead_positions) >> positions) == 0,
                "decision mask extent");
        seen[r.member] = true;
    }
    for (std::size_t i = 0; i < rows.size(); ++i)
        require(rows[i].constrained == requested[i] && requested[i] == seen[i],
                "missing decision member");
}

void WorkContract::validate(const Work& work, const Result& result, const Commit& commit) const {
    const auto* computed = std::get_if<ComputeResult>(&result);
    const bool score     = std::holds_alternative<ScoreWork>(work.operation);
    require(score ? std::holds_alternative<ScoreResult>(result) : computed != nullptr,
            "result kind");
    validate_commit(
        work, computed ? std::span<const RowResult>(computed->rows) : std::span<const RowResult>{},
        commit);
}

void WorkContract::validate_commit(const Work& work, std::span<const RowResult> result,
                                   const Commit& commit) const {
    const auto rows    = work_rows(work);
    const auto* f      = std::get_if<ForwardWork>(&work.operation);
    const auto* m      = std::get_if<MtpWork>(&work.operation);
    const bool sampled = m || (f && f->sample != SampleKind::None);
    require(commit.rows.size() == rows.size() &&
                (std::holds_alternative<ScoreWork>(work.operation) || result.size() == rows.size()),
            "commit membership");
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& c = commit.rows[i];
        require(same_member(c.member, rows[i].member) && c.states.size() == rows[i].states.size(),
                "commit identity");
        for (const auto& [rank, s] : c.states) {
            const auto before = rows[i].states.at(rank);
            require(s.source == before.source && s.destination == before.destination,
                    "commit changes State binding");
        }
        require(!(c.cancelled && c.failed), "conflicting terminal disposition");
        if (c.cancelled || c.failed) {
            require(c.terminal && c.accepted == 0 &&
                        (!c.failed || (sampled && result[i].dead_positions)),
                    "invalid rejected result");
        } else if (sampled) {
            const auto produced = result[i].tokens.size();
            const auto reached  = produced >= 32 ? UINT32_MAX : (std::uint32_t{1} << produced) - 1;
            require(produced && produced <= (m ? m->draft_extents[i] + 1 : 1) && c.accepted &&
                        c.accepted <= produced && (c.terminal || c.accepted == produced) &&
                        !(result[i].dead_positions & reached),
                    "invalid sampled commit");
        } else
            require(!c.failed && c.accepted == rows[i].ids.size(),
                    "invalid processed-input commit");
    }
}
} // namespace ninfer::models::glm5next
