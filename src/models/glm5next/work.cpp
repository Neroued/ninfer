#include "models/glm5next/work.h"

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
    return out;
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
            require(b.generated.size() <= shape.max_context, "generated history exceeds context");
            tokens(b.generated, public_tokens);
        }
        require(work.batch.members.empty() && work.batch.owners.empty(),
                "context has compute rows");
        return;
    }
    const auto rows = work_rows(work);
    require(kind == runtime::FrameKind::Compute && !rows.empty() &&
                rows.size() <= shape.max_concurrency,
            "compute frame extent");
    require(work.batch == plan_batch(rows, ranks), "physical row plan differs from membership");
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        member(r.member, shape.max_concurrency);
        require(r.states.size() == ranks && r.main_table.valid() && r.main_table.pool_id == 0,
                "incomplete context binding");
        for (std::uint32_t rank = 0; rank < ranks; ++rank) {
            const auto found = r.states.find(rank);
            require(found != r.states.end() && found->second.source >= 0 &&
                        found->second.destination >= 0,
                    "invalid State selector");
        }
        require(r.main_frontier <= shape.max_context &&
                    r.ids.size() <= shape.max_context - r.main_frontier &&
                    r.count_tokens.size() <= shape.max_context,
                "input exceeds context");
        tokens(r.ids, vocabulary);
        tokens(r.count_tokens, public_tokens);
        std::uint32_t previous = 0;
        for (auto end : r.window_ends) {
            require(end > previous && end < r.ids.size(), "invalid window boundary");
            previous = end;
        }
        for (std::size_t j = 0; j < i; ++j) {
            require(r.member.sequence != rows[j].member.sequence &&
                        r.member.residency.lane != rows[j].member.residency.lane &&
                        r.main_table.index != rows[j].main_table.index,
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
            require(n && f->sample == SampleKind::None, "control extent");
            break;
        default:
            require(false, "unknown forward kind");
        }
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
                    r.drafts.empty(),
                "decision member");
        requested[r.member] = true;
    }
    for (const auto& r : input.rows) {
        require(r.member < rows.size() && requested[r.member] && !seen[r.member] &&
                    r.positions == 1 && r.words.size() == (std::size_t(public_tokens) + 31) / 32 &&
                    r.dead_positions <= 1,
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
    const bool sampled = f && f->sample != SampleKind::None;
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
            require(result[i].tokens.size() == 1 && c.accepted == 1 && !result[i].dead_positions,
                    "invalid sampled commit");
        } else
            require(!c.failed && c.accepted == rows[i].ids.size(),
                    "invalid processed-input commit");
    }
}
} // namespace ninfer::models::glm5next
