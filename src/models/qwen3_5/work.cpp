#include "models/qwen3_5/work.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
DecodeWork::DecodeWork(speculative::DecodeRoundPlan selected, ExecutionPath path)
    : plan(std::make_shared<const speculative::DecodeRoundPlan>(std::move(selected))),
      geometry(std::make_shared<const SpeculativeExecutionGeometry>(plan->geometry(path))) {}

namespace {
void validate_prefill_work(const ForwardWork& op, std::uint32_t prefill_chunk) {
    if (std::ranges::any_of(op.inputs, [](const auto& w) { return w.visual.has_value(); }) &&
        (op.rows.size() != 1 || op.inputs.front().tokens.empty()))
        throw std::invalid_argument("Qwen3.5 visual prefill requires one nonempty input window");
    std::size_t total = 0;
    bool any_sample   = false;
    for (std::size_t i = 0; i < op.rows.size(); ++i) {
        const auto& row = op.rows[i];
        const auto n    = op.inputs[i].tokens.size();
        if (n > prefill_chunk - total)
            throw std::invalid_argument("Qwen3.5 prefill exceeds its packed token budget");
        total += n;
        const bool sampled     = !op.sampling.empty() && op.sampling[i].sample;
        const bool constrained = !op.sampling.empty() && op.sampling[i].constrained;
        any_sample |= sampled;
        if ((!n && (!sampled || !row.main_frontier)) || (constrained && !sampled))
            throw std::invalid_argument("Qwen3.5 prefill row has invalid sampling or extent");
        for (std::size_t j = 0; j < i; ++j) {
            const auto& prev        = op.rows[j];
            const bool prev_updates = !op.inputs[j].tokens.empty();
            if ((n && prev_updates &&
                 (row.destination_state == prev.destination_state ||
                  row.state_offsets.destination == prev.state_offsets.destination)) ||
                (prev_updates && (row.source_state == prev.destination_state ||
                                  row.state_offsets.source == prev.state_offsets.destination)) ||
                (n && (row.destination_state == prev.source_state ||
                       row.state_offsets.destination == prev.state_offsets.source)) ||
                row.main_table.index == prev.main_table.index ||
                (row.backend_table && prev.backend_table &&
                 row.backend_table->index == prev.backend_table->index))
                throw std::invalid_argument("Qwen3.5 prefill rows overlap State or KV writes");
        }
    }
    if (op.sample != (any_sample ? SamplePurpose::FirstToken : SamplePurpose::None))
        throw std::invalid_argument("Qwen3.5 prefill sample purpose differs from its rows");
}

template <class Count>
void validate_commit_rows_impl(const Work& work, const Commit& commit,
                               std::span<const Count> counts) {
    const auto* forward   = std::get_if<ForwardWork>(&work.operation);
    const auto* decode    = std::get_if<DecodeWork>(&work.operation);
    const auto* normalize = std::get_if<NormalizeWork>(&work.operation);
    const auto batch      = forward     ? forward->rows.size()
                            : decode    ? decode->rows.size()
                            : normalize ? normalize->rows.size()
                                        : 1;
    if (!decode && (!forward || forward->sample == SamplePurpose::None)) {
        if (!commit.rows.empty() && commit.rows.size() != batch)
            throw std::invalid_argument("Qwen3.5 non-generating commit has invalid membership");
        return;
    }
    const bool packed = forward && forward->purpose == ForwardPurpose::Prefill;
    if (commit.rows.size() != batch || (counts.empty() ? !packed : counts.size() < batch))
        throw std::invalid_argument("Qwen3.5 generating commit has invalid membership");
    for (std::size_t i = 0; i < batch; ++i) {
        const auto& row = commit.rows[i];
        if (packed) {
            const bool sampled  = forward->sampling[i].sample;
            const auto expected = row.discarded ? 0
                                  : sampled     ? 1
                                                : forward->inputs[i].tokens.size();
            if (row.accepted_tokens != expected || (!sampled && row.terminal != row.discarded) ||
                (!counts.empty() && counts[i] != (sampled ? 1U : 0U)))
                throw std::invalid_argument(
                    "Qwen3.5 prefill commit has an invalid accepted prefix");
            continue;
        }
        const auto count = static_cast<std::uint32_t>(counts[i]);
        if (row.accepted_tokens > count || (row.discarded && row.accepted_tokens != 0) ||
            (!row.discarded && !row.accepted_tokens) ||
            (!row.terminal && !row.discarded && row.accepted_tokens != count)) {
            throw std::invalid_argument("Qwen3.5 commit contains an invalid accepted prefix");
        }
    }
}
} // namespace

void validate_forward_work(const ForwardWork& work, std::uint32_t prefill_chunk,
                           std::uint32_t control_limit, std::uint32_t public_tokens) {
    if (work.rows.empty() || work.inputs.size() != work.rows.size() ||
        (!work.sampling.empty() && work.sampling.size() != work.rows.size()) ||
        (work.sample != SamplePurpose::None && work.sampling.size() != work.rows.size()))
        throw std::invalid_argument("Qwen3.5 forward inputs do not match membership");
    if (work.purpose != ForwardPurpose::Control && !work.control_last_tokens.empty())
        throw std::invalid_argument("Qwen3.5 control output outside control work");
    switch (work.purpose) {
    case ForwardPurpose::Prefill:
        validate_prefill_work(work, prefill_chunk);
        return;
    case ForwardPurpose::Replay:
        if (work.rows.size() != 1 || work.sample != SamplePurpose::None)
            throw std::invalid_argument("Qwen3.5 Replay requires one unsampled sequence");
        return;
    case ForwardPurpose::Control: {
        const auto width = work.inputs.front().tokens.size();
        if (work.sample != SamplePurpose::None ||
            work.control_last_tokens.size() != work.rows.size() || !width ||
            width > control_limit ||
            !std::ranges::all_of(work.inputs,
                                 [width](const auto& w) { return w.tokens.size() == width; }) ||
            !std::ranges::all_of(work.control_last_tokens, [public_tokens](TokenId id) {
                return id >= 0 && static_cast<std::uint32_t>(id) < public_tokens;
            }))
            throw std::invalid_argument("Qwen3.5 control has invalid membership or token windows");
        return;
    }
    }
    throw std::invalid_argument("unknown Qwen3.5 forward purpose");
}

void validate_narrowing(const DecodeWork& work, std::uint32_t max_context) {
    if (!work.narrowing) return;
    const auto& narrowing = *work.narrowing;
    if (narrowing.rows.size() != work.rows.size() || narrowing.max_context != max_context)
        throw std::invalid_argument("Qwen3.5 narrowing differs from decode geometry");
    for (const auto& row : narrowing.rows) {
        if (row.member.owner != 0 || row.member.granted_limit > row.member.hard_limit ||
            row.frontier > max_context)
            throw std::invalid_argument("Qwen3.5 invalid narrowing row");
    }
}

void validate_decision_rows(const Work& work, const runtime::DecisionInput& input,
                            std::uint32_t vocabulary,
                            std::span<const std::uint32_t> verify_drafts) {
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    const auto* decode  = std::get_if<DecodeWork>(&work.operation);
    if ((!forward && !decode) || (forward && forward->sample == SamplePurpose::None))
        throw std::invalid_argument("Qwen3.5 decision supplied for non-sampling work");
    const auto batch = forward ? forward->rows.size() : decode->rows.size();
    if ((forward && forward->sampling.size() != batch) || (decode && verify_drafts.size() != batch))
        throw std::invalid_argument("Qwen3.5 decision geometry differs from membership");
    const auto words      = (std::size_t(vocabulary) + 31) / 32;
    const auto needs_mask = [&](std::size_t i) {
        const auto& row = forward ? forward->sampling[i] : decode->rows[i].sampling;
        return row.constrained &&
               (!forward || forward->purpose != ForwardPurpose::Prefill || row.sample);
    };
    std::vector<bool> present(batch);
    for (const auto& row : input.rows) {
        if (row.member >= batch || present[row.member] || !needs_mask(row.member) ||
            row.positions != (forward ? 1U : verify_drafts[row.member] + 1U) ||
            row.positions >= 32 || row.words.size() != words * row.positions ||
            (row.dead_positions >> row.positions) != 0)
            throw std::invalid_argument("Qwen3.5 decision mask has invalid shape or membership");
        present[row.member] = true;
    }
    for (std::size_t i = 0; i < batch; ++i)
        if (present[i] != needs_mask(i))
            throw std::invalid_argument("Qwen3.5 decision omitted a constrained row");
}

void validate_commit_rows(const Work& work, const Commit& commit,
                          std::span<const std::int32_t> counts) {
    validate_commit_rows_impl(work, commit, counts);
}

void validate_commit_rows(const Work& work, const Commit& commit,
                          std::span<const std::uint32_t> counts) {
    validate_commit_rows_impl(work, commit, counts);
}

void plan_work_batch(Work& work, const BatchExecutionPlan& plan) {
    std::visit(
        [&](auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, ScoreWork>) {
                op.shape = plan.exact(1, 1);
            } else if constexpr (!std::is_same_v<T, ContextWork> &&
                                 !std::is_same_v<T, DecodeWork>) {
                const auto rows = static_cast<std::uint32_t>(op.rows.size());
                op.shape        = plan.exact(rows, rows);
            }
        },
        work.operation);
}

ExecutionWork ExecutionTypes::describe_work(const Work& work) {
    ExecutionWork result;
    std::visit(
        [&](const auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, ForwardWork>) {
                result.kind = op.purpose == ForwardPurpose::Prefill  ? ExecutionKind::Prefill
                              : op.purpose == ForwardPurpose::Replay ? ExecutionKind::Replay
                                                                     : ExecutionKind::Control;
                result.rows = op.rows.size();
                for (const auto& input : op.inputs) result.input_tokens += input.tokens.size();
                result.scheduled_positions = result.input_tokens;
            } else if constexpr (std::is_same_v<T, DecodeWork>) {
                result.kind         = ExecutionKind::Decode;
                result.rows         = op.rows.size();
                result.input_tokens = op.rows.size();
                result.scheduled_positions = op.plan->valid_target_tokens();
                result.speculative         = op.geometry;
            } else if constexpr (std::is_same_v<T, ScoreWork>) {
                result.kind         = ExecutionKind::Score;
                result.rows         = 1;
                result.input_tokens = result.scheduled_positions = op.input.tokens.size();
            }
        },
        work.operation);
    return result;
}
} // namespace ninfer::models::qwen3_5
