#include "models/qwen3_5/work.h"

namespace ninfer::models::qwen3_5 {
DecodeWork::DecodeWork(speculative::DecodeRoundPlan selected, ExecutionPath path)
    : plan(std::make_shared<const speculative::DecodeRoundPlan>(std::move(selected))),
      geometry(std::make_shared<const SpeculativeExecutionGeometry>(plan->geometry(path))) {}

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
