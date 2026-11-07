#include "models/qwen3_5/work.h"

namespace ninfer::models::qwen3_5 {
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
                for (const auto& row : op.rows) result.scheduled_positions += row.draft_extent + 1;
            } else if constexpr (std::is_same_v<T, MixedWork>) {
                // The execution is a prefill whose segments are joined by decode rows.
                result.kind = ExecutionKind::Prefill;
                result.rows = op.prefill.rows.size() + op.decode.rows.size();
                for (const auto& input : op.prefill.inputs) result.input_tokens += input.tokens.size();
                result.scheduled_positions = result.input_tokens;
                result.input_tokens += op.decode.rows.size();
                for (const auto& row : op.decode.rows) result.scheduled_positions += row.draft_extent + 1;
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
