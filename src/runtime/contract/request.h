#pragma once

#include "ninfer/types.h"
#include <atomic>
#include <cstdint>
#include <optional>

namespace ninfer::runtime {

// Engine has already selected the model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class ContinuationAction : std::uint8_t {
    Decode,
    ApplyTargetControl,
};

struct OutputDecision {
    std::uint32_t accepted_tokens   = 0;
    FinishReason finish_reason      = FinishReason::None;
    ContinuationAction continuation = ContinuationAction::Decode;
    // Empty or one token-aligned frontier within the accepted span where model-history
    // reconstruction gains an execution split. Frontend owns detection; Engine only transports
    // this relative position.
    std::optional<std::uint32_t> prefix_execution_split_after;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Non-owning cancellation observation used while the worker advances a context transaction. The
// request record owns the flag for longer than Program can retain this view.
struct CancellationFlagView {
    const std::atomic<bool>* flag = nullptr;

    [[nodiscard]] bool requested() const noexcept {
        return flag != nullptr && flag->load(std::memory_order_acquire);
    }
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::Root;
    std::uint64_t service_work_quanta     = 0;
    bool publish_continuation             = true;
};

} // namespace ninfer::runtime
