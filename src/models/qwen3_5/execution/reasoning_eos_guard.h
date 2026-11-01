#pragma once

// Reasoning EOS guard (serve --reasoning-eos-guard, on in --profile flash-next). Qwen3.8-Flash-Next
// sometimes draws its end-of-turn token as the very first token of a reply that starts in thinking
// ("<think>\n" at the end of the prompt): the reply is empty and an agent harness stops. With the
// guard, that first draw cannot be a terminal token: when it is, the token is redrawn from the same
// distribution without the terminal ids (ops::replace_banned_token). A first token that is not
// terminal is left untouched.
// Process-wide, configured once before serving. Only the first generated token is guarded (the
// prefill's final sample, fresh or from an exact cache hit); a stop later in the reasoning is the
// job of the serve-side reasoning-close retry.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <cstdint>
#include <span>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::models::qwen3_5::execution {

struct ReasoningEosGuardConfig {
    bool enabled         = false;
    std::int32_t banned0 = -1;
    std::int32_t banned1 = -1;
    std::vector<TokenId> think_open; // the prompt suffix of a reply that starts in thinking
};

// Not thread-safe against running requests: call before the first request.
void configure_reasoning_eos_guard(ReasoningEosGuardConfig config);

// True when the guard is on and the prompt ends with the thinking opener.
[[nodiscard]] bool reasoning_eos_guard_applies(std::span<const int> prompt) noexcept;

// After the prefill's ops::sample of the first generated token (B = 1).
void guard_first_token(Tensor& logits, Tensor& token, std::int32_t token_domain,
                       const ops::SamplingConfig* configs, const Tensor& logical_positions,
                       WorkspaceArena& workspace, cudaStream_t stream);

// Replacements since the process started.
[[nodiscard]] std::uint64_t reasoning_eos_guard_replacements() noexcept;

} // namespace ninfer::models::qwen3_5::execution
