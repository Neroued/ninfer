#include "ops/linear/fp8/fp8_dispatch.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/fp8/fp8_launch.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearRoute : std::uint8_t {
    A16,
    A8,
};

Fp8LinearRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy,
                             std::int32_t tokens) {
    if (tokens <= 0 || !is_fp8_linear_problem(output_rows, input_rows)) {
        throw std::invalid_argument("fp8 linear: unsupported shape");
    }
    const Fp8Problem problem = resolve_fp8_problem(output_rows, input_rows);
    if (policy == LinearPolicy::A16Only) { return Fp8LinearRoute::A16; }
    // A permissive policy does not require a lower-precision route. Vocabulary logits retain
    // BF16 activation compute for every policy, matching the existing Q6/W8 output heads.
    if (problem == Fp8Problem::Vocabulary &&
        (policy == LinearPolicy::AllowA8 || policy == LinearPolicy::AllowA4)) {
        return Fp8LinearRoute::A16;
    }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 linear: unsupported policy");
    }

    switch (problem) {
    case Fp8Problem::AttnInput:
        return tokens >= 12 ? Fp8LinearRoute::A8 : Fp8LinearRoute::A16;
    case Fp8Problem::GdnInput:
        return tokens >= 11 ? Fp8LinearRoute::A8 : Fp8LinearRoute::A16;
    case Fp8Problem::MlpGateUp:
        return tokens == 1 || tokens >= 5 ? Fp8LinearRoute::A8 : Fp8LinearRoute::A16;
    case Fp8Problem::Vocabulary:
        return Fp8LinearRoute::A16;
    case Fp8Problem::Residual6144:
    case Fp8Problem::Residual17408:
        return tokens >= 25 ? Fp8LinearRoute::A8 : Fp8LinearRoute::A16;
    }
    throw std::logic_error("unreachable FP8 linear problem");
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const Fp8Problem problem = resolve_fp8_problem(weight.n, weight.k);
#ifndef NINFER_VOLTA_BUILD
    if (problem == Fp8Problem::Vocabulary && x.ne[1] >= kFp8VocabularyFirstA16GemmT) {
        launch_fp8_vocabulary_a16_gemm(x, weight, out, stream);
        return;
    }
    const std::int32_t chunk = problem == Fp8Problem::Vocabulary ? kFp8VocabularyLastA16SmallTMmaT
                                                                 : fp8_linear_small_t_max(problem);
#else
    const bool qpn = fp8_volta_qpn_supported(weight.n, weight.k, kFp8VoltaQpnMaxTokens);
    const std::int32_t chunk = qpn ? kFp8VoltaQpnMaxTokens : fp8_linear_small_t_max(problem);
#endif
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += chunk) {
        const std::int32_t active = std::min(chunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {weight.n, active});
        #ifdef NINFER_VOLTA_BUILD
        if (fp8_volta_qpn_supported(weight.n, weight.k, active)) {
            launch_fp8_volta_qpn(input_chunk, weight, output_chunk, stream);
        } else if (active == 1) {
            launch_fp8_decode(input_chunk, weight, output_chunk, stream);
        } else {
            launch_fp8_small_t(input_chunk, weight, output_chunk, stream);
        }
        #else
        if (problem == Fp8Problem::Vocabulary) {
            launch_fp8_vocabulary_a16_small_t(input_chunk, weight, output_chunk, stream);
        } else if (active == 1) {
            launch_fp8_decode(input_chunk, weight, output_chunk, stream);
        } else {
            launch_fp8_small_t(input_chunk, weight, output_chunk, stream);
        }
        #endif
    }
}

bool interval_uses_a8(Fp8Problem problem, LinearPolicy policy, std::int32_t min_tokens,
                      std::int32_t max_tokens) {
    if (policy == LinearPolicy::A16Only) { return false; }
    switch (problem) {
    case Fp8Problem::AttnInput:
        return max_tokens >= 12;
    case Fp8Problem::GdnInput:
        return max_tokens >= 11;
    case Fp8Problem::MlpGateUp:
        return min_tokens == 1 || max_tokens >= 5;
    case Fp8Problem::Vocabulary:
        return false;
    case Fp8Problem::Residual6144:
    case Fp8Problem::Residual17408:
        return max_tokens >= 25;
    }
    throw std::logic_error("unreachable FP8 linear problem");
}

} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                LinearPolicy policy, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    }
    const Fp8Problem problem = resolve_fp8_problem(output_rows, input_rows);
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    (void)resolve_route(output_rows, input_rows, policy, max_tokens);
    return interval_uses_a8(problem, policy, min_tokens, max_tokens)
               ? fp8_a8_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    const Fp8LinearRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Fp8LinearRoute::A16) {
        launch_a16(x, weight, out, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
    launch_fp8_a8(x, weight, out, scratch, stream);
}

} // namespace ninfer::ops::detail
