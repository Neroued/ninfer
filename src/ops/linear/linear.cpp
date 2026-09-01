#include "ninfer/ops/linear.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <cstdint>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4G64_F16S:
        detail::q4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::Q5G64_F16S:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6G64_F16S:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::W8G32_F16S:
        detail::w8_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16_CTRL:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16S:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::Q4G64_F16S: {
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
#ifdef NINFER_VOLTA_BUILD
        // The fused tensor-core route needs an fp32 split-K accumulator. Size for the widest
        // token count in the band it is routed for; q4_dispatch falls back to SIMT when the
        // arena cannot supply it, so this is an opportunity, not a requirement.
        const std::int32_t banded = std::min<std::int32_t>(max_tokens, 64);
        if (banded >= 16 && detail::q4_volta_mma_supported(output_rows, input_rows, banded)) {
            return detail::q4_volta_mma_workspace_bytes(output_rows, input_rows, banded);
        }
#endif
        return 0;
    }
    case QType::Q5G64_F16S:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6G64_F16S:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::W8G32_F16S:
        (void)detail::select_w8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_w8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16_CTRL:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        if (!detail::is_nvfp4_linear_problem(output_rows, input_rows) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

} // namespace ninfer::ops
