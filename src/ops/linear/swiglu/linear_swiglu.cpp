#include "core/platform.h"
#include "core/weight.h"
#include "ninfer/ops/linear_swiglu.h"

#include "ops/linear/swiglu/fp8/dispatch.h"
#include "ops/linear/swiglu/nvfp4/dispatch.h"
#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/common/operands.h"
#include "ops/common/host/operands.h"

#include <cstdint>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

} // namespace

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    detail::validate_swiglu_geometry(qtype, gate_up_rows, input_rows);
    if (qtype == QType::BF16) return 0;
    if (qtype == QType::Q8_G32_FP16) return 0;
    if (qtype == QType::Q4_G64_FP16) {
        return detail::q4_linear_swiglu_capacity_workspace_bytes(gate_up_rows, input_rows,
                                                                 min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4) {
#if NINFER_TARGET_SPARK
        if (gate_up_rows == 12288 && input_rows == 4096)
            return detail::nvfp4_swiglu_n12288_k4096_capacity(policy, min_tokens, max_tokens);
#endif
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16) {
        return detail::fp8_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

static void execute_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, float limit,
                           bool clamped, LinearPolicy policy, MemoryArena& ws,
                           CudaExecutionView execution) {
    validate_policy(policy);
    if (x.device != execution.device() || out.device != execution.device() ||
        gate_up_weight.device != execution.device() || ws.device() != execution.device()) {
        throw std::invalid_argument(
            "linear_swiglu: operands and workspace must belong to execution");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const auto& w = gate_up_weight;
    if (clamped && (!std::isfinite(limit) || limit <= 0.0f))
        throw std::invalid_argument("linear_swiglu: clamp limit must be finite and positive");
    detail::validate_swiglu_geometry(w.qtype, w.n, w.k, clamped);
    const int t = x.ne[1];
    if (t <= 0 || x.ne[0] != w.k || out.ne[0] != w.n / 2 || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[1] != t || out.ne[2] != 1 || out.ne[3] != 1 || w.padded_shape[0] != w.n ||
        w.padded_shape[1] != w.k)
        throw std::invalid_argument("linear_swiglu: expected [K,T] and [M,T] for weight [2*M,K]");
    if (!x.is_contiguous() || !out.is_contiguous() || !common::operand::aligned(x.data, 16) ||
        !common::operand::aligned(out.data, 16))
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous and 16-byte aligned");

#if NINFER_TARGET_SPARK
    if (w.qtype == QType::BF16) {
        detail::validate_bf16_projection(x, w, {&out, 1}, nullptr, execution, "linear_swiglu");
        detail::bf16_linear_swiglu_dispatch(x, w, out, limit, clamped, execution.stream());
        return;
    }
#endif
    const bool q8_weight    = w.qtype == QType::Q8_G32_FP16;
    const bool fp8_weight   = w.qtype == QType::FP8_E4M3FN_ROW_BF16;
    const bool nvfp4_weight = w.qtype == QType::NVFP4;
    if (!fp8_weight && !nvfp4_weight) {
        const int group = q8_weight ? 32 : 64;
        if (w.layout != QuantLayout::RowSplit || w.scale_dtype != DType::FP16 || w.ndim != 2 ||
            w.shape[0] != w.n || w.shape[1] != w.k || w.group_size != group || w.group != group ||
            w.qdata == nullptr || w.scales == nullptr ||
            (q8_weight && (w.qhigh != nullptr || w.high_plane_bytes != 0)))
            throw std::invalid_argument("linear_swiglu: invalid row-split weight representation");
    }

    if (fp8_weight) {
        (void)common::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, execution.stream());
        return;
    }

    if (nvfp4_weight) {
        (void)common::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
#if NINFER_TARGET_SPARK
        if (w.n == 12288 && w.k == 4096) {
            detail::launch_nvfp4_swiglu_n12288_k4096(x, w, out, limit, clamped, policy, ws,
                                                     execution.stream());
            return;
        }
#endif
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws,
                                             execution.stream());
        return;
    }

    if (!common::operand::aligned(gate_up_weight.qdata, 16) ||
        !common::operand::aligned(gate_up_weight.scales, q8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }

    if (q8_weight) {
        detail::q8_linear_swiglu_dispatch(x, gate_up_weight, out, limit, clamped,
                                          execution.stream());
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, execution);
    }
}

void linear_swiglu(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   MemoryArena& ws, CudaExecutionView execution) {
    execute_swiglu(x, weight, out, 0.0f, false, policy, ws, execution);
}

void linear_swiglu(const Tensor& x, const Weight& weight, Tensor& out, float limit,
                   LinearPolicy policy, MemoryArena& ws, CudaExecutionView execution) {
    execute_swiglu(x, weight, out, limit, true, policy, ws, execution);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, MemoryArena& ws,
                   CudaExecutionView execution) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, execution);
}

} // namespace ninfer::ops
