#include "core/weight.h"
#include "ninfer/ops/linear_swiglu.h"

#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/q3/q3_arena.h"
#include "ops/linear/q3/q3_linear.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/ffn14336/linear_swiglu_ffn14336.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/tp2/linear_swiglu_half.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

// The NVFP4 and FP8 gate/up problems: the single-device [34816,5120] and its two-device
// output-row half [17408,5120]. Both formats register the same two geometries.
constexpr std::int32_t kGateUpRows      = detail::Nvfp4N34816K5120::kOutputRows;
constexpr std::int32_t kShardGateUpRows = detail::Nvfp4N17408K5120::kOutputRows;
constexpr std::int32_t kGateUpInputRows = detail::Nvfp4N34816K5120::kInputRows;
static_assert(detail::Fp8N34816K5120::kOutputRows == kGateUpRows &&
              detail::Fp8N17408K5120::kOutputRows == kShardGateUpRows &&
              detail::Fp8N34816K5120::kInputRows == kGateUpInputRows &&
              detail::Fp8N17408K5120::kInputRows == kGateUpInputRows &&
              detail::Nvfp4N17408K5120::kInputRows == kGateUpInputRows);

bool fused_gate_up_problem(std::int32_t gate_up_rows, std::int32_t input_rows) {
    return (gate_up_rows == kGateUpRows || gate_up_rows == kShardGateUpRows) &&
           input_rows == kGateUpInputRows;
}

// NVFP4 only: the dense FFN-14336 gate|up [28672,5120] and its two-device half [14336,5120]
// (nvfp4/ffn14336/linear_swiglu_ffn14336.h).
bool ffn14336_gate_up_problem(std::int32_t gate_up_rows, std::int32_t input_rows) {
    return (gate_up_rows == detail::kLinearSwiGluFfn14336Rows ||
            gate_up_rows == detail::kLinearSwiGluFfn14336HalfRows) &&
           input_rows == kGateUpInputRows;
}

} // namespace

bool linear_swiglu_registered(QType qtype, std::int32_t gate_up_rows, std::int32_t input_rows) {
    const bool large = gate_up_rows == kGateUpRows && input_rows == kGateUpInputRows;
    const bool fused = fused_gate_up_problem(gate_up_rows, input_rows);
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return large;
    case QType::Q8_G32_FP16:
        return large || (gate_up_rows == 12288 && input_rows == 2048);
    case QType::NVFP4:
        return fused || ffn14336_gate_up_problem(gate_up_rows, input_rows);
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::Q3_G128_FP16:
        return fused;
    default:
        return false;
    }
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (qtype == QType::Q8_G32_FP16) {
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q3_G128_FP16) {
        if (input_rows % 128 != 0) {
            throw std::invalid_argument("linear_swiglu workspace: Q3 needs K a multiple of 128");
        }
        return detail::q3_workspace_bytes(input_rows, max_tokens);
    }
    if (qtype == QType::Q4_G64_FP16) {
        return detail::q4_linear_swiglu_capacity_workspace_bytes(
            gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4 && fused_gate_up_problem(gate_up_rows, input_rows)) {
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4 && ffn14336_gate_up_problem(gate_up_rows, input_rows)) {
        return gate_up_rows == detail::kLinearSwiGluFfn14336Rows
                   ? detail::nvfp4_linear_swiglu_n28672_workspace_capacity_bytes(policy, min_tokens,
                                                                                 max_tokens)
                   : detail::nvfp4_linear_swiglu_n14336_workspace_capacity_bytes(policy, min_tokens,
                                                                                 max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16 && fused_gate_up_problem(gate_up_rows, input_rows)) {
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

namespace {

bool matches_problem(const Tensor& x, const Weight& w, const Tensor& out, std::int32_t gate_up_rows,
                     std::int32_t input_rows) {
    return x.ne[0] == input_rows && out.ne[0] == gate_up_rows / 2 && w.n == gate_up_rows &&
           w.k == input_rows && w.padded_shape[0] == gate_up_rows &&
           w.padded_shape[1] == input_rows;
}

// Every check linear_swiglu() makes, so that a split form can reject a rank pair before either
// rank issues work.
void validate_linear_swiglu(const Tensor& x, const Weight& gate_up_weight, const Tensor& out,
                            LinearPolicy policy) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t   = x.ne[1];
    const bool large_shape = matches_problem(x, gate_up_weight, out, kGateUpRows, kGateUpInputRows);
    const bool shard_shape =
        matches_problem(x, gate_up_weight, out, kShardGateUpRows, kGateUpInputRows);
    const bool q8_shape = matches_problem(x, gate_up_weight, out, 12288, 2048);
    const bool ffn14336_shape =
        matches_problem(x, gate_up_weight, out, detail::kLinearSwiGluFfn14336Rows,
                        kGateUpInputRows) ||
        matches_problem(x, gate_up_weight, out, detail::kLinearSwiGluFfn14336HalfRows,
                        kGateUpInputRows);
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !shard_shape && !q8_shape && !ffn14336_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight = large_shape && gate_up_weight.qtype == QType::Q4_G64_FP16 &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                           common_row_split;
    const bool q8_weight =
        (q8_shape || large_shape) && gate_up_weight.qtype == QType::Q8_G32_FP16 &&
        gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
        gate_up_weight.qhigh == nullptr && gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool fused_shape = large_shape || shard_shape;
    const bool nvfp4_weight =
        (fused_shape || ffn14336_shape) && gate_up_weight.qtype == QType::NVFP4;
    const bool fp8_weight = fused_shape && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16;
    // v0.7 27b-16gb: the GSQ-3 gate|up parent [34816,5120] (and its [17408,5120] half).
    const bool q3_weight = fused_shape && gate_up_weight.qtype == QType::Q3_G128_FP16;
    if (q3_weight) {
        detail::validate_q3_weight(gate_up_weight, "linear_swiglu");
        return;
    }
    if (!q4_weight && !q8_weight && !nvfp4_weight && !fp8_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (fp8_weight) {
        (void)detail::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        return;
    }

    if (nvfp4_weight) {
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
        return;
    }

    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, q8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }
}

// Issues a validated call. `ws` may be null when the resolved route needs no workspace.
void dispatch_linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                            LinearPolicy policy, WorkspaceArena* ws, cudaStream_t stream) {
    switch (gate_up_weight.qtype) {
    case QType::Q3_G128_FP16: {
        const detail::Q3ArenaWorkspace q3ws(ws, gate_up_weight.k, x.ne[1]);
        detail::q3_linear_swiglu_dispatch(x, gate_up_weight, out, stream, q3ws.get());
        return;
    }
    case QType::FP8_E4M3FN_ROW_BF16:
        if (gate_up_weight.n == kShardGateUpRows) {
            return detail::fp8_linear_swiglu_half_dispatch(x, gate_up_weight, out, policy, ws,
                                                           stream);
        }
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    case QType::NVFP4:
        if (gate_up_weight.n == kShardGateUpRows) {
            return detail::nvfp4_linear_swiglu_half_dispatch(x, gate_up_weight, out, policy, ws,
                                                             stream);
        }
        if (gate_up_weight.n == detail::kLinearSwiGluFfn14336Rows) {
            return detail::nvfp4_linear_swiglu_n28672_dispatch(x, gate_up_weight, out, policy, ws,
                                                               stream);
        }
        if (gate_up_weight.n == detail::kLinearSwiGluFfn14336HalfRows) {
            return detail::nvfp4_linear_swiglu_n14336_dispatch(x, gate_up_weight, out, policy, ws,
                                                               stream);
        }
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    case QType::Q8_G32_FP16:
        detail::q8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
        return;
    case QType::Q4_G64_FP16:
        if (ws == nullptr) {
            throw std::invalid_argument("linear_swiglu: Q4 requires caller workspace");
        }
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, *ws, stream);
        return;
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::BF16:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear_swiglu: unsupported weight");
}

} // namespace

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_linear_swiglu(x, gate_up_weight, out, policy);
    dispatch_linear_swiglu(x, gate_up_weight, out, policy, &ws, stream);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

} // namespace ninfer::ops

#include "ops/wrapper/tp2/linear_swiglu_tp2.inc"
