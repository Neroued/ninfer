#include "targets/qwen3_6_27b_rtx5090/impl/kernels/linear_swiglu/linear_swiglu.h"
#include "targets/qwen3_6_27b_rtx5090/impl/kernels/linear_swiglu/launch.h"

#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_gate_up_34816.cuh"
#include "kernels/linear/linear.h"
#include "kernels/silu_mul/silu_mul.h"

#include <stdexcept>

namespace ninfer::kernels {

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t = x.ne[1];
    if (x.ne[0] != 5120 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[0] != 17408 || out.ne[1] != t ||
        out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("linear_swiglu: x/out data must be non-null");
    }
    if (gate_up_weight.qtype != QType::Q4G64_F16S ||
        gate_up_weight.layout != QuantLayout::RowSplit ||
        gate_up_weight.scale_dtype != DType::FP16 || gate_up_weight.group_size != 64 ||
        gate_up_weight.group != 64 || gate_up_weight.qdata == nullptr ||
        gate_up_weight.scales == nullptr) {
        throw std::invalid_argument("linear_swiglu: weight must be Q4 rowsplit");
    }

    if (gate_up_weight.n != 34816 || gate_up_weight.k != 5120 ||
        gate_up_weight.padded_shape[0] != 34816 || gate_up_weight.padded_shape[1] != 5120) {
        throw std::invalid_argument("linear_swiglu: weight must be 34816x5120");
    }
    if (t == 1) {
        detail::linear_rowsplit_gemv_mlp_gate_up_silu_17408_q4_launch(x, gate_up_weight, out,
                                                                      stream);
        return;
    }
    if (t > 16) {
        detail::linear_rowsplit_q4_gate_up_silu_gemm_mma_launch(x, gate_up_weight, out, stream);
        return;
    }

    auto scratch_scope = ws.scope();
    Tensor gate_up     = ws.alloc(DType::BF16, {34816, t});
    linear(x, gate_up_weight, gate_up, ws, stream);
    silu_mul(gate_up.slice(0, 0, 17408), gate_up.slice(0, 17408, 17408), out, stream);
}

} // namespace ninfer::kernels
