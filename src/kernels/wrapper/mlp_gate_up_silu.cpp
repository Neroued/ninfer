#include "qus/kernels/mlp_gate_up_silu.h"

#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_gate_up_34816.cuh"

#include <stdexcept>

namespace qus::kernels {

void mlp_gate_up_silu_decode(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                             cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("mlp_gate_up_silu_decode: x/out must be BF16");
    }
    if (x.ne[0] != 5120 || x.ne[1] != 1 || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[0] != 17408 || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("mlp_gate_up_silu_decode: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("mlp_gate_up_silu_decode: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("mlp_gate_up_silu_decode: x/out data must be non-null");
    }
    if (gate_up_weight.qtype != QType::Q4G64_F16S ||
        gate_up_weight.layout != QuantLayout::RowSplit || gate_up_weight.qdata == nullptr ||
        gate_up_weight.scales == nullptr) {
        throw std::invalid_argument("mlp_gate_up_silu_decode: weight must be Q4 rowsplit");
    }

    detail::linear_rowsplit_gemv_mlp_gate_up_silu_17408_q4_launch(x, gate_up_weight, out,
                                                                  stream);
}

} // namespace qus::kernels
