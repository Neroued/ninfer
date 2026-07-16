#include "ninfer/ops/linear_swiglu.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

} // namespace

std::size_t linear_swiglu_workspace_bytes(std::int32_t gate_up_rows, std::int32_t max_tokens) {
    return detail::q4_linear_swiglu_capacity_workspace_bytes(gate_up_rows, gate_up_rows / 2, 5120,
                                                             5120, max_tokens);
}

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
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16) ||
        !aligned_to(gate_up_weight.qdata, 16) || !aligned_to(gate_up_weight.scales, 4)) {
        throw std::invalid_argument("linear_swiglu: Q4 requires aligned x/out/code/scale storage");
    }

    detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
}

} // namespace ninfer::ops
