#include "ninfer/ops/linear_swiglu.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

} // namespace

std::size_t linear_swiglu_workspace_bytes(std::int32_t gate_up_rows, std::int32_t max_tokens) {
    if (gate_up_rows == 12288) {
        (void)detail::w8_linear_swiglu_resolve_plan({12288, 6144, 2048, 2048, max_tokens});
        return 0;
    }
    return detail::q4_linear_swiglu_capacity_workspace_bytes(gate_up_rows, gate_up_rows / 2, 5120,
                                                             5120, max_tokens);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t = x.ne[1];
    const bool q4_shape  = x.ne[0] == 5120 && out.ne[0] == 17408 && gate_up_weight.n == 34816 &&
                          gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 34816 &&
                          gate_up_weight.padded_shape[1] == 5120;
    const bool w8_shape = x.ne[0] == 2048 && out.ne[0] == 6144 && gate_up_weight.n == 12288 &&
                          gate_up_weight.k == 2048 && gate_up_weight.padded_shape[0] == 12288 &&
                          gate_up_weight.padded_shape[1] == 2048;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!q4_shape && !w8_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("linear_swiglu: x/out data must be non-null");
    }
    const bool common_weight = gate_up_weight.layout == QuantLayout::RowSplit &&
                               gate_up_weight.scale_dtype == DType::FP16 &&
                               gate_up_weight.ndim == 2 &&
                               gate_up_weight.shape[0] == gate_up_weight.n &&
                               gate_up_weight.shape[1] == gate_up_weight.k &&
                               gate_up_weight.qdata != nullptr && gate_up_weight.scales != nullptr;
    const bool q4_weight = q4_shape && gate_up_weight.qtype == QType::Q4G64_F16S &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64;
    const bool w8_weight = w8_shape && gate_up_weight.qtype == QType::W8G32_F16S &&
                           gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
                           gate_up_weight.qhigh == nullptr && gate_up_weight.high_plane_bytes == 0;
    if (!common_weight || (!q4_weight && !w8_weight)) {
        throw std::invalid_argument("linear_swiglu: unsupported RowSplit weight");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16) ||
        !aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, w8_weight ? 16 : 4)) {
        throw std::invalid_argument(
            "linear_swiglu: required x/out/code/scale alignment is missing");
    }

    if (w8_weight) {
        (void)ws;
        detail::w8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
    }
}

} // namespace ninfer::ops
