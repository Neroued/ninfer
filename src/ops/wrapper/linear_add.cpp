#include "ninfer/ops/linear_add.h"

#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t columns,
                    const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != columns || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("linear_add: invalid ") + name);
    }
}

void require_q5(const Weight& w) {
    if (w.qtype != QType::Q5G64_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 64 || w.group != 64 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.qdata == nullptr ||
        w.qhigh == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_add: weight must be Q5G64_F16S row-split");
    }
}

void require_w8(const Weight& w) {
    if (w.qtype != QType::W8G32_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 32 || w.group != 32 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.qdata == nullptr ||
        w.qhigh != nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_add: weight must be W8G32_F16S row-split");
    }
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

} // namespace

std::size_t linear_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                       std::int32_t max_tokens) {
    if (output_rows == 2048 && (input_rows == 4096 || input_rows == 6144)) {
        (void)detail::w8_linear_add_resolve_plan({output_rows, input_rows, input_rows, max_tokens});
        return 0;
    }
    return detail::q5_linear_add_capacity_workspace_bytes(output_rows, input_rows, input_rows,
                                                          max_tokens);
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, WorkspaceArena& ws,
                cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    require_tensor(x, DType::BF16, w.k, t, "x");
    require_tensor(residual_out, DType::BF16, w.n, t, "residual_out");

    if (w.qtype == QType::Q5G64_F16S) {
        require_q5(w);
        const bool supported_shape = (w.n == 5120 && w.k == 17408) || (w.n == 5120 && w.k == 6144);
        if (!supported_shape) { throw std::invalid_argument("linear_add: unsupported Q5 shape"); }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16) || !aligned_to(w.qhigh, 16) || !aligned_to(w.scales, 16)) {
            throw std::invalid_argument(
                "linear_add: Q5 requires 16-byte x/residual/code/high/scale alignment");
        }
        detail::q5_linear_add_dispatch(x, w, residual_out, ws, stream);
        return;
    }

    if (w.qtype == QType::W8G32_F16S) {
        require_w8(w);
        if (w.n != 2048 || (w.k != 4096 && w.k != 6144)) {
            throw std::invalid_argument("linear_add: unsupported W8 shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16) || !aligned_to(w.scales, 16)) {
            throw std::invalid_argument(
                "linear_add: W8 requires 16-byte x/residual/code/scale alignment");
        }
        (void)ws;
        detail::w8_linear_add_dispatch(x, w, residual_out, stream);
        return;
    }

    throw std::invalid_argument("linear_add: unsupported weight format");
}

} // namespace ninfer::ops
