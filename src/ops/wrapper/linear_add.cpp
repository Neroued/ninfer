#include "ninfer/ops/linear_add.h"

#include "ninfer/ops/residual_add.h"
#include "ops/linear/q5/q5_rowsplit_add.h"
#include "ops/linear/q5/q5_rowsplit_plan.h"

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

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

} // namespace

std::size_t linear_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                       std::int32_t tokens) {
    return detail::q5_rowsplit_add_workspace_bytes(output_rows, input_rows, tokens);
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, WorkspaceArena& ws,
                cudaStream_t stream) {
    require_q5(w);
    const std::int32_t t = x.ne[1];
    require_tensor(x, DType::BF16, w.k, t, "x");
    require_tensor(residual_out, DType::BF16, w.n, t, "residual_out");

    const bool supported_shape = (w.n == 5120 && w.k == 17408) || (w.n == 5120 && w.k == 6144);
    if (!supported_shape) { throw std::invalid_argument("linear_add: unsupported Q5 shape"); }
    if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) || !aligned_to(w.qdata, 16) ||
        !aligned_to(w.qhigh, 16) || !aligned_to(w.scales, 16)) {
        throw std::invalid_argument(
            "linear_add: Q5 requires 16-byte x/residual/code/high/scale alignment");
    }

    if (detail::q5_rowsplit_add_route(t) != detail::Q5AddRoute::Materialized) {
        detail::q5_rowsplit_add_dispatch(x, w, residual_out, stream);
        return;
    }

    auto scratch_scope = ws.scope();
    Tensor linear_out  = ws.alloc(DType::BF16, {w.n, t});
    detail::q5_rowsplit_dispatch(x, w, linear_out, stream);
    residual_add(linear_out, residual_out, stream);
}

} // namespace ninfer::ops
