#include "ninfer/ops/rope.h"
#include "ops/rope/apply/dispatch.h"
#include "ops/common/host/operands.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kTextHeadDim = 256;

int position_axes(const Tensor& positions, std::int32_t tokens) {
    if (positions.ne[0] != tokens || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        (positions.ne[1] != 1 && positions.ne[1] != 2 && positions.ne[1] != 3)) {
        throw std::invalid_argument("rope: positions must have shape [T], [T,2], or [T,3]");
    }
    return positions.ne[1];
}

void require_tensor_layout(const Tensor& tensor, const char* label, std::int32_t head_dim,
                           std::int32_t heads, std::int32_t tokens) {
    if (heads <= 0) {
        throw std::invalid_argument(std::string("rope: ") + label + " must have positive heads");
    }
    if (tensor.ne[0] != head_dim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * static_cast<std::int64_t>(head_dim) * heads ||
        (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " strides");
    }
}

void require_common(const Tensor& positions, int rotary_dim, float theta) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope: positions must be I32");
    }
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: theta must be positive and finite");
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0) {
        throw std::invalid_argument("rope: rotary_dim must be positive and even");
    }
}

void require_positions_storage(const Tensor& positions) {
    if (!positions.is_contiguous()) {
        throw std::invalid_argument("rope: positions must be contiguous");
    }
    if (positions.data == nullptr) {
        throw std::invalid_argument("rope: positions data must be non-null");
    }
}

void require_rotary_geometry(int axes, int rotary_dim, std::int32_t head_dim) {
    if (axes == 2) {
        if ((head_dim != 64 && head_dim != 72) || rotary_dim != head_dim) {
            throw std::invalid_argument(
                "rope: 2-D Vision mode requires head_dim=rotary_dim=64 or 72");
        }
        return;
    }
    if (axes == 1 && head_dim == 128 && rotary_dim == 128) { return; }
    if (head_dim != kTextHeadDim || rotary_dim > kTextHeadDim) {
        throw std::invalid_argument("rope: Text mode requires D256 or one-dimensional D128/R128");
    }
    if (axes == 3 && rotary_dim != 64) {
        throw std::invalid_argument("rope: 3-D Text MRoPE requires rotary_dim=64");
    }
}

} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          CudaExecutionView execution) {
    require_common(positions, rotary_dim, theta);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    (void)common::operand::numel(positions, "rope");
    const std::int64_t q_numel = common::operand::numel(q, "rope");
    (void)common::operand::numel(k, "rope");
    const std::int32_t tokens   = q.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t k_heads  = k.ne[1];
    require_rotary_geometry(axes, rotary_dim, head_dim);
    require_tensor_layout(q, "q", head_dim, q_heads, tokens);
    require_tensor_layout(k, "k", head_dim, k_heads, tokens);
    if (q_numel == 0) { return; }
    common::operand::device(positions.device, execution.device(), "rope", "positions");
    require_positions_storage(positions);
    common::operand::device(q.device, execution.device(), "rope", "query");
    common::operand::device(k.device, execution.device(), "rope", "key");
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: q/k data must be non-null");
    }
    detail::rope_dispatch(positions, rotary_dim, theta, q, k, execution);
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          CudaExecutionView execution) {
    require_common(positions, rotary_dim, theta);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    (void)common::operand::numel(positions, "rope");
    const std::int64_t x_numel  = common::operand::numel(x, "rope");
    const std::int32_t tokens   = x.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = x.ne[0];
    const std::int32_t heads    = x.ne[1];
    require_rotary_geometry(axes, rotary_dim, head_dim);
    require_tensor_layout(x, "tensor", head_dim, heads, tokens);
    if (x_numel == 0) { return; }
    common::operand::device(positions.device, execution.device(), "rope", "positions");
    require_positions_storage(positions);
    common::operand::device(x.device, execution.device(), "rope", "input");
    if (x.data == nullptr) { throw std::invalid_argument("rope: tensor data must be non-null"); }
    detail::rope_dispatch(positions, rotary_dim, theta, x, execution);
}

} // namespace ninfer::ops
