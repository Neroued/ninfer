#include "ninfer/kernels/rope.h"

#include "kernels/launcher/rope.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::kernels {
namespace {

constexpr std::int32_t kTextHeadDim = 256;
constexpr std::int32_t kTextQHeads  = 24;
constexpr std::int32_t kTextKHeads  = 4;
constexpr std::int32_t kVisionDim   = 72;
constexpr std::int32_t kVisionHeads = 16;

std::int64_t numel_allow_zero(const Tensor& tensor, const char* label) {
    bool zero      = false;
    std::int64_t n = 1;
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] < 0) {
            throw std::invalid_argument(std::string("rope: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (tensor.ne[dim] == 0) {
            zero = true;
            continue;
        }
        if (n > std::numeric_limits<std::int64_t>::max() / tensor.ne[dim]) {
            throw std::overflow_error("rope: tensor size overflows int64");
        }
        n *= tensor.ne[dim];
    }
    return zero ? 0 : n;
}

int position_axes(const Tensor& positions, std::int32_t tokens) {
    if (positions.ne[0] != tokens || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        (positions.ne[1] != 1 && positions.ne[1] != 2 && positions.ne[1] != 3)) {
        throw std::invalid_argument("rope: positions must have shape [T], [T,2], or [T,3]");
    }
    return positions.ne[1];
}

void require_tensor_layout(const Tensor& tensor, const char* label, std::int32_t head_dim,
                           std::int32_t heads, std::int32_t tokens) {
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

void require_model_mode(int axes, int rotary_dim, std::int32_t head_dim) {
    if (axes == 2) {
        if (head_dim != kVisionDim || rotary_dim != kVisionDim) {
            throw std::invalid_argument("rope: 2-D Vision mode requires head_dim=rotary_dim=72");
        }
        return;
    }
    if (head_dim != kTextHeadDim || rotary_dim > kTextHeadDim) {
        throw std::invalid_argument("rope: Text mode requires head_dim=256");
    }
    if (axes == 3 && rotary_dim != 64) {
        throw std::invalid_argument("rope: 3-D Text MRoPE requires rotary_dim=64");
    }
}

} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream) {
    require_common(positions, rotary_dim, theta);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    (void)numel_allow_zero(positions, "positions");
    const std::int64_t q_numel = numel_allow_zero(q, "q");
    (void)numel_allow_zero(k, "k");
    const std::int32_t tokens   = q.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = axes == 2 ? kVisionDim : kTextHeadDim;
    const std::int32_t q_heads  = axes == 2 ? kVisionHeads : kTextQHeads;
    const std::int32_t k_heads  = axes == 2 ? kVisionHeads : kTextKHeads;
    require_model_mode(axes, rotary_dim, head_dim);
    require_tensor_layout(q, "q", head_dim, q_heads, tokens);
    require_tensor_layout(k, "k", head_dim, k_heads, tokens);
    if (q_numel == 0) { return; }
    require_positions_storage(positions);
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: q/k data must be non-null");
    }
    if (axes == 1 && q.is_contiguous() && k.is_contiguous()) {
        detail::rope_launch(positions, rotary_dim, theta, q, k, stream);
    } else {
        detail::rope_nd_launch(positions, rotary_dim, theta, q, k, stream);
    }
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream) {
    require_common(positions, rotary_dim, theta);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    (void)numel_allow_zero(positions, "positions");
    const std::int64_t x_numel  = numel_allow_zero(x, "tensor");
    const std::int32_t tokens   = x.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = axes == 2 ? kVisionDim : kTextHeadDim;
    std::int32_t heads          = 0;
    bool text_q                 = false;
    if (axes == 2) {
        heads = kVisionHeads;
    } else if (x.ne[1] == kTextQHeads) {
        heads  = kTextQHeads;
        text_q = true;
    } else if (x.ne[1] == kTextKHeads) {
        heads = kTextKHeads;
    } else {
        throw std::invalid_argument("rope: Text tensor must have 24 Q heads or 4 K heads");
    }
    require_model_mode(axes, rotary_dim, head_dim);
    require_tensor_layout(x, "tensor", head_dim, heads, tokens);
    if (x_numel == 0) { return; }
    require_positions_storage(positions);
    if (x.data == nullptr) { throw std::invalid_argument("rope: tensor data must be non-null"); }
    if (axes != 1 || !x.is_contiguous()) {
        detail::rope_nd_single_launch(positions, rotary_dim, theta, x, stream);
    } else if (text_q) {
        detail::rope_q_launch(positions, rotary_dim, theta, x, stream);
    } else {
        detail::rope_k_launch(positions, rotary_dim, theta, x, stream);
    }
}

} // namespace ninfer::kernels
