#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/common/host/operands.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::noncausal {

inline constexpr std::int32_t kHeads = 16;

inline void require_geometry(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) ||
        (geometry.head_dim != 64 && geometry.head_dim != 72) || geometry.query_heads != kHeads ||
        geometry.kv_heads != kHeads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

inline void require_profile(AttentionHeadGeometry geometry, float scale, const char* op) {
    require_geometry(geometry, op);
    const float expected = geometry.head_dim == 64 ? 0.125F : 0.11785113019775792073F;
    if (!std::isfinite(scale) || std::abs(scale - expected) > 1.0e-7f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(head_dim)");
    }
}

inline void require_qkv(const Tensor& tensor, std::int32_t head_dim, std::int32_t tokens,
                        const char* op, const char* name, Device device) {
    common::operand::device(tensor.device, device, op, name);
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != head_dim || tensor.ne[1] != kHeads ||
        tensor.ne[2] != tokens || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * head_dim * kHeads || (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " strides");
    }
    if (!common::operand::aligned(tensor.data, 2)) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must have natural BF16 alignment");
    }
}

inline std::int32_t validate_qkv(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& out, AttentionHeadGeometry geometry, float scale,
                                 const char* op, Device device) {
    require_profile(geometry, scale, op);
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_qkv(q, geometry.head_dim, tokens, op, "q", device);
    require_qkv(k, geometry.head_dim, tokens, op, "k", device);
    require_qkv(v, geometry.head_dim, tokens, op, "v", device);
    require_qkv(out, geometry.head_dim, tokens, op, "out", device);
    if (!out.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": out must be contiguous");
    }
    return tokens;
}

} // namespace ninfer::ops::detail::noncausal
