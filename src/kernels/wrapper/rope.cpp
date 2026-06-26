// qus::kernels - rope wrapper: public api validation and launcher dispatch.
#include "qus/kernels/rope.h"

#include "kernels/launcher/rope.h"  // detail::rope_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads = 24;
constexpr std::int32_t kKHeads = 4;

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("rope: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("rope: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_rope_tensor_shape(const Tensor& t, const char* label, std::int32_t heads,
                               std::int32_t T) {
    if (t.ne[0] != kHeadDim || t.ne[1] != heads || t.ne[2] != T || t.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope: ") + label +
                                    " must have shape [256,heads,T]");
    }
}

void require_positions_shape(const Tensor& positions) {
    if (positions.ne[1] != 1 || positions.ne[2] != 1 || positions.ne[3] != 1) {
        throw std::invalid_argument("rope: positions must have shape [T]");
    }
}

} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope: positions must be I32");
    }
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: theta must be positive and finite");
    }

    (void) numel_allow_zero(positions, "positions");
    const std::int64_t q_numel = numel_allow_zero(q, "q");
    (void) numel_allow_zero(k, "k");

    require_positions_shape(positions);
    const std::int32_t T = positions.ne[0];
    require_rope_tensor_shape(q, "q", kQHeads, T);
    require_rope_tensor_shape(k, "k", kKHeads, T);

    if (rotary_dim <= 0 || rotary_dim > q.ne[0] || (rotary_dim & 1) != 0) {
        throw std::invalid_argument("rope: rotary_dim must be positive, even, and <= head dim");
    }
    if (q_numel == 0) { return; }

    if (!positions.is_contiguous() || !q.is_contiguous() || !k.is_contiguous()) {
        throw std::invalid_argument("rope: positions/q/k must be contiguous");
    }
    if (positions.data == nullptr || q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: positions/q/k data must be non-null");
    }

    detail::rope_launch(positions, rotary_dim, theta, q, k, stream);
}

} // namespace qus::kernels
