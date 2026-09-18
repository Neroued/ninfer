#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q5/q5_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q5Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q5_n1024_k5120},   ShapeEntry{6144, 5120, select_q5_n6144_k5120},
    ShapeEntry{7168, 5120, select_q5_n7168_k5120},   ShapeEntry{5120, 6144, select_q5_n5120_k6144},
    ShapeEntry{5120, 17408, select_q5_n5120_k17408}, ShapeEntry{1152, 1152, select_q5_n1152_k1152},
    ShapeEntry{1152, 4304, select_q5_n1152_k4304},
};
} // namespace

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q5 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q5 linear: unsupported shape");
}

Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q5 linear: unsupported policy");
    return select_q5_a16_launch(n, k, t);
}

void q5_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q5_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
