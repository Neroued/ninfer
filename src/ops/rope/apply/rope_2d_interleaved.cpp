#include "ninfer/ops/rope.h"
#include "ops/common/host/operands.h"
#include "ops/rope/apply/rope_2d_interleaved.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

namespace operand    = common::operand;
constexpr char kOp[] = "rope_2d_interleaved";

void validate_plane(const Tensor& value, int tokens, const char* label) {
    operand::dtype(value, DType::BF16, kOp, "input must be BF16");
    operand::shape(value, {64, value.ne[1], tokens, 1}, kOp, label);
    if (value.ne[1] <= 0 || value.nb[0] != 2 || value.nb[1] != 128 ||
        value.nb[2] < std::int64_t(128) * value.ne[1] || value.nb[2] % 2 != 0)
        throw std::invalid_argument("rope_2d_interleaved: invalid head extent or strides");
}

void require_storage(const Tensor& value, Device device, std::size_t alignment, const char* label) {
    operand::device(value.device, device, kOp, label);
    if (!operand::aligned(value.data, alignment))
        throw std::invalid_argument("rope_2d_interleaved: null or misaligned operand");
}

operand::Range storage_range(const Tensor& value) {
    const auto blocks = operand::blocks(value, kOp);
    // Include inter-token padding in the disjointness contract.
    return {blocks.first.begin, blocks.first.end + (blocks.count - 1) * blocks.stride};
}

void apply_interleaved(const Tensor& positions, float theta, Tensor& q, Tensor* k,
                       CudaExecutionView execution) {
    const int tokens = q.ne[2];
    if (tokens < 0 || !(theta > 0.F) || !std::isfinite(theta))
        throw std::invalid_argument("rope_2d_interleaved: invalid token extent or theta");
    operand::dtype(positions, DType::I32, kOp, "positions must be I32");
    operand::shape(positions, {tokens, 2, 1, 1}, kOp, "positions");
    if ((tokens != 1 && positions.nb[0] != 4) || positions.nb[1] != 4LL * tokens)
        throw std::invalid_argument("rope_2d_interleaved: positions must be contiguous");
    validate_plane(q, tokens, "q/x");
    if (k) validate_plane(*k, tokens, "k");
    if (tokens == 0) return;

    require_storage(positions, execution.device(), 4, "positions");
    require_storage(q, execution.device(), 2, "q/x");
    const auto pos_range = operand::range(positions, kOp);
    const auto q_range   = storage_range(q);
    operand::disjoint(q_range, pos_range, kOp);
    if (k) {
        require_storage(*k, execution.device(), 2, "k");
        const auto k_range = storage_range(*k);
        operand::disjoint(k_range, pos_range, kOp);
        operand::disjoint(q_range, k_range, kOp);
    }

    detail::launch_rope_2d_interleaved(positions, theta, q, k, execution);
}

} // namespace

void rope_2d_interleaved(const Tensor& positions, float theta, Tensor& q, Tensor& k,
                         CudaExecutionView execution) {
    apply_interleaved(positions, theta, q, &k, execution);
}

void rope_2d_interleaved(const Tensor& positions, float theta, Tensor& x,
                         CudaExecutionView execution) {
    apply_interleaved(positions, theta, x, nullptr, execution);
}

} // namespace ninfer::ops
