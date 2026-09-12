#include "ninfer/ops/rmsnorm_pack_tail.h"

#include "ops/launcher/rmsnorm_pack_tail.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16_contiguous_aligned(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(tensor.data) & 15U) != 0) {
        throw std::invalid_argument(std::string("rmsnorm_pack_tail: ") + name +
                                    " must be contiguous 16-byte-aligned BF16");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

} // namespace

void rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, Tensor& output,
                       cudaStream_t stream) {
    require_bf16_contiguous_aligned(input, "input");
    require_bf16_contiguous_aligned(weight, "weight");
    require_bf16_contiguous_aligned(output, "output");

    const std::int32_t batch = input.ne[2];
    const std::int32_t width = input.ne[1];
    if (input.ne[0] != 5120 || (width < 2 || width > 16) || batch < 1 || batch > 8 ||
        input.ne[3] != 1) {
        throw std::invalid_argument(
            "rmsnorm_pack_tail: input must have shape [5120,W,B], W=2..16, B=1..8");
    }
    if (weight.ne[0] != 5120 || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm_pack_tail: weight must have shape [5120]");
    }
    if (output.ne[0] != 5120 || output.ne[1] != (width - 1) * batch || output.ne[2] != 1 ||
        output.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm_pack_tail: output must have shape [5120,(W-1)B]");
    }
    if (overlaps(input, weight) || overlaps(input, output) || overlaps(weight, output)) {
        throw std::invalid_argument("rmsnorm_pack_tail: tensors must not overlap");
    }

    detail::rmsnorm_pack_tail_launch(input, weight, output, stream);
}

} // namespace ninfer::ops
