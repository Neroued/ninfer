#include "ninfer/ops/prepare_masked_block.h"

#include "ops/launcher/prepare_masked_block.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_i32_scalar(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != 1 || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument("prepare_masked_block: " + std::string(name) +
                                    " must be a contiguous I32 scalar");
    }
}

void require_i32_vector(const Tensor& tensor, std::int32_t size, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != size || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument("prepare_masked_block: " + std::string(name) +
                                    " must be a contiguous I32 vector");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

} // namespace

void prepare_masked_block(const Tensor& anchor, const Tensor& length, std::int32_t mask_id,
                          Tensor& ids, Tensor& positions, cudaStream_t stream) {
    const std::int32_t block_size = ids.ne[0];
    if (block_size < 2 || block_size > 16) {
        throw std::invalid_argument("prepare_masked_block: B must be 2..16");
    }
    if (mask_id < 0) {
        throw std::invalid_argument("prepare_masked_block: mask_id must be nonnegative");
    }
    require_i32_scalar(anchor, "anchor");
    require_i32_scalar(length, "length");
    require_i32_vector(ids, block_size, "ids");
    require_i32_vector(positions, block_size, "positions");
    if (overlaps(ids, positions) || overlaps(anchor, ids) || overlaps(anchor, positions) ||
        overlaps(length, ids) || overlaps(length, positions)) {
        throw std::invalid_argument("prepare_masked_block: inputs and outputs must not overlap");
    }
    const auto plan = detail::prepare_masked_block_resolve_plan(block_size);
    detail::prepare_masked_block_launch(anchor, length, mask_id, ids, positions, plan, stream);
}

} // namespace ninfer::ops
