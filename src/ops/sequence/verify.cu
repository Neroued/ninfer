#include "ninfer/ops/sequence.h"
#include "ops/common/host/launch.h"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/math.h"

#include <algorithm>

namespace ninfer::ops {
namespace {

__global__ void prepare_verify_kernel(const std::int32_t* anchors, const std::int32_t* drafts,
                                      const std::int32_t* base_positions,
                                      const std::int32_t* current_extents, std::int32_t* verify_ids,
                                      std::int32_t* positions, int width, int max_extent,
                                      std::int64_t draft_column, std::int64_t draft_row) {
    const int row    = blockIdx.y;
    const int extent = min(max(current_extents[row], 0), max_extent);
    const int j      = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= width) return;
    const auto index = std::int64_t(row) * width + j;
    verify_ids[index] =
        j > 0 && j <= extent ? drafts[row * draft_row + (j - 1) * draft_column] : anchors[row];
    if (positions) positions[index] = base_positions[row] + min(j, extent);
}

void prepare_verify(const Tensor& anchors, const Tensor& drafts, const Tensor& current_extents,
                    Tensor& verify_ids, const Tensor* base_positions, Tensor* positions,
                    CudaExecutionView execution) {
    const char* op    = positions ? "prepare_verify_inputs" : "prepare_verify_ids";
    const int width   = verify_ids.ne[0];
    const int batch   = verify_ids.ne[1];
    const int steps   = drafts.ne[0];
    const auto device = execution.device();
    if (width < 2 || steps < 1 || batch < 1)
        throw std::invalid_argument(std::string(op) + ": invalid verification geometry");
    common::operand::tensor(anchors, DType::I32, {batch}, 4, op, "anchors", device);
    common::operand::strided_tensor(drafts, DType::I32, {steps, batch}, 4, op, "drafts", device);
    common::operand::tensor(current_extents, DType::I32, {batch}, 4, op, "current_extents", device);
    common::operand::tensor(verify_ids, DType::I32, {width, batch}, 4, op, "verify_ids", device);
    if (positions) {
        common::operand::tensor(*base_positions, DType::I32, {batch}, 4, op, "base_positions",
                                device);
        common::operand::tensor(*positions, DType::I32, {width, batch}, 4, op, "positions", device);
    }
    constexpr int block = 32;
    common::for_each_token_slice(batch, 1, [&](int offset, int count) {
        const auto begin = std::int64_t(offset) * width;
        const dim3 grid(common::div_up(width, block), count);
        prepare_verify_kernel<<<grid, block, 0, execution.stream()>>>(
            static_cast<const int*>(anchors.data) + offset,
            static_cast<const int*>(drafts.data) + offset * (drafts.nb[1] / 4),
            base_positions ? static_cast<const int*>(base_positions->data) + offset : nullptr,
            static_cast<const int*>(current_extents.data) + offset,
            static_cast<int*>(verify_ids.data) + begin,
            positions ? static_cast<int*>(positions->data) + begin : nullptr, width,
            std::min(width - 1, steps), drafts.nb[0] / 4, drafts.nb[1] / 4);
    });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                           const Tensor& base_positions, const Tensor& current_extents,
                           Tensor& verify_ids, Tensor& positions, CudaExecutionView execution) {
    prepare_verify(anchors, drafts, current_extents, verify_ids, &base_positions, &positions,
                   execution);
}

void prepare_verify_ids(const Tensor& anchors, const Tensor& drafts, const Tensor& current_extents,
                        Tensor& verify_ids, CudaExecutionView execution) {
    prepare_verify(anchors, drafts, current_extents, verify_ids, nullptr, nullptr, execution);
}
} // namespace ninfer::ops
