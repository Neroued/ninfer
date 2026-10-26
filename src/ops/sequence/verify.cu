#include "ops/common/host/operands.h"
#include "ninfer/ops/sequence.h"
#include <algorithm>
#include <stdexcept>
#include "ops/common/numeric/math.h"
#include "core/device.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cstddef>

namespace ninfer::ops {
using namespace common;

namespace {

__global__ void prepare_verify_inputs_kernel(
    const std::int32_t* anchors, const std::int32_t* drafts, const std::int32_t* base_positions,
    const std::int32_t* current_extents, std::int32_t* verify_ids, std::int32_t* positions,
    std::int32_t k, std::int64_t draft_column, std::int64_t draft_row) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        const int off   = row * T + j;
        verify_ids[off] = j == 0 ? anchors[row]
                                 : (j <= extent ? drafts[row * draft_row + (j - 1) * draft_column]
                                                : anchors[row]);
        if (positions != nullptr) {
            positions[off] = base_positions[row] + (j <= extent ? j : extent);
        }
    }
}

void prepare_verify_inputs_launch(const Tensor& anchors, const Tensor& drafts,
                                  const Tensor& base_positions, const Tensor& current_extents,
                                  Tensor& verify_ids, Tensor& positions, cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(base_positions.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), static_cast<std::int32_t*>(positions.data), k,
        drafts.nb[0] / 4, drafts.nb[1] / 4);
    CUDA_CHECK(cudaGetLastError());
}

void prepare_verify_ids_launch(const Tensor& anchors, const Tensor& drafts,
                               const Tensor& current_extents, Tensor& verify_ids,
                               cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data), nullptr,
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), nullptr, k, drafts.nb[0] / 4,
        drafts.nb[1] / 4);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                           const Tensor& base_positions, const Tensor& current_extents,
                           Tensor& verify_ids, Tensor& positions, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "prepare_verify_inputs";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1) { throw std::invalid_argument("prepare_verify_inputs: K must be >=1"); }
    if (batch < 1) { throw std::invalid_argument("prepare_verify_inputs: B must be >=1"); }
    common::operand::tensor(anchors, DType::I32, {batch}, 1, op, "anchors", execution.device());
    common::operand::strided_tensor(drafts, DType::I32, {k, batch}, 4, op, "drafts",
                                    execution.device());
    common::operand::tensor(base_positions, DType::I32, {batch}, 1, op, "base_positions",
                            execution.device());
    common::operand::tensor(current_extents, DType::I32, {batch}, 1, op, "current_extents",
                            execution.device());
    common::operand::tensor(verify_ids, DType::I32, {k + 1, batch}, 1, op, "verify_ids",
                            execution.device());
    common::operand::tensor(positions, DType::I32, {k + 1, batch}, 1, op, "positions",
                            execution.device());
    prepare_verify_inputs_launch(anchors, drafts, base_positions, current_extents, verify_ids,
                                 positions, stream);
}

void prepare_verify_ids(const Tensor& anchors, const Tensor& drafts, const Tensor& current_extents,
                        Tensor& verify_ids, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "prepare_verify_ids";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1 || batch < 1) {
        throw std::invalid_argument("prepare_verify_ids: K and B must be positive");
    }
    common::operand::tensor(anchors, DType::I32, {batch}, 1, op, "anchors", execution.device());
    common::operand::strided_tensor(drafts, DType::I32, {k, batch}, 4, op, "drafts",
                                    execution.device());
    common::operand::tensor(current_extents, DType::I32, {batch}, 1, op, "current_extents",
                            execution.device());
    common::operand::tensor(verify_ids, DType::I32, {k + 1, batch}, 1, op, "verify_ids",
                            execution.device());
    prepare_verify_ids_launch(anchors, drafts, current_extents, verify_ids, stream);
}

} // namespace ninfer::ops
