#include "ops/common/host/operands.h"
#include "ninfer/ops/target_logprobs.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "core/device.h"
#include "core/platform.h"
#include "ops/common/cuda/reduce.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>

namespace ninfer::ops {
using namespace common;

namespace {

inline constexpr int kTargetLogprobsBlock = 256;

template <int BlockSize, bool FastExp>
__launch_bounds__(BlockSize) __global__
    void target_logprobs_kernel(const __nv_bfloat16* logits, const std::int32_t* target_ids,
                                float* output, std::int32_t valid_rows,
                                std::int32_t physical_rows) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base   = static_cast<std::int64_t>(column) * physical_rows;

    float local_max = -CUDART_INF_F;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        local_max = fmaxf(local_max, __bfloat162float(logits[base + row]));
    }
    const float maximum = block_max_broadcast<BlockSize>(local_max);

    float local_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value = __bfloat162float(logits[base + row]) - maximum;
        local_sum += FastExp ? __expf(value) : expf(value);
    }
    __shared__ float warp_sums[BlockSize / kWarpSize];
    const float sum = block_reduce_sum<BlockSize>(local_sum, warp_sums);
    if (threadIdx.x == 0) {
        const float target = __bfloat162float(logits[base + target_ids[column]]);
        output[column]     = target - maximum - logf(sum);
    }
}

template <int Block, bool FastExp>
void launch_columns(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                    Tensor& output, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    target_logprobs_kernel<Block, FastExp>
        <<<columns, Block, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits.data),
                                        static_cast<const std::int32_t*>(target_ids.data),
                                        static_cast<float*>(output.data), valid_rows, logits.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void target_logprobs_launch(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                            Tensor& output, cudaStream_t stream) {
#if NINFER_TARGET_SPARK
    if (logits.ne[0] == 248320 && valid_rows == 248077) {
        launch_columns<1024, true>(logits, target_ids, valid_rows, output, stream);
        return;
    }
#endif
    launch_columns<kTargetLogprobsBlock, false>(logits, target_ids, valid_rows, output, stream);
}

} // namespace

void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     Tensor& output, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "target_logprobs";
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || valid_rows <= 0 || valid_rows > logits.ne[0])
        throw std::invalid_argument("target_logprobs: invalid column or row domain");
    const auto columns = logits.ne[1];
    common::operand::tensor(logits, DType::BF16, {logits.ne[0], columns}, alignof(std::uint16_t),
                            op, "logits", execution.device());
    common::operand::tensor(target_ids, DType::I32, {columns}, alignof(std::int32_t), op,
                            "target_ids", execution.device());
    common::operand::tensor(output, DType::FP32, {columns}, alignof(float), op, "output",
                            execution.device());
    common::operand::disjoint(common::operand::range(output, op),
                              common::operand::range(logits, op), op);
    common::operand::disjoint(common::operand::range(output, op),
                              common::operand::range(target_ids, op), op);

    target_logprobs_launch(logits, target_ids, valid_rows, output, stream);
}

} // namespace ninfer::ops
