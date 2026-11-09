#include "ninfer/ops/rmsnorm_gathered_features.h"
#include "ops/common/host/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include <cmath>

namespace ninfer::ops {
namespace {

template <int Threads, int Vector>
__global__ __launch_bounds__(Threads) void rmsnorm_gathered_features_kernel(
    const __nv_bfloat16* __restrict__ input, const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out, int columns, float epsilon) {
    constexpr int Items = 4096 / (Threads * Vector);
    static_assert(4096 % (Threads * Vector) == 0);
    const auto column = std::int64_t(blockIdx.x);
    common::Bf16Vector<Vector> values[Items], gains[Items];
    float even = 0, odd = 0;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int h       = (item * Threads + threadIdx.x) * Vector;
        const auto source = std::int64_t(h / 2048) * 2048 * columns + column * 2048 + h % 2048;
        values[item]      = common::Bf16Vector<Vector>::load(input + source);
        gains[item]       = common::Bf16Vector<Vector>::load(gamma + h);
#pragma unroll
        for (int j = 0; j < Vector; j += 2) {
            const float x = __bfloat162float(values[item].word[j]);
            const float y = __bfloat162float(values[item].word[j + 1]);
            even          = fmaf(x, x, even);
            odd           = fmaf(y, y, odd);
        }
    }
    __shared__ float sums[Threads / 32];
    const float sum = common::block_reduce_sum<Threads>(even + odd, sums);
    if (threadIdx.x == 0) sums[0] = rsqrtf(sum / 4096.F + epsilon);
    __syncthreads();
    const float inverse = sums[0];
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        common::Bf16Vector<Vector> result;
#pragma unroll
        for (int j = 0; j < Vector; ++j)
            result.word[j] = __float2bfloat16_rn(__bfloat162float(values[item].word[j]) * inverse *
                                                 __bfloat162float(gains[item].word[j]));
        result.store(out + column * 4096 + (item * Threads + threadIdx.x) * Vector);
    }
}

template <int Threads, int Vector>
void launch(const Tensor& input, const Tensor& gamma, float epsilon, Tensor& out,
            cudaStream_t stream) {
    rmsnorm_gathered_features_kernel<Threads, Vector><<<input.ne[1], Threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(gamma.data), static_cast<__nv_bfloat16*>(out.data),
        input.ne[1], epsilon);
}

} // namespace

void rmsnorm_gathered_features(const Tensor& input, const Tensor& gamma, float epsilon, Tensor& out,
                               CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "rmsnorm_gathered_features";
    const int columns        = input.ne[1];
    if (columns < 1 || !(epsilon > 0) || !std::isfinite(epsilon))
        throw std::invalid_argument("rmsnorm_gathered_features: invalid columns or epsilon");
    operand::tensor(input, DType::BF16, {2048, columns, 2}, 16, op, "gathered", execution.device());
    operand::tensor(gamma, DType::BF16, {4096}, 16, op, "gamma", execution.device());
    operand::tensor(out, DType::BF16, {4096, columns}, 16, op, "out", execution.device());
    for (const auto* source : {&input, &gamma})
        operand::disjoint(operand::range(out, op), operand::range(*source, op), op);
    if (columns <= 128)
        launch<512, 8>(input, gamma, epsilon, out, execution.stream());
    else
        launch<256, 8>(input, gamma, epsilon, out, execution.stream());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
