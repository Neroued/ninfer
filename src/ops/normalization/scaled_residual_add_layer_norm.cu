#include "ninfer/ops/scaled_residual_add_layer_norm.h"

#include "ops/common/cuda/reduce.cuh"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops {
namespace {

using Bf16        = __nv_bfloat16;
namespace operand = common::operand;

template <class Gamma>
__device__ __forceinline__ Bf16 scaled_update(Bf16 residual, Bf16 update, Gamma gamma) {
    const Bf16 scaled = __float2bfloat16_rn(__bfloat162float(update) * float(gamma));
    return __float2bfloat16_rn(__bfloat162float(residual) + __bfloat162float(scaled));
}

template <int Width, class Gamma>
__device__ __forceinline__ void update_values(const Bf16* residual, const Bf16* update,
                                              const Gamma* gamma, Bf16* out, std::int64_t offset,
                                              int channel, float (&values)[Width]) {
    if constexpr (Width == 1) {
        const Bf16 r = scaled_update(residual[offset], update[offset], gamma[channel]);
        out[offset]  = r;
        values[0]    = __bfloat162float(r);
    } else {
        const common::Bf16x8Bits r{.raw = common::load_vec<uint4>(residual + offset)};
        const common::Bf16x8Bits u{.raw = common::load_vec<uint4>(update + offset)};
        alignas(16) float g[8];
        if constexpr (std::is_same_v<Gamma, float>) {
            common::store_vec(g, common::load_vec<float4>(gamma + channel));
            common::store_vec(g + 4, common::load_vec<float4>(gamma + channel + 4));
        } else {
            const common::Bf16x8Bits gamma_pack{.raw = common::load_vec<uint4>(gamma + channel)};
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 pair = __bfloat1622float2(gamma_pack.pair[p]);
                g[2 * p]          = pair.x;
                g[2 * p + 1]      = pair.y;
            }
        }

        common::Bf16x8Bits result{.raw = {}};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 v       = __bfloat1622float2(u.pair[p]);
            const auto scaled    = __floats2bfloat162_rn(v.x * g[2 * p], v.y * g[2 * p + 1]);
            result.pair[p]       = __hadd2_rn(r.pair[p], scaled);
            const float2 written = __bfloat1622float2(result.pair[p]);
            values[2 * p]        = written.x;
            values[2 * p + 1]    = written.y;
        }
        common::store_vec(out + offset, result.raw);
    }
}

template <int Block, int Width, class Gamma, bool PowerOfTwo>
__launch_bounds__(Block) __global__
    void scaled_add_kernel(const Bf16* residual, const Bf16* update, const Gamma* gamma, Bf16* out,
                           std::int64_t packs, int row_packs, common::FastDivisorU31 divisor) {
    const auto start  = std::int64_t(blockIdx.x) * Block + threadIdx.x;
    const auto stride = std::int64_t(gridDim.x) * Block;
    for (auto pack = start; pack < packs; pack += stride) {
        int channel;
        if constexpr (PowerOfTwo) {
            channel = pack & (row_packs - 1);
        } else if (packs < (1LL << 31)) {
            const auto index = static_cast<std::uint32_t>(pack);
            channel          = index - divisor.divide(index) * row_packs;
        } else {
            channel = pack % row_packs;
        }
        float unused[Width];
        update_values(residual, update, gamma, out, pack * Width, channel * Width, unused);
    }
}

template <int Block>
__device__ __forceinline__ float broadcast_sum(float value, float* shared_warps) {
    if constexpr (Block == 32) {
        return common::warp_sum(value);
    } else {
        constexpr int warps = Block / 32;
        const int lane      = threadIdx.x % 32;
        value               = common::warp_reduce_sum(value);
        if (lane == 0) shared_warps[threadIdx.x / 32] = value;
        __syncthreads();
        return common::warp_sum<warps>(shared_warps[lane % warps]);
    }
}

template <int Block>
__device__ __forceinline__ void normalize_large_values(const Bf16* published, const Bf16* weight,
                                                       const Bf16* bias, Bf16* out, int d,
                                                       float anchor, float inverse_d, float eps,
                                                       float* mean_warps, float* variance_warps) {
    // Even for opposite-sign maximum BF16 values and INT32_MAX channels, this
    // power-of-two rescaling bounds the centered sum of squares below 2^100.
    constexpr float scale     = 0x1p-96f;
    const float scaled_anchor = anchor * scale;
    const auto shifted        = [&](std::int64_t i) {
        return __bfloat162float(published[i]) * scale - scaled_anchor;
    };
    float sum = 0.f;
    for (std::int64_t i = threadIdx.x; i < d; i += Block) sum += shifted(i);
    const float mean = broadcast_sum<Block>(sum, mean_warps) * inverse_d;
    float squares    = 0.f;
    for (std::int64_t i = threadIdx.x; i < d; i += Block) {
        const float centered = shifted(i) - mean;
        squares              = fmaf(centered, centered, squares);
    }
    const float variance = broadcast_sum<Block>(squares, variance_warps) * inverse_d;
    // A vanishing scaled epsilon is harmless for these large-variance rows.
    const float inverse = variance == 0.f ? 0.f : rsqrtf(variance + (eps * scale) * scale);
    for (std::int64_t i = threadIdx.x; i < d; i += Block) {
        const float normalized = (shifted(i) - mean) * inverse;
        out[i]                 = __float2bfloat16_rn(normalized * __bfloat162float(weight[i]) +
                                                     __bfloat162float(bias[i]));
    }
}

template <int Block, int Width, int Items, class Gamma>
__launch_bounds__(Block) __global__
    void scaled_add_norm_tile(const Bf16* residual, const Bf16* update, const Gamma* gamma,
                              const Bf16* weight, const Bf16* bias, Bf16* residual_out,
                              Bf16* normalized_out, int d, float inverse_d, float eps) {
    const int lane = threadIdx.x;
    const auto row = std::int64_t(blockIdx.x) * d;
    float values[Items][Width]{};
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int channel = (lane + item * Block) * Width;
        if (channel < d)
            update_values(residual, update, gamma, residual_out, row + channel, channel,
                          values[item]);
    }

    // Subtract a represented residual before both reductions. This retains small
    // deviations even when the rounded BF16 residual has a large common offset.
    float anchor;
    if constexpr (Block == 32) {
        anchor = __shfl_sync(common::kFullWarpMask, values[0][0], 0);
    } else {
        __shared__ float shared_anchor;
        if (lane == 0) shared_anchor = values[0][0];
        __syncthreads();
        anchor = shared_anchor;
    }

    float sum = 0.f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        if ((lane + item * Block) * Width < d) {
#pragma unroll
            for (int i = 0; i < Width; ++i) values[item][i] -= anchor;
            if constexpr (Width == 8) {
                const auto& v = values[item];
                sum += ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
            } else {
                sum += values[item][0];
            }
        }
    }

    __shared__ float mean_warps[Block / 32], variance_warps[Block / 32];
    const float mean_sum = broadcast_sum<Block>(sum, mean_warps);
    const float mean     = mean_sum * inverse_d;
    float squares = 0.f, odd = 0.f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        if ((lane + item * Block) * Width < d) {
#pragma unroll
            for (int i = 0; i < Width; ++i) {
                values[item][i] -= mean;
                if (i % 2 == 0)
                    squares = fmaf(values[item][i], values[item][i], squares);
                else
                    odd = fmaf(values[item][i], values[item][i], odd);
            }
        }
    }
    const float variance_sum = broadcast_sum<Block>(squares + odd, variance_warps);
    const float denominator  = fmaf(variance_sum, inverse_d, eps);
    if (!isfinite(mean_sum) || !isfinite(denominator)) {
        __syncthreads();
        normalize_large_values<Block>(residual_out + row, weight, bias, normalized_out + row, d,
                                      anchor, inverse_d, eps, mean_warps, variance_warps);
        return;
    }
    const float inverse = rsqrtf(denominator);

#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int channel = (lane + item * Block) * Width;
        if (channel < d) {
            if constexpr (Width == 1) {
                normalized_out[row + channel] = __float2bfloat16_rn(
                    values[item][0] * inverse * __bfloat162float(weight[channel]) +
                    __bfloat162float(bias[channel]));
            } else {
                const common::Bf16x8Bits gain{.raw = common::load_vec<uint4>(weight + channel)};
                const common::Bf16x8Bits shift{.raw = common::load_vec<uint4>(bias + channel)};
                common::Bf16x8Bits result{.raw = {}};
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 w = __bfloat1622float2(gain.pair[p]);
                    const float2 b = __bfloat1622float2(shift.pair[p]);
                    result.pair[p] =
                        __floats2bfloat162_rn(values[item][2 * p] * inverse * w.x + b.x,
                                              values[item][2 * p + 1] * inverse * w.y + b.y);
                }
                common::store_vec(normalized_out + row + channel, result.raw);
            }
        }
    }
}

template <int Block, class Gamma>
__launch_bounds__(Block) __global__
    void scaled_add_norm_large(const Bf16* residual, const Bf16* update, const Gamma* gamma,
                               const Bf16* weight, const Bf16* bias, Bf16* residual_out,
                               Bf16* normalized_out, int d, float inverse_d, float eps) {
    const auto row = std::int64_t(blockIdx.x) * d;
    __shared__ float anchor_value;
    if (threadIdx.x == 0) {
        const Bf16 r      = scaled_update(residual[row], update[row], gamma[0]);
        residual_out[row] = r;
        anchor_value      = __bfloat162float(r);
    }
    __syncthreads();
    const float anchor = anchor_value;

    float sum = 0.f;
    for (std::int64_t i = threadIdx.x ? threadIdx.x : Block; i < d; i += Block) {
        const Bf16 r          = scaled_update(residual[row + i], update[row + i], gamma[i]);
        residual_out[row + i] = r;
        sum += __bfloat162float(r) - anchor;
    }
    __shared__ float mean_warps[Block / 32], variance_warps[Block / 32];
    const float mean_sum = broadcast_sum<Block>(sum, mean_warps);
    const float mean     = mean_sum * inverse_d;

    // Wide rows use the required BF16 output as staging once the row is published.
    float squares = 0.f;
    for (std::int64_t i = threadIdx.x; i < d; i += Block) {
        const float centered = (__bfloat162float(residual_out[row + i]) - anchor) - mean;
        squares              = fmaf(centered, centered, squares);
    }
    const float variance_sum = broadcast_sum<Block>(squares, variance_warps);
    const float denominator  = fmaf(variance_sum, inverse_d, eps);
    if (!isfinite(mean_sum) || !isfinite(denominator)) {
        __syncthreads();
        normalize_large_values<Block>(residual_out + row, weight, bias, normalized_out + row, d,
                                      anchor, inverse_d, eps, mean_warps, variance_warps);
        return;
    }
    const float inverse = rsqrtf(denominator);
    for (std::int64_t i = threadIdx.x; i < d; i += Block) {
        const float centered    = (__bfloat162float(residual_out[row + i]) - anchor) - mean;
        normalized_out[row + i] = __float2bfloat16_rn(
            centered * inverse * __bfloat162float(weight[i]) + __bfloat162float(bias[i]));
    }
}

bool aligned_pack(const Tensor& x, const Tensor& update, const Tensor& gamma, const Tensor& out) {
    return x.ne[0] % 8 == 0 && operand::aligned(x.data, 16) && operand::aligned(update.data, 16) &&
           operand::aligned(gamma.data, 16) && operand::aligned(out.data, 16);
}

template <class Gamma>
void launch_add(const Tensor& residual, const Tensor& update, const Tensor& gamma, Tensor& out,
                cudaStream_t stream) {
    const auto launch = [&]<int Width> {
        constexpr int block = 128;
        const int row_packs = residual.ne[0] / Width;
        const auto count    = std::int64_t(row_packs) * residual.ne[1];
        const int grid = std::min<std::int64_t>(common::div_up(count, std::int64_t(block)), 32768);
        const auto submit = [&]<bool PowerOfTwo> {
            scaled_add_kernel<block, Width, Gamma, PowerOfTwo><<<grid, block, 0, stream>>>(
                static_cast<const Bf16*>(residual.data), static_cast<const Bf16*>(update.data),
                static_cast<const Gamma*>(gamma.data), static_cast<Bf16*>(out.data), count,
                row_packs, common::FastDivisorU31::make(row_packs));
        };
        if ((row_packs & (row_packs - 1)) == 0)
            submit.template operator()<true>();
        else
            submit.template operator()<false>();
    };
    if (aligned_pack(residual, update, gamma, out))
        launch.template operator()<8>();
    else
        launch.template operator()<1>();
}

template <class Gamma>
void launch_norm(const Tensor& residual, const Tensor& update, const Tensor& gamma,
                 const Tensor& weight, const Tensor& bias, float eps, Tensor& out, Tensor& norm,
                 cudaStream_t stream) {
    const int d = residual.ne[0], m = residual.ne[1];
    const auto* r         = static_cast<const Bf16*>(residual.data);
    const auto* u         = static_cast<const Bf16*>(update.data);
    const auto* g         = static_cast<const Gamma*>(gamma.data);
    const auto* w         = static_cast<const Bf16*>(weight.data);
    const auto* b         = static_cast<const Bf16*>(bias.data);
    auto* ro              = static_cast<Bf16*>(out.data);
    auto* y               = static_cast<Bf16*>(norm.data);
    const float inverse_d = 1.f / static_cast<float>(d);
    const auto launch     = [&]<int Block, int Width, int Items> {
        scaled_add_norm_tile<Block, Width, Items>
            <<<m, Block, 0, stream>>>(r, u, g, w, b, ro, y, d, inverse_d, eps);
    };
    if (d > 4096) {
        scaled_add_norm_large<256><<<m, 256, 0, stream>>>(r, u, g, w, b, ro, y, d, inverse_d, eps);
    } else if (aligned_pack(residual, update, gamma, out) && operand::aligned(w, 16) &&
               operand::aligned(b, 16) && operand::aligned(y, 16)) {
        if (d <= 256)
            launch.template operator()<32, 8, 1>();
        else if (d <= 512)
            launch.template operator()<64, 8, 1>();
        else if (d <= 1024)
            launch.template operator()<128, 8, 1>();
        else if (d <= 2048)
            launch.template operator()<128, 8, 2>();
        else if (d <= 3072)
            launch.template operator()<128, 8, 3>();
        else
            launch.template operator()<128, 8, 4>();
    } else {
        if (d <= 32)
            launch.template operator()<32, 1, 1>();
        else if (d <= 128)
            launch.template operator()<128, 1, 1>();
        else if (d <= 512)
            launch.template operator()<128, 1, 4>();
        else if (d <= 1024)
            launch.template operator()<128, 1, 8>();
        else if (d <= 2048)
            launch.template operator()<128, 1, 16>();
        else
            launch.template operator()<128, 1, 32>();
    }
}

void validate_add_operands(const Tensor& residual, const Tensor& update, const Tensor& gamma,
                           const Tensor& out, Device device, const char* op) {
    const int d = residual.ne[0], m = residual.ne[1];
    if (d <= 0 || m < 0) throw std::invalid_argument("scaled residual add: invalid extents");
    for (const auto* t : {&residual, &update, &out}) {
        operand::shape(*t, {d, m, 1, 1}, op, "residual/update/residual_out");
        operand::dtype(*t, DType::BF16, op, "residual/update/residual_out must be BF16");
    }
    if (gamma.dtype != DType::FP32 && gamma.dtype != DType::BF16)
        throw std::invalid_argument("scaled residual add: gamma must be BF16 or FP32");
    operand::tensor(gamma, gamma.dtype, {d}, dtype_size(gamma.dtype), op, "gamma", device);
}

operand::Range validate_add_storage(const Tensor& residual, const Tensor& update,
                                    const Tensor& gamma, const Tensor& out, Device device,
                                    const char* op) {
    const int d = residual.ne[0], m = residual.ne[1];
    operand::tensor(residual, DType::BF16, {d, m}, 2, op, "residual", device);
    operand::tensor(update, DType::BF16, {d, m}, 2, op, "update", device);
    operand::tensor(out, DType::BF16, {d, m}, 2, op, "residual_out", device);
    const auto range = operand::range(out, op);
    if (residual.data != out.data) operand::disjoint(operand::range(residual, op), range, op);
    operand::disjoint(operand::range(update, op), range, op);
    operand::disjoint(operand::range(gamma, op), range, op);
    return range;
}

} // namespace

void scaled_residual_add(const Tensor& residual, const Tensor& update, const Tensor& gamma,
                         Tensor& residual_out, CudaExecutionView execution) {
    constexpr char op[] = "scaled_residual_add";
    const auto device   = execution.device();
    validate_add_operands(residual, update, gamma, residual_out, device, op);
    if (residual.ne[1] == 0) return;
    validate_add_storage(residual, update, gamma, residual_out, device, op);
    if (gamma.dtype == DType::FP32)
        launch_add<float>(residual, update, gamma, residual_out, execution.stream());
    else
        launch_add<Bf16>(residual, update, gamma, residual_out, execution.stream());
    CUDA_CHECK(cudaGetLastError());
}

void scaled_residual_add_layer_norm(const Tensor& residual, const Tensor& update,
                                    const Tensor& gamma, const Tensor& weight, const Tensor& bias,
                                    float eps, Tensor& residual_out, Tensor& normalized_out,
                                    CudaExecutionView execution) {
    constexpr char op[] = "scaled_residual_add_layer_norm";
    const auto device   = execution.device();
    validate_add_operands(residual, update, gamma, residual_out, device, op);
    const int d = residual.ne[0], m = residual.ne[1];
    if (!(eps > 0.f) || !std::isfinite(eps))
        throw std::invalid_argument("scaled_residual_add_layer_norm: invalid epsilon");
    operand::shape(normalized_out, {d, m, 1, 1}, op, "normalized_out");
    operand::dtype(normalized_out, DType::BF16, op, "normalized_out must be BF16");
    operand::tensor(weight, DType::BF16, {d}, 2, op, "weight", device);
    operand::tensor(bias, DType::BF16, {d}, 2, op, "bias", device);
    if (m == 0) return;

    const auto published = validate_add_storage(residual, update, gamma, residual_out, device, op);
    operand::tensor(normalized_out, DType::BF16, {d, m}, 2, op, "normalized_out", device);
    operand::disjoint(operand::range(weight, op), published, op);
    operand::disjoint(operand::range(bias, op), published, op);
    const auto norm = operand::range(normalized_out, op);
    operand::disjoint(published, norm, op);
    for (const auto* input : {&residual, &update, &gamma, &weight, &bias})
        operand::disjoint(operand::range(*input, op), norm, op);

    if (gamma.dtype == DType::FP32)
        launch_norm<float>(residual, update, gamma, weight, bias, eps, residual_out, normalized_out,
                           execution.stream());
    else
        launch_norm<Bf16>(residual, update, gamma, weight, bias, eps, residual_out, normalized_out,
                          execution.stream());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
