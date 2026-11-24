#include "ops/common/host/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include "ninfer/ops/argmax.h"
#include <cstdint>
#include <stdexcept>
#include "ops/common/numeric/math.h"
#include "ops/common/host/launch.h"
#include "core/device.h"
#include "core/platform.h"
#include "ops/common/numeric/convert.cuh"
#include <cuda_bf16.h>
#include <climits>
#include <math_constants.h>

namespace ninfer::ops {
using namespace common;

namespace {

// Small-column execution uses 512 threads to reduce atomic contenders. Aggregate
// execution dispatches registered vocab profiles to a smaller block so the much
// larger 2-D grid exposes enough resident CTAs without oversized reductions.
inline constexpr int kArgmaxBlock          = 512;
inline constexpr int kArgmaxItemsPerThread = 1;

__launch_bounds__(kArgmaxBlock) __global__
    void argmax_kernel(const __nv_bfloat16* logits, std::int32_t* out, std::int32_t valid_rows,
                       std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;

    float best_value        = __bfloat162float(logits[base]);
    std::int32_t best_index = 0;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += blockDim.x) {
        const float value = __bfloat162float(logits[base + v]);
        if (score_id_better(value, v, best_value, best_index)) {
            best_value = value;
            best_index = v;
        }
    }

    __shared__ float values[kArgmaxBlock];
    __shared__ std::int32_t indices[kArgmaxBlock];
    values[threadIdx.x]  = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value        = values[threadIdx.x + stride];
            const std::int32_t other_index = indices[threadIdx.x + stride];
            if (score_id_better(other_value, other_index, values[threadIdx.x],
                                indices[threadIdx.x])) {
                values[threadIdx.x]  = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { out[t] = indices[0]; }
}

template <int MaxBlock, int Items>
__launch_bounds__(MaxBlock) __global__
    void argmax_tiled_atomic_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                                    std::int32_t valid_rows, std::int32_t physical_rows) {
    const std::int32_t t          = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base       = static_cast<std::int64_t>(t) * physical_rows;
    const std::int32_t tile_start = static_cast<std::int32_t>(blockIdx.x) * blockDim.x * Items;

    float best_value        = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
    if constexpr (Items == 1) {
        const auto v = tile_start + threadIdx.x;
        if (v < valid_rows) {
            const float value = __bfloat162float(logits[base + v]);
            if (score_id_better(value, v, best_value, best_index)) {
                best_value = value;
                best_index = v;
            }
        }
    } else {
        const auto start = tile_start + threadIdx.x * Items;
        if (start + Items <= valid_rows) {
            const auto values = Bf16Vector<Items>::load(logits + base + start);
#pragma unroll
            for (int item = 0; item < Items; ++item) {
                const float value = __bfloat162float(values.word[item]);
                if (score_id_better(value, start + item, best_value, best_index)) {
                    best_value = value;
                    best_index = start + item;
                }
            }
        } else {
#pragma unroll
            for (int item = 0; item < Items; ++item) {
                const auto v = start + item;
                if (v < valid_rows) {
                    const float value = __bfloat162float(logits[base + v]);
                    if (score_id_better(value, v, best_value, best_index)) {
                        best_value = value;
                        best_index = v;
                    }
                }
            }
        }
    }

    block_reduce_argmax(best_value, best_index);
    if (threadIdx.x != 0 || best_index == INT32_MAX) { return; }

    int current = out[t];
    while (true) {
        const float current_value = __bfloat162float(logits[base + current]);
        if (!score_id_better(best_value, best_index, current_value, current)) { break; }
        const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, best_index);
        if (observed == current) { break; }
        current = observed;
    }
}

constexpr std::int32_t kFullPhysicalRows      = 248320;
constexpr std::int32_t kFullValidRows         = 248077;
constexpr std::int32_t kShortlistRows         = 131072;
constexpr std::int32_t kSmallAggregateColumns = 8;
constexpr int kFullAggregateBlock             = 128;
constexpr int kShortlistAggregateBlock        = 256;

int select_tiled_block(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t t_count) {
    if (t_count <= kSmallAggregateColumns) { return kArgmaxBlock; }
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows) {
        return kFullAggregateBlock;
    }
    if (physical_rows == kShortlistRows && valid_rows == kShortlistRows) {
        return kShortlistAggregateBlock;
    }
    return kArgmaxBlock;
}

template <int MaxBlock, int Items>
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream);

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                   cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    if (t_count == 0) { return; }

    constexpr int kTileElems = kArgmaxBlock * kArgmaxItemsPerThread;
    const int tiled_blocks   = div_up(valid_rows, kTileElems);
    if (tiled_blocks < 2) {
        argmax_kernel<<<static_cast<unsigned int>(t_count), kArgmaxBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

#if NINFER_TARGET_SPARK
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows &&
        reinterpret_cast<std::uintptr_t>(logits.data) % 16 == 0) {
        argmax_tiled_atomic_launch<256, 8>(logits, out, valid_rows, 256, stream);
        return;
    }
#endif
    argmax_tiled_atomic_launch<kArgmaxBlock, 1>(
        logits, out, valid_rows, select_tiled_block(physical_rows, valid_rows, t_count), stream);
}

template <int MaxBlock, int Items>
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    const int tiled_blocks           = div_up(valid_rows, block * Items);
    for_each_token_slice(t_count, 1, [&](int token_offset, int token_count) {
        const Tensor logits_slice = logits.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(0, token_offset, token_count);
        CUDA_CHECK(cudaMemsetAsync(out_slice.data, 0,
                                   static_cast<std::size_t>(token_count) * sizeof(std::int32_t),
                                   stream));
        const dim3 grid(static_cast<unsigned int>(tiled_blocks),
                        static_cast<unsigned int>(token_count));
        argmax_tiled_atomic_kernel<MaxBlock, Items><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits_slice.data),
            static_cast<std::int32_t*>(out_slice.data), valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace

void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
            CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("argmax: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("argmax: out must be I32"); }

    const std::int64_t logits_n = common::operand::numel(logits, "argmax");
    (void)common::operand::numel(out, "argmax");

    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("argmax: logits must be rank-2 [vocab,T]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("argmax: out must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("argmax: physical rows must be positive");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("argmax: valid_rows must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("argmax: out shape must be [logits.ne[1]]");
    }
    if (logits_n == 0) { return; }
    if (logits.device != execution.device() || out.device != execution.device()) {
        throw std::invalid_argument("argmax: operands must belong to the execution device");
    }

    if (!logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("argmax: logits/out must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("argmax: logits/out data must be non-null");
    }

    argmax_launch(logits, out, valid_rows, stream);
}

} // namespace ninfer::ops
