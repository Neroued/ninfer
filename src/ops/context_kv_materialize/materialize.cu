#include "ops/context_kv_materialize/launch.h"
#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/common/dflash_rope.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {
namespace {
constexpr int kLayers  = static_cast<int>(kContextKVMaterializeLayers);
constexpr int kRows    = 1024;
constexpr int kHeadDim = 128;

__device__ __forceinline__ int context_column(int column, int width, int prefix) {
    return width == prefix ? column : column / prefix * width + column % prefix;
}

struct DeviceLayerView {
    const std::uint8_t* key_codes;
    const std::uint8_t* key_scales;
    const std::uint8_t* value_codes;
    const std::uint8_t* value_scales;
    const __nv_bfloat16* key_norm;
    __nv_bfloat16* cache_k;
    __half* cache_v;
    std::int32_t padded_capacity;
};

struct DeviceLayers {
    DeviceLayerView layer[kContextKVMaterializeLayers];
};

__device__ __forceinline__ void store_key_head(const float* input, DeviceLayerView layer,
                                               const int* positions, const int* slots, int column,
                                               int width, int head) {
    const int lane = threadIdx.x & 31;
    const int j    = lane * 2;
    float x0 = input[j], x1 = input[j + 1], y0 = input[j + 64], y1 = input[j + 65];
    float sum           = warp_reduce_sum(x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1);
    const float inverse = rsqrtf(__shfl_sync(0xffffffffU, sum, 0) / 128.0f + 1.e-6f);
    x0 *= inverse * __bfloat162float(layer.key_norm[j]);
    x1 *= inverse * __bfloat162float(layer.key_norm[j + 1]);
    y0 *= inverse * __bfloat162float(layer.key_norm[j + 64]);
    y1 *= inverse * __bfloat162float(layer.key_norm[j + 65]);
    float sin0, cos0, sin1, cos1;
    dflash_rope_sincos(positions, column, j, &sin0, &cos0);
    dflash_rope_sincos(positions, column, j + 1, &sin1, &cos1);
    const auto dst = 128LL * ((positions[column] & 2047) + (long long)layer.padded_capacity *
                                                               (head + 8 * slots[column / width]));
    auto* out      = reinterpret_cast<__nv_bfloat162*>(layer.cache_k + dst);
    out[lane]      = __floats2bfloat162_rn(x0 * cos0 - y0 * sin0, x1 * cos1 - y1 * sin1);
    out[lane + 32] = __floats2bfloat162_rn(y0 * cos0 + x0 * sin0, y1 * cos1 + x1 * sin1);
}

union alignas(16) Bf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

__device__ __forceinline__ int swizzle_128(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

template <int Rows, int Columns, int BlockK>
union alignas(16) MaterializeStorage {
    struct {
        __nv_bfloat16 code_values[Rows][BlockK];
        __nv_bfloat16 activations[Columns][BlockK];
        std::uint8_t codes[Rows][BlockK];
        std::uint16_t scales[Rows][BlockK / 32];
    } mainloop;

    float scores[Columns][Rows];
};

template <int Rows, int Columns, int BlockK, int ColumnWarps>
__global__ __launch_bounds__(Rows / 16 * ColumnWarps * 32, 1) void context_kv_mma_kernel(
    const __nv_bfloat16* hidden, const int* positions, const int* counts, const int* slots,
    DeviceLayers layers, float* key_scratch, int width, int batch, int min_count, int max_count) {
    constexpr int kBlockRows = Rows, kBlockK = BlockK;
    constexpr int kBlockColumns = Columns;
    constexpr int kColumnWarps  = ColumnWarps;
    constexpr int kWarps = Rows / 16 * kColumnWarps, kThreads = kWarps * 32;
    constexpr int kWarpColumns = Columns / kColumnWarps, kTokenMmas = kWarpColumns / 8;
    constexpr int kKTiles = 5120 / kBlockK;
    static_assert((Rows == 64 || Rows == 128) && Columns % (8 * ColumnWarps) == 0);
    static_assert(5120 % kBlockK == 0 && kThreads <= 1024);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int warp_row = warp / kColumnWarps, warp_col = warp % kColumnWarps;
    const int gid = lane >> 2, lid = lane & 3;
    const int a_matrix = lane >> 3, a_rowoff = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_coloff = (a_matrix >> 1) << 3, b_row = lane & 7, b_coloff = ((lane >> 3) & 1) << 3;
    const int column_begin = blockIdx.y * Columns,
              live_columns = min(Columns, max_count * batch - column_begin);
    const int row_begin = blockIdx.x * Rows, layer_index = blockIdx.z >> 1;
    const bool value          = (blockIdx.z & 1) != 0;
    const auto layer          = layers.layer[layer_index];
    const auto* weight_codes  = value ? layer.value_codes : layer.key_codes;
    const auto* weight_scales = value ? layer.value_scales : layer.key_scales;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto& storage  = *reinterpret_cast<MaterializeStorage<Rows, Columns, BlockK>*>(shared_bytes);
    auto& mainloop = storage.mainloop;
    {
        float accumulators[kTokenMmas][4] = {};

        const auto stage_activation = [&](int k_tile) {
            const int k_begin    = k_tile * kBlockK;
            constexpr int kItems = kBlockColumns * (kBlockK / 8);
            for (int item = tid; item < kItems; item += kThreads) {
                const int column  = item / (kBlockK / 8);
                const int k8      = item - column * (kBlockK / 8);
                auto* destination = &mainloop.activations[column][swizzle_128(column, k8 * 8)];
                if (column < live_columns) {
                    cp_async<16, Cache::ca>(destination,
                                            hidden +
                                                static_cast<std::int64_t>(context_column(
                                                    column_begin + column, width, max_count)) *
                                                    5120 +
                                                k_begin + k8 * 8);
                } else {
                    cp_async_zfill<16, Cache::ca>(destination, hidden + k_begin + k8 * 8, 0);
                }
            }
        };

        const auto stage_weight = [&](int k_tile) {
            const int k_begin     = k_tile * kBlockK;
            constexpr int kChunks = kBlockRows * (kBlockK / 16);
            for (int item = tid; item < kChunks; item += kThreads) {
                const int local_row = item / (kBlockK / 16);
                const int chunk     = item - local_row * (kBlockK / 16);
                cp_async<16, Cache::cg>(
                    &mainloop.codes[local_row][chunk * 16],
                    weight_codes + static_cast<std::int64_t>(row_begin + local_row) * 5120 +
                        k_begin + chunk * 16);
            }
            for (int local_row = tid; local_row < kBlockRows; local_row += kThreads) {
                const std::int64_t group =
                    static_cast<std::int64_t>(row_begin + local_row) * (5120 / 32) + k_begin / 32;
                cp_async<kBlockK / 32 * sizeof(std::uint16_t)>(
                    &mainloop.scales[local_row][0], weight_scales + group * sizeof(std::uint16_t));
            }
        };

        // Signed W8 codes are exactly representable in BF16. Apply the exact stored FP16
        // scale in FP32 after each 32-wide MMA group; never round a scaled weight to BF16.
        const auto decode_signed_codes = [&]() {
            constexpr int kChunksPerRow = kBlockK / 8;
            for (int item = tid; item < kBlockRows * kChunksPerRow; item += kThreads) {
                const int row      = item / kChunksPerRow;
                const int chunk    = item - row * kChunksPerRow;
                const int col      = chunk * 8;
                const uint2 packed = *reinterpret_cast<const uint2*>(&mainloop.codes[row][col]);
                Bf16x8 decoded;
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const unsigned word = (pair < 2 ? packed.x : packed.y) >> ((pair & 1) * 16);
                    const int q0        = static_cast<int>(static_cast<std::int8_t>(word & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>((word >> 8) & 0xffu));
                    decoded.pair[pair] =
                        __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
                }
                store_vec(&mainloop.code_values[row][swizzle_128(row, col)], decoded.raw);
            }
        };

        stage_activation(0);
        stage_weight(0);
        cp_commit();

#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
            cp_wait<0>();
            __syncthreads();
            decode_signed_codes();
            __syncthreads();

            // Capture scales before the next async weight stage reuses their shared plane.
            float top_scales[kBlockK / 32], bottom_scales[kBlockK / 32];
#pragma unroll
            for (int g = 0; g < kBlockK / 32; ++g) {
                top_scales[g] =
                    __half2float(__ushort_as_half(mainloop.scales[warp_row * 16 + gid][g]));
                bottom_scales[g] =
                    __half2float(__ushort_as_half(mainloop.scales[warp_row * 16 + gid + 8][g]));
            }
            __syncthreads();
            const int next = k_tile + 1;
            if (next < kKTiles) {
                stage_weight(next);
                cp_commit();
            }

            const auto load_fragments = [&](int k_step, unsigned(&a)[4],
                                            unsigned(&b)[kTokenMmas][2]) {
                const int weight_row = warp_row * 16 + a_rowoff;
                const int weight_col = k_step * 16 + a_coloff;
                ldmatrix_x4(
                    a[0], a[1], a[2], a[3],
                    smem_addr(
                        &mainloop.code_values[weight_row][swizzle_128(weight_row, weight_col)]));
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    const int activation_row = warp_col * kWarpColumns + token_mma * 8 + b_row;
                    const int activation_col = k_step * 16 + b_coloff;
                    ldmatrix_x2(b[token_mma][0], b[token_mma][1],
                                smem_addr(&mainloop.activations[activation_row][swizzle_128(
                                    activation_row, activation_col)]));
                }
            };

            unsigned a_fragments[2][4];
            unsigned b_fragments[2][kTokenMmas][2];
            load_fragments(0, a_fragments[0], b_fragments[0]);
#pragma unroll
            for (int group = 0; group < kBlockK / 32; ++group) {
                float group_acc[kTokenMmas][4] = {};
#pragma unroll
                for (int step = 0; step < 2; ++step) {
                    const int k_step = 2 * group + step;
                    if (k_step + 1 < kBlockK / 16)
                        load_fragments(k_step + 1, a_fragments[step ^ 1], b_fragments[step ^ 1]);
#pragma unroll
                    for (int t = 0; t < kTokenMmas; ++t)
                        mma_bf16(group_acc[t][0], group_acc[t][1], group_acc[t][2], group_acc[t][3],
                                 a_fragments[step][0], a_fragments[step][1], a_fragments[step][2],
                                 a_fragments[step][3], b_fragments[step][t][0],
                                 b_fragments[step][t][1]);
                }
#pragma unroll
                for (int t = 0; t < kTokenMmas; ++t) {
                    accumulators[t][0] =
                        fmaf(group_acc[t][0], top_scales[group], accumulators[t][0]);
                    accumulators[t][1] =
                        fmaf(group_acc[t][1], top_scales[group], accumulators[t][1]);
                    accumulators[t][2] =
                        fmaf(group_acc[t][2], bottom_scales[group], accumulators[t][2]);
                    accumulators[t][3] =
                        fmaf(group_acc[t][3], bottom_scales[group], accumulators[t][3]);
                }
            }

            if (next < kKTiles) {
                __syncthreads();
                stage_activation(next);
                cp_commit();
            }
        }

        __syncthreads();
        auto& scores         = storage.scores;
        const int local_row0 = warp_row * 16 + gid;
        const int local_row1 = local_row0 + 8;
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            const int column0 = warp_col * kWarpColumns + token_mma * 8 + 2 * lid;
            if (column0 < Columns) {
                scores[column0][local_row0] = accumulators[token_mma][0];
                scores[column0][local_row1] = accumulators[token_mma][2];
            }
            if (column0 + 1 < Columns) {
                scores[column0 + 1][local_row0] = accumulators[token_mma][1];
                scores[column0 + 1][local_row1] = accumulators[token_mma][3];
            }
        }
    }
    __syncthreads();
    for (int local = warp; local < live_columns; local += kWarps) {
        const int packed_column = column_begin + local;
        const int column        = context_column(packed_column, width, max_count);
        const int request       = column / width;
        const int count         = counts[request];
        if (count < min_count || count > max_count || column % width >= count) continue;
        if constexpr (Rows == 128) {
            if (!value) {
                store_key_head(storage.scores[local], layer, positions, slots, column, width,
                               row_begin / 128);
                continue;
            }
        }
        for (int r = lane; r < Rows; r += 32) {
            const int row      = row_begin + r;
            const float result = storage.scores[local][r];
            if (!value) {
                key_scratch[row + 1024LL * (packed_column + max_count * batch * layer_index)] =
                    result;
            } else {
                const auto dst     = row % 128 + 128LL * ((positions[column] & 2047) +
                                                      (long long)layer.padded_capacity *
                                                          (row / 128 + 8 * slots[request]));
                layer.cache_v[dst] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(result)));
            }
        }
    }
}

template <int Rows, int Columns, int BlockK, int ColumnWarps>
void launch_mma(const Tensor& x, const Tensor& positions, const Tensor& counts, const Tensor& slots,
                DeviceLayers layers, ContextKVMaterializeExecutionEnvelope envelope,
                const Tensor& scratch, cudaStream_t stream) {
    constexpr int bytes = sizeof(MaterializeStorage<Rows, Columns, BlockK>);
    if constexpr (bytes > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(context_kv_mma_kernel<Rows, Columns, BlockK, ColumnWarps>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    context_kv_mma_kernel<Rows, Columns, BlockK, ColumnWarps>
        <<<dim3(1024 / Rows, (envelope.max_count * x.ne[2] + Columns - 1) / Columns, 10),
           Rows / 16 * ColumnWarps * 32, bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const int*>(positions.data),
            static_cast<const int*>(counts.data), static_cast<const int*>(slots.data), layers,
            static_cast<float*>(scratch.data), x.ne[1], x.ne[2], envelope.min_count,
            envelope.max_count);
    CUDA_CHECK(cudaGetLastError());
}

struct MaterializeProjectionEpilogue {
    DeviceLayerView layer;
    const int* positions;
    const int* counts;
    const int* slots;
    float* scratch;
    int layer_index, width, batch, min_count, max_count;
    bool value;

    __device__ void store(int row, int packed_column, float result) const {
        if (packed_column >= max_count * batch) return;
        const int column  = context_column(packed_column, width, max_count);
        const int request = column / width, count = counts[request];
        if (count < min_count || count > max_count || column % width >= count) return;
        if (!value)
            scratch[row + 1024LL * (packed_column + max_count * batch * layer_index)] = result;
        else {
            const auto dst     = row % 128 + 128LL * ((positions[column] & 2047) +
                                                  (long long)layer.padded_capacity *
                                                      (row / 128 + 8 * slots[request]));
            layer.cache_v[dst] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(result)));
        }
    }

    __device__ void store_pair(int row, int col, float4 sum, int columns) const {
        if (col < columns) {
            store(row, col, sum.x);
            store(row + 8, col, sum.z);
        }
        if (col + 1 < columns) {
            store(row, col + 1, sum.y);
            store(row + 8, col + 1, sum.w);
        }
    }
};

struct ContextPrefixColumns {
    int width, prefix;

    __device__ __forceinline__ int operator()(int column) const {
        return context_column(column, width, prefix);
    }
};

template <int Columns, int KWarps = 8>
using GroupedSchedule = W8SmallTMmaSchedule<KWarps, Columns, 1, W8SmallTMmaScaleAccess::Shared>;

template <int Columns, int KWarps = 8>
__global__ __launch_bounds__(KWarps * 32, 1) void context_kv_grouped_kernel(
    const __nv_bfloat16* x, const int* positions, const int* counts, const int* slots,
    DeviceLayers layers, float* scratch, int width, int batch, int min_count, int max_count) {
    const int l        = blockIdx.z >> 1;
    const bool value   = (blockIdx.z & 1) != 0;
    const auto layer   = layers.layer[l];
    const auto* codes  = value ? layer.value_codes : layer.key_codes;
    const auto* scales = value ? layer.value_scales : layer.key_scales;
    const MaterializeProjectionEpilogue epilogue{layer, positions, counts,    slots,     scratch, l,
                                                 width, batch,     min_count, max_count, value};
    w8_small_t_mma<W8LinearGeometry<1024, 5120>, Columns, GroupedSchedule<Columns, KWarps>,
                   W8ContiguousOutput, MaterializeProjectionEpilogue, W8SmallTMmaIdentityRows, true,
                   true>(x, codes, scales, {nullptr, 0}, epilogue, {}, max_count * batch,
                         ContextPrefixColumns{width, max_count});
}

template <int Columns, int KWarps = 8>
void launch_grouped(const Tensor& x, const Tensor& positions, const Tensor& counts,
                    const Tensor& slots, DeviceLayers layers,
                    ContextKVMaterializeExecutionEnvelope envelope, const Tensor& scratch,
                    cudaStream_t stream) {
    context_kv_grouped_kernel<Columns, KWarps>
        <<<dim3(64, (envelope.max_count * x.ne[2] + Columns - 1) / Columns, 10), KWarps * 32, 0,
           stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                     static_cast<const int*>(positions.data), static_cast<const int*>(counts.data),
                     static_cast<const int*>(slots.data), layers, static_cast<float*>(scratch.data),
                     x.ne[1], x.ne[2], envelope.min_count, envelope.max_count);
    CUDA_CHECK(cudaGetLastError());
}

__global__ __launch_bounds__(256) void context_kv_key_post_kernel(
    const float* __restrict__ key_scratch, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ state_slots,
    DeviceLayers layers, std::int32_t batch_size, std::int32_t width, std::int32_t min_count,
    std::int32_t max_count) {
    const int packed_column   = static_cast<int>(blockIdx.x);
    const int physical_column = context_column(packed_column, width, max_count);
    const int layer_index     = static_cast<int>(blockIdx.y);
    const int batch           = physical_column / width;
    const int local           = physical_column % width;
    const int count           = counts[batch];
    if (count < min_count || count > max_count || local >= count) return;
    const auto layer   = layers.layer[layer_index];
    const int head     = threadIdx.x >> 5;
    const float* input = key_scratch +
                         kRows * (packed_column + max_count * batch_size * layer_index) +
                         head * kHeadDim;
    store_key_head(input, layer, positions, state_slots, physical_column, width, head);
}

DeviceLayers make_device_layers(
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers) {
    DeviceLayers result{};
    for (int index = 0; index < kLayers; ++index) {
        const ContextKVMaterializeLayerView& source = layers[static_cast<std::size_t>(index)];
        result.layer[index]                         = {
            static_cast<const std::uint8_t*>(source.key_weight.qdata),
            static_cast<const std::uint8_t*>(source.key_weight.scales),
            static_cast<const std::uint8_t*>(source.value_weight.qdata),
            static_cast<const std::uint8_t*>(source.value_weight.scales),
            static_cast<const __nv_bfloat16*>(source.key_norm_weight.data),
            static_cast<__nv_bfloat16*>(source.cache.k.data),
            static_cast<__half*>(source.cache.v.data),
            static_cast<std::int32_t>(source.cache.padded_capacity),
        };
    }
    return result;
}

} // namespace

void context_kv_materialize_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, ContextKVMaterializeRoute route,
    const Tensor& key_scratch, cudaStream_t stream) {
    const DeviceLayers device_layers = make_device_layers(layers);
    using Route                      = ContextKVMaterializeRoute;
    switch (route) {
    case Route::KSplit16:
        launch_grouped<16>(context, positions, counts, state_slots, device_layers, envelope,
                           key_scratch, stream);
        break;
    case Route::KSplit24:
        launch_grouped<24>(context, positions, counts, state_slots, device_layers, envelope,
                           key_scratch, stream);
        break;
    case Route::Mma32:
        launch_mma<64, 32, 128, 2>(context, positions, counts, state_slots, device_layers, envelope,
                                   key_scratch, stream);
        break;
    case Route::Mma80:
        launch_mma<64, 80, 128, 5>(context, positions, counts, state_slots, device_layers, envelope,
                                   key_scratch, stream);
        break;
    case Route::Mma96:
        launch_mma<64, 96, 128, 6>(context, positions, counts, state_slots, device_layers, envelope,
                                   key_scratch, stream);
        break;
    case Route::Fused64:
        launch_mma<128, 64, 128, 2>(context, positions, counts, state_slots, device_layers,
                                    envelope, key_scratch, stream);
        return;
    case Route::Mma64:
        launch_mma<64, 64, 64, 2>(context, positions, counts, state_slots, device_layers, envelope,
                                  key_scratch, stream);
        break;
    }
    context_kv_key_post_kernel<<<dim3(envelope.max_count * context.ne[2], kLayers), 256, 0,
                                 stream>>>(
        static_cast<const float*>(key_scratch.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(counts.data), static_cast<const int*>(state_slots.data),
        device_layers, context.ne[2], context.ne[1], envelope.min_count, envelope.max_count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
