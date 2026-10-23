#pragma once

#include "ops/kv_cache/context_materialize/epilogue.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/linear/kernels/q8/q8_a16_sliced_k_mma.cuh"

namespace ninfer::ops::detail::context_kv {

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
    const __nv_bfloat16* hidden, const int* positions, const int* counts,
    const std::int64_t* offsets, DeviceLayers layers, float* key_scratch, int width, int batch,
    int min_count, int max_count) {
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
                const int column = item / (kBlockK / 8);
                const int k8     = item - column * (kBlockK / 8);
                auto* destination =
                    &mainloop.activations[column][common::swizzle_xor_64(column, k8 * 8)];
                if (column < live_columns) {
                    common::cp_async<16, Cache::ca>(
                        destination, hidden +
                                         static_cast<std::int64_t>(context_column(
                                             column_begin + column, width, max_count)) *
                                             5120 +
                                         k_begin + k8 * 8);
                } else {
                    common::cp_async_zfill<16, Cache::ca>(destination, hidden + k_begin + k8 * 8,
                                                          0);
                }
            }
        };

        const auto stage_weight = [&](int k_tile) {
            const int k_begin     = k_tile * kBlockK;
            constexpr int kChunks = kBlockRows * (kBlockK / 16);
            for (int item = tid; item < kChunks; item += kThreads) {
                const int local_row = item / (kBlockK / 16);
                const int chunk     = item - local_row * (kBlockK / 16);
                common::cp_async<16, Cache::cg>(
                    &mainloop.codes[local_row][chunk * 16],
                    weight_codes + static_cast<std::int64_t>(row_begin + local_row) * 5120 +
                        k_begin + chunk * 16);
            }
            for (int local_row = tid; local_row < kBlockRows; local_row += kThreads) {
                const std::int64_t group =
                    static_cast<std::int64_t>(row_begin + local_row) * (5120 / 32) + k_begin / 32;
                common::cp_async<kBlockK / 32 * sizeof(std::uint16_t)>(
                    &mainloop.scales[local_row][0], weight_scales + group * sizeof(std::uint16_t));
            }
        };

        // Signed Q8 codes are exactly representable in BF16. Apply the exact stored FP16
        // scale in FP32 after each 32-wide MMA group; never round a scaled weight to BF16.
        const auto decode_signed_codes = [&]() {
            constexpr int kChunksPerRow = kBlockK / 8;
            for (int item = tid; item < kBlockRows * kChunksPerRow; item += kThreads) {
                const int row      = item / kChunksPerRow;
                const int chunk    = item - row * kChunksPerRow;
                const int col      = chunk * 8;
                const uint2 packed = *reinterpret_cast<const uint2*>(&mainloop.codes[row][col]);
                common::Bf16x8Bits decoded;
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const unsigned word = (pair < 2 ? packed.x : packed.y) >> ((pair & 1) * 16);
                    const int q0        = static_cast<int>(static_cast<std::int8_t>(word & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>((word >> 8) & 0xffu));
                    decoded.pair[pair] =
                        __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
                }
                common::store_vec(&mainloop.code_values[row][common::swizzle_xor_64(row, col)],
                                  decoded.raw);
            }
        };

        stage_activation(0);
        stage_weight(0);
        common::cp_commit();

#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
            common::cp_wait<0>();
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
                common::cp_commit();
            }

            const auto load_fragments = [&](int k_step, unsigned(&a)[4],
                                            unsigned(&b)[kTokenMmas][2]) {
                const int weight_row = warp_row * 16 + a_rowoff;
                const int weight_col = k_step * 16 + a_coloff;
                common::ldmatrix_x4(
                    a[0], a[1], a[2], a[3],
                    common::smem_addr(&mainloop.code_values[weight_row][common::swizzle_xor_64(
                        weight_row, weight_col)]));
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    const int activation_row = warp_col * kWarpColumns + token_mma * 8 + b_row;
                    const int activation_col = k_step * 16 + b_coloff;
                    common::ldmatrix_x2(
                        b[token_mma][0], b[token_mma][1],
                        common::smem_addr(
                            &mainloop.activations[activation_row][common::swizzle_xor_64(
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
                        common::mma_bf16(
                            group_acc[t][0], group_acc[t][1], group_acc[t][2], group_acc[t][3],
                            a_fragments[step][0], a_fragments[step][1], a_fragments[step][2],
                            a_fragments[step][3], b_fragments[step][t][0], b_fragments[step][t][1]);
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
                common::cp_commit();
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
                store_key_head(storage.scores[local], layer, positions, offsets, column, width,
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
                store_value(row, column, offsets[request], positions, layer, result);
            }
        }
    }
}

template <int Columns, int KWarps = 8>
using GroupedSchedule =
    Q8A16SlicedKMmaSchedule<Columns, KWarps, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                            Q8ActivationStage::ActiveOnly, 5120>;

template <int Columns, int KWarps = 8>
__global__ __launch_bounds__(KWarps * 32, 1) void context_kv_grouped_kernel(
    const __nv_bfloat16* x, const int* positions, const int* counts, const std::int64_t* offsets,
    DeviceLayers layers, float* scratch, int width, int batch, int min_count, int max_count) {
    const int l        = blockIdx.z >> 1;
    const bool value   = (blockIdx.z & 1) != 0;
    const auto layer   = layers.layer[l];
    const auto* codes  = value ? layer.value_codes : layer.key_codes;
    const auto* scales = value ? layer.value_scales : layer.key_scales;
    const MaterializeProjectionEpilogue epilogue{layer, positions, counts,    offsets,   scratch, l,
                                                 width, batch,     min_count, max_count, value};
    q8_a16_sliced_k_mma<GroupedSchedule<Columns, KWarps>, true>(
        Q8LinearOperands{x, codes, scales, 1024, 5120, max_count * batch, 5120},
        LinearBf16Output{nullptr, 0}, epilogue, Q8SlicedKIdentityRows{}, 0,
        ContextPrefixColumns{width, max_count});
}

} // namespace ninfer::ops::detail::context_kv
