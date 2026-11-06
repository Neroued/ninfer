#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/a16.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/csf32.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::a16 {

template <int Intermediate, bool Gate, int Rows, int Columns, int KStep, int Stages,
          bool Fp16 = false, bool RegisterPipeline = false>
struct Schedule {
    static constexpr int intermediate = Intermediate;
    static constexpr bool gate        = Gate;
    static constexpr int rows = Rows, columns = Columns, k_step = KStep, stages = Stages;
    static constexpr int k             = Gate ? kHidden : Intermediate;
    static constexpr int feature_rows  = Gate ? Rows / 2 : Rows;
    static constexpr int threads       = Rows;
    static constexpr int mma_tokens    = Columns / 8;
    static constexpr bool fp16_product = Fp16;
    static constexpr bool bf16_weights = Gate || !Fp16;
    static constexpr int planes        = Gate || Fp16 ? 1 : 2;
    static constexpr bool register_pipeline = RegisterPipeline;
};

template <class S>
struct Tile {
    alignas(16) std::uint16_t inputs[S::stages][S::planes][S::columns * S::k_step];
    alignas(16) std::uint8_t codes[S::stages][S::rows * (S::k_step / 2)];
    alignas(16) std::uint8_t scales[S::stages][S::register_pipeline ? 1 : 128 * (S::k_step / 16)];
    int sources[S::columns];
};

template <bool Fp16>
__device__ __forceinline__ void store_product(A16Product product, std::int64_t offset,
                                              float value) {
    if constexpr (Fp16) {
        reinterpret_cast<float*>(product.values)[offset] = value;
    } else {
        const auto high = __float2bfloat16_rn(value);
        const auto low  = __float2bfloat16_rn(value - __bfloat162float(high));
        reinterpret_cast<__nv_bfloat16*>(product.values)[offset]   = high;
        reinterpret_cast<__nv_bfloat16*>(product.residual)[offset] = low;
    }
}

template <int Intermediate, bool Fp16>
__device__ __forceinline__ float product_inverse(A16Product product, int column) {
    if constexpr (Fp16) {
        const auto* exponent = reinterpret_cast<const std::uint8_t*>(
            product.values + std::int64_t(column) * (2 * Intermediate) + Intermediate);
        return __uint_as_float(unsigned(127 - *exponent) << 23);
    }
    return 1.F;
}

template <int Intermediate>
__global__ __launch_bounds__(256) void normalize_product(A16Product product, Groups groups,
                                                          const int* ids, int vector_limit,
                                                          Nvfp4A16Dispatch dispatch) {
    static_assert(Intermediate % 1024 == 0);
    constexpr int parts = Intermediate / 1024;
    __shared__ float maxima[8];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    if (groups.load_range[1] <= vector_limit) return;
    const int cohort = dispatch.cohort_load(groups.load_range[0], groups.load_range[1]);
    const int columns = groups.offsets[kExperts];
    for (int column = blockIdx.x; column < columns; column += gridDim.x) {
        if (!cohort) {
            const int expert = ids[groups.assignments[column]];
            if (groups.offsets[expert + 1] - groups.offsets[expert] <= vector_limit) continue;
        }
        float4 values[parts];
        float maximum = 0.F;
#pragma unroll
        for (int part = 0; part < parts; ++part) {
            const auto v = common::load_vec<float4>(reinterpret_cast<const float*>(product.values) +
                                                    std::int64_t(column) * Intermediate +
                                                    part * 1024 + tid * 4);
            values[part] = v;
            maximum =
                fmaxf(maximum, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
        maximum = common::warp_max(maximum);
        if (!lane) maxima[warp] = maximum;
        __syncthreads();
        maximum      = common::warp_max(lane < 8 ? maxima[lane] : 0.F);
        int exponent = 9;
        if (maximum > 0.F) {
            const int power = int((__float_as_uint(maximum) >> 23) & 255) - 127;
            exponent        = max(9, min(120, 14 - power));
        }
        const float scale = __uint_as_float(unsigned(127 + exponent) << 23);
        // All original FP32 values are now private registers. Each column has
        // its own FP32-sized row, so writing the half-sized prefix cannot race
        // a different column's reads. Its spare half owns the exponent byte.
        auto* destination = product.values + std::int64_t(column) * (2 * Intermediate);
#pragma unroll
        for (int part = 0; part < parts; ++part) {
            const auto v = values[part];
            common::store_vec(destination + part * 1024 + tid * 4,
                              make_uint2(common::pack_f16x2(v.x * scale, v.y * scale),
                                         common::pack_f16x2(v.z * scale, v.w * scale)));
        }
        if (!tid)
            *reinterpret_cast<std::uint8_t*>(destination + Intermediate) =
                static_cast<std::uint8_t>(exponent);
        __syncthreads();
    }
}

template <bool Bf16>
__device__ __forceinline__ unsigned scale_pair(unsigned bits) {
    if constexpr (Bf16) {
        return common::fp8_e4m3x2_to_bf16x2_bits(bits);
    } else {
        unsigned value;
        asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(value) : "h"(static_cast<std::uint16_t>(bits)));
        return value;
    }
}

template <bool Bf16>
__device__ __forceinline__ unsigned weight_pair(std::uint8_t code, unsigned scale) {
    if constexpr (Bf16) {
        return common::nvfp4_pair_bf16_scaled(code, scale);
    } else {
        unsigned value, result;
        asm("{.reg .b8 p;mov.b32 {p,_,_,_},%1;cvt.rn.f16x2.e2m1x2 %0,p;}"
            : "=r"(value)
            : "r"(unsigned(code)));
        asm("mul.rn.f16x2 %0,%1,%2;" : "=r"(result) : "r"(value), "r"(scale));
        return result;
    }
}

template <class S>
__device__ __forceinline__ int weight_row(int row0, int row) {
    if constexpr (S::gate)
        return row0 + (row < S::rows / 2 ? row : row - S::rows / 2 + S::intermediate);
    return row0 + row;
}

template <class S>
__device__ __forceinline__ int code_offset(int row, int byte) {
    const int offset = row * (S::k_step / 2) + byte;
    if constexpr (S::register_pipeline) {
        static_assert(S::k_step == 128);
        // Permute aligned 16-byte chunks within each 128-byte row pair so
        // lanes caching different rows do not all hit the same banks.
        return offset ^ ((row & 7) * 16);
    }
    return offset;
}

template <class S, bool Compressed, bool CopyCodes = true, bool CopyInputs = true>
__device__ __forceinline__ void stage(Tile<S>& tile, int slot, int kt, int row0,
                                      common::Nvfp4MultiplierMatrix weight, const __nv_bfloat16* x,
                                      A16Product product, int active_columns) {
    const int tid = threadIdx.x, k0 = kt * S::k_step;
    if constexpr (CopyCodes) {
        for (int v = tid; v < S::rows * (S::k_step / 32); v += S::threads) {
            const int row = v / (S::k_step / 32), k = v % (S::k_step / 32) * 32;
            common::cp_async<16, Cache::cg>(
                tile.codes[slot] + code_offset<S>(row, k / 2),
                weight.codes + std::int64_t(weight_row<S>(row0, row)) * (S::k / 2) + (k0 + k) / 2);
        }
        if constexpr (Compressed) {
            const int warp = tid / 32;
            for (int b = warp; b < (S::rows / 32) * (S::k_step / 32); b += S::threads / 32) {
                const int row   = b / (S::k_step / 32) * 32;
                const int group = b % (S::k_step / 32) * 2;
                common::csf32_stage_block<S::k_step, S::gate ? 2 * S::intermediate : kHidden,
                                          S::k / 16>(weight.csf, weight_row<S>(row0, row),
                                                     k0 / 16 + group, tile.scales[slot], row, group);
            }
        } else if constexpr (!S::register_pipeline) {
            for (int v = tid; v < S::rows * (S::k_step / 64); v += S::threads) {
                const int row = v / (S::k_step / 64), group = v % (S::k_step / 64) * 4;
                common::cp_async<4>(
                    tile.scales[slot] + common::nvfp4_scale_byte_offset(row, group, S::k_step),
                    weight.scales + common::nvfp4_scale_byte_offset(weight_row<S>(row0, row),
                                                                    k0 / 16 + group, S::k));
            }
        }
    }
    if constexpr (CopyInputs) {
        for (int v = tid; v < active_columns * (S::k_step / 8); v += S::threads) {
            const int column = v / (S::k_step / 8), k = v % (S::k_step / 8) * 8;
            const int source      = tile.sources[column];
            const int destination = column * S::k_step + common::swizzle_xor_64(column, k);
            if constexpr (S::gate) {
                common::cp_async_zfill<16>(tile.inputs[slot][0] + destination,
                                           x + std::int64_t(source >= 0 ? source : 0) * S::k + k0 + k,
                                           source >= 0 ? 16 : 0);
            } else {
                const auto offset =
                    std::int64_t(source >= 0 ? source : 0) * S::k * (S::fp16_product ? 2 : 1) + k0 + k;
                common::cp_async_zfill<16>(tile.inputs[slot][0] + destination, product.values + offset,
                                           source >= 0 ? 16 : 0);
                if constexpr (!S::fp16_product)
                    common::cp_async_zfill<16>(tile.inputs[slot][1] + destination,
                                               product.residual + offset, source >= 0 ? 16 : 0);
            }
        }
    }
    common::cp_commit();
}

template <class S, bool Compressed>
__device__ __forceinline__ void
contract(Tile<S>& tile, int row0, common::Nvfp4MultiplierMatrix weight, const __nv_bfloat16* x,
         A16Product product, int valid_columns, float (&acc)[2][S::mma_tokens][4]) {
    const int lane = threadIdx.x % 32, wm = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < S::stages; ++i) {
        if constexpr (S::register_pipeline) {
            stage<S, Compressed, true, false>(tile, i, i, row0, weight, x, product,
                                               (valid_columns + 7) / 8 * 8);
            stage<S, Compressed, false, true>(tile, i, i, row0, weight, x, product,
                                               (valid_columns + 7) / 8 * 8);
        } else {
            stage<S, Compressed>(tile, i, i, row0, weight, x, product,
                                 (valid_columns + 7) / 8 * 8);
        }
    }
#pragma unroll 1
    for (int kt = 0; kt < S::k / S::k_step; ++kt) {
        const int slot = kt % S::stages;
        const int owned_row = S::gate ? wm * 16 + (lane & 15) + (lane >> 4) * (S::rows / 2)
                                      : wm * 32 + (lane & 15) + (lane >> 4) * 16;
        unsigned scale_words[S::k_step / 64];
        if constexpr (S::register_pipeline) {
#pragma unroll
            for (int word = 0; word < S::k_step / 64; ++word)
                scale_words[word] = common::load_vec<unsigned>(
                    weight.scales + common::nvfp4_scale_byte_offset(
                                        weight_row<S>(row0, owned_row),
                                        kt * (S::k_step / 16) + word * 4, S::k));
        }
        // The register pipeline commits codes and inputs as separate groups.
        common::cp_wait<(S::register_pipeline ? 2 : 1) * (S::stages - 1)>();
        __syncthreads();
        unsigned code_words[S::k_step / 8];
        const int next = kt + S::stages;
        if constexpr (S::register_pipeline) {
#pragma unroll
            for (int part = 0; part < S::k_step / 32; ++part) {
                const auto v = common::load_vec<uint4>(tile.codes[slot] +
                                                       code_offset<S>(owned_row, part * 16));
                code_words[part * 4]     = v.x;
                code_words[part * 4 + 1] = v.y;
                code_words[part * 4 + 2] = v.z;
                code_words[part * 4 + 3] = v.w;
            }
            // All lanes own their code row now; reuse its shared slot while
            // contracting. The copy wrappers do not carry a memory clobber.
            __syncthreads();
            asm volatile("" ::: "memory");
            if (next < S::k / S::k_step)
                stage<S, Compressed, true, false>(tile, slot, next, row0, weight, x, product,
                                                   (valid_columns + 7) / 8 * 8);
            else
                common::cp_commit();
        }
#pragma unroll
        for (int k = 0; k < S::k_step; k += 16) {
            unsigned a[2][4];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int base   = S::gate ? wm * 16 + m * (S::rows / 2) : wm * 32 + m * 16;
                const int row    = base + (lane >> 2);
                const int column = k + (lane & 3) * 2;
                const auto scale = [&](int half) {
                    if constexpr (S::register_pipeline)
                        return static_cast<std::uint8_t>(
                            __shfl_sync(0xffffffff, scale_words[k / 64],
                                        m * 16 + half * 8 + (lane >> 2)) >>
                            (8 * ((k / 16) & 3)));
                    else
                        return tile.scales[slot][common::nvfp4_scale_byte_offset(
                            row + half * 8, k / 16, S::k_step)];
                };
                const auto low_scale  = scale(0);
                const auto high_scale = scale(1);
                const unsigned low_factor =
                    scale_pair<S::bf16_weights>(unsigned(low_scale) * 0x101U);
                const unsigned high_factor =
                    scale_pair<S::bf16_weights>(unsigned(high_scale) * 0x101U);
                if constexpr (S::register_pipeline) {
                    const auto code = [&](int half, int part) {
                        return static_cast<std::uint8_t>(
                            __shfl_sync(0xffffffff, code_words[k / 8 + part],
                                        m * 16 + half * 8 + (lane >> 2)) >>
                            (8 * (lane & 3)));
                    };
                    a[m][0] = weight_pair<S::bf16_weights>(code(0, 0), low_factor);
                    a[m][1] = weight_pair<S::bf16_weights>(code(1, 0), high_factor);
                    a[m][2] = weight_pair<S::bf16_weights>(code(0, 1), low_factor);
                    a[m][3] = weight_pair<S::bf16_weights>(code(1, 1), high_factor);
                } else {
                    const auto* low  = tile.codes[slot] + row * (S::k_step / 2) + column / 2;
                    const auto* high = tile.codes[slot] + (row + 8) * (S::k_step / 2) + column / 2;
                    a[m][0] = weight_pair<S::bf16_weights>(low[0], low_factor);
                    a[m][1] = weight_pair<S::bf16_weights>(high[0], high_factor);
                    a[m][2] = weight_pair<S::bf16_weights>(low[4], low_factor);
                    a[m][3] = weight_pair<S::bf16_weights>(high[4], high_factor);
                }
            }
            // High and low input contractions share one FP32 accumulator.
#pragma unroll
            for (int plane = 0; plane < S::planes; ++plane)
#pragma unroll
                for (int n = 0; n < S::mma_tokens; ++n) {
                    if (n * 8 >= valid_columns) continue;
                    const int row    = n * 8 + (lane & 7);
                    const int column = k + ((lane >> 3) & 1) * 8;
                    unsigned b[2];
                    common::ldmatrix_x2(b[0], b[1],
                                        common::smem_addr(tile.inputs[slot][plane] +
                                                          row * S::k_step +
                                                          common::swizzle_xor_64(row, column)));
#pragma unroll
                    for (int m = 0; m < 2; ++m) {
                        if constexpr (S::bf16_weights)
                            common::mma_bf16(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                                             a[m][0], a[m][1], a[m][2], a[m][3], b[0], b[1]);
                        else
                            common::mma_f16(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                                            a[m][0], a[m][1], a[m][2], a[m][3], b[0], b[1]);
                    }
                }
        }
        __syncthreads();
        if (next < S::k / S::k_step) {
            if constexpr (S::register_pipeline)
                stage<S, Compressed, false, true>(tile, slot, next, row0, weight, x, product,
                                                   (valid_columns + 7) / 8 * 8);
            else
                stage<S, Compressed>(tile, slot, next, row0, weight, x, product,
                                     (valid_columns + 7) / 8 * 8);
        }
        else
            common::cp_commit();
    }
    common::cp_wait<0>();
}

template <int Intermediate, bool Gate, int Rows, int Columns, int KStep, int Stages,
          bool Compressed, bool Fp16 = false, bool RegisterPipeline = false>
__global__ __launch_bounds__(
    Schedule<Intermediate, Gate, Rows, Columns, KStep, Stages, Fp16>::threads,
    Fp16&& Rows == 128 && Columns == 64 && KStep == 128 && Stages == 1
        ? 3
        : 0) void grouped(const __nv_bfloat16* __restrict__ x, ExpertMatrixDeviceView bank,
                          Groups groups, A16Product product, float* __restrict__ contributions,
                          const float* __restrict__ coefficients, float* __restrict__ sum,
                          int job_columns, float limit, int minimum_load, int maximum_load,
                          int row_window, Nvfp4A16Dispatch dispatch) {
    using S = Schedule<Intermediate, Gate, Rows, Columns, KStep, Stages, Fp16, RegisterPipeline>;
    static_assert(Rows == 64 || Rows == 128);
    static_assert(KStep == 64 || KStep == 128);
    static_assert(S::k % KStep == 0 && S::k / KStep >= Stages);
    const int cohort = Fp16 ? dispatch.cohort_load(groups.load_range[0], groups.load_range[1]) : 0;
    if (cohort) {
        if (cohort <= minimum_load || cohort > maximum_load) return;
    } else if (groups.load_range &&
               (groups.load_range[1] <= minimum_load || groups.load_range[0] > maximum_load)) {
        return;
    }
    const int subtiles      = (min(job_columns, maximum_load) + Columns - 1) / Columns;
    constexpr int row_tiles = (Gate ? Intermediate : kHidden) / S::feature_rows;
    auto& tile     = *reinterpret_cast<Tile<S>*>(common::shared_storage<sizeof(Tile<S>)>());
    const int lane = threadIdx.x % 32, wm = threadIdx.x / 32;
    const int jobs    = *groups.jobs;
    const bool full_rows = dispatch.full_rows_when_mixed &&
                           groups.load_range[1] - groups.load_range[0] > 1;
    const int window = Gate || full_rows || row_window <= 0 ? row_tiles : min(row_tiles, row_window);
    const int windows = (row_tiles + window - 1) / window;
    const int tasks   = jobs * windows * window * subtiles;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int row_tile = task % window + (task / (window * subtiles * jobs)) * window;
        const int job      = task / (window * subtiles) % jobs;
        const int subtile  = task / window % subtiles;
        if (row_tile >= row_tiles) continue;
        const int expert = groups.job_experts[job];
        const int start = groups.offsets[expert], end = groups.offsets[expert + 1];
        const int load = cohort ? cohort : end - start;
        if (load <= minimum_load || load > maximum_load) continue;
        const int begin = start + groups.job_columns[job] + subtile * Columns;
        if (begin >= end) continue;
        const auto weight = common::nvfp4_mul_expert_matrix(bank, expert);
        if (!weight.valid) continue;
        for (int column = threadIdx.x; column < Columns; column += S::threads) {
            const int packed = begin + column;
            tile.sources[column] =
                packed < end ? (Gate ? groups.assignments[packed] / kTopK : packed) : -1;
        }
        __syncthreads();
        const int row0 = row_tile * S::feature_rows;
        float acc[2][S::mma_tokens][4]{};
        contract<S, Compressed>(tile, row0, weight, x, product, min(Columns, end - begin), acc);
#pragma unroll
        for (int n = 0; n < S::mma_tokens; ++n) {
            const int column = begin + n * 8 + (lane & 3) * 2;
            if constexpr (Gate) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    if (column + t >= end) continue;
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int r      = row0 + wm * 16 + (lane >> 2) + half * 8;
                        const float gate = acc[0][n][t + 2 * half] * weight.multiplier;
                        const float up   = acc[1][n][t + 2 * half] * weight.multiplier;
                        const float value =
                            common::silu(fminf(gate, limit)) * fmaxf(-limit, fminf(up, limit));
                        store_product<Fp16>(product, std::int64_t(column + t) * Intermediate + r,
                                            value);
                    }
                }
            } else {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    if (column + t >= end) continue;
                    const int assignment = groups.assignments[column + t];
                    const float inverse  = product_inverse<Intermediate, Fp16>(product, column + t);
                    const float factor = weight.multiplier * (sum ? coefficients[assignment] : 1.F);
                    float* destination = sum ? sum + std::int64_t(assignment / kTopK) * kHidden
                                             : contributions + std::int64_t(column + t) * kHidden;
#pragma unroll
                    for (int m = 0; m < 2; ++m)
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const int r       = row0 + wm * 32 + m * 16 + (lane >> 2) + half * 8;
                            const float value = (acc[m][n][t + half * 2] * inverse) * factor;
                            if (sum)
                                atomicAdd(destination + r, value);
                            else
                                destination[r] = value;
                        }
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void decode_group(common::Nvfp4MultiplierMatrix matrix, int row,
                                             int group, int k, float (&values)[16]) {
    const auto decoded =
        common::nvfp4_dequant_f16x16(matrix.codes + std::int64_t(row) * (k / 2) + group * 8,
                                     matrix.scales[common::nvfp4_scale_byte_offset(row, group, k)]);
    const unsigned words[8]{
        static_cast<unsigned>(decoded.lo.x), static_cast<unsigned>(decoded.lo.y),
        static_cast<unsigned>(decoded.lo.z), static_cast<unsigned>(decoded.lo.w),
        static_cast<unsigned>(decoded.hi.x), static_cast<unsigned>(decoded.hi.y),
        static_cast<unsigned>(decoded.hi.z), static_cast<unsigned>(decoded.hi.w)};
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const auto value     = __half22float2(*reinterpret_cast<const __half2*>(&words[pair]));
        values[pair * 2]     = value.x;
        values[pair * 2 + 1] = value.y;
    }
}

template <int Intermediate, int MaxColumns, bool Fp16 = false>
__global__ __launch_bounds__(128) void vector_gate(const __nv_bfloat16* __restrict__ x,
                                                   ExpertMatrixDeviceView bank, Groups groups,
                                                   A16Product product, float limit,
                                                   int maximum_load, Nvfp4A16Dispatch dispatch) {
    const int cohort = Fp16 ? dispatch.cohort_load(groups.load_range[0], groups.load_range[1]) : 0;
    if (cohort > maximum_load) return;
    if (groups.load_range && groups.load_range[0] > maximum_load) return;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int jobs = *groups.jobs;
    for (int task = blockIdx.x; task < jobs * (Intermediate / 4); task += gridDim.x) {
        const int job = task / (Intermediate / 4), row = task % (Intermediate / 4) * 4 + warp;
        const int expert = groups.job_experts[job];
        const int begin = groups.offsets[expert], end = groups.offsets[expert + 1];
        if (end - begin > maximum_load) continue;
        const auto weight = common::nvfp4_mul_expert_matrix(bank, expert);
        if (!weight.valid) continue;
        float gate[MaxColumns]{}, up[MaxColumns]{};
        for (int group = lane; group < kHidden / 16; group += 32) {
            float g[16], u[16];
            decode_group(weight, row, group, kHidden, g);
            decode_group(weight, row + Intermediate, group, kHidden, u);
#pragma unroll
            for (int col = 0; col < MaxColumns; ++col) {
                if (begin + col >= end) continue;
                const int token = groups.assignments[begin + col] / kTopK;
                const auto first =
                    common::load_vec<uint4>(x + std::int64_t(token) * kHidden + group * 16);
                const auto second =
                    common::load_vec<uint4>(x + std::int64_t(token) * kHidden + group * 16 + 8);
                const unsigned words[8]{first.x,  first.y,  first.z,  first.w,
                                        second.x, second.y, second.z, second.w};
#pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const auto value = common::bf16x2_bits_to_float2(words[pair]);
                    gate[col]        = fmaf(g[pair * 2], value.x, gate[col]);
                    gate[col]        = fmaf(g[pair * 2 + 1], value.y, gate[col]);
                    up[col]          = fmaf(u[pair * 2], value.x, up[col]);
                    up[col]          = fmaf(u[pair * 2 + 1], value.y, up[col]);
                }
            }
        }
#pragma unroll
        for (int col = 0; col < MaxColumns; ++col) {
            const float g = common::warp_reduce_sum(gate[col]) * weight.multiplier;
            const float u = common::warp_reduce_sum(up[col]) * weight.multiplier;
            if (!lane && begin + col < end)
                store_product<Fp16>(product, std::int64_t(begin + col) * Intermediate + row,
                                    common::silu(fminf(g, limit)) * fmaxf(-limit, fminf(u, limit)));
        }
    }
}

template <int Intermediate, int MaxColumns, bool Fp16 = false>
__global__ __launch_bounds__(128) void vector_down(ExpertMatrixDeviceView bank, Groups groups,
                                                   A16Product product,
                                                   float* __restrict__ contributions,
                                                   const float* __restrict__ coefficients,
                                                   float* __restrict__ sum, int maximum_load,
                                                   Nvfp4A16Dispatch dispatch) {
    const int cohort = Fp16 ? dispatch.cohort_load(groups.load_range[0], groups.load_range[1]) : 0;
    if (cohort > maximum_load) return;
    if (groups.load_range && groups.load_range[0] > maximum_load) return;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int jobs = *groups.jobs;
    for (int task = blockIdx.x; task < jobs * (kHidden / 8); task += gridDim.x) {
        const int job = task / (kHidden / 8), flat = task % (kHidden / 8) * 8 + warp * 2;
        const int expert = groups.job_experts[job];
        const int begin = groups.offsets[expert], end = groups.offsets[expert + 1];
        if (end - begin > maximum_load) continue;
        const auto weight = common::nvfp4_mul_expert_matrix(bank, expert);
        if (!weight.valid) continue;
        int rows[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int index = flat + r;
            rows[r]         = index / 128 * 128 + ((index & 127) >> 2) + (index & 3) * 32;
        }
        float accum[2][MaxColumns]{};
        for (int group = lane; group < Intermediate / 16; group += 32) {
            int2 codes[2];
            unsigned scales[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const auto* address =
                    weight.codes + std::int64_t(rows[r]) * (Intermediate / 2) + group * 8;
                asm("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                    : "=r"(codes[r].x), "=r"(codes[r].y)
                    : "l"(address));
                const unsigned scale =
                    weight.scales[common::nvfp4_scale_byte_offset(rows[r], group, Intermediate)];
                scales[r] = scale_pair<true>(scale | (scale << 8));
            }
            unsigned high[MaxColumns][Fp16 ? 16 : 8], low[Fp16 ? 1 : MaxColumns][8];
#pragma unroll
            for (int col = 0; col < MaxColumns; ++col) {
                if (begin + col >= end) continue;
                const auto offset = std::int64_t(begin + col) * Intermediate + group * 16;
                if constexpr (Fp16) {
                    // Short expert groups keep the original FP32 product. SIMT
                    // needs no FP16 staging or separate normalization pass.
#pragma unroll
                    for (int part = 0; part < 4; ++part) {
                        const auto v = common::load_vec<uint4>(
                            reinterpret_cast<const float*>(product.values) + offset + part * 4);
                        high[col][part * 4]     = v.x;
                        high[col][part * 4 + 1] = v.y;
                        high[col][part * 4 + 2] = v.z;
                        high[col][part * 4 + 3] = v.w;
                    }
                } else {
                    const auto a = common::load_vec<uint4>(product.values + offset);
                    const auto b = common::load_vec<uint4>(product.values + offset + 8);
                    high[col][0] = a.x;
                    high[col][1] = a.y;
                    high[col][2] = a.z;
                    high[col][3] = a.w;
                    high[col][4] = b.x;
                    high[col][5] = b.y;
                    high[col][6] = b.z;
                    high[col][7] = b.w;
                    const auto c = common::load_vec<uint4>(product.residual + offset);
                    const auto d = common::load_vec<uint4>(product.residual + offset + 8);
                    low[col][0]  = c.x;
                    low[col][1]  = c.y;
                    low[col][2]  = c.z;
                    low[col][3]  = c.w;
                    low[col][4]  = d.x;
                    low[col][5]  = d.y;
                    low[col][6]  = d.z;
                    low[col][7]  = d.w;
                }
            }
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                unsigned values[2];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const unsigned word = pair < 4 ? codes[r].x : codes[r].y;
                    const auto code     = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                    values[r]           = weight_pair<true>(code, scales[r]);
                }
#pragma unroll
                for (int col = 0; col < MaxColumns; ++col) {
                    if (begin + col >= end) continue;
                    if constexpr (Fp16) {
                        const float x = __uint_as_float(high[col][2 * pair]);
                        const float y = __uint_as_float(high[col][2 * pair + 1]);
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const auto w = common::bf16x2_bits_to_float2(values[r]);
                            accum[r][col] = fmaf(w.x, x, accum[r][col]);
                            accum[r][col] = fmaf(w.y, y, accum[r][col]);
                        }
                    } else {
                        const auto h = common::bf16x2_bits_to_float2(high[col][pair]);
                        const auto l = common::bf16x2_bits_to_float2(low[col][pair]);
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const auto w  = common::bf16x2_bits_to_float2(values[r]);
                            accum[r][col] = fmaf(w.x, h.x + l.x, accum[r][col]);
                            accum[r][col] = fmaf(w.y, h.y + l.y, accum[r][col]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int col = 0; col < MaxColumns; ++col) {
                const float dot = common::warp_reduce_sum(accum[r][col]);
                if (!lane && begin + col < end) {
                    const float value = dot * weight.multiplier;
                    if (sum) {
                        const int assignment = groups.assignments[begin + col];
                        atomicAdd(sum + std::int64_t(assignment / kTopK) * kHidden + rows[r],
                                  value * coefficients[assignment]);
                    } else {
                        contributions[std::int64_t(begin + col) * kHidden + rows[r]] = value;
                    }
                }
            }
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::a16
