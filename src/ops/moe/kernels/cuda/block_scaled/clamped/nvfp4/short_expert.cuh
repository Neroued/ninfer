#pragma once
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/compute.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::short_expert {
// E2M1 times a finite nonnegative E4M3 scale is exact in FP16. Native
// half-input, FP32-accumulator FMA avoids scalar widening without changing
// the represented operands or the FP32 matrix/calibration factors.
__device__ __forceinline__ float dot_pair(float accum, unsigned w, unsigned x) {
    asm("{.reg .b16 ax,ay,bx,by;mov.b32 {ax,ay},%1;mov.b32 {bx,by},%2;"
        "fma.rn.f32.f16 %0,ax,bx,%0;fma.rn.f32.f16 %0,ay,by,%0;}"
        : "+f"(accum)
        : "r"(w), "r"(x));
    return accum;
}

__device__ __forceinline__ unsigned scales_f16_pair(unsigned packed) {
    unsigned result;
    asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(result) : "h"(static_cast<std::uint16_t>(packed)));
    return result;
}

__device__ __forceinline__ unsigned pair_low(std::uint8_t code, unsigned scales) {
    unsigned value, result;
    asm("{.reg .b8 p;mov.b32 {p,_,_,_},%1;cvt.rn.f16x2.e2m1x2 %0,p;}"
        : "=r"(value)
        : "r"(unsigned(code)));
    asm("{.reg .b16 x,y;.reg .b32 s;mov.b32 {x,y},%2;mov.b32 s,{x,x};mul.rn.f16x2 %0,%1,s;}"
        : "=r"(result)
        : "r"(value), "r"(scales));
    return result;
}

__device__ __forceinline__ unsigned pair_high(std::uint8_t code, unsigned scales) {
    unsigned value, result;
    asm("{.reg .b8 p;mov.b32 {p,_,_,_},%1;cvt.rn.f16x2.e2m1x2 %0,p;}"
        : "=r"(value)
        : "r"(unsigned(code)));
    asm("{.reg .b16 x,y;.reg .b32 s;mov.b32 {x,y},%2;mov.b32 s,{y,y};mul.rn.f16x2 %0,%1,s;}"
        : "=r"(result)
        : "r"(value), "r"(scales));
    return result;
}

template <int I, bool Gate, int Columns>
struct Shape {
    static constexpr int k = Gate ? kHidden : I, features = Gate ? 32 : 64, groups = k / 16;
    static constexpr int threads = 256, warps = 8, columns = Columns, pages = Gate ? 4 : 1,
                         capacity = 256;
};

template <class S>
struct Tile {
    struct alignas(16) Encoded {
        std::uint8_t nibbles[4][32];
        unsigned bounds[4];
    };

    // Encoded metadata and expanded scales share storage. restore() consumes
    // batches backwards, loading every overlapping source before writing it.
    union alignas(16) Data {
        Encoded encoded[S::k / 64];
        unsigned scales[(S::k / 64) * 64];
    } data;

    // Exception pages stay compressed. Pages above this bounded cache use
    // the original records, including dense real-weight exception tails.
    alignas(16) std::uint16_t exceptions[S::pages][S::capacity + 8];
    common::Csf32RowBases128 bases;
    unsigned starts[S::pages];
    int lengths[S::pages];
};

__device__ __forceinline__ unsigned scale_word(unsigned row, unsigned group4) {
    // Consecutive rows are conflict-free during decode; the XOR also spreads
    // the two-K16-group word loads of each consumer warp over shared banks.
    return group4 * 64 + (row ^ ((group4 & 15) * 2));
}

template <class S>
struct Destination {
    Tile<S>& tile;
    unsigned row, group;

    __device__ __forceinline__ void store_exception(unsigned lane, unsigned col,
                                                    std::uint8_t value) const {
        const unsigned g = group + col;
        reinterpret_cast<std::uint8_t*>(
            tile.data.scales)[scale_word(row + lane, g / 4) * 4 + g % 4] = value;
    }
};

template <int I, bool Gate, class S>
__device__ __forceinline__ void restore(Tile<S>& tile, common::Nvfp4MultiplierMatrix weight,
                                        int row0) {
    const auto& source = weight.csf;
    const auto rows    = make_uint4(row0, row0 + (Gate ? I : 32), 0, 0);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    if (warp == 0) common::csf32_copy_row_bases_async<2>(source, rows, tile.bases);
    for (int item = tid; item < (S::k / 64) * 8; item += S::threads) {
        const int group = item / 8, index = item / 2 % 4, half = item & 1;
        const auto b = common::csf32_detail::locate<Gate ? 2 * I : kHidden, S::groups>(
            source, common::csf32_detail::row_origin(rows, index / 2), group * 4 + (index & 1) * 2);
        common::cp_async<16>(tile.data.encoded[group].nibbles[index] + half * 16,
                             source.nibbles + b.nibbles + half * 16);
    }
    common::cp_commit();
    for (int item = tid; item < (S::k / 64) * 4; item += S::threads) {
        const int group = item / 4, index = item % 4;
        const auto b = common::csf32_detail::locate<Gate ? 2 * I : kHidden, S::groups>(
            source, common::csf32_detail::row_origin(rows, index / 2), group * 4 + (index & 1) * 2);
        tile.data.encoded[group].bounds[index] =
            unsigned(source.offsets[b.offset]) | (unsigned(source.offsets[b.offset + 1]) << 16);
    }
    if (tid < S::pages) {
        const auto b = common::csf32_detail::locate<Gate ? 2 * I : kHidden, S::groups>(
            source, Gate && tid >= 2 ? rows.y : rows.x, Gate ? (tid & 1) * 128 : 0);
        const int count   = source.offsets[(b.page + 1) * 257 - 1];
        tile.starts[tid]  = source.page_bases[b.page] / 2;
        tile.lengths[tid] = count <= S::capacity ? count : -1;
    }
    __syncthreads();
    constexpr int vectors = (S::capacity + 8) / 8;
    for (int item = tid; item < S::pages * vectors; item += S::threads) {
        const int page = item / vectors, segment = item % vectors, count = tile.lengths[page];
        const unsigned start = tile.starts[page];
        if (count > 0 && segment * 8 < count + (start & 7)) {
            const std::uint64_t off = std::uint64_t(start & ~7U) + segment * 8;
            common::cp_async_zfill<16>(
                tile.exceptions[page] + segment * 8, source.exceptions + off,
                int(min(std::uint64_t(16), (source.exception_count - off) * 2)));
        }
    }
    common::cp_commit();
    common::cp_wait<0>();
    __syncthreads();
    // Expand backwards in groups of four. Every warp loads its source fields
    // before a barrier permits destination writes to overlap the current batch.
    // Earlier source batches end below the current destination batch.
    for (int high = S::k / 64 - 1; high >= 0; high -= 4) {
        const int group4 = high - warp / 2, row_part = warp % 2, row = row_part * 32 + lane;
        const auto& e     = tile.data.encoded[group4];
        const unsigned lo = e.nibbles[row_part * 2][lane], hi = e.nibbles[row_part * 2 + 1][lane];
        const unsigned repeated = __byte_perm(lo | (hi << 8), 0, 0x1100);
        const unsigned decoded  = ((repeated & 0x000f000fU) | ((repeated >> 4) & 0x0f000f00U)) +
                                 unsigned(tile.bases.values[row_part][lane]) * 0x01010101U;
        const unsigned first = e.bounds[row_part * 2], second = e.bounds[row_part * 2 + 1];
        const unsigned begin0 = first & 65535, begin1 = second & 65535,
                       count0 = (first >> 16) - begin0, count1 = (second >> 16) - begin1;
        __syncthreads();
        tile.data.scales[scale_word(row, group4)] = decoded;
        if (count0 || count1) {
            __syncwarp();
            const int page       = Gate ? row_part * 2 + group4 / 32 : 0;
            const unsigned start = tile.starts[page], bias = start & 7;
            const bool cached = tile.lengths[page] >= 0;
            const Destination<S> d0{tile, unsigned(row_part * 32), unsigned(group4 * 4)};
            const Destination<S> d1{tile, unsigned(row_part * 32), unsigned(group4 * 4 + 2)};
            for (unsigned offset = 0; offset < max(count0, count1); offset += 32) {
                const unsigned index = offset + lane;
                unsigned v0 = 0, v1 = 0;
                if (index < count0)
                    v0 = cached ? tile.exceptions[page][bias + begin0 + index]
                                : source.exceptions[std::uint64_t(start) + begin0 + index];
                if (index < count1)
                    v1 = cached ? tile.exceptions[page][bias + begin1 + index]
                                : source.exceptions[std::uint64_t(start) + begin1 + index];
                if (index < count0) d0.store_exception((v0 & 255) / 2, v0 & 1, v0 >> 8);
                if (index < count1) d1.store_exception((v1 & 255) / 2, v1 & 1, v1 >> 8);
            }
        }
    }
    __syncthreads();
}

template <int I, bool Gate, int Columns = 2>
__global__ __launch_bounds__(256, Columns == 1 ? 4 : 3) void contract(
    const std::uint8_t* __restrict__ input, const std::uint8_t* __restrict__ input_scales,
    ExpertMatrixDeviceView bank, ActivationScale input_scale, ActivationScale product_scale,
    Groups groups, std::uint8_t* product, std::uint8_t* product_scales,
    const float* __restrict__ coefficients, float* __restrict__ sum, float limit,
    int pool_min_experts, int pool_expert_limit, bool grouped_input = false) {
    using S = Shape<I, Gate, Columns>;
    static_assert((Columns == 1 || Columns == 2) && S::k % 256 == 0);
    // Pool down owns every selected expert, including any short tails.
    if constexpr (!Gate)
        if (pool_expert_limit && groups.load_range[2] > pool_min_experts &&
            groups.load_range[2] <= pool_expert_limit)
            return;
    const int population_load = grouped_input ? balanced_group_load(groups) : 0;
    if (groups.load_range && (population_load ? population_load : groups.load_range[0]) > Columns)
        return;
    __shared__ Tile<S> tile;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    constexpr int row_tiles = (Gate ? I : kHidden) / S::features;
    const int jobs          = *groups.jobs;
    for (int task = blockIdx.x; task < jobs * row_tiles; task += gridDim.x) {
        const int job = task / row_tiles, row0 = task % row_tiles * S::features;
        const int expert = groups.job_experts[job], begin = groups.offsets[expert],
                  end = groups.offsets[expert + 1];
        if ((population_load ? population_load : end - begin) > Columns || end == begin) continue;
        const auto weight = common::nvfp4_mul_expert_matrix<true>(bank, expert);
        if (!weight.valid) continue;
        restore<I, Gate, S>(tile, weight, row0);
        // Consume compressed activations directly through L1. A separate
        // decoded shared tile costs capacity and conflicts for short experts.
        int sources[Columns];
#pragma unroll
        for (int c = 0; c < Columns; ++c)
            sources[c] = begin + c < end
                             ? (Gate && !grouped_input ? groups.assignments[begin + c] : begin + c)
                             : 0;
        const float factor = weight.multiplier * input_scale.get(expert);
        uint4 carry[Gate ? 2 : 1];
#pragma unroll 1
        for (int r = 0; r < S::features / S::warps; ++r) {
            const int row = warp + r * S::warps;
            if (Gate || r == 0) {
#pragma unroll
                for (int part = 0; part < (Gate ? 2 : 1); ++part) {
                    const auto* first =
                        weight.codes + std::int64_t(row0 + row + part * I) * (S::k / 2) + lane * 16;
                    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
                                 : "=r"(carry[part].x), "=r"(carry[part].y), "=r"(carry[part].z),
                                   "=r"(carry[part].w)
                                 : "l"(first)
                                 : "memory");
                }
            }
            float acc[Gate ? 2 : 1][Columns]{};
            for (int group = lane * 2; group < S::groups; group += 64) {
                uint4 next_codes[Gate ? 2 : 1], xcodes[Columns];
                const bool within    = group + 64 < S::groups;
                const int next_row   = within ? row : row + S::warps,
                          next_group = within ? group + 64 : lane * 2;
                const bool following = within || (!Gate && next_row < S::features);
                unsigned scales[Gate ? 2 : 1], xscales[Columns];
#pragma unroll
                for (int part = 0; part < (Gate ? 2 : 1); ++part) {
                    const int local = row + part * 32;
                    if (following) {
                        const auto* next = weight.codes +
                                           std::int64_t(row0 + next_row + part * I) * (S::k / 2) +
                                           next_group * 8;
                        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
                                     : "=r"(next_codes[part].x), "=r"(next_codes[part].y),
                                       "=r"(next_codes[part].z), "=r"(next_codes[part].w)
                                     : "l"(next)
                                     : "memory");
                    }
                    const unsigned scale = reinterpret_cast<const std::uint16_t*>(
                        tile.data.scales)[scale_word(local, group / 4) * 2 + (group % 4) / 2];
                    scales[part] = scales_f16_pair(scale);
                }
#pragma unroll
                for (int col = 0; col < Columns; ++col) {
                    if (begin + col >= end) continue;
                    const auto* addr = input + std::int64_t(sources[col]) * (S::k / 2) + group * 8;
                    asm("ld.global.ca.v4.u32 {%0,%1,%2,%3},[%4];"
                        : "=r"(xcodes[col].x), "=r"(xcodes[col].y), "=r"(xcodes[col].z),
                          "=r"(xcodes[col].w)
                        : "l"(addr));
                    xscales[col] = scales_f16_pair(*reinterpret_cast<const std::uint16_t*>(
                        input_scales + std::int64_t(sources[col]) * S::groups + group));
                }
#pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    unsigned w[Gate ? 2 : 1];
#pragma unroll
                    for (int part = 0; part < (Gate ? 2 : 1); ++part) {
                        const unsigned word = pair < 4    ? carry[part].x
                                              : pair < 8  ? carry[part].y
                                              : pair < 12 ? carry[part].z
                                                          : carry[part].w;
                        w[part] =
                            pair < 8
                                ? pair_low(std::uint8_t(word >> (8 * (pair & 3))), scales[part])
                                : pair_high(std::uint8_t(word >> (8 * (pair & 3))), scales[part]);
                    }
#pragma unroll
                    for (int col = 0; col < Columns; ++col) {
                        if (begin + col >= end) continue;
                        const unsigned word = pair < 4    ? xcodes[col].x
                                              : pair < 8  ? xcodes[col].y
                                              : pair < 12 ? xcodes[col].z
                                                          : xcodes[col].w;
                        const unsigned x =
                            pair < 8
                                ? pair_low(std::uint8_t(word >> (8 * (pair & 3))), xscales[col])
                                : pair_high(std::uint8_t(word >> (8 * (pair & 3))), xscales[col]);
#pragma unroll
                        for (int part = 0; part < (Gate ? 2 : 1); ++part)
                            acc[part][col] = dot_pair(acc[part][col], w[part], x);
                    }
                }
                if (following) {
#pragma unroll
                    for (int part = 0; part < (Gate ? 2 : 1); ++part)
                        carry[part] = next_codes[part];
                }
            }
            // Finish every lane's scale reads before lane 0 reuses the storage for products.
            if constexpr (Gate) __syncwarp();
#pragma unroll
            for (int col = 0; col < Columns; ++col) {
                if (begin + col >= end) continue;
                const float first = common::warp_reduce_sum(acc[0][col]) * factor;
                if constexpr (Gate) {
                    const float second = common::warp_reduce_sum(acc[1][col]) * factor;
                    // This row's contraction is complete: its scale slots can
                    // hold FP32 products until all rows reach the quantizer.
                    if (!lane)
                        reinterpret_cast<float*>(tile.data.scales)[scale_word(row, col)] =
                            common::silu(fminf(first, limit)) * fmaxf(-limit, fminf(second, limit));
                } else if (!lane) {
                    const int assignment = groups.assignments[begin + col];
                    atomicAdd(sum + std::int64_t(assignment / kTopK) * kHidden + row0 + row,
                              first * coefficients[assignment]);
                }
            }
        }
        if constexpr (Gate) {
            __syncthreads();
            if (warp < 2) {
                const int col = lane & 3, feature = warp * 16 + (lane >> 2);
                const bool valid = col < Columns && begin + col < end;
                const float lo =
                    valid
                        ? reinterpret_cast<const float*>(tile.data.scales)[scale_word(feature, col)]
                        : 0;
                const float hi = valid ? reinterpret_cast<const float*>(
                                             tile.data.scales)[scale_word(feature + 8, col)]
                                       : 0;
                store_product<I, true>(lo, hi, product_scale.get(expert), product, product_scales,
                                       valid ? begin + col : -1, row0 + warp * 16);
            }
        }
        __syncthreads();
    }
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::short_expert
