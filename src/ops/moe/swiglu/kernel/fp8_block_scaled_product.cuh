#pragma once

#include "ops/linear/kernels/fp8_block/a16_tma.cuh"
#include "ops/linear/swiglu/product_epilogue.cuh"

namespace ninfer::ops::detail::swiglu_ffn {

__device__ __forceinline__ unsigned scaled_product_half2_bits(__half2 value) {
    const __half2_raw raw = value;
    return unsigned(raw.x) | (unsigned(raw.y) << 16);
}

// Private block-floating FP16 product; per-column groups retain FP32 range.
struct ScaledF16Output {
    __half* values;
    float* scales;
};

template <int Group>
__device__ __forceinline__ void store_scaled_f16_product(ScaledF16Output output, float4 v, int row,
                                                         int token, bool valid) {
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int d = Group / 8; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d, Group / 4));
    if (m == 0) m = 1;
    const bool normal = m >= 1e-30F && m <= 1e30F;
    const auto divide = [&](float value) {
        return normal ? __fdividef(value, m) : __fdiv_rn(value, m);
    };
    const auto h0 = __floats2half2_rn(divide(v.x), divide(v.y)),
               h1 = __floats2half2_rn(divide(v.z), divide(v.w));
    if (valid) {
        common::store_vec(output.values + std::int64_t(token) * 1024 + row,
                          make_uint2(scaled_product_half2_bits(h0), scaled_product_half2_bits(h1)));
        if (row % Group == 0) output.scales[std::int64_t(token) * (1024 / Group) + row / Group] = m;
    }
}

// The sliced-K contraction has already reduced both branches into its FP32 tile.
// One four-lane group normalizes sixteen products before any observable rounding.
struct ScaledF16SlicedProductEpilogue {
    static constexpr bool kSwizzleRowStorage = true;
    ClampedSwiGluProduct product;
    int physical_tokens;

    template <class C, class Tile>
    __device__ void finish_rows(ScaledF16Output output, Tile tile, int rb, int tb, int rows,
                                int) const {
        constexpr int Group = C::kBlockRows / 2;
        static_assert(Group == 16);
        for (int i = threadIdx.x; i < C::kBlockTokens * (Group / 4); i += C::kThreads) {
            const int t = i / (Group / 4), r = i % (Group / 4) * 4;
            const float4 value{product(tile.at(r, t), tile.at(r + Group, t)),
                               product(tile.at(r + 1, t), tile.at(r + 1 + Group, t)),
                               product(tile.at(r + 2, t), tile.at(r + 2 + Group, t)),
                               product(tile.at(r + 3, t), tile.at(r + 3 + Group, t))};
            store_scaled_f16_product<Group>(output, value, rb + r, tb + t,
                                            rb + r + 3 < rows / 2 && tb + t < physical_tokens);
        }
    }
};

template <int Group>
struct ScaledF16ProductEpilogue {
    ClampedSwiGluProduct product;
    template <class C>
    static constexpr int kSharedBytes = C::kBlockRows / 2 * C::kBlockTokens * 4;

    template <class C, bool Full>
    __device__ void finish_tile(ScaledF16Output output, unsigned char* scratch,
                                float (&acc)[C::kMmaRows][C::kMmaTokens][4], int rb, int tb,
                                int rows, int tokens) const {
        constexpr int P = C::kBlockRows / 2;
        static_assert(C::kWarpsRows == 2 && P % 32 == 0);
        auto* tile = reinterpret_cast<float*>(scratch);
        int tid = threadIdx.x, warp = tid / 32, lane = tid & 31, wr = warp / C::kWarpsTokens,
            wt  = warp % C::kWarpsTokens;
        auto at = [&](int r, int t) -> float& { return tile[t * P + (r ^ ((t & 6) << 2))]; };
        if (wr == 1)
            for (int mi = 0; mi < C::kMmaRows; ++mi)
                for (int ni = 0; ni < C::kMmaTokens; ++ni) {
                    int r            = mi * 16 + (lane >> 2),
                        t            = wt * C::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    auto& a          = acc[mi][ni];
                    at(r, t)         = a[0];
                    at(r, t + 1)     = a[1];
                    at(r + 8, t)     = a[2];
                    at(r + 8, t + 1) = a[3];
                }
        __syncthreads();
        if (wr == 0)
            for (int mi = 0; mi < C::kMmaRows; ++mi)
                for (int ni = 0; ni < C::kMmaTokens; ++ni) {
                    int r            = mi * 16 + (lane >> 2),
                        t            = wt * C::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    auto& a          = acc[mi][ni];
                    at(r, t)         = product(a[0], at(r, t));
                    at(r, t + 1)     = product(a[1], at(r, t + 1));
                    at(r + 8, t)     = product(a[2], at(r + 8, t));
                    at(r + 8, t + 1) = product(a[3], at(r + 8, t + 1));
                }
        __syncthreads();
        for (int i = tid; i < C::kBlockTokens * (P / 4); i += C::kThreads) {
            int t = i / (P / 4), r = i % (P / 4) * 4;
            auto v = common::load_vec<float4>(&at(r, t));
            store_scaled_f16_product<Group>(output, v, rb + r, tb + t,
                                            rb + r + 3 < rows / 2 && (Full || tb + t < tokens));
        }
    }
};

struct ScaledF16BlockOperands {
    const __half* x;
    const std::uint8_t* codes;
    const float* scales;
    const float* input_scales;
    int rows, k, tokens;
};

// F16 tensor maps transfer the normalized product without conversion.
template <class S>
Fp8TmaDescriptors scaled_f16_block_maps(const ScaledF16BlockOperands& p) {
    validate_native_fp8<S>(p);
    CUtensorMap activation{};
    const std::uint64_t dims[]{64, std::uint64_t(p.k / 64), std::uint64_t(p.tokens)};
    const std::uint64_t strides[]{128, std::uint64_t(p.k) * 2};
    const std::uint32_t box[]{64, std::uint32_t(S::kBlockK / 64), std::uint32_t(S::kBlockTokens)};
    const std::uint32_t steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &activation, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<__half*>(p.x), dims, strides,
        box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("scaled F16 FFN TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return {activation, fp8_tma_map(p.codes, p.rows, p.k, S::kBlockRows, S::kBlockK)};
}

template <class S, class O, class E>
__device__ __forceinline__ void scaled_f16_block_tma_body(const Fp8TmaDescriptors& maps,
                                                          ScaledF16BlockOperands p, O output,
                                                          E epilogue, int tile_idx) {
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, BK = S::kBlockK, ST = S::kStages;
    constexpr int Group = S::kInputGroup;
    static_assert(S::kBlockRows <= 128 && 128 % S::kBlockRows == 0);
    static_assert(Group == 16 || Group == 32 || Group == 64 || Group == 128);
    extern __shared__ __align__(128) unsigned char raw[];
    auto* weights = raw;
    auto* inputs  = reinterpret_cast<__half*>(raw + ST * BR * BK);
    auto* full    = reinterpret_cast<std::uint64_t*>(raw + fp8_block_tma_scratch_bytes<S, E>);
    auto* empty   = full + ST;
    int ri, ti;
    native_fp8_tile<S>(tile_idx, p.rows, p.tokens, ri, ti);
    const int rb = ri * BR, tb = ti * BT, K = S::kStaticK ? S::kStaticK : p.k;
    const int tiles = K / BK;
    const int tid = threadIdx.x, lane = tid & 31;
    const bool producer = tid >= S::kConsumerThreads;
    if (tid == 0) {
#pragma unroll
        for (int slot = 0; slot < ST; ++slot) {
            common::cta_mbarrier_init(full + slot, 1);
            common::cta_mbarrier_init(empty + slot, S::kConsumerWarps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    float accum[S::kMmaRows][S::kMmaTokens][4]{};
    if (producer) {
#pragma unroll 1
        for (int kt = 0; kt < tiles; ++kt) {
            int slot = kt % ST;
            common::cta_mbarrier_wait(empty + slot, 1U ^ ((kt / ST) & 1U));
            if (lane == 0) {
                common::cta_mbarrier_arrive_expect_tx(full + slot, (BR + 2 * BT) * BK);
                common::tma_load_2d(weights + slot * BR * BK, &maps.weight, kt * BK, rb,
                                    full + slot);
                common::tma_load_3d(inputs + slot * BT * BK, &maps.activation, 0, kt * (BK / 64),
                                    tb, full + slot);
            }
        }
    } else {
        const int warp = tid >> 5, wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
        // Restore each private input group before the represented FP32 weight scale.
        // A G128 sum spans both K64 transfers; no BF16 product rounding is introduced.
        float partial[S::kMmaRows][S::kMmaTokens][4]{};
#pragma unroll 1
        for (int kt = 0; kt < tiles; ++kt) {
            int slot = kt % ST;
            common::cta_mbarrier_wait(full + slot, (kt / ST) & 1U);
            const auto* ws = weights + slot * BR * BK;
            const auto* xs = inputs + slot * BT * BK;
            unsigned af[S::kMmaRows][4], bf[S::kMmaTokens][2];
            const auto load = [&](int kk) {
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const int row   = wr * S::kWarpRows + r * 16 + (lane >> 2),
                              col   = kk + (lane & 3) * 2;
                    const auto pair = [&](int row, int col) {
                        const auto bits =
                            common::fp8_code2_to_half2(common::load_vec<std::uint16_t>(
                                ws + row * BK + fp8_mma_shared_byte<S>(row, col)));
                        return scaled_product_half2_bits(bits);
                    };
                    af[r][0] = pair(row, col);
                    af[r][1] = pair(row + 8, col);
                    af[r][2] = pair(row, col + 8);
                    af[r][3] = pair(row + 8, col + 8);
                }
#pragma unroll
                for (int t = 0; t < S::kMmaTokens; ++t) {
                    const int row = wt * S::kWarpTokens + t * 8 + (lane & 7),
                              col = kk + ((lane >> 3) & 1) * 8;
                    common::ldmatrix_x2(
                        bf[t][0], bf[t][1],
                        common::smem_addr(xs + row * BK + bf16_mma_shared_col<S>(row, col)));
                }
            };
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                load(kk);
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
#pragma unroll
                    for (int t = 0; t < S::kMmaTokens; ++t) {
                        auto& a = partial[r][t];
                        common::mma_f16(a[0], a[1], a[2], a[3], af[r][0], af[r][1], af[r][2],
                                        af[r][3], bf[t][0], bf[t][1]);
                    }
                }
                if (((kt * BK + kk + 16) % Group) == 0) {
                    const float weight_scale =
                        p.scales[std::int64_t(rb / 128) * (K / 128) + kt * BK / 128];
#pragma unroll
                    for (int t = 0; t < S::kMmaTokens; ++t) {
                        const int token = tb + wt * S::kWarpTokens + t * 8 + 2 * (lane & 3);
                        const int group = (kt * BK + kk) / Group;
                        const float sx =
                            token < p.tokens
                                ? p.input_scales[std::int64_t(token) * (K / Group) + group]
                                : 0.F;
                        const float sy =
                            token + 1 < p.tokens
                                ? p.input_scales[std::int64_t(token + 1) * (K / Group) + group]
                                : 0.F;
#pragma unroll
                        for (int r = 0; r < S::kMmaRows; ++r) {
                            accum[r][t][0] =
                                fmaf(partial[r][t][0] * sx, weight_scale, accum[r][t][0]);
                            accum[r][t][1] =
                                fmaf(partial[r][t][1] * sy, weight_scale, accum[r][t][1]);
                            accum[r][t][2] =
                                fmaf(partial[r][t][2] * sx, weight_scale, accum[r][t][2]);
                            accum[r][t][3] =
                                fmaf(partial[r][t][3] * sy, weight_scale, accum[r][t][3]);
#pragma unroll
                            for (int j = 0; j < 4; ++j) partial[r][t][j] = 0.F;
                        }
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) common::cta_mbarrier_arrive(empty + slot);
        }
    }
    __syncthreads();
    if (producer) return;
    native_fp8_finish<Fp8BlockTmaConsumerSchedule<S>, true>(output, epilogue, raw, accum, rb, tb,
                                                            p.rows, p.tokens);
}

template <class S, class O, class E>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void scaled_f16_block_tma_kernel(
    const __grid_constant__ Fp8TmaDescriptors maps, ScaledF16BlockOperands p, O output,
    E epilogue) {
    scaled_f16_block_tma_body<S>(maps, p, output, epilogue, int(blockIdx.x));
}
} // namespace ninfer::ops::detail::swiglu_ffn
