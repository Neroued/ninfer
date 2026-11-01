#pragma once
#include "ops/linear/kernels/nvfp4/nvfp4_a4_quantize.cuh"
#include "ops/linear/swiglu/paired_output.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
struct QuantizedProductOutput {
    std::uint8_t* codes;
    std::uint8_t* scales;
    float divisor;
    Nvfp4ScaleLayout layout;
    int tokens;
    int scale_tokens;

    __device__ void store_group(int row, int token, uint2 value, std::uint8_t scale) const {
        if (token < tokens)
            common::store_vec(codes + std::int64_t(token) * (6144 / 2) + row / 2, value);
        else if (layout == Nvfp4ScaleLayout::RowMajor || token >= scale_tokens)
            return;
        const int group = row / 16;
        std::int64_t offset;
        if (layout == Nvfp4ScaleLayout::RowMajor)
            offset = std::int64_t(token) * (6144 / 16) + group;
        else if (layout == Nvfp4ScaleLayout::Tiled128)
            offset =
                nvfp4_tiled_scale_offset<Nvfp4ActivationGeometry<6144>, Nvfp4ScaleLayout::Tiled128>(
                    token, group);
        else
            offset =
                nvfp4_tiled_scale_offset<Nvfp4ActivationGeometry<6144>, Nvfp4ScaleLayout::Tiled256>(
                    token, group);
        scales[offset] = token < tokens ? scale : 0;
    }
};

// Four adjacent lanes own one token and one G16 product group. Each lane has rows
// 2*l,2*l+1,8+2*l,9+2*l in two matching gate/up fragments. A four-lane maximum and
// gather quantize the FP32 products directly; no BF16 or full product scratch is introduced.
template <class Product>
struct QuantizedProductEpilogue {
    Product product;
    template <class S>
    static constexpr int kSharedBytes = 0;

    __device__ void group(QuantizedProductOutput out, int row, int token, float a, float b, float c,
                          float d, int lane) const {
        float maximum    = fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c), fabsf(d)));
        maximum          = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1, 4));
        maximum          = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2, 4));
        const auto scale = common::nvfp4_activation_scale(maximum, out.divisor);
        const unsigned word =
            common::quantize_nvfp4_activation_pair(make_float2(a, b), out.divisor, scale) |
            (unsigned(common::quantize_nvfp4_activation_pair(make_float2(c, d), out.divisor, scale))
             << 8);
        const int first   = lane & ~3;
        const unsigned b1 = __shfl_sync(0xffffffff, word, first + 1),
                       b2 = __shfl_sync(0xffffffff, word, first + 2),
                       b3 = __shfl_sync(0xffffffff, word, first + 3);
        const unsigned lo =
            (word & 255) | ((b1 & 255) << 8) | ((b2 & 255) << 16) | ((b3 & 255) << 24);
        const unsigned hi = (word >> 8) | ((b1 >> 8) << 8) | ((b2 >> 8) << 16) | ((b3 >> 8) << 24);
        if ((lane & 3) == 0) out.store_group(row, token, make_uint2(lo, hi), scale);
    }

    template <class S, bool Full>
    __device__ void finish_tile(QuantizedProductOutput out, unsigned char*,
                                float (&acc)[S::kMmaTokens][S::kMmaRows][4], int row_begin,
                                int token_begin, int, int) const {
        constexpr int producers = [] {
            if constexpr (requires { S::kProducerThreads; })
                return S::kProducerThreads;
            else
                return 0;
        }();
        constexpr int gate_fragments = S::kMmaRows / 2;
        static_assert(gate_fragments % 2 == 0);
        const int tid = int(threadIdx.x) - producers, warp = tid / 32, lane = tid % 32;
        const int wt = warp / S::kWarpsRows, wr = warp % S::kWarpsRows;
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
            const int token = token_begin + wt * S::kWarpTokens + mt * 16 + (lane >> 2);
#pragma unroll
            for (int mr = 0; mr < gate_fragments; mr += 2) {
                const int row  = row_begin + wr * (S::kWarpRows / 2) + mr * 8;
                const auto& g0 = acc[mt][mr];
                const auto& u0 = acc[mt][mr + gate_fragments];
                const auto& g1 = acc[mt][mr + 1];
                const auto& u1 = acc[mt][mr + 1 + gate_fragments];
                group(out, row, token, product(g0[0], u0[0]), product(g0[1], u0[1]),
                      product(g1[0], u1[0]), product(g1[1], u1[1]), lane);
                group(out, row, token + 8, product(g0[2], u0[2]), product(g0[3], u0[3]),
                      product(g1[2], u1[2]), product(g1[3], u1[3]), lane);
            }
        }
    }
};
} // namespace ninfer::ops::detail::swiglu_ffn
