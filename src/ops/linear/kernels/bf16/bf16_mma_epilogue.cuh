#pragma once

#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {

// Identity BF16 output for complete 16-row MMA atoms. Shuffle pairs of rows
// into adjacent BF16 stores; token tails remain predicated after the shuffle.
struct Bf16MmaWarpStoreEpilogue {
private:
    __device__ __forceinline__ static __nv_bfloat16* pointer(const LinearBf16Output& output,
                                                             int row, int token) {
        return output.at(row, token);
    }

    __device__ __forceinline__ static __nv_bfloat16* pointer(const LinearBf16SegmentOutput& output,
                                                             int row, int token) {
        return output.data +
               (static_cast<std::int64_t>(token) * output.rows + row - output.parent_row_begin);
    }

    template <int... Rows>
    __device__ __forceinline__ static __nv_bfloat16*
    pointer(const LinearBf16SegmentedOutput<Rows...>& output, int row, int token) {
        static_assert(((Rows % 2 == 0) && ...), "warp-pair stores require even segments");
        return pointer(output.segment(row), row, token);
    }

protected:
    template <bool Streaming, class Output>
    __device__ __forceinline__ void store_fragment_impl(const Output& output, int row, int token,
                                                        float4 v, int tokens) const {
        const unsigned lane = threadIdx.x & 31;
        const int rb = row - int(lane >> 2), tb = token - 2 * int(lane & 3);
        const unsigned source = (lane & 3) * 8 + (lane >> 3);
        const unsigned shift  = (lane & 4) * 4;
        const auto packed     = [](float lo, float hi) {
            const __nv_bfloat162_raw bits = __floats2bfloat162_rn(lo, hi);
            return unsigned(bits.x) | (unsigned(bits.y) << 16);
        };
        const unsigned even = packed(v.x, v.z), odd = packed(v.y, v.w);
        const unsigned e0 = __shfl_sync(0xffffffffU, even, source);
        const unsigned e1 = __shfl_sync(0xffffffffU, even, source + 4);
        const unsigned o0 = __shfl_sync(0xffffffffU, odd, source);
        const unsigned o1 = __shfl_sync(0xffffffffU, odd, source + 4);
        const unsigned ev = ((e0 >> shift) & 0xffffU) | (((e1 >> shift) & 0xffffU) << 16);
        const unsigned od = ((o0 >> shift) & 0xffffU) | (((o1 >> shift) & 0xffffU) << 16);
        const int rr = rb + 2 * int(lane & 7), tt = tb + 2 * int(lane >> 3);
        if (tt < tokens) {
            if constexpr (Streaming)
                asm volatile("st.global.cs.u32 [%0], %1;" ::"l"(pointer(output, rr, tt)), "r"(ev)
                             : "memory");
            else
                common::store_vec(pointer(output, rr, tt), ev);
        }
        if (tt + 1 < tokens) {
            if constexpr (Streaming)
                asm volatile("st.global.cs.u32 [%0], %1;" ::"l"(pointer(output, rr, tt + 1)),
                             "r"(od)
                             : "memory");
            else
                common::store_vec(pointer(output, rr, tt + 1), od);
        }
    }

public:
    template <class Output>
    __device__ __forceinline__ void store_fragment(const Output& output, int row, int token,
                                                   float4 v, int, int tokens) const {
        store_fragment_impl<false>(output, row, token, v, tokens);
    }
};

// Evict streaming output first so large head projections can retain reused operands in L2.
struct Bf16MmaStreamingWarpStoreEpilogue : Bf16MmaWarpStoreEpilogue {
    template <class Output>
    __device__ __forceinline__ void store_fragment(const Output& output, int row, int token,
                                                   float4 v, int, int tokens) const {
        store_fragment_impl<true>(output, row, token, v, tokens);
    }
};

// Reuse the retired TMA staging area to transpose the accumulator fragments into
// contiguous BF16 output vectors. All threads, including the producer, participate.
struct Bf16TmaVectorEpilogue {
    static constexpr bool kTmaCollective = true;

    template <class Schedule>
    static constexpr int kSharedBytes =
        ((Schedule::kBlockRows + 63) / 64 * 64) * Schedule::kBlockTokens * 2;

    template <class Schedule, bool FullTokens, bool PredicatedRows = false, class Output>
    __device__ __forceinline__ void
    finish_tile(const Output& output, unsigned char* scratch,
                float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int row_begin,
                int token_begin, int rows, int tokens) const {
        constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens;
        constexpr int PaddedRows = (BR + 63) / 64 * 64;
        static_assert(BR % 8 == 0 && Schedule::kProducerThreads == 32);
        auto* tile    = reinterpret_cast<__nv_bfloat16*>(scratch);
        const auto at = [&](int row, int token) {
            return tile + token * PaddedRows + (row ^ ((token & 7) << 3));
        };
        if (threadIdx.x >= Schedule::kProducerThreads) {
            const int tid  = int(threadIdx.x) - Schedule::kProducerThreads;
            const int warp = tid / 32, lane = tid & 31;
            const int wm = warp / Schedule::kWarpsTokens, wn = warp % Schedule::kWarpsTokens;
#pragma unroll
            for (int mi = 0; mi < Schedule::kMmaRows; ++mi) {
                const int row = wm * Schedule::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
                for (int ni = 0; ni < Schedule::kMmaTokens; ++ni) {
                    const int token         = wn * Schedule::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    *at(row, token)         = __float2bfloat16_rn(accum[mi][ni][0]);
                    *at(row, token + 1)     = __float2bfloat16_rn(accum[mi][ni][1]);
                    *at(row + 8, token)     = __float2bfloat16_rn(accum[mi][ni][2]);
                    *at(row + 8, token + 1) = __float2bfloat16_rn(accum[mi][ni][3]);
                }
            }
        }
        __syncthreads();
        for (int vector = threadIdx.x; vector < BT * (BR / 8); vector += Schedule::kThreads) {
            const int token = vector / (BR / 8), row = (vector % (BR / 8)) * 8;
            if (FullTokens || token_begin + token < tokens) {
                if constexpr (PredicatedRows) {
                    // Compact output strides may be unaligned when rows is not a multiple of 8.
                    if (rows % 8 == 0 && row_begin + row + 8 <= rows) {
                        output.store_vector(row_begin + row, token_begin + token,
                                            common::load_vec<uint4>(at(row, token)));
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; ++i)
                            if (row_begin + row + i < rows)
                                output.store(row_begin + row + i, token_begin + token,
                                             __bfloat162float(at(row, token)[i]));
                    }
                } else {
                    output.store_vector(row_begin + row, token_begin + token,
                                        common::load_vec<uint4>(at(row, token)));
                }
            }
        }
    }
};

// Opt in explicitly: existing full-row schedules retain the original specialization.
struct Bf16TmaRowVectorEpilogue : Bf16TmaVectorEpilogue {
    static constexpr bool kPredicatedRows = true;

    template <class Schedule, bool FullTokens, class Output>
    __device__ __forceinline__ void
    finish_tile(const Output& output, unsigned char* scratch,
                float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int row_begin,
                int token_begin, int rows, int tokens) const {
        Bf16TmaVectorEpilogue::template finish_tile<Schedule, FullTokens, true>(
            output, scratch, accum, row_begin, token_begin, rows, tokens);
    }
};

} // namespace ninfer::ops::detail
