#pragma once
#include "ops/linear/swiglu/product_epilogue.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_instances.cuh"
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
// Pair an MMA atom's first eight rows with its second eight rows.
// Persistent gate and up planes stay unchanged.
template <class Schedule, int Intermediate>
struct SwiGluBf16SlicedSchedule : Schedule {
    __device__ static int weight_row(int row) {
        return (row / 16) * 8 + row % 8 + (row % 16 >= 8 ? Intermediate : 0);
    }
};

template <class Product>
struct SwiGluBf16FragmentEpilogue {
    Product product;

    template <class Output>
    __device__ void store_fragment(Output out, int row, int token, float4 v, int,
                                   int tokens) const {
        const int r = (row / 16) * 8 + row % 8;
        if (token < tokens) out.store(r, token, swiglu_product(product, r, v.x, v.z));
        if (token + 1 < tokens) out.store(r, token + 1, swiglu_product(product, r, v.y, v.w));
    }
};

// TMA fills one contiguous gate panel and one contiguous up panel per CTA.
template <int Intermediate>
struct SwiGluBf16TmaWeights {
    template <class Schedule>
    static CUtensorMap map(const Bf16A16Operands& p) {
        return bf16_tma_map<Schedule>(p.weight, p.rows, p.k, Schedule::kBlockRows / 2,
                                      Schedule::kBlockK);
    }

    template <class Schedule>
    __device__ static void load(void* dst, const CUtensorMap* map, int sector, int row,
                                std::uint64_t* barrier) {
        bf16_tma_load<Schedule>(dst, map, sector, row / 2, barrier);
        bf16_tma_load<Schedule>(static_cast<__nv_bfloat16*>(dst) +
                                    Schedule::kBlockRows / 2 * Schedule::kBlockK,
                                map, sector, row / 2 + Intermediate, barrier);
    }
};

template <int Hidden, int Intermediate, class Product, class Output>
__global__ __launch_bounds__(128) void swiglu_bf16_paired_gemv(const __nv_bfloat16* x,
                                                               const __nv_bfloat16* w, Output out,
                                                               Product product) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    float gate[4] = {}, up[4] = {};
    for (int k = lane * 8; k < Hidden; k += 256) {
        common::Bf16x8Bits xv, gv, uv;
        xv.raw = common::load_vec<uint4>(x + k);
        gv.raw = common::load_vec<uint4>(w + std::int64_t(row) * Hidden + k);
        uv.raw = common::load_vec<uint4>(w + std::int64_t(row + Intermediate) * Hidden + k);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const auto a = __bfloat1622float2(xv.pair[pair]);
            const auto g = __bfloat1622float2(gv.pair[pair]);
            const auto u = __bfloat1622float2(uv.pair[pair]);
            gate[pair]   = fmaf(a.x, g.x, gate[pair]);
            gate[pair]   = fmaf(a.y, g.y, gate[pair]);
            up[pair]     = fmaf(a.x, u.x, up[pair]);
            up[pair]     = fmaf(a.y, u.y, up[pair]);
        }
    }
    const float g = common::warp_reduce_sum((gate[0] + gate[1]) + (gate[2] + gate[3]));
    const float u = common::warp_reduce_sum((up[0] + up[1]) + (up[2] + up[3]));
    if (lane == 0) out.store(row, 0, swiglu_product(product, row, g, u));
}

template <class Product>
struct SwiGluBf16TmaProduct {
    enum { kTmaCollective = true };

    Product product;
    template <class S>
    static constexpr int kSharedBytes =
        BasicSwiGluRowMajorMmaEpilogue<Product>::template kSharedBytes<S>;

    template <class S, bool Full, class Output>
    __device__ void finish_tile(Output out, unsigned char* scratch,
                                float (&acc)[S::kMmaRows][S::kMmaTokens][4], int row, int token,
                                int rows, int tokens) const {
        if (threadIdx.x >= S::kProducerThreads)
            BasicSwiGluRowMajorMmaEpilogue<Product>{product}.template finish_tile<S, Full>(
                out, scratch, acc, row / 2, token, rows, tokens);
    }
};

template <class Schedule, int Intermediate, class Product, class Output>
void launch_swiglu_bf16_tma(const Bf16A16Operands& p, Output out, Product product,
                            cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule, 1, Output, SwiGluBf16TmaProduct<Product>,
                            SwiGluBf16TmaWeights<Intermediate>>(p, out, {product}, stream);
}

} // namespace ninfer::ops::detail
