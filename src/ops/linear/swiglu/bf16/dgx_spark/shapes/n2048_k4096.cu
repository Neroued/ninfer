#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/bf16/paired.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Product>
__global__ __launch_bounds__(128) void paired_gemv(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                                   LinearBf16Output out, Product product) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    float gate[4] = {}, up[4] = {};
    for (int k = lane * 8; k < 4096; k += 256) {
        common::Bf16x8Bits xv, gv, uv;
        xv.raw = common::load_vec<uint4>(x + k);
        gv.raw = common::load_vec<uint4>(w + std::int64_t(row) * 4096 + k);
        uv.raw = common::load_vec<uint4>(w + std::int64_t(row + 1024) * 4096 + k);
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
    if (lane == 0) out.store(row, 0, product(g, u));
}

template <class Schedule, class Product>
struct TmaProduct {
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

template <class Schedule, class Product>
void tma(const Bf16A16Operands& p, LinearBf16Output out, Product product, cudaStream_t stream) {
    using S = Bf16ScheduleInstance<Schedule, 4096>;
    launch_bf16_a16_tma_mma<S, 1, LinearBf16Output, TmaProduct<S, Product>,
                            SwiGluBf16TmaWeights<1024>>(p, out, {product}, stream);
}

template <class Product>
void launch(const Tensor& x, const Weight& w, Tensor& y, Product product, cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, w);
    const LinearBf16Output out{static_cast<__nv_bfloat16*>(y.data), 1024};
    if (p.tokens == 1) {
        paired_gemv<<<256, 128, 0, stream>>>(p.x, p.weight, out, product);
        CUDA_CHECK(cudaGetLastError());
    } else if (p.tokens <= 8) {
        using S = SwiGluBf16SlicedSchedule<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>, 4096>, 1024>;
        launch_bf16_a16_sliced_k_mma<S>(p, out, SwiGluBf16FragmentEpilogue<Product>{product},
                                        stream);
    } else if (p.tokens <= 16) {
        using S = SwiGluBf16SlicedSchedule<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>, 4096>, 1024>;
        launch_bf16_a16_sliced_k_mma<S>(p, out, SwiGluBf16FragmentEpilogue<Product>{product},
                                        stream);
    } else if (p.tokens <= 32) {
        tma<Bf16A16TmaMmaSchedule<32, 16, 128, 16, 8, 2, 2>>(p, out, product, stream);
    } else if (p.tokens <= 64) {
        tma<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 2, 2>>(p, out, product, stream);
    } else if (p.tokens <= 128) {
        tma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 2, 2>>(p, out, product, stream);
    } else if (p.tokens <= 512) {
        tma<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 2, 1>>(p, out, product, stream);
    } else if (p.tokens <= 2048) {
        tma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1>>(p, out, product, stream);
    } else {
        tma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>>(
            p, out, product, stream);
    }
}
} // namespace

void launch_bf16_swiglu_n2048_k4096(const Tensor& x, const Weight& w, Tensor& y, float limit,
                                    bool clamped, cudaStream_t stream) {
    if (clamped)
        launch(x, w, y, ClampedSwiGluProduct{limit}, stream);
    else
        launch(x, w, y, SwiGluProduct{}, stream);
}
} // namespace ninfer::ops::detail
