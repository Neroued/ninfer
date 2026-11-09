#include "ops/dynamic_grouped_conv/finish/launch.h"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail::dynamic_grouped_conv::finish {
namespace {
constexpr int kOutputRows = 4096, kOutputGroups = 256;

struct Outputs {
    const __nv_bfloat16* base;
    const __nv_bfloat16* delta;
    __nv_bfloat16* partial;
    int width, columns;
};

template <bool Whole>
struct Convolution {
    enum { kSlicedTile = true };

    template <class S>
    __device__ void finish_sliced_tile(Outputs out, const float* z, int row0, int token0, int,
                                       int tokens) const {
        const int count = Whole ? tokens : min(out.columns, tokens - token0);
        for (int item = threadIdx.x; item < S::kBlockRows * count; item += S::kThreads) {
            const int h = row0 + item % S::kBlockRows, token = token0 + item / S::kBlockRows;
            const int g = h / 16, offset = (token - token0) * S::kBlockRows + h - row0;
            const int di = token * 2 * kOutputGroups + g;
            float value  = (__bfloat162float(out.base[2 * kOutputRows + h]) +
                           __bfloat162float(out.delta[di])) *
                          z[offset];
            if (token % out.width)
                value = fmaf(__bfloat162float(out.base[3 * kOutputRows + h]) +
                                 __bfloat162float(out.delta[di + kOutputGroups]),
                             z[offset - S::kBlockRows], value);
            out.partial[std::int64_t(token) * kOutputRows + h] = __float2bfloat16_rn(value);
        }
    }
};

template <class S>
__global__ __launch_bounds__(S::kThreads, 1) void kernel(Bf16A16Operands p, Outputs out) {
    extern __shared__ __align__(16) unsigned char scratch[];
    const int tiles = (p.tokens + out.columns - 1) / out.columns;
    bf16_a16_sliced_k_mma_body<S>(p.x, p.weight, out, Convolution<false>{}, p.rows, p.k, p.tokens,
                                  blockIdx.x / tiles, (blockIdx.x % tiles) * out.columns, scratch);
}

template <int K, int R, int T, int KW, int WK, int Phase, bool Reuse = false, bool Whole = false>
void run(const Tensor& x, const Weight& w, const Tensor& base, const Tensor& delta, Tensor& y,
         cudaStream_t stream) {
    using S =
        Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<R, T, KW, WK, 2, Cache::cg, Cache::cg, 1,
                                                       Phase, Bf16MmaRaster::RowFast, Reuse>,
                             K>;
    const int m = x.ne[1] * x.ne[2];
    // A CTA owns whole requests; previous taps never cross its tile.
    const int step = x.ne[1] * std::min(x.ne[2], T / x.ne[1]);
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(x.data),
                            static_cast<const __nv_bfloat16*>(w.qdata), 4096, K, m};
    const Outputs out{static_cast<const __nv_bfloat16*>(base.data),
                      static_cast<const __nv_bfloat16*>(delta.data),
                      static_cast<__nv_bfloat16*>(y.data), x.ne[1], step};
    if constexpr (Whole) {
        launch_bf16_a16_sliced_k_mma<S>(p, out, Convolution<true>{}, stream);
        return;
    }
    constexpr auto entry = kernel<S>;
    const auto bytes     = common::prepare_shared<S::kSharedBytes, entry, true>();
    entry<<<(4096 / R) * ((m + step - 1) / step), S::kThreads, bytes, stream>>>(p, out);
    CUDA_CHECK(cudaGetLastError());
}

template <int K>
void launch_profile(const Tensor& x, const Weight& w, const Tensor& base, const Tensor& delta,
                    Tensor& y, cudaStream_t stream) {
    const int m = x.ne[1] * x.ne[2];
    if (m <= 32) {
        if constexpr (K == 2048)
            return run<K, 16, 32, 16, 32, 7, false, true>(x, w, base, delta, y, stream);
        else
            return run<K, 16, 32, 16, 32, 3, false, true>(x, w, base, delta, y, stream);
    }
    if (m <= 64) {
        if constexpr (K == 2048)
            return run<K, 16, 32, 16, 16, 11>(x, w, base, delta, y, stream);
        else
            return run<K, 16, 32, 16, 32, 3>(x, w, base, delta, y, stream);
    }
    return run<K, 32, 64, 8, 16, 7, true>(x, w, base, delta, y, stream);
}
} // namespace

void launch(const Tensor& x, const Weight& w, const Tensor& base, const Tensor& delta, Tensor& y,
            cudaStream_t stream) {
    if (w.k == 2048) return launch_profile<2048>(x, w, base, delta, y, stream);
    launch_profile<6144>(x, w, base, delta, y, stream);
}
} // namespace ninfer::ops::detail::dynamic_grouped_conv::finish
