#include "ops/linear/projection/q4/q4_shapes.h"
#include "ops/linear/kernels/q4/q4_instance_launch.cuh"
#include "ops/linear/kernels/q4/q4_a16_tma_prefetch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv =
    Q4A16GemvSchedule<4, 2, 16, 1, Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedWord8,
                      Q4GemvDecodeMode::Fp16Mantissa, Q4GemvCodeTransfer::AsyncVector16,
                      Q4GemvScaleAccess::SharedPair32, Cache::ca, 64, 1>;
template <int Tokens>
using Few      = Q4A16SimtSchedule<4, Tokens, 2, 8, 2, Cache::ca, 1>;
using Small16  = Q4A16SlicedKMmaSchedule<16, 16, 8, 1, Cache::cg, Cache::ca, 2, 4096, 16>;
using T48      = Q4A16TmaPrefetchSchedule<32, 48, 32, 8, 2, 4096, 1, 16, 4, 1, 128>;
using T64      = Q4A16TmaPrefetchSchedule<32, 64, 32, 16, 3, 4096>;
using T96      = Q4A16TmaPrefetchSchedule<64, 96, 32, 24, 2, 4096, 1, 16, 4, 1, 128, 1>;
using T128     = Q4A16TmaPrefetchSchedule<64, 128, 64, 16, 2, 4096, 1>;
using Mid96    = Q4A16TmaPrefetchSchedule<128, 96, 64, 24, 2, 4096, 1, 16, 4, 1, 64, 1>;
using T256     = Q4A16TmaPrefetchSchedule<64, 256, 64, 32, 2, 4096, 1, 16, 4, 1, 64, 1>;
using Large256 = Q4A16TmaPrefetchSchedule<64, 256, 64, 32, 2, 4096, 1, 64, 4, 1, 64, 1>;

template <class Schedule>
void tma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_tma_prefetch<Schedule>(
        q4_linear_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
}

template <class Schedule>
void bulk(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int tail = x.ne[1] % Schedule::kBlockTokens;
    if (!tail || tail > T128::kBlockTokens) {
        tma<Schedule>(x, w, out, stream);
        return;
    }
    const int prefix = x.ne[1] - tail;
    auto input       = x.slice(1, 0, prefix);
    auto output      = out.slice(1, 0, prefix);
    tma<Schedule>(input, w, output, stream);
    input  = x.slice(1, prefix, tail);
    output = out.slice(1, prefix, tail);
    // Narrow tails still use BF16 MMA, including one to four columns.
    if (tail <= Small16::kBlockTokens)
        launch_q4_a16_sliced_instance<Small16>(input, w, output, stream);
    else
        select_q4_n77440_k4096(tail)(input, w, output, stream);
}
} // namespace

Q4Launch select_q4_n77440_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_instance<Gemv>;
    // Full two/three-column SIMT tiles avoid the masked four-column route's
    // control overhead at the core decode points.
    if (tokens == 2) return launch_q4_a16_simt_instance<Few<2>>;
    if (tokens == 3) return launch_q4_a16_simt_instance<Few<3>>;
    if (tokens == 4) return launch_q4_a16_simt_instance<Few<4>>;
    if (tokens <= 16) return launch_q4_a16_sliced_instance<Small16>;
    if (tokens <= 48) return tma<T48>;
    if (tokens <= 64) return tma<T64>;
    if (tokens <= 96) return tma<T96>;
    if (tokens <= 128) return tma<T128>;
    if (tokens <= 192) return tma<Mid96>;
    if (tokens <= 640) return tma<T128>;
    if (tokens < 3072) return bulk<T256>;
    return bulk<Large256>;
}
} // namespace ninfer::ops::detail
