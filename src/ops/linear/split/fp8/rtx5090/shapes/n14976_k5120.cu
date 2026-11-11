#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
using Output = LinearBf16SegmentedOutput<1024, 1024, 128, 6144, 6144, 512>;

template <class Schedule, int Rows = 14976>
struct ShapeSchedule : Fp8ScheduleInstance<Schedule, 5120> {};

using Hot     = ShapeSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>>;
using T256    = ShapeSchedule<Fp8A8TmaMmaSchedule<256, 128, 128, 2, 4, 2, 1>>;
using Bulk128 = ShapeSchedule<
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 4>>;
using Bulk256 = ShapeSchedule<
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<256, 128, 128, 2, 4, 2, 1>, 170, 4, 4>>;

bool quantized(LinearPolicy policy, int tokens) { return allows_a8(policy) && tokens >= 17; }

std::size_t partial_capacity(int last) {
    if (last >= 897) return Bulk256::kPartialBytes;
    return last >= 257 ? Bulk128::kPartialBytes : 0;
}

std::size_t capacity(LinearPolicy policy, int, int last) {
    return quantized(policy, last)
               ? fp8_a8_workspace_capacity_bytes(last, 5120, partial_capacity(last))
               : 0;
}

void a16(const Fp8A16Operands& p, Output out, cudaStream_t stream) {
    const auto sliced = [&]<class S>() {
        launch_fp8_a16_sliced_k_mma<ShapeSchedule<S>>(p, out, LinearIdentityEpilogue{}, stream);
    };
    const auto mma = [&]<class S>() {
        launch_fp8_a16_mma<ShapeSchedule<S>>(p, out, LinearIdentityEpilogue{}, stream);
    };
    if (p.tokens == 1)
        return launch_fp8_a16_gemv<
            ShapeSchedule<Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>>>(
            p, out, LinearIdentityEpilogue{}, stream);
    if (p.tokens <= 16) return sliced.template operator()<Fp8SlicedInstance<16, 4, 1>>();
    if (p.tokens <= 24) return sliced.template operator()<Fp8SlicedInstance<32, 4, 2>>();
    if (p.tokens <= 32) return sliced.template operator()<Fp8SlicedInstance<32, 4, 1>>();
    if (p.tokens <= 64)
        return mma.template operator()<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>();
    if (p.tokens <= 128)
        return mma.template operator()<Fp8A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>();
    mma.template operator()<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    const Output out{
        static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
        static_cast<__nv_bfloat16*>(outputs[2].data), static_cast<__nv_bfloat16*>(outputs[3].data),
        static_cast<__nv_bfloat16*>(outputs[4].data), static_cast<__nv_bfloat16*>(outputs[5].data)};
    const auto& w    = weights.first;
    const int tokens = x.ne[1];
    if (!quantized(weights.policy, tokens)) return a16(fp8_a16_operands(x, w), out, stream);
    const auto scratch =
        allocate_fp8_a8_workspace(*workspace, tokens, 5120, partial_capacity(tokens));
    launch_fp8_a8_quantize(x, w, scratch, stream);
    const auto p = fp8_a8_operands(w, scratch, tokens);
    if (tokens <= 32)
        return launch_fp8_a8_mma<ShapeSchedule<Fp8A8T32R32K128>>(p, out, LinearIdentityEpilogue{},
                                                                 stream);
    const auto tma = [&]<class S>() {
        launch_fp8_a8_tma_mma<S>(p, out, LinearIdentityEpilogue{}, stream, scratch.partials);
    };
    if (tokens <= 128) return tma.template operator()<Hot>();
    if (tokens <= 256) return tma.template operator()<T256>();
    if (tokens <= 896 || (tokens >= 1025 && tokens <= 1408))
        return tma.template operator()<Bulk128>();
    tma.template operator()<Bulk256>();
}
} // namespace

const SplitShape kFp8SplitN14976K5120{14976, 5120, kSplitRows<1024, 1024, 128, 6144, 6144, 512>,
                                      launch, capacity};
} // namespace ninfer::ops::detail::linear_split
