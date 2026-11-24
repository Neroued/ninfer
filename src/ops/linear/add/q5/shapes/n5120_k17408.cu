#include "ops/linear/add/dispatch.h"
#include "ops/linear/add/q5/launch.cuh"
#include "ops/linear/kernels/q5/q5_instances.cuh"

namespace ninfer::ops::detail {
namespace {
void launch_wide_tail(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    constexpr auto wide_launch = launch_q5_add_collective<q5_instances::MmaR64T128>;
    const int wide             = (x.ne[1] / 512) * 512;
    const int tail             = x.ne[1] - wide;
    if (wide == 0 || tail == 0 || tail > 192) return wide_launch(x, w, residual, stream);
    const auto prefix_x = x.slice(1, 0, wide);
    auto prefix_out     = residual.slice(1, 0, wide);
    wide_launch(prefix_x, w, prefix_out, stream);
    const auto tail_x = x.slice(1, wide, tail);
    auto tail_out     = residual.slice(1, wide, tail);
    select_q5_linear_add_n5120_k17408(tail)(tail_x, w, tail_out, stream);
}
} // namespace

LinearAddLaunch select_q5_linear_add_n5120_k17408(std::int32_t tokens) {
    if (tokens == 1)
        return launch_q5_add_simt<Q5A16DirectSimtSchedule<1, 1, 2, 8, 16, 17408, true>>;
    if (tokens == 2)
        return launch_q5_add_simt<Q5A16DirectSimtSchedule<1, 2, 2, 8, 16, 17408, true>>;
    if (tokens == 3)
        return launch_q5_add_simt<Q5A16DirectSimtSchedule<1, 3, 2, 8, 16, 17408, true>>;
    if (tokens == 4)
        return launch_q5_add_simt<Q5A16DirectSimtSchedule<1, 4, 2, 8, 16, 17408, true>>;
    if (tokens <= 8) return launch_q5_add_sliced<q5_instances::SlicedR16T8W4S2>;
    if (tokens <= 16) return launch_q5_add_sliced<q5_instances::SlicedR16T16W4S2>;
    if (tokens <= 24) return launch_q5_add_sliced<q5_instances::SlicedR16T24W4S2>;
    if (tokens <= 32) return launch_q5_add_sliced<q5_instances::SlicedR32T32W4S2>;
    if (tokens <= 48) return launch_q5_add_sliced<q5_instances::SlicedR32T24W4S2Pairwise>;
    if (tokens <= 64) return launch_q5_add_sliced<q5_instances::SlicedR32T32W4S1>;
    if (tokens <= 96) return launch_q5_add_sliced<q5_instances::SlicedR32T32W4S1>;
    if (tokens <= 128) return launch_q5_add_sliced<q5_instances::SlicedR32T32W2S2>;
    if (tokens <= 160) return launch_q5_add_sliced<q5_instances::SlicedR32T64W2S1>;
    if (tokens <= 256) return launch_q5_add_pointwise<q5_instances::MmaR32T128>;
    if (tokens <= 512) return launch_q5_add_collective<q5_instances::MmaR64T128>;
    return launch_wide_tail;
}
} // namespace ninfer::ops::detail
