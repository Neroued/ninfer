#include "ops/linear/q4/q4_simt_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using Q4SimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4SimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;


} // namespace

void launch_q4_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_simt<Q4SimtR8C4Schedule>(x, w, out, stream);
}

void launch_q4_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_simt<Q4SimtR8C8Schedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
