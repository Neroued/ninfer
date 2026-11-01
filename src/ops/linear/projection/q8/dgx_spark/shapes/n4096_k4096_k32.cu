#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"
#include "ops/linear/projection/q8/dgx_spark/launch.h"

namespace ninfer::ops::detail {
namespace {
template <int Raster>
struct Schedule : Q8A16TmaK32Schedule<Raster, 4096> {};
} // namespace

void launch_q8_n4096_k4096_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    const auto launch = [&]<int Raster>() {
        launch_q8_a16_tma_k32<Schedule<Raster>>(
            q8_linear_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
            LinearIdentityEpilogue{}, stream);
    };
    if (x.ne[1] >= 4096)
        launch.template operator()<16>();
    else
        launch.template operator()<4>();
}
} // namespace ninfer::ops::detail
