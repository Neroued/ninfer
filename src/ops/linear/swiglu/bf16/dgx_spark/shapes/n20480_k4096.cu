#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/bf16/paired.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 4096>;
using S16    = SwiGluBf16SlicedSchedule<Static<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>, 10240>;
using S32    = SwiGluBf16SlicedSchedule<Static<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 1>>, 10240>;

template <class S, class Product>
void tma(const Bf16A16Operands& p, LinearBf16Output out, Product product, cudaStream_t stream) {
    launch_swiglu_bf16_tma<Static<S>, 10240>(p, out, product, stream);
}

template <class Product>
void launch(const Tensor& x, const Weight& weight, Tensor& y, Product product,
            cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, weight);
    const LinearBf16Output out{static_cast<__nv_bfloat16*>(y.data), 10240};
    if (p.tokens <= 16) {
        launch_bf16_a16_sliced_k_mma<S16>(p, out, SwiGluBf16FragmentEpilogue<Product>{product},
                                          stream);
    } else if (p.tokens <= 32) {
        launch_bf16_a16_sliced_k_mma<S32>(p, out, SwiGluBf16FragmentEpilogue<Product>{product},
                                          stream);
    } else if (p.tokens <= 64) {
        tma<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 2, 2>>(p, out, product, stream);
    } else if (p.tokens <= 256) {
        tma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3, 1>>(p, out, product, stream);
    } else if (p.tokens <= 512) {
        tma<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 2, 1>>(p, out, product, stream);
    } else if (p.tokens <= 2560) {
        tma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>(p, out, product, stream);
    } else {
        tma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12>>(
            p, out, product, stream);
    }
}
} // namespace

void launch_bf16_swiglu_n20480_k4096(const Tensor& x, const Weight& weight, Tensor& out,
                                     float limit, bool clamped, cudaStream_t stream) {
    if (clamped)
        launch(x, weight, out, ClampedSwiGluProduct{limit}, stream);
    else
        launch(x, weight, out, SwiGluProduct{}, stream);
}

} // namespace ninfer::ops::detail
