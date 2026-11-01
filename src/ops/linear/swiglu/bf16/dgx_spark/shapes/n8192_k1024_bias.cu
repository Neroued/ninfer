#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/bf16/paired.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 1024>;

template <class S, class Product>
void tma(const Bf16A16Operands& p, LinearBf16Output out, Product product, cudaStream_t stream) {
    launch_swiglu_bf16_tma<Static<S>, 4096>(p, out, product, stream);
}

template <class Product>
void launch(const Tensor& x, const Weight& weight, Tensor& y, Product product,
            cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, weight);
    const LinearBf16Output out{static_cast<__nv_bfloat16*>(y.data), 4096};
    if (p.tokens == 1) {
        swiglu_bf16_paired_gemv<1024, 4096><<<1024, 128, 0, stream>>>(p.x, p.weight, out, product);
        CUDA_CHECK(cudaGetLastError());
    } else if (p.tokens <= 8) {
        using S =
            SwiGluBf16SlicedSchedule<Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>>, 4096>;
        launch_bf16_a16_sliced_k_mma<S>(p, out, SwiGluBf16FragmentEpilogue<Product>{product},
                                        stream);
    } else if (p.tokens <= 16) {
        using S =
            SwiGluBf16SlicedSchedule<Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>, 4096>;
        launch_bf16_a16_sliced_k_mma<S>(p, out, SwiGluBf16FragmentEpilogue<Product>{product},
                                        stream);
    } else if (p.tokens <= 32) {
        using S =
            SwiGluBf16SlicedSchedule<Static<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 1>>, 4096>;
        launch_bf16_a16_sliced_k_mma<S>(p, out, SwiGluBf16FragmentEpilogue<Product>{product},
                                        stream);
    } else if (p.tokens <= 64) {
        tma<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 2, 2>>(p, out, product, stream);
    } else if (p.tokens <= 128) {
        tma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 2, 2>>(p, out, product, stream);
    } else {
        tma<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 2, 1>>(p, out, product, stream);
    }
}
} // namespace

void launch_bf16_bias_swiglu_n8192_k1024(const Tensor& x, const Weight& weight, const Tensor& bias,
                                         Tensor& out, float limit, cudaStream_t stream) {
    launch(x, weight, out,
           BiasedClampedSwiGluProduct{static_cast<const __nv_bfloat16*>(bias.data), 4096, limit},
           stream);
}
} // namespace ninfer::ops::detail
