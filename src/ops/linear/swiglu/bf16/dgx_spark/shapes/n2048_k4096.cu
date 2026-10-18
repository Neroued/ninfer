#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/bf16/paired.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule, class Product, class Output>
void tma(const Bf16A16Operands& p, Output out, Product product, cudaStream_t stream) {
    launch_swiglu_bf16_tma<Bf16ScheduleInstance<Schedule, 4096>, 1024>(p, out, product, stream);
}

template <class Product>
void launch(const Tensor& x, const Weight& w, Tensor& y, Product product, cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, w);
    const LinearBf16Output out{static_cast<__nv_bfloat16*>(y.data), 1024};
    if (p.tokens == 1) {
        swiglu_bf16_paired_gemv<4096, 1024><<<256, 128, 0, stream>>>(p.x, p.weight, out, product);
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
