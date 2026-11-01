#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/q8/paired.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"

namespace ninfer::ops::detail {
namespace {
using S16    = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                       Q8ActivationStage::PaddedZero, 4096>;
using S32    = Q8A16SlicedKMmaSchedule<32, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                       Q8ActivationStage::PaddedZero, 4096>;
using T64    = Q8A16TmaMmaSchedule<32, 64, 32, 16, 1, 8, 4096, 128, true, true, true>;
using Medium = Q8A16TmaMmaSchedule<64, 64, 64, 16, 3, 8, 4096, 64, false, false, true>;
using T128   = Q8A16TmaMmaSchedule<64, 128, 32, 32, 2, 8, 4096, 64, true, true, true>;
using Bulk   = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 4096, 64, true, true, true>;
using Large  = Q8A16TmaK32Schedule<16, 4096, true>;

template <class Product>
void run(const Tensor& x, const Weight& w, Tensor& out, Product product, cudaStream_t stream) {
    const auto p = q8_linear_operands(x, w);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), 1024};
    if (p.tokens == 1) return launch_q8_swiglu_vector<4096, 1024, 2, 8>(p, output, product, stream);
    if (p.tokens <= 16) return launch_q8_swiglu_sliced<S16, 1024>(p, output, product, stream);
    if (p.tokens <= 32) return launch_q8_swiglu_sliced<S32, 1024>(p, output, product, stream);
    if (p.tokens <= 64) return launch_q8_swiglu_tma<T64>(p, output, product, stream);
    if (p.tokens <= 192) return launch_q8_swiglu_tma<Medium>(p, output, product, stream);
    if (p.tokens <= 512) return launch_q8_swiglu_tma<T128>(p, output, product, stream);
    if (p.tokens <= 2560) return launch_q8_swiglu_tma<Bulk>(p, output, product, stream);
    launch_q8_a16_tma_k32<Large, SwiGluRowMajorMmaRows<Large>>(
        p, output, BasicSwiGluRowMajorMmaEpilogue<Product>{product}, stream);
}
} // namespace

void launch_q8_swiglu_n2048_k4096(const Tensor& x, const Weight& w, Tensor& out, float limit,
                                  bool clamped, cudaStream_t stream) {
    if (clamped) return run(x, w, out, ClampedSwiGluProduct{limit}, stream);
    run(x, w, out, SwiGluProduct{}, stream);
}
} // namespace ninfer::ops::detail
