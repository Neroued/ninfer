#include "ops/linear/swiglu/q4/launch.h"
#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/kernels/q4/q4_instances.cuh"
#include "ops/linear/kernels/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/kernels/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S>
void launch_sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_sliced_k_mma<S>(
        q4_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class S>
void launch_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma<S>(q4_linear_operands(x, w),
                         LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                         LinearIdentityEpilogue{}, stream);
}

LinearSwiGluLaunch select_projection(int tokens) {
    if (tokens <= 63) return launch_sliced<q4_instances::SlicedR32T32W4S1>;
    if (tokens <= 64) return launch_mma<q4_instances::MmaR64T64K128S2A1>;
    if (tokens <= 80) return launch_mma<q4_instances::MmaR64T80K64Wr16Wt40S2A2B1>;
    if (tokens <= 96) return launch_mma<q4_instances::MmaR64T96K64Wr32Wt16S2A2B1>;
    if (tokens <= 120)
        return tokens % 120 == 0 ? launch_mma<q4_instances::MmaR64T120K64Wr16Wt120S2A2B1>
                                 : launch_mma<q4_instances::MmaR64T120K64Wr32Wt24S2A2B1>;
    if (tokens <= 128) return launch_mma<q4_instances::MmaR64T128K64Wr64Wt32S2A2B1>;
    if (tokens <= 192) return launch_mma<q4_instances::MmaR64T96K64Wr32Wt16S2A2B1>;
    if (tokens <= 224)
        return tokens % 112 == 0 ? launch_mma<q4_instances::MmaR64T112K64Wr16Wt112S2A2B1>
                                 : launch_mma<q4_instances::MmaR64T112K64Wr32Wt16S2A2B1>;
    if (tokens <= 240)
        return tokens % 120 == 0 ? launch_mma<q4_instances::MmaR64T120K64Wr16Wt120S2A2B1>
                                 : launch_mma<q4_instances::MmaR64T120K64Wr32Wt24S2A2B1>;
    if (tokens <= 256) return launch_mma<q4_instances::MmaR64T128K64Wr64Wt32S2A2B1>;
    if (tokens <= 288) return launch_mma<q4_instances::MmaR64T96K64Wr32Wt16S2A2B1>;
    return launch_mma<q4_instances::MmaR64T128K64Wr64Wt32S2A2B1>;
}
} // namespace

void launch_q4_swiglu_materialized_projection(const Tensor& x, const Weight& w, Tensor& out,
                                              cudaStream_t stream) {
    select_projection(x.ne[1])(x, w, out, stream);
}
} // namespace ninfer::ops::detail
