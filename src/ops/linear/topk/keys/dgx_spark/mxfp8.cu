#include "ops/linear/topk/keys/epilogue.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"

namespace ninfer::ops::detail::topk_keys {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 4096>;

template <class S, bool A8 = false>
void run(const Tensor& x, const Weight& w, int begin, int domain, const LinearTopKWorkspace& ws,
         Mxfp8A8Workspace activation, cudaStream_t stream) {
    if constexpr (A8) {
        launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), w.k, x.ne[1],
                                 activation, stream);
        launch_mxfp8_a8_tma<S>(mxfp8_a8_operands(w, activation, x.ne[1]), output(ws, begin, domain),
                               PanelEpilogue<64>{}, stream);
    } else
        launch_native_fp8_a16_sliced_k<S>(mxfp8_a16_operands(x, w), output(ws, begin, domain),
                                          LinearIdentityEpilogue{}, stream);
}

using Small  = Static<NativeFp8A16SlicedKSchedule<16, 16, 8, 256, 1>>;
using Medium = Static<Mxfp8A8TmaSchedule<64, 64, 256, 32, 16, 2, 1, 8, false, 2>>;
using Large  = Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 32, 2, 1, 8, false, 2>>;
} // namespace

Plan select_mxfp8(int columns) {
    if (columns <= 16) return {run<Small>, Small::kBlockRows};
    if (columns <= 96) return {run<Medium, true>, Medium::kBlockRows, true};
    return {run<Large, true>, Large::kBlockRows, true};
}
} // namespace ninfer::ops::detail::topk_keys
