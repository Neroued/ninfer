#include "ops/linear/topk/keys/epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail::topk_keys {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 4096>;
using Small  = Static<Bf16A16SlicedKMmaSchedule<16, 16, 16, 32, 2, Cache::cg, Cache::cg, 1, 7>>;

template <class S>
void run(const Tensor& x, const Weight& w, int begin, int domain, const LinearTopKWorkspace& ws,
         Mxfp8A8Workspace, cudaStream_t stream) {
    if constexpr (S::kBlockRows == 16)
        launch_bf16_a16_sliced_k_mma<S>(bf16_a16_operands(x, w), output(ws, begin, domain),
                                        LinearIdentityEpilogue{}, stream);
    else
        launch_bf16_a16_sliced_k_mma<S>(bf16_a16_operands(x, w), output(ws, begin, domain),
                                        SlicedEpilogue{}, stream);
}

using Medium = Static<Bf16A16SlicedKMmaSchedule<32, 16, 8, 64, 1, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
using Large  = Static<Bf16A16SlicedKMmaSchedule<64, 32, 8, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                Bf16MmaRaster::TokenFast>>;
} // namespace

Plan select_bf16(int columns) {
    if (columns <= 16) return {run<Small>, Small::kBlockRows};
    if (columns <= 32) return {run<Medium>, Medium::kBlockRows};
    return {run<Large>, Large::kBlockRows};
}
} // namespace ninfer::ops::detail::topk_keys
