#include "ops/linear/topk/keys/epilogue.cuh"
#include "ops/linear/kernels/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/kernels/q4/q4_a16_tma_prefetch.cuh"

namespace ninfer::ops::detail::topk_keys {
namespace {
using Small = Q4A16SlicedKMmaSchedule<16, 16, 8, 2, Cache::cg, Cache::ca, 2, 4096, 16>;

template <class S, int Panel = 64, bool Sliced = false>
void run(const Tensor& x, const Weight& w, int begin, int domain, const LinearTopKWorkspace& ws,
         Mxfp8A8Workspace, cudaStream_t stream) {
    if constexpr (Sliced)
        launch_q4_a16_sliced_k_mma<S>(q4_linear_operands(x, w), output(ws, begin, domain),
                                      LinearIdentityEpilogue{}, stream);
    else
        launch_q4_a16_tma_prefetch<S>(q4_linear_operands(x, w), output(ws, begin, domain),
                                      PanelEpilogue<Panel>{}, stream);
}

using Mid16 = Q4A16TmaPrefetchSchedule<64, 16, 16, 8, 2, 4096, 1, 16, 4, 1, 128, 2>;
using Mid32 = Q4A16TmaPrefetchSchedule<64, 32, 32, 8, 2, 4096, 1, 16, 4, 1, 128, 2>;
using Mid64 = Q4A16TmaPrefetchSchedule<128, 64, 128, 8, 2, 4096, 1, 16, 4, 1, 64, 1>;
using Mid96 = Q4A16TmaPrefetchSchedule<64, 96, 32, 24, 2, 4096, 1, 16, 4, 1, 128, 1>;
using Large = Q4A16TmaPrefetchSchedule<64, 128, 64, 16, 2, 4096, 1>;
} // namespace

Plan select_q4(int columns) {
    if (columns <= 8) return {run<Small, 16, true>, Small::kBlockRows};
    if (columns <= 16) return {run<Mid16, 16>, Mid16::kBlockRows};
    if (columns <= 32) return {run<Mid32>, Mid32::kBlockRows};
    if (columns <= 64) return {run<Mid64, 16>, Mid64::kBlockRows};
    if (columns <= 96) return {run<Mid96, 128>, Mid96::kBlockRows};
    return {run<Large, 128>, Large::kBlockRows};
}
} // namespace ninfer::ops::detail::topk_keys
