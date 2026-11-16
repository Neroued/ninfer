#include "ops/linear/topk/keys/epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail::topk_keys {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 4096>;
using Small  = Static<Bf16A16SlicedKMmaSchedule<16, 16, 16, 32, 2, Cache::cg, Cache::cg, 1, 7>>;

template <class S>
void run(const Tensor& x, const Weight& w, int begin, int domain, const Workspace& ws,
         cudaStream_t stream) {
    if constexpr (S::kBlockRows == 16)
        launch_bf16_a16_sliced_k_mma<S>(bf16_a16_operands(x, w),
                                        output(ws.candidates, begin, domain),
                                        LinearIdentityEpilogue{}, stream);
    else
        launch_bf16_a16_sliced_k_mma<S>(bf16_a16_operands(x, w),
                                        output(ws.candidates, begin, domain), SlicedEpilogue{},
                                        stream);
}

using Medium  = Static<Bf16A16SlicedKMmaSchedule<32, 16, 8, 64, 1, Cache::cg, Cache::cg, 1, 7,
                                                 Bf16MmaRaster::TokenFast>>;
using Large   = Static<Bf16A16SlicedKMmaSchedule<64, 32, 8, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                                 Bf16MmaRaster::TokenFast>>;
using Bulk    = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1>>;
using Prefill = Static<Bf16RowTailSchedule<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1>>>;

template <class S>
void tma(const Tensor& x, const Weight& w, int begin, int domain, const Workspace& ws,
         cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w), output(ws.candidates, begin, domain),
                               TopKPanelEpilogue<64, true>{}, stream);
}
} // namespace

Plan select_bf16(int columns) {
    if (columns <= 16) return {run<Small>, Small::kBlockRows};
    if (columns <= 32) return {run<Medium>, Medium::kBlockRows};
    if (columns <= 128) return {run<Large>, Large::kBlockRows};
    if (columns <= 256) return {run<Large>, Large::kBlockRows, false, nullptr, 8192};
    if (columns <= 512) return {tma<Bulk>, Bulk::kBlockRows, false, nullptr, 8192};
    return {tma<Prefill>, Prefill::kBlockRows, false, nullptr, 8192};
}
} // namespace ninfer::ops::detail::topk_keys
