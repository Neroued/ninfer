#include "ops/linear/pair/dispatch.h"
#include "ops/linear/pair/bf16/rtx5090/small_launch.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n1024_k2304.cuh"

namespace ninfer::ops::detail {
namespace {

using bf16_n1024_k2304::Mma16;
using bf16_n1024_k2304::StaticK;
using bf16_n1024_k2304::Tma32;
using bf16_n1024_k2304::Tma48;
using bf16_n1024_k2304::Tma80;
using PairGemv =
    StaticK<Bf16A16GemvSchedule<4, 1, 1, 8, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 2, 4>>;
using PairSimt = StaticK<
    Bf16A16SimtSchedule<4, 1, 2, 8, 4, 4, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2, 4>>;
using PairSliced = StaticK<Bf16A16SlicedKMmaSchedule<16, 8, 3, 128, 2, Cache::cg, Cache::cg, 1, 1>>;
using NarrowTma32 = StaticK<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 3>>;
using PairTma64 =
    StaticK<Bf16A16TmaMmaSchedule<64, 64, 64, 16, 32, 3, 1, Bf16MmaRaster::Grouped, 4>>;
using PairTma96 =
    StaticK<Bf16A16TmaMmaSchedule<64, 96, 64, 32, 24, 3, 1, Bf16MmaRaster::Grouped, 8>>;
using NarrowTma112 =
    StaticK<Bf16A16TmaMmaSchedule<64, 112, 64, 32, 56, 3, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using PairTma112 =
    StaticK<Bf16A16TmaMmaSchedule<128, 112, 64, 32, 56, 3, 1, Bf16MmaRaster::Grouped, 4, 3>>;
using PairTma128 =
    StaticK<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 64, 2, 1, Bf16MmaRaster::Grouped, 4>>;

using Launch = void (*)(Bf16A16Operands, Bf16A16Operands, LinearBf16Output, LinearBf16Output,
                        cudaStream_t);

Launch select(int tokens) {
    using WarpStore   = Bf16MmaWarpStoreEpilogue;
    using VectorStore = Bf16TmaVectorEpilogue;
    if (tokens == 1) return launch_bf16_pair_gemv<PairGemv>;
    if (tokens <= 8) return launch_bf16_pair_simt<PairSimt>;
    if (tokens <= 16) return launch_bf16_pair_sliced<PairSliced>;
    if (tokens <= 32) return launch_bf16_pair_mma<Mma16, LinearIdentityEpilogue>;
    if (tokens <= 64) return launch_bf16_pair_tma<NarrowTma32, LinearIdentityEpilogue>;
    if (tokens <= 128) return launch_bf16_pair_tma<Tma32, WarpStore>;
    // Reuse tile sizes at successive CTA wave boundaries on the paired grid.
    if (tokens <= 192) return launch_bf16_pair_tma<Tma48, WarpStore>;
    if (tokens <= 320) return launch_bf16_pair_tma<PairTma64, WarpStore>;
    if (tokens <= 480) return launch_bf16_pair_tma<PairTma96, VectorStore>;
    if (tokens <= 560) return launch_bf16_pair_tma<NarrowTma112, WarpStore>;
    if (tokens <= 640) return launch_bf16_pair_tma<PairTma64, WarpStore>;
    if (tokens <= 800) return launch_bf16_pair_tma<Tma80, WarpStore>;
    if (tokens <= 960) return launch_bf16_pair_tma<PairTma96, VectorStore>;
    if (tokens <= 1120) return launch_bf16_pair_tma<PairTma112, WarpStore>;
    if (tokens <= 1280) return launch_bf16_pair_tma<PairTma128, WarpStore>;
    if (tokens <= 1440) return launch_bf16_pair_tma<PairTma96, VectorStore>;
    if (tokens <= 1680) return launch_bf16_pair_tma<Tma80, WarpStore>;
    if (tokens <= 1920) return launch_bf16_pair_tma<PairTma128, WarpStore>;
    if (tokens <= 2352) return launch_bf16_pair_tma<PairTma112, WarpStore>;
    if (tokens <= 2560) return launch_bf16_pair_tma<PairTma128, WarpStore>;
    if (tokens <= 2816) return launch_bf16_pair_tma<PairTma64, WarpStore>;
    if (tokens <= 3840) return launch_bf16_pair_tma<PairTma128, WarpStore>;
    return launch_bf16_pair_tma<Tma80, WarpStore>;
}

} // namespace

void launch_bf16_pair_n1024_k2304(const Tensor& x0, const Tensor& x1, const Weight& w0,
                                  const Weight& w1, Tensor& y0, Tensor& y1, cudaStream_t stream) {
    const auto a = bf16_a16_operands(x0, w0), b = bf16_a16_operands(x1, w1);
    const LinearBf16Output first{static_cast<__nv_bfloat16*>(y0.data), a.rows};
    const LinearBf16Output second{static_cast<__nv_bfloat16*>(y1.data), b.rows};
    select(a.tokens)(a, b, first, second, stream);
}

} // namespace ninfer::ops::detail
