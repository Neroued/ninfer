#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Bf16Geometry<14336, 5120>;
using Gemv = Bf16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                              Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using Mma  = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                             Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using C2   = Bf16SimtSchedule<4, 1, 8, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C4   = Bf16SimtSchedule<4, 1, 8, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C8   = Bf16SimtSchedule<4, 1, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C12  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 2>;
using C16  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 4>;
using C20  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 3>;
using C24  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 3>;

} // namespace

Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return launch_bf16_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_bf16_simt<Geometry, 4, C4>;
    if (tokens <= 8) return launch_bf16_simt<Geometry, 8, C8>;
    if (tokens <= 12) return launch_bf16_simt<Geometry, 12, C12>;
    if (tokens <= 16) return launch_bf16_simt<Geometry, 16, C16>;
    if (tokens <= 20) return launch_bf16_simt<Geometry, 20, C20>;
    if (tokens <= 24) return launch_bf16_simt<Geometry, 24, C24>;
    return launch_bf16_mma<Geometry, Mma>;
}

} // namespace ninfer::ops::detail
