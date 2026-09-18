#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Bf16Geometry<5120, 6144>;
using Gemv = Bf16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                              Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
using Mma  = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                             Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using C2   = Bf16SimtSchedule<4, 1, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C4   = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C8   = Bf16SimtSchedule<4, 1, 4, 16, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C12  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 2>;
using C16  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 4>;
using C20  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C24  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 4>;
using C28  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 4>;
using C32  = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;

} // namespace

Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return launch_bf16_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_bf16_simt<Geometry, 4, C4>;
    if (tokens <= 8) return launch_bf16_simt<Geometry, 8, C8>;
    if (tokens <= 12) return launch_bf16_simt<Geometry, 12, C12>;
    if (tokens <= 16) return launch_bf16_simt<Geometry, 16, C16>;
    if (tokens <= 20) return launch_bf16_simt<Geometry, 20, C20>;
    if (tokens <= 24) return launch_bf16_simt<Geometry, 24, C24>;
    if (tokens <= 28) return launch_bf16_simt<Geometry, 28, C28>;
    if (tokens <= 32) return launch_bf16_simt<Geometry, 32, C32>;
    return launch_bf16_mma<Geometry, Mma>;
}

} // namespace ninfer::ops::detail
