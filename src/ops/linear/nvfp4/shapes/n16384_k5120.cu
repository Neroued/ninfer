#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<16384, 5120>;
using Gemv =
    Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
template <int Tokens>
using Exact     = Nvfp4SimtSchedule<4, 1, 2, (Tokens >= 17 && Tokens <= 20) ? 8 : 16, Tokens, 1,
                                    Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct,
                                    Nvfp4CodeCache::Default, 1, Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C2        = Nvfp4SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C4        = Nvfp4SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C32       = Nvfp4SimtSchedule<4, 1, 2, 16, 8, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::TokenTilesContiguous, 4>;
using FullChunk = Nvfp4SimtSchedule<4, 1, 2, 16, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using T32R64    = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128   = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R128   = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using T128R128Resident  = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_nvfp4_gemv<Geometry, Gemv>;
    if (tokens == 32) return launch_nvfp4_simt<Geometry, 32, FullChunk, true>;
    if (tokens >= 5 && tokens <= 28) return select_nvfp4_exact<Geometry, 5, 28, Exact>(tokens);
    if (tokens <= 2) return launch_nvfp4_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return launch_nvfp4_simt<Geometry, 4, C4, false>;
    if (tokens <= 32) return launch_nvfp4_simt<Geometry, 32, C32, false>;
    throw std::logic_error("nvfp4 A16 chunk exceeds shape capacity");
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 1024) return nvfp4_a4_tma_route<Nvfp4GeometryId::N16384K5120>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 96) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t) { return true; }
} // namespace

const Nvfp4LinearShape kNvfp4N16384K5120{16384, 5120, launch_nvfp4_a16_chunks<32, select_a16>,
                                         launch_nvfp4_a4<select_a4>, uses_a4};
} // namespace ninfer::ops::detail
