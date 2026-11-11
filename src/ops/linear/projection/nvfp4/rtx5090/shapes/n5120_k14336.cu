#include "ops/linear/projection/nvfp4/nvfp4_shapes.h"
#include "ops/linear/projection/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include <algorithm>
#include <array>

namespace ninfer::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<5120, 14336>;
using T32R32   = Nvfp4A4MmaSchedule<32, 32, 256, 2, 2, 4, 2>;
using T32R64   = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2>;
using T64R64   = Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 3, 1>;

struct SplitRange {
    int begin, end;
    Nvfp4A4Route route;
};

constexpr std::array kSplitRanges{
    SplitRange{
        129, 256,
        nvfp4_a4_tma_split_route<Nvfp4GeometryId::N5120K14336, Nvfp4ScaleLayout::Tiled128>()},
    SplitRange{
        257, 512,
        nvfp4_a4_tma_split_route<Nvfp4GeometryId::N5120K14336, Nvfp4ScaleLayout::Tiled256>()},
    SplitRange{
        1025, 1536,
        nvfp4_a4_tma_split_route<Nvfp4GeometryId::N5120K14336, Nvfp4ScaleLayout::Tiled256>()},
};

Nvfp4Launch select_a16(int tokens) {
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 4, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 48) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    for (const auto& range : kSplitRanges)
        if (tokens >= range.begin && tokens <= range.end) return range.route;
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, T32R32>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T64R64>();
    return nvfp4_a4_tma_route<Nvfp4GeometryId::N5120K14336>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 5; }

std::size_t partial_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens) {
    std::size_t bytes = 0;
    for (const auto& range : kSplitRanges) {
        if (min_tokens > range.end || max_tokens < range.begin) continue;
        bytes = std::max(bytes, nvfp4_a4_split_k_partial_bytes(Geometry::kOutputRows,
                                                               std::min(max_tokens, range.end),
                                                               kNvfp4LinearTmaKPartitions));
    }
    return bytes;
}
} // namespace

const Nvfp4LinearShape kNvfp4N5120K14336{
    5120, 14336, launch_a16, launch_nvfp4_a4<select_a4>, uses_a4, partial_capacity_bytes};
} // namespace ninfer::ops::detail
