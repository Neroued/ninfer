#include "ops/softmax_attention/relative/attention/rtx5090/tiled.cuh"

namespace ninfer::ops::relative {
namespace {

constexpr int kRaggedGroup = 1;
constexpr int kRaggedWarps = kRaggedRows / 16 * kRaggedGroup;

__global__ void build_tiles(const int* prefix, int segments, RaggedTile* tiles, int count) {
    const auto i = blockIdx.x * std::uint64_t(blockDim.x) + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    int lower = 0, upper = segments;
    const auto tile_begin = [&](int seq) { return ragged_tiles(prefix[seq], seq); };
    while (lower + 1 < upper) {
        const int mid = lower + (upper - lower) / 2;
        if (tile_begin(mid) <= i)
            lower = mid;
        else
            upper = mid;
    }
    const auto first = prefix[lower] + (i - tile_begin(lower)) * kRaggedRows;
    const int end    = prefix[lower + 1];
    tiles[i]         = {static_cast<int>(min(first, std::uint64_t(end))),
                first < static_cast<unsigned>(end) ? min(kRaggedRows, end - int(first)) : 0, lower,
                        prefix[lower]};
}

template <bool Swa, bool Fp8>
__global__ __launch_bounds__(kRaggedWarps * 32) void relative_ragged(Query q, CacheView cache,
                                                                     int tokens, int tile_count,
                                                                     const RaggedTile* tiles) {
    constexpr int Warps = kRaggedWarps;
    constexpr int Keys  = kKeyTile;
    __shared__ TileStorage<Warps, true, Fp8, Keys, kRaggedRows, Swa> smem;
    const int index = tile_count - 1 - blockIdx.x;
    const int first = index * kRaggedRows;
    const auto tile =
        tiles ? tiles[index] : RaggedTile{first, min(kRaggedRows, tokens - first), 0, 0};
    if (!tile.count) return;
    tiled_attention<Fp8, Swa, true, Warps, Warps * 32, true, Keys, kRaggedRows, kRaggedGroup>(
        q, cache, tile.sequence, tile.sequence_first, tile.first, tile.count, tile.count,
        blockIdx.y, 0, 1, {}, 0, smem);
}

} // namespace

void launch_ragged(Query q, CacheView cache, bool swa, KvCacheStorage storage, const int* prefix,
                   int tokens, int segments, RaggedTile* descriptors, int tiles,
                   cudaStream_t stream) {
    if (descriptors) {
        build_tiles<<<(unsigned(tiles) + 127) / 128, 128, 0, stream>>>(prefix, segments,
                                                                       descriptors, tiles);
        CUDA_CHECK(cudaGetLastError());
    }
    const dim3 grid(tiles, kHeads / kRaggedGroup);
    const bool fp8    = storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    const auto launch = [&]<bool Swa, bool Fp8>() {
        relative_ragged<Swa, Fp8>
            <<<grid, kRaggedWarps * 32, 0, stream>>>(q, cache, tokens, tiles, descriptors);
    };
    if (swa) {
        if (fp8)
            launch.template operator()<true, true>();
        else
            launch.template operator()<true, false>();
    } else {
        if (fp8)
            launch.template operator()<false, true>();
        else
            launch.template operator()<false, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::relative
