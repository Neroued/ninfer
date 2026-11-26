#include "ops/softmax_attention/relative/attention/rtx5090/tiled.cuh"

namespace ninfer::ops::relative {
namespace {

template <int Rows, int Heads = kGroup>
inline constexpr int kRaggedWarps = (Rows * Heads + 15) / 16;

template <bool Fp8, int Rows, int Keys, int Heads, bool RegisterBias = false>
using RaggedStorage =
    TileStorage<kRaggedWarps<Rows, Heads>, true, Fp8, Keys, Rows, Heads != 1, RegisterBias>;

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

template <bool Swa, bool Fp8, int Rows, int Keys, int Heads, bool RegisterBias = false>
__global__ __launch_bounds__(kRaggedWarps<Rows, Heads> *
                             32) void relative_ragged(Query q, CacheView cache, int tokens,
                                                      int tile_count, const RaggedTile* tiles) {
    using Storage = RaggedStorage<Fp8, Rows, Keys, Heads, RegisterBias>;
    Storage* smem;
    if constexpr (Fp8 || sizeof(Storage) > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char memory[];
        smem = reinterpret_cast<Storage*>(memory);
    } else {
        __shared__ Storage storage;
        smem = &storage;
    }
    const int index = tile_count - 1 - blockIdx.x;
    const int first = index * kRaggedRows;
    const auto tile =
        tiles ? tiles[index] : RaggedTile{first, min(kRaggedRows, tokens - first), 0, 0};
    constexpr int HeadTiles = kHeads / Heads;
    const int offset        = Rows == kRaggedRows ? 0 : (blockIdx.y / HeadTiles) * Rows;
    const int head          = Rows == kRaggedRows ? blockIdx.y : blockIdx.y % HeadTiles;
    const int count         = min(Rows, tile.count - offset);
    if (count <= 0) return;
    constexpr int Warps = kRaggedWarps<Rows, Heads>;
    const auto start    = std::int64_t(tile.first) + offset;
    if constexpr (RegisterBias) {
        // Inspect device positions so graph replay can still use irregular inputs.
        const std::int64_t low = q.positions[start];
        const bool consecutive =
            low >= 0 && low + count - 1 <= position_limit<Swa>(cache) &&
            (threadIdx.x >= count || q.positions[start + threadIdx.x] == low + threadIdx.x);
        if (__syncthreads_and(consecutive)) {
            tiled_attention<Fp8, Swa, true, Warps, Warps * 32, true, Keys, Rows, Heads,
                            RegisterBias, true>(q, cache, tile.sequence, tile.sequence_first, start,
                                                count, count, head, 0, 1, {}, 0, *smem);
            return;
        }
    }
    tiled_attention<Fp8, Swa, true, Warps, Warps * 32, true, Keys, Rows, Heads, RegisterBias>(
        q, cache, tile.sequence, tile.sequence_first, start, count, count, head, 0, 1, {}, 0,
        *smem);
}

} // namespace

void launch_ragged(Query q, CacheView cache, bool swa, KvCacheStorage storage, const int* prefix,
                   int tokens, int segments, RaggedTile* descriptors, int tiles, int sms,
                   cudaStream_t stream) {
    if (descriptors) {
        build_tiles<<<(unsigned(tiles) + 127) / 128, 128, 0, stream>>>(prefix, segments,
                                                                       descriptors, tiles);
        CUDA_CHECK(cudaGetLastError());
    }
    const bool fp8    = storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    const auto launch = [&]<bool Swa, bool Fp8>() {
        const auto submit = [&]<int Rows, int Keys, int Heads, bool RegisterBias = false>() {
            static_assert(kRaggedRows % Rows == 0);
            constexpr int bytes  = sizeof(RaggedStorage<Fp8, Rows, Keys, Heads, RegisterBias>);
            constexpr int shared = Fp8 || bytes > 48 * 1024 ? bytes : 0;
            const dim3 grid(tiles, kHeads / Heads * (kRaggedRows / Rows));
            if constexpr (shared > 48 * 1024)
                CUDA_CHECK(
                    cudaFuncSetAttribute(relative_ragged<Swa, Fp8, Rows, Keys, Heads, RegisterBias>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
            relative_ragged<Swa, Fp8, Rows, Keys, Heads, RegisterBias>
                <<<grid, kRaggedWarps<Rows, Heads> * 32, shared, stream>>>(q, cache, tokens, tiles,
                                                                           descriptors);
        };
        if constexpr (Fp8) {
            if constexpr (!Swa) {
                // Long paged histories benefit from more query reuse once the
                // wider tiles supply at least two waves of independent CTAs.
                if (std::int64_t(tokens) * kKvHeads >= 2 * std::int64_t(sms) * 32) {
                    submit.template operator()<32, 32, kGroup>();
                    return;
                }
            }
            submit.template operator()<16, 32, kGroup>();
        } else {
            // Long Full tiles spend most of their time beyond the support of the relative
            // bias. Keep their streaming path, and retain grouped tiles for very small grids.
            const int min_independent_ctas = Swa ? sms / (2 * kGroup) : sms;
            if (std::int64_t(tiles) * kHeads < min_independent_ctas) {
                submit.template operator()<16, 64, kGroup>();
            } else if (Swa || std::int64_t(cache.stride) * kPagedKVPageSize <= 8 * kExtent) {
                submit.template operator()<kRaggedRows, kKeyTile, 1, true>();
            } else {
                submit.template operator()<kRaggedRows, kKeyTile, 1>();
            }
        }
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
