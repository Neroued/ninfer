#pragma once

#include "core/device.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail::loop_message {

constexpr int kHidden         = 2304;
constexpr int kRank           = 64;
constexpr int kUpFeatures     = 64;
constexpr int kUpTiles        = kHidden / kUpFeatures;
constexpr int kDecodeFeatures = 32;
constexpr int kDecodeTiles    = kHidden / kDecodeFeatures;
constexpr int kDecodeSplits   = 2;
constexpr int kMediumSplits   = 18;
constexpr int kShortSplits    = 8;
constexpr int kLongSplits     = 4;
constexpr int kShortRankTile  = 32;

enum class Route { Decode, Medium, PrefillShort, Cluster4, Cluster8, ClusterWide, Vector };

struct Schedule {
    Route route;
    int tokens;
    int streams;
};

struct ScratchLayout {
    std::size_t bytes      = 0;
    std::size_t projection = 0;
    std::size_t inv        = 0;
    std::size_t blended    = 0;
    std::size_t sums       = 0;
    std::size_t done       = 0;
    std::size_t latent     = 0;
    int sum_tiles          = 0;
};

struct Arguments {
    const float* dest;
    const float* message;
    const __nv_bfloat16* down;
    const __nv_bfloat16* up;
    const __nv_bfloat16* gamma;
    __nv_bfloat16* out;
    __nv_bfloat16* latent;
    int hidden, rank, tokens, streams;
    bool broadcast_dest, broadcast_gamma;
    float pair_eps, norm_eps;
};

struct Chunk {
    Arguments args;
    int first_token, first_stream;
    int tokens, streams, sum_tiles;
    float* projection;
    float2* inv;
    float* blended;
    float2* sums;
    unsigned* done;
};

void launch_decode(Chunk a, cudaStream_t stream);
void launch_normalize(Chunk a, cudaStream_t stream);
void launch_prefill(Chunk a, Route route, cudaStream_t stream);
void launch_generic(Arguments a, cudaStream_t stream);
void launch_cluster_prefill(Arguments a, Route route, cudaStream_t stream);

} // namespace ninfer::ops::detail::loop_message
