// Implements: include/ninfer/ops/sparse_attention.h
// Qwen Sparse Attention: paged append with pooled block-key publication, indexer block selection
// and selected-key attention.
//
// Provenance: the append codecs and the Tensor Core selected-key attention are ported from
// giveen/ninfer-ext (src/ops/sparse_attention/kernels.cu @9c933f01, Apache-2.0). The pooled
// block key published at append time follows ninfer6000 (flash_next_qsa.cu, compressed index
// groups); the column-tiled scorer and the three-pass radix top-k are written for this fork.
#include "ops/sparse_attention/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/launcher/kernel_attr_once.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim      = 256;
constexpr int kIndexDim     = kQsaIndexKeyDim;
constexpr int kRecord       = kQsaIndexRecordWords;
constexpr int kBlockWords   = kQsaBlockKeyWords;
constexpr int kMaxIndexHead = 4; // registered indexer: 4 heads
constexpr int kMaxPairs     = kIndexDim / 2;

__device__ __forceinline__ std::int32_t lane_valid(const std::int32_t* valid, std::int32_t lane,
                                                   std::int32_t width) {
    if (valid == nullptr) { return width; }
    return max(0, min(width, valid[lane]));
}

// Kernel-side indexer geometry: the FP32 inverse-frequency table is computed on the host.
struct IndexGeometry {
    std::int32_t index_heads;
    std::int32_t ratio;
    std::int32_t block_topk;
    std::int32_t rotary_dim;
    float rms_norm_eps;
    float inv_freq[kMaxPairs];
    std::uint8_t pair_axes[kMaxPairs];
};

IndexGeometry index_geometry(const QsaGeometry& g) {
    IndexGeometry out{};
    out.index_heads  = g.index_heads;
    out.ratio        = g.ratio;
    out.block_topk   = g.block_topk();
    out.rotary_dim   = g.rotary_dim;
    out.rms_norm_eps = g.rms_norm_eps;
    for (int i = 0; i < g.rotary_dim / 2; ++i) {
        // Hugging Face: 1 / theta ** (arange(0, dim, 2).float() / dim), all in FP32.
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(g.rotary_dim);
        out.inv_freq[i]      = 1.0F / std::pow(g.rope_theta, exponent);
        out.pair_axes[i]     = g.pair_axes[static_cast<std::size_t>(i)];
    }
    return out;
}

__device__ __forceinline__ std::int32_t read_record_position(const __nv_bfloat16* record,
                                                             int axis) {
    const auto* words = reinterpret_cast<const std::uint16_t*>(record + kIndexDim);
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(words[2 * axis]) |
                                     (static_cast<std::uint32_t>(words[2 * axis + 1]) << 16));
}

template <typename T>
__device__ __forceinline__ T* record_at(T* records, const std::int32_t* table, std::int32_t token) {
    const std::int64_t page = paged_kv_physical_page(table, token);
    return records + (page * kPagedKVPageSize + (token & kPagedKVPageMask)) * kRecord;
}

// The 128 words of block b live in the four token slots of the block (ratio 4, 32 words each);
// a block never straddles a page.
__device__ __forceinline__ std::int64_t block_key_offset(const std::int32_t* table,
                                                         std::int32_t block, std::int32_t ratio) {
    const std::int32_t token = block * ratio;
    const std::int64_t page  = paged_kv_physical_page(table, token);
    return (page * kPagedKVPageSize + (token & kPagedKVPageMask)) * kBlockWords;
}

// Rotates the warp-held 128-feature row (lane owns features 4*lane .. 4*lane+3) in place.
// Pair i rotates features (i, i + rotary/2) with angle positions[axis_i] * inv_freq_i.
__device__ __forceinline__ void rope_row(float (&x)[4], const IndexGeometry& g,
                                         const std::int32_t (&positions)[3], int lane) {
    const int half  = g.rotary_dim / 2;
    const int shift = half / 4; // lanes between pair members
    float other[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) { other[r] = __shfl_xor_sync(0xffffffffu, x[r], shift); }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int d = 4 * lane + r;
        if (d < g.rotary_dim) {
            const int pair    = d < half ? d : d - half;
            const float angle = static_cast<float>(positions[g.pair_axes[pair]]) * g.inv_freq[pair];
            float s, c;
            sincosf(angle, &s, &c);
            x[r] = d < half ? x[r] * c - other[r] * s : x[r] * c + other[r] * s;
        }
    }
}

__device__ __forceinline__ void offset_rmsnorm_row(float (&x)[4], const __nv_bfloat16* weight,
                                                   float eps, int lane) {
    float sum = 0.0F;
#pragma unroll
    for (int r = 0; r < 4; ++r) { sum += x[r] * x[r]; }
    const float inv = rsqrtf(warp_sum(sum) / static_cast<float>(kIndexDim) + eps);
#pragma unroll
    for (int r = 0; r < 4; ++r) { x[r] *= inv * (1.0F + __bfloat162float(weight[4 * lane + r])); }
}

// ---------------------------------------------------------------------------------------------
// Cache codecs (ninfer-ext): every quantized profile stores K Hadamard-rotated; V is rotated where
// its codec is NVFP4.

enum class Codec { Fp8Row, Int8G64, Nvfp4G16 };

struct ProfileCodecs {
    Codec k;
    Codec v;
    bool v_rotated;
};

__host__ __device__ constexpr ProfileCodecs profile_codecs(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::Int8Group64:
        return {Codec::Int8G64, Codec::Int8G64, false};
    case KvCacheStorage::Nvfp4Group16:
        return {Codec::Nvfp4G16, Codec::Nvfp4G16, true};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {Codec::Fp8Row, Codec::Nvfp4G16, true};
    default:
        return {Codec::Fp8Row, Codec::Fp8Row, false};
    }
}

__host__ __device__ constexpr int code_bytes(Codec codec) {
    return codec == Codec::Nvfp4G16 ? 128 : 256;
}

// Encodes one row held by a warp as values[r] = x[lane + 32 r] into cache row `row`. NVFP4 groups
// gather through the warp's 256-float `scratch`.
template <Codec C>
__device__ __forceinline__ void encode_row(const float (&values)[8], std::int64_t row, void* codes,
                                           void* scales, float* scratch, int lane) {
    constexpr unsigned kMask = 0xffffffffu;
    if constexpr (C == Codec::Fp8Row) {
        float absmax = 0.0F;
#pragma unroll
        for (float x : values) { absmax = fmaxf(absmax, fabsf(x)); }
        const auto q = kv_cache_fp8_quant_params(warp_max(absmax, kMask));
        auto* c      = static_cast<std::uint8_t*>(codes) + row * kHeadDim;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            c[lane + 32 * r] = kv_cache_fp8_quant_code(values[r], q.inverse_scale);
        }
        if (lane == 0) { static_cast<__half*>(scales)[row] = q.scale; }
    } else if constexpr (C == Codec::Int8G64) {
        // Group g holds values[2g] (feature 64g + lane) and values[2g + 1] (feature 64g + 32 +
        // lane).
        auto* c = static_cast<std::int8_t*>(codes) + row * kHeadDim;
#pragma unroll
        for (int group = 0; group < kKVCacheInt8Groups; ++group) {
            const float x0 = values[2 * group], x1 = values[2 * group + 1];
            const auto q = kv_cache_int8_quant_params(warp_max(fmaxf(fabsf(x0), fabsf(x1)), kMask));
            c[group * kKVCacheInt8Group + lane] = kv_cache_int8_quant_code(x0, q.inverse_scale);
            c[group * kKVCacheInt8Group + lane + 32] =
                kv_cache_int8_quant_code(x1, q.inverse_scale);
            if (lane == 0) {
                static_cast<__half*>(scales)[row * kKVCacheInt8Groups + group] = q.scale;
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < 8; ++r) { scratch[lane + 32 * r] = values[r]; }
        __syncwarp();
        if (lane < kKVCacheNvfp4Groups) {
            const auto q = kv_cache_nvfp4_quantize_group16(scratch + lane * kKVCacheNvfp4Group);
            store_vec(static_cast<std::uint8_t*>(codes) + row * kKVCacheNvfp4CodeBytes + lane * 8,
                      make_uint2(q.codes_lo, q.codes_hi));
            static_cast<std::uint8_t*>(scales)[row * kKVCacheNvfp4Groups + lane] = q.scale;
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------------------------
// Append

constexpr int kAppendThreads = 256;

struct AppendArgs {
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const __nv_bfloat16* index_keys;
    const std::int32_t* rope_positions;
    std::int32_t rope_axes;
    const std::int32_t* cache;
    const std::int32_t* valid;
    const std::int32_t* table_rows;
    const std::int32_t* kv_tables;
    std::int32_t kv_table_stride;
    const std::int32_t* index_tables;
    std::int32_t index_table_stride;
    void* k_pages;
    void* v_pages;
    void* k_scales;
    void* v_scales;
    __nv_bfloat16* records;
    __nv_bfloat16* block_keys;
    const __nv_bfloat16* key_norm;
    std::int32_t kv_heads;
    std::int32_t width;
    std::int32_t lanes;
};

// One warp per (column, lane, kv head) plus one warp per (column, lane) for the index record.
template <KvCacheStorage Storage>
__global__ void __launch_bounds__(kAppendThreads) qsa_append_kernel(AppendArgs a) {
    constexpr bool kNvfp4 =
        Storage == KvCacheStorage::Nvfp4Group16 || Storage == KvCacheStorage::Fp8KeyNvfp4Value;
    __shared__ float scratch[kNvfp4 ? kAppendThreads / 32 : 1][kNvfp4 ? kHeadDim : 1];
    const int warp_global = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane_id     = static_cast<int>(threadIdx.x) & 31;
    const int units       = a.width * a.lanes * (a.kv_heads + 1);
    if (warp_global >= units) { return; }
    const int head   = warp_global % (a.kv_heads + 1);
    const int column = (warp_global / (a.kv_heads + 1)) % a.width;
    const int lane   = warp_global / ((a.kv_heads + 1) * a.width);
    if (column >= lane_valid(a.valid, lane, a.width)) { return; }
    const std::int64_t token    = static_cast<std::int64_t>(lane) * a.width + column;
    const std::int32_t position = a.cache[token];

    if (head == a.kv_heads) {
        const std::int32_t* table =
            a.index_tables + static_cast<std::int64_t>(a.table_rows[lane]) * a.index_table_stride;
        __nv_bfloat16* record = record_at(a.records, table, position);
        for (int d = lane_id; d < kIndexDim; d += 32) {
            record[d] = a.index_keys[token * kIndexDim + d];
        }
        auto* words = reinterpret_cast<std::uint16_t*>(record + kIndexDim);
        if (lane_id < 3) {
            const std::int64_t tokens = static_cast<std::int64_t>(a.width) * a.lanes;
            const std::int32_t p =
                a.rope_positions[(a.rope_axes == 3 ? lane_id : 0) * tokens + token];
            words[2 * lane_id] = static_cast<std::uint16_t>(static_cast<std::uint32_t>(p));
            words[2 * lane_id + 1] =
                static_cast<std::uint16_t>(static_cast<std::uint32_t>(p) >> 16);
        } else if (lane_id < 5) {
            words[3 + lane_id] = 0; // pad words 6 and 7
        }
        return;
    }

    const std::int32_t* table =
        a.kv_tables + static_cast<std::int64_t>(a.table_rows[lane]) * a.kv_table_stride;
    const std::int32_t page   = paged_kv_physical_page(table, position);
    const std::int32_t offset = position & kPagedKVPageMask;
    const std::int64_t source = (token * a.kv_heads + head) * kHeadDim;
    const std::int64_t row =
        (static_cast<std::int64_t>(page) * a.kv_heads + head) * kPagedKVPageSize + offset;
    if constexpr (Storage == KvCacheStorage::BFloat16) {
        auto* kb = static_cast<__nv_bfloat16*>(a.k_pages);
        auto* vh = static_cast<__half*>(a.v_pages);
        for (int d = lane_id; d < kHeadDim; d += 32) {
            kb[row * kHeadDim + d] = a.k[source + d];
            vh[row * kHeadDim + d] = __float2half_rn(__bfloat162float(a.v[source + d]));
        }
    } else {
        constexpr ProfileCodecs codecs = profile_codecs(Storage);
        float* warp_scratch            = scratch[kNvfp4 ? static_cast<int>(threadIdx.x) >> 5 : 0];
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = __bfloat162float(a.k[source + lane_id + 32 * r]);
        }
        normalized_hadamard_d256_inplace(values, lane_id);
        encode_row<codecs.k>(values, row, a.k_pages, a.k_scales, warp_scratch, lane_id);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = __bfloat162float(a.v[source + lane_id + 32 * r]);
        }
        if constexpr (codecs.v_rotated) { normalized_hadamard_d256_inplace(values, lane_id); }
        encode_row<codecs.v>(values, row, a.v_pages, a.v_scales, warp_scratch, lane_id);
    }
}

// One warp per column that completes a block: pool, normalize, rotate and publish its key.
__global__ void __launch_bounds__(32) qsa_publish_block_kernel(AppendArgs a, IndexGeometry g) {
    const int column  = static_cast<int>(blockIdx.x);
    const int lane_id = static_cast<int>(threadIdx.x);
    const int lane    = column / a.width;
    if (column % a.width >= lane_valid(a.valid, lane, a.width)) { return; }
    const std::int32_t position = a.cache[column];
    if (position % g.ratio != g.ratio - 1) { return; }
    const std::int32_t block = position / g.ratio;
    const std::int32_t* table =
        a.index_tables + static_cast<std::int64_t>(a.table_rows[lane]) * a.index_table_stride;

    float x[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    std::int32_t first_position[3];
    for (int t = 0; t < g.ratio; ++t) {
        const __nv_bfloat16* record =
            record_at(static_cast<const __nv_bfloat16*>(a.records), table, block * g.ratio + t);
        const uint2 raw         = *reinterpret_cast<const uint2*>(record + 4 * lane_id);
        const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        x[0] += __low2float(lo);
        x[1] += __high2float(lo);
        x[2] += __low2float(hi);
        x[3] += __high2float(hi);
        if (t == 0) {
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                first_position[axis] = read_record_position(record, axis);
            }
        }
    }
    const float inv_ratio = 1.0F / static_cast<float>(g.ratio);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        // Upstream rounds the pooled mean to the activation dtype before normalizing.
        x[r] = __bfloat162float(__float2bfloat16_rn(x[r] * inv_ratio));
    }
    offset_rmsnorm_row(x, a.key_norm, g.rms_norm_eps, lane_id);
    rope_row(x, g, first_position, lane_id);
    const __nv_bfloat162 lo = __floats2bfloat162_rn(x[0], x[1]);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(x[2], x[3]);
    uint2 packed;
    packed.x                               = *reinterpret_cast<const unsigned*>(&lo);
    packed.y                               = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(a.block_keys + block_key_offset(table, block, g.ratio) +
                              4 * lane_id) = packed;
}

// ---------------------------------------------------------------------------------------------
// Selection: score kernel

constexpr int kTileColumns    = 8;  // maximum columns of one lane per score CTA
constexpr int kScoreBlocks    = 64; // blocks per score iteration (one thread per block/column)
constexpr int kScoreKeyStride = kIndexDim / 2 + 1; // bf16x2 words per staged key, padded
constexpr int kScoreKeyBytes  = kScoreBlocks * kScoreKeyStride * 4;
// A full tile (prefill) scores four blocks of an iteration per thread (j, j + 16, ...) for one
// column, so one load of the prepared query feeds four dots (the kernel was bound by shared-memory
// loads: one key word and four query pairs per eight FMAs). A narrower tile (decode) keeps one
// block per thread: it has few columns, so it needs the threads. Every dot keeps its FMA order: the
// same bits.
constexpr int kScoreRegBlocksFull = 4;
static_assert(kScoreBlocks % kScoreRegBlocksFull == 0 && (kIndexDim / 2) % 2 == 0);

[[nodiscard]] __host__ __device__ constexpr int score_reg_blocks(int tile_columns) {
    return tile_columns >= kTileColumns ? kScoreRegBlocksFull : 1;
}

// Threads of a score CTA: kScoreBlocks / reg_blocks per column, at least one warp (the query
// preparation runs per warp).
__host__ __device__ constexpr int score_threads(int tile_columns) {
    const int lanes = kScoreBlocks / score_reg_blocks(tile_columns);
    return lanes * tile_columns < 32 ? 32 : lanes * tile_columns;
}

constexpr int kScoreTargetCtas = 512;
constexpr int kPassTiles       = 32;

struct SelectArgs {
    const __nv_bfloat16* index_query;
    const std::int32_t* query_positions;
    std::int32_t query_axes;
    const std::int32_t* cache;
    const std::int32_t* valid;
    const std::int32_t* table_rows;
    const std::int32_t* tables;
    std::int32_t table_stride;
    const __nv_bfloat16* block_keys;
    const __nv_bfloat16* query_norm;
    float* scores;
    std::int32_t* selected;
    std::int32_t* counts;
    std::int32_t width;
    std::int32_t lanes;
    std::int32_t max_selected;
    std::int32_t tile_columns;
    std::int32_t tiles_per_lane;
    std::int32_t tile_begin; // first global tile of this pass
    std::int32_t pass_tiles;
    std::int32_t max_blocks;
    std::int32_t chunk_blocks; // blocks per score CTA
};

struct TileColumn {
    std::int32_t lane;
    std::int32_t w;      // column within the lane
    std::int32_t column; // global column lane*W + w
    std::int32_t visible;
    std::int32_t blocks;
    bool live;
};

__device__ __forceinline__ TileColumn tile_column(const SelectArgs& a, std::int32_t tile,
                                                  std::int32_t c, std::int32_t ratio) {
    TileColumn out{};
    out.lane    = tile / a.tiles_per_lane;
    out.w       = (tile % a.tiles_per_lane) * a.tile_columns + c;
    out.column  = out.lane * a.width + out.w;
    out.live    = c < a.tile_columns && out.w < lane_valid(a.valid, out.lane, a.width);
    out.visible = out.live ? a.cache[out.column] + 1 : 0;
    out.blocks  = out.visible / ratio;
    return out;
}

// CTA (x, y): pass tile y, blocks [x * chunk_blocks, +chunk_blocks). Thread (j, c) of the
// kScoreBlocks x tile_columns grid scores block j of the current iteration for tile column c.
template <int kScoreRegBlocks>
__global__ void __launch_bounds__((kScoreBlocks / kScoreRegBlocks) * kTileColumns)
    qsa_score_kernel(SelectArgs a, IndexGeometry g) {
    constexpr int kScoreLanes = kScoreBlocks / kScoreRegBlocks; // threads per column
    // Dynamic: staged keys, then the prepared queries of this launch's tile_columns only, so a
    // narrow decode tile keeps several CTAs per SM.
    extern __shared__ __align__(16) std::uint8_t raw_shared[];
    auto* keys  = reinterpret_cast<std::uint32_t*>(raw_shared);
    auto* query = reinterpret_cast<float*>(raw_shared + kScoreKeyBytes);
    __shared__ std::int32_t block_count[kTileColumns];

    const int tid           = static_cast<int>(threadIdx.x);
    const int threads       = static_cast<int>(blockDim.x);
    const int warp          = tid >> 5;
    const int lane_id       = tid & 31;
    const std::int32_t tile = a.tile_begin + static_cast<std::int32_t>(blockIdx.y);
    const int lane          = tile / a.tiles_per_lane;
    const std::int32_t* table =
        a.tables + static_cast<std::int64_t>(a.table_rows[lane]) * a.table_stride;

    if (tid < kTileColumns) {
        const TileColumn col = tile_column(a, tile, tid, g.ratio);
        // Only columns that must choose among their blocks are scored.
        block_count[tid] = col.blocks > g.block_topk ? col.blocks : 0;
    }
    __syncthreads();
    std::int32_t tile_blocks = 0;
    for (int c = 0; c < a.tile_columns; ++c) { tile_blocks = max(tile_blocks, block_count[c]); }
    const std::int32_t first = static_cast<std::int32_t>(blockIdx.x) * a.chunk_blocks;
    const std::int32_t last  = min(tile_blocks, first + a.chunk_blocks);
    if (first >= last) { return; }

    // Prepared queries: offset RMSNorm then RoPE at the query position, one warp per (c, head).
    const int warps = threads >> 5;
    for (int item = warp; item < a.tile_columns * g.index_heads; item += warps) {
        const int c = item / g.index_heads;
        const int h = item % g.index_heads;
        if (block_count[c] == 0) { continue; }
        const TileColumn col      = tile_column(a, tile, c, g.ratio);
        const std::int64_t tokens = static_cast<std::int64_t>(a.width) * a.lanes;
        std::int32_t qpos[3];
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            qpos[axis] = a.query_positions[(a.query_axes == 3 ? axis : 0) * tokens + col.column];
        }
        const __nv_bfloat16* raw =
            a.index_query + (static_cast<std::int64_t>(col.column) * g.index_heads + h) * kIndexDim;
        float x[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) { x[r] = __bfloat162float(raw[4 * lane_id + r]); }
        offset_rmsnorm_row(x, a.query_norm, g.rms_norm_eps, lane_id);
        rope_row(x, g, qpos, lane_id);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            query[(c * kMaxIndexHead + h) * kIndexDim + 4 * lane_id + r] = x[r];
        }
    }

    const int j = tid % kScoreLanes;
    const int c = min(tid / kScoreLanes, kTileColumns - 1); // threads past the tile: staging only
    const float* qc   = query + c * kMaxIndexHead * kIndexDim;
    const float scale = rsqrtf(static_cast<float>(kIndexDim));
    float* scores_row =
        a.scores + (static_cast<std::int64_t>(blockIdx.y) * kTileColumns + c) * a.max_blocks;
    const std::int32_t my_blocks = tid / kScoreLanes < kTileColumns ? block_count[c] : 0;
    for (std::int32_t base = first; base < last; base += kScoreBlocks) {
        __syncthreads(); // queries ready / previous keys consumed
        // Stage up to 64 pooled keys, 16-byte vectors coalesced per 256-byte key.
        constexpr int kKeyVectors = kIndexDim * 2 / 16;
#pragma unroll 4
        for (int item = tid; item < kScoreBlocks * kKeyVectors; item += threads) {
            const int b = item / kKeyVectors;
            const int v = item % kKeyVectors;
            uint4 value = make_uint4(0, 0, 0, 0);
            if (base + b < last) {
                value = reinterpret_cast<const uint4*>(
                    a.block_keys + block_key_offset(table, base + b, g.ratio))[v];
            }
            std::uint32_t* staged = keys + b * kScoreKeyStride + 4 * v;
            staged[0]             = value.x;
            staged[1]             = value.y;
            staged[2]             = value.z;
            staged[3]             = value.w;
        }
        __syncthreads();
        const std::int32_t limit = min(last, my_blocks);
        if (base + j >= limit) {
            continue;
        } // its later blocks (j + kScoreLanes r) are past the limit too
        float dot[kScoreRegBlocks][kMaxIndexHead];
#pragma unroll
        for (int r = 0; r < kScoreRegBlocks; ++r)
#pragma unroll
            for (int h = 0; h < kMaxIndexHead; ++h) { dot[r][h] = 0.0F; }
#pragma unroll 4
        for (int d2 = 0; d2 < kIndexDim / 2; d2 += 2) {
            float2 k0[kScoreRegBlocks], k1[kScoreRegBlocks];
#pragma unroll
            for (int r = 0; r < kScoreRegBlocks; ++r) {
                const std::uint32_t* key = keys + (j + r * kScoreLanes) * kScoreKeyStride;
                const std::uint32_t w0 = key[d2], w1 = key[d2 + 1];
                k0[r] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w0));
                k1[r] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w1));
            }
#pragma unroll
            for (int h = 0; h < kMaxIndexHead; ++h) {
                if (h < g.index_heads) {
                    const float4 qq = *reinterpret_cast<const float4*>(qc + h * kIndexDim + 2 * d2);
#pragma unroll
                    for (int r = 0; r < kScoreRegBlocks; ++r) {
                        dot[r][h] = fmaf(qq.x, k0[r].x, dot[r][h]);
                        dot[r][h] = fmaf(qq.y, k0[r].y, dot[r][h]);
                        dot[r][h] = fmaf(qq.z, k1[r].x, dot[r][h]);
                        dot[r][h] = fmaf(qq.w, k1[r].y, dot[r][h]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kScoreRegBlocks; ++r) {
            const std::int32_t block = base + j + r * kScoreLanes;
            if (block >= limit) { continue; }
            float score = 0.0F;
#pragma unroll
            for (int h = 0; h < kMaxIndexHead; ++h) {
                if (h < g.index_heads) { score += fmaxf(dot[r][h], 0.0F); }
            }
            scores_row[block] = score * scale;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Selection: exact top-k of one column's block scores.

constexpr int kSelectThreads = 1024;
constexpr int kStagedBlocks  = 20480; // 80 KiB of staged scores (plus one pad word per 32)

// One pad word per 32 keeps both strided (radix passes) and contiguous-range (emission) reads of
// the staged keys free of shared-memory bank conflicts.
__device__ __forceinline__ std::int32_t staged_index(std::int32_t b) { return b + (b >> 5); }

constexpr std::size_t kStagedWords = kStagedBlocks + kStagedBlocks / 32;
constexpr int kRadixBins           = 2048;

// Scores are sums of ReLU terms, so they are non-negative; the order key still handles any sign.
__device__ __forceinline__ std::uint32_t order_key(float x) {
    const std::uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

struct SelectShared {
    using Scan = cub::BlockScan<int, kSelectThreads>;
    typename Scan::TempStorage scan;
    int histogram[kRadixBins];
    std::uint32_t prefix;
    int remaining;
};

// Radix digit passes: [31:21] 11 bits, [20:10] 11 bits, [9:0] 10 bits.
__device__ __forceinline__ void radix_pass(SelectShared& sh, const std::uint32_t* staged,
                                           const float* row, bool use_staged, std::int32_t blocks,
                                           int shift, int bits, std::uint32_t mask) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kRadixBins; i += kSelectThreads) { sh.histogram[i] = 0; }
    __syncthreads();
    const std::uint32_t prefix     = sh.prefix;
    const std::uint32_t digit_mask = (1u << bits) - 1u;
    // Scores of one column crowd a few leading digits, so lanes aggregate equal digits before the
    // shared-memory atomic (one atomic per distinct digit per warp).
    const auto count = [&](std::uint32_t key, bool live) {
        const bool counted    = live && (key & mask) == prefix;
        const unsigned voters = __ballot_sync(__activemask(), counted);
        if (counted) {
            const int digit      = static_cast<int>((key >> shift) & digit_mask);
            const unsigned peers = __match_any_sync(voters, digit);
            if ((tid & 31) == __ffs(peers) - 1) { atomicAdd(&sh.histogram[digit], __popc(peers)); }
        }
    };
    if (use_staged) {
        for (std::int32_t b = tid; b < blocks; b += kSelectThreads) {
            count(staged[staged_index(b)], true);
        }
    } else {
        // Unstaged rows (long contexts) are read from L2 four scores per load.
        for (std::int32_t b = 4 * tid; b < blocks; b += 4 * kSelectThreads) {
            const float4 v = reinterpret_cast<const float4*>(row)[b / 4];
            count(order_key(v.x), true);
            count(order_key(v.y), b + 1 < blocks);
            count(order_key(v.z), b + 2 < blocks);
            count(order_key(v.w), b + 3 < blocks);
        }
    }
    __syncthreads();
    // Thread t owns the two digits 2047-2t (hi) and 2046-2t (lo): an inclusive scan in thread
    // order counts the keys at or above each owned digit.
    const int hi_bin = kRadixBins - 1 - 2 * tid;
    const int hi     = hi_bin < (1 << bits) ? sh.histogram[hi_bin] : 0;
    const int lo     = hi_bin - 1 < (1 << bits) ? sh.histogram[hi_bin - 1] : 0;
    int inclusive    = hi + lo;
    SelectShared::Scan(sh.scan).InclusiveSum(inclusive, inclusive);
    const int before = inclusive - hi - lo; // keys above hi
    const int need   = sh.remaining;
    __syncthreads(); // every thread read `remaining`
    if (before < need && need <= before + hi) {
        sh.prefix    = prefix | (static_cast<std::uint32_t>(hi_bin) << shift);
        sh.remaining = need - before;
    } else if (before + hi < need && need <= before + hi + lo) {
        sh.prefix    = prefix | (static_cast<std::uint32_t>(hi_bin - 1) << shift);
        sh.remaining = need - before - hi;
    }
    __syncthreads();
}

// One CTA per pass column.
__global__ void __launch_bounds__(kSelectThreads) qsa_topk_kernel(SelectArgs a, IndexGeometry g) {
    extern __shared__ __align__(16) std::uint8_t raw_shared[];
    SelectShared& sh = *reinterpret_cast<SelectShared*>(raw_shared);
    auto* staged     = reinterpret_cast<std::uint32_t*>(raw_shared + sizeof(SelectShared));

    const int tid           = static_cast<int>(threadIdx.x);
    const int slot          = static_cast<int>(blockIdx.x); // pass column slot
    const std::int32_t c    = slot % kTileColumns;
    const std::int32_t tile = a.tile_begin + slot / kTileColumns;
    if (c >= a.tile_columns) { return; }
    const TileColumn col = tile_column(a, tile, c, g.ratio);
    if (col.w >= a.width) { return; }
    std::int32_t* selected = a.selected + static_cast<std::int64_t>(col.column) * a.max_selected;
    if (!col.live) {
        if (tid == 0) { a.counts[col.column] = 0; }
        return;
    }
    const std::int32_t ratio  = g.ratio;
    const std::int32_t blocks = col.blocks;
    const std::int32_t keep   = min(g.block_topk, blocks);
    const std::int32_t tail   = col.visible - blocks * ratio;
    if (blocks <= g.block_topk) {
        for (int i = tid; i < col.visible; i += kSelectThreads) { selected[i] = i; }
        if (tid == 0) { a.counts[col.column] = col.visible; }
        return;
    }
    const float* row      = a.scores + static_cast<std::int64_t>(slot) * a.max_blocks;
    const bool use_staged = blocks <= kStagedBlocks;
    if (use_staged) {
        for (std::int32_t b = tid; b < blocks; b += kSelectThreads) {
            staged[staged_index(b)] = order_key(row[b]);
        }
    }
    if (tid == 0) {
        sh.prefix    = 0;
        sh.remaining = keep;
    }
    __syncthreads();
    radix_pass(sh, staged, row, use_staged, blocks, 21, 11, 0u);
    radix_pass(sh, staged, row, use_staged, blocks, 10, 11, 0xFFE00000u);
    radix_pass(sh, staged, row, use_staged, blocks, 0, 10, 0xFFFFFC00u);
    const std::uint32_t threshold = sh.prefix;
    const int ties                = sh.remaining; // keys equal to threshold to admit, lowest first

    // Each thread owns a contiguous block range (a multiple of four, so an unstaged row is read
    // as aligned float4); greater keys and the first `ties` equal keys are emitted in ascending
    // block order.
    const std::int32_t per   = ((blocks + 4 * kSelectThreads - 1) / (4 * kSelectThreads)) * 4;
    const std::int32_t begin = min(blocks, tid * per);
    const std::int32_t end   = min(blocks, begin + per);
    const auto key_at        = [&](std::int32_t b) {
        return use_staged ? staged[staged_index(b)] : order_key(row[b]);
    };
    const auto for_range = [&](auto&& visit) {
        if (use_staged) {
            for (std::int32_t b = begin; b < end; ++b) { visit(b, key_at(b)); }
            return;
        }
        for (std::int32_t b = begin; b < end; b += 4) {
            const float4 v = reinterpret_cast<const float4*>(row)[b / 4];
            visit(b, order_key(v.x));
            if (b + 1 < end) { visit(b + 1, order_key(v.y)); }
            if (b + 2 < end) { visit(b + 2, order_key(v.z)); }
            if (b + 3 < end) { visit(b + 3, order_key(v.w)); }
        }
    };
    int greater = 0, equal = 0;
    for_range([&](std::int32_t, std::uint32_t key) {
        greater += key > threshold;
        equal += key == threshold;
    });
    int greater_before = 0, equal_before = 0;
    SelectShared::Scan(sh.scan).ExclusiveSum(greater, greater_before);
    __syncthreads();
    SelectShared::Scan(sh.scan).ExclusiveSum(equal, equal_before);
    int taken_ties = max(0, min(equal, ties - equal_before));
    int out        = greater_before + min(equal_before, ties);
    for_range([&](std::int32_t b, std::uint32_t key) {
        bool take = key > threshold;
        if (!take && key == threshold && taken_ties > 0) {
            take = true;
            --taken_ties;
        }
        if (take) {
            for (int t = 0; t < ratio; ++t) { selected[out * ratio + t] = b * ratio + t; }
            ++out;
        }
    });
    for (int t = tid; t < tail; t += kSelectThreads) {
        selected[keep * ratio + t] = blocks * ratio + t;
    }
    if (tid == 0) { a.counts[col.column] = keep * ratio + tail; }
}

// ---------------------------------------------------------------------------------------------
// Attention (ninfer-ext Tensor Core selected-key attention)

constexpr int kKeyTile = 16;

// Split policy: a call whose (column, KV head) CTAs cannot fill the GPU splits each selection
// into contiguous ranges of at least kMinSplitKeys keys until about kSplitTargetCtas CTAs run.
constexpr std::int32_t kSplitTargetCtas = 256;
constexpr std::int32_t kMinSplitKeys    = 2 * kKeyTile;

// Partial state of one (column, KV head, split, head): the unnormalized FP32 value sum, then
// the running max and the softmax denominator.
constexpr std::int32_t kPartialWords = kHeadDim + 2;

// Tensor Core attention. One CTA of four warps per (column, KV head, split) runs the query group
// as a 16-row MMA tile (rows past the group are zero) against gathered 32-key tiles: S = Q K^T, an
// FP32 online softmax, then O += P V with FP16 MMA where P is rounded to FP16.
//   BF16 cache (BF16 K, FP16 V): S uses BF16 MMA on the stored rows; K/V tiles are double-buffered
//   through cp.async.
//   Quantized caches (K always Hadamard-rotated): the codes are double-buffered through cp.async
//   and widened to FP16 tiles; Q is rotated like K and rounded to FP16 and S uses FP16 MMA. A
//   row-scaled E4M3 plane widens exactly and applies its per-key scale afterwards (K to S, V folded
//   into P before rounding); an int8 G64 plane widens with its group scales (one FP16 rounding);
//   an NVFP4 G16 plane widens exactly with its group scales. A rotated V leaves O in the rotated
//   domain, so the epilogue rotates each head's row back (the rotation is its own inverse).
// With `partial` null the CTA owns the whole selection and writes normalized output; otherwise it
// covers keys [split * split_keys, +split_keys) and writes its partial state.
namespace mma_attention {

constexpr int kRows      = 16;
constexpr int kKeys      = 32;
constexpr int kThreads   = 128;
constexpr int kChunks    = kHeadDim / 8; // 16-byte chunks per 256-element row
constexpr int kPStride   = kKeys + 8;    // FP16 P row stride, off the 128-byte bank period
constexpr int kRowBytes  = kHeadDim * 2;
constexpr int kTileBytes = kKeys * kRowBytes;

// A staged key: its physical page and its row inside the page, kv_head * 64 + offset; row is -1
// when the key is absent.
struct KeyRow {
    std::int32_t page;
    std::int32_t row;
};

template <typename T>
__device__ __forceinline__ const T*
key_row_data(const void* pages, KeyRow key, std::int32_t page_rows, std::int32_t row_elements) {
    const std::int64_t page_elements = static_cast<std::int64_t>(page_rows) * row_elements;
    return static_cast<const T*>(pages) + page_elements * key.page +
           static_cast<std::int64_t>(key.row) * row_elements;
}

// 16-bit K/V tiles: two BF16/FP16 buffers, or one FP16 buffer the quantized codes widen into.
template <bool Quantized>
struct Shared;

template <>
struct Shared<false> {
    static constexpr int kTiles = 2;
    alignas(16) std::uint8_t q[kRows * kRowBytes];
    alignas(16) std::uint8_t k[kTiles][kTileBytes];
    alignas(16) std::uint8_t v[kTiles][kTileBytes];
    alignas(16) std::uint8_t codes[1][2][16];
    KeyRow rows[1][kKeys];
    alignas(16) __half p[kRows * kPStride];
    float s[kRows * kKeys];
    float m[kRows], l[kRows], alpha[kRows];
    float k_scale[kKeys], v_scale[kKeys];

    __device__ std::uint8_t* q_ptr() { return q; }

    __device__ __half* p_ptr() { return p; }

    __device__ float* s_ptr() { return s; }
};

// Quantized caches: 49,856 bytes, so that two CTAs share an SM (78,016 before: one CTA of
// four warps, 8 % occupancy, no eligible warp 85 % of the cycles). Regions in use at the same time
// stay apart: Q (prologue only, then in registers) and P (written after S has read K) live in the K
// tile; S (written after the widening has read the codes, read by the softmax before the next
// stage) lives in the single codes buffer.
template <>
struct Shared<true> {
    static constexpr int kTiles = 1;
    alignas(16) std::uint8_t k[kTiles][kTileBytes];
    alignas(16) std::uint8_t v[kTiles][kTileBytes];
    alignas(16) std::uint8_t codes[1][2][kKeys * kHeadDim];
    KeyRow rows[1][kKeys];
    float m[kRows], l[kRows], alpha[kRows];
    float k_scale[kKeys], v_scale[kKeys];

    __device__ std::uint8_t* q_ptr() { return k[0]; }

    __device__ __half* p_ptr() { return reinterpret_cast<__half*>(k[0]); }

    __device__ float* s_ptr() { return reinterpret_cast<float*>(codes[0][0]); }
};

static_assert(kRows * kRowBytes <= kTileBytes && kRows * kPStride * 2 <= kTileBytes &&
              kRows * kKeys * 4 <= kKeys * kHeadDim);
// Two CTAs per SM on sm_120 (102,400 bytes of shared memory, 1,024 reserved per CTA).
static_assert(2 * (sizeof(Shared<true>) + 1024) <= 102400,
              "the quantized layout must leave room for two CTAs");

// Byte offset of 16-byte chunk `chunk` of row `row` in a swizzled 512-byte-row tile.
__device__ __forceinline__ int swizzle(int row, int chunk) {
    return row * kRowBytes + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ unsigned pack_half2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Widens the staged codes of one plane (`codes` [kKeys][256 bytes], row j at j * 256) into the
// swizzled FP16 tile `target`. Group scales are read from `scales`; a row-scaled plane widens
// unscaled.
template <Codec C>
__device__ __forceinline__ void widen_tile(const std::uint8_t* codes, const KeyRow* rows,
                                           const void* scales, std::int32_t page_rows,
                                           std::uint8_t* target, int tid) {
    if constexpr (C == Codec::Nvfp4G16) {
        // 16 bytes hold 32 features: groups 2c and 2c + 1 of the row.
        static_assert((kKeys * 8) % kThreads == 0);
#pragma unroll
        for (int item = tid; item < kKeys * 8; item += kThreads) {
            const int j      = item / 8;
            const int chunk  = item % 8;
            const KeyRow key = rows[j];
            const std::uint8_t* key_scales =
                key.row >= 0
                    ? key_row_data<std::uint8_t>(scales, key, page_rows, kKVCacheNvfp4Groups)
                    : nullptr;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int group      = 2 * chunk + half;
                const std::uint8_t s = key.row >= 0 ? key_scales[group] : std::uint8_t{0};
                const auto wide =
                    kv_cache_nvfp4_dequant_f16x16(&codes[j * kHeadDim + chunk * 16 + half * 8], s);
                *reinterpret_cast<int4*>(&target[swizzle(j, 2 * group)])     = wide.lo;
                *reinterpret_cast<int4*>(&target[swizzle(j, 2 * group + 1)]) = wide.hi;
            }
        }
    } else {
        // Unrolled (kKeys * 16 / kThreads = 4 items per thread): the group-scale loads of all items
        // issue before the first is used instead of one dependent round trip per item.
        static_assert((kKeys * 16) % kThreads == 0);
#pragma unroll
        for (int item = tid; item < kKeys * 16; item += kThreads) {
            const int j       = item / 16;
            const int chunk   = item % 16;
            const uint4 bytes = *reinterpret_cast<const uint4*>(&codes[j * kHeadDim + chunk * 16]);
            uint4 wide[2];
            auto* halves = reinterpret_cast<__half2*>(wide);
            if constexpr (C == Codec::Fp8Row) {
                const auto* pairs = reinterpret_cast<const std::uint16_t*>(&bytes);
#pragma unroll
                for (int i = 0; i < 8; ++i) { halves[i] = kv_cache_fp8_code2_to_half2(pairs[i]); }
            } else {
                const KeyRow key = rows[j];
                const __half s   = key.row >= 0 ? key_row_data<__half>(scales, key, page_rows,
                                                                       kKVCacheInt8Groups)[chunk / 4]
                                                : __float2half_rn(0.0F);
                const __half2 s2 = __halves2half2(s, s);
                const auto* ints = reinterpret_cast<const std::int8_t*>(&bytes);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const __half2 x = __halves2half2(__short2half_rn(ints[2 * i]),
                                                     __short2half_rn(ints[2 * i + 1]));
                    halves[i]       = __hmul2(x, s2);
                }
            }
            *reinterpret_cast<uint4*>(&target[swizzle(j, 2 * chunk)])     = wide[0];
            *reinterpret_cast<uint4*>(&target[swizzle(j, 2 * chunk + 1)]) = wide[1];
        }
    }
}

} // namespace mma_attention

struct AttentionArgs {
    const __nv_bfloat16* q;
    const std::int32_t* selected;
    const std::int32_t* counts;
    std::int32_t max_selected;
    const std::int32_t* table_rows;
    const std::int32_t* tables;
    std::int32_t table_stride;
    const void* k_pages;
    const void* v_pages;
    const void* k_scales;
    const void* v_scales;
    std::int32_t query_heads;
    std::int32_t kv_heads;
    std::int32_t width;
    float scale;
    std::int32_t split_keys;
    float* partial;
    __nv_bfloat16* out;
};

template <KvCacheStorage Storage>
__global__ void __launch_bounds__(mma_attention::kThreads)
    qsa_attention_mma_kernel(AttentionArgs a) {
    using namespace mma_attention;
    constexpr bool kQuantized      = Storage != KvCacheStorage::BFloat16;
    constexpr ProfileCodecs codecs = profile_codecs(Storage);
    constexpr bool kKeyRowScale    = kQuantized && codecs.k == Codec::Fp8Row;
    constexpr bool kValueRowScale  = kQuantized && codecs.v == Codec::Fp8Row;
    constexpr bool kRotatedValue   = kQuantized && codecs.v_rotated;
    extern __shared__ __align__(16) std::uint8_t raw_shared[];
    Shared<kQuantized>& sh = *reinterpret_cast<Shared<kQuantized>*>(raw_shared);

    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane_id        = tid & 31;
    const int gid            = lane_id >> 2;
    const int lid            = lane_id & 3;
    const int kv_head        = static_cast<int>(blockIdx.x) % a.kv_heads;
    const int column         = static_cast<int>(blockIdx.x) / a.kv_heads;
    const int group          = a.query_heads / a.kv_heads;
    const std::int32_t* list = a.selected + static_cast<std::int64_t>(column) * a.max_selected;
    float* partial           = a.partial;
    const std::int32_t first = partial == nullptr ? 0 : static_cast<int>(blockIdx.y) * a.split_keys;
    const std::int32_t count =
        partial == nullptr ? a.counts[column] : min(a.counts[column], first + a.split_keys);
    const std::int32_t* table =
        a.tables + static_cast<std::int64_t>(a.table_rows[column / a.width]) * a.table_stride;

    const std::int32_t page_rows = a.kv_heads * kPagedKVPageSize;
    const auto key_row           = [&](int base, int j) -> KeyRow {
        if (base + j >= count) { return {0, -1}; }
        const std::int32_t token = list[base + j];
        return {paged_kv_physical_page(table, token),
                kv_head * kPagedKVPageSize + (token & kPagedKVPageMask)};
    };
    // Gather keys [base, base + kKeys) of the selection into buffer `buf`; missing keys are zero.
    auto stage = [&](int buf, int base) {
        if constexpr (!kQuantized) {
#pragma unroll 4
            for (int item = tid; item < kKeys * kChunks; item += kThreads) {
                const int j      = item / kChunks;
                const int chunk  = item - j * kChunks;
                const KeyRow key = key_row(base, j);
                const int offset = swizzle(j, chunk);
                const int bytes  = key.row >= 0 ? 16 : 0;
                const auto* k_src =
                    key.row >= 0 ? key_row_data<__nv_bfloat16>(a.k_pages, key, page_rows, kHeadDim)
                                 : static_cast<const __nv_bfloat16*>(a.k_pages);
                const auto* v_src = key.row >= 0
                                        ? key_row_data<__half>(a.v_pages, key, page_rows, kHeadDim)
                                        : static_cast<const __half*>(a.v_pages);
                cp_async_zfill<16, Cache::cg>(&sh.k[buf][offset], k_src + chunk * 8, bytes);
                cp_async_zfill<16, Cache::cg>(&sh.v[buf][offset], v_src + chunk * 8, bytes);
            }
        } else {
            // Each row's codes in 16-byte chunks, unswizzled; the widening pass swizzles.
            if (tid < kKeys) { sh.rows[buf][tid] = key_row(base, tid); }
            const auto plane = [&](std::uint8_t* dst, const void* pages, int row_bytes) {
                const int chunks = row_bytes / 16;
#pragma unroll 4
                for (int item = tid; item < kKeys * chunks; item += kThreads) {
                    const int j      = item / chunks;
                    const int chunk  = item - j * chunks;
                    const KeyRow key = key_row(base, j);
                    const std::uint8_t* src =
                        key.row >= 0 ? key_row_data<std::uint8_t>(pages, key, page_rows, row_bytes)
                                     : static_cast<const std::uint8_t*>(pages);
                    cp_async_zfill<16, Cache::cg>(&dst[j * kHeadDim + chunk * 16], src + chunk * 16,
                                                  key.row >= 0 ? 16 : 0);
                }
            };
            plane(sh.codes[buf][0], a.k_pages, code_bytes(codecs.k));
            plane(sh.codes[buf][1], a.v_pages, code_bytes(codecs.v));
        }
    };

    if (first < count) { stage(0, first); }
    cp_commit();

    // The query group, rows past the group zero.
    if constexpr (!kQuantized) {
        for (int item = tid; item < kRows * kChunks; item += kThreads) {
            const int h     = item / kChunks;
            const int chunk = item - h * kChunks;
            uint4 value     = make_uint4(0, 0, 0, 0);
            if (h < group) {
                value = *reinterpret_cast<const uint4*>(
                    a.q +
                    (static_cast<std::int64_t>(column) * a.query_heads + kv_head * group + h) *
                        kHeadDim +
                    chunk * 8);
            }
            *reinterpret_cast<uint4*>(&sh.q_ptr()[swizzle(h, chunk)]) = value;
        }
    } else {
        // One warp per head: lane l rotates dimensions l + 32 r, then stores them as FP16.
        for (int h = warp; h < kRows; h += kThreads / 32) {
            float values[8] = {};
            if (h < group) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(column) * a.query_heads + kv_head * group + h) *
                    kHeadDim;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    values[r] = __bfloat162float(a.q[base + lane_id + 32 * r]);
                }
                normalized_hadamard_d256_inplace(values, lane_id);
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane_id + 32 * r;
                *reinterpret_cast<__half*>(&sh.q_ptr()[swizzle(h, d / 8) + (d % 8) * 2]) =
                    __float2half_rn(values[r]);
            }
        }
    }
    if (tid < kRows) {
        sh.m[tid] = -CUDART_INF_F;
        sh.l[tid] = 0.0F;
    }
    __syncthreads();

    // Q fragments for all sixteen K-steps stay in registers.
    unsigned qf[kHeadDim / 16][4];
#pragma unroll
    for (int step = 0; step < kHeadDim / 16; ++step) {
        ldmatrix_x4(qf[step][0], qf[step][1], qf[step][2], qf[step][3],
                    smem_addr(&sh.q_ptr()[swizzle(lane_id & 15, 2 * step + (lane_id >> 4))]));
    }
    // Every warp holds its Q fragments before Q's region is reused (the first widened K tile, or
    // the rotated-V epilogue's row buffer when the CTA has no key at all).
    __syncthreads();

    // This warp's output features [64 * warp, +64): eight n8 tiles.
    float acc[8][4];
#pragma unroll
    for (int t = 0; t < 8; ++t) { acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0F; }

    int buf = 0;
    for (int base = first; base < count; base += kKeys, buf ^= 1) {
        if constexpr (kQuantized) {
            buf = 0; // one codes buffer: tile t + 1 is staged after the softmax of tile t
            cp_wait<0>();
        } else if (base + kKeys < count) {
            stage(buf ^ 1, base + kKeys);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const int tile = kQuantized ? 0 : buf;

        if constexpr (kQuantized) {
            // Widen this tile's codes to FP16 (swizzled) and read the keys' row scales.
            widen_tile<codecs.k>(sh.codes[buf][0], sh.rows[buf], a.k_scales, page_rows, sh.k[0],
                                 tid);
            widen_tile<codecs.v>(sh.codes[buf][1], sh.rows[buf], a.v_scales, page_rows, sh.v[0],
                                 tid);
            if (tid < kKeys) {
                const KeyRow key = sh.rows[buf][tid];
                if constexpr (kKeyRowScale) {
                    sh.k_scale[tid] =
                        key.row >= 0
                            ? __half2float(*key_row_data<__half>(a.k_scales, key, page_rows, 1))
                            : 0.0F;
                }
                if constexpr (kValueRowScale) {
                    sh.v_scale[tid] =
                        key.row >= 0
                            ? __half2float(*key_row_data<__half>(a.v_scales, key, page_rows, 1))
                            : 0.0F;
                }
            }
            __syncthreads();
        }

        // S for keys [8 * warp, +8).
        {
            float c[4]    = {0.0F, 0.0F, 0.0F, 0.0F};
            const int key = warp * 8 + (lane_id & 7);
#pragma unroll
            for (int pair = 0; pair < kHeadDim / 32; ++pair) {
                unsigned b[4];
                ldmatrix_x4(b[0], b[1], b[2], b[3],
                            smem_addr(&sh.k[tile][swizzle(key, 4 * pair + (lane_id >> 3))]));
                if constexpr (kQuantized) {
                    mma_f16(c[0], c[1], c[2], c[3], qf[2 * pair][0], qf[2 * pair][1],
                            qf[2 * pair][2], qf[2 * pair][3], b[0], b[1]);
                    mma_f16(c[0], c[1], c[2], c[3], qf[2 * pair + 1][0], qf[2 * pair + 1][1],
                            qf[2 * pair + 1][2], qf[2 * pair + 1][3], b[2], b[3]);
                } else {
                    mma_bf16(c[0], c[1], c[2], c[3], qf[2 * pair][0], qf[2 * pair][1],
                             qf[2 * pair][2], qf[2 * pair][3], b[0], b[1]);
                    mma_bf16(c[0], c[1], c[2], c[3], qf[2 * pair + 1][0], qf[2 * pair + 1][1],
                             qf[2 * pair + 1][2], qf[2 * pair + 1][3], b[2], b[3]);
                }
            }
            const int col                 = warp * 8 + 2 * lid;
            const float s0                = kKeyRowScale ? a.scale * sh.k_scale[col] : a.scale;
            const float s1                = kKeyRowScale ? a.scale * sh.k_scale[col + 1] : a.scale;
            sh.s_ptr()[gid * kKeys + col] = c[0] * s0;
            sh.s_ptr()[gid * kKeys + col + 1]       = c[1] * s1;
            sh.s_ptr()[(gid + 8) * kKeys + col]     = c[2] * s0;
            sh.s_ptr()[(gid + 8) * kKeys + col + 1] = c[3] * s1;
        }
        __syncthreads();

        // Online softmax: eight threads per row, four keys apiece.
        {
            const int row  = tid >> 3;
            const int col0 = (tid & 7) * 4;
            float s[4];
            float tile_max = -CUDART_INF_F;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i] = base + col0 + i < count ? sh.s_ptr()[row * kKeys + col0 + i] : -CUDART_INF_F;
                tile_max = fmaxf(tile_max, s[i]);
            }
#pragma unroll
            for (int offset = 1; offset < 8; offset <<= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
            }
            const float m_old = sh.m[row];
            const float m_new = fmaxf(m_old, tile_max);
            float sum         = 0.0F;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i] = base + col0 + i < count ? __expf(s[i] - m_new) : 0.0F;
                sum += s[i];
            }
#pragma unroll
            for (int offset = 1; offset < 8; offset <<= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            float pv[4] = {s[0], s[1], s[2], s[3]};
            if constexpr (kValueRowScale) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { pv[i] *= sh.v_scale[col0 + i]; }
            }
            *reinterpret_cast<uint2*>(&sh.p_ptr()[row * kPStride + col0]) =
                make_uint2(pack_half2(pv[0], pv[1]), pack_half2(pv[2], pv[3]));
            __syncwarp();
            if ((tid & 7) == 0) {
                const float alpha = __expf(m_old - m_new);
                sh.alpha[row]     = alpha;
                sh.l[row]         = sh.l[row] * alpha + sum;
                sh.m[row]         = m_new;
            }
        }
        __syncthreads();

        if constexpr (kQuantized) {
            if (base + kKeys < count) { stage(0, base + kKeys); }
            cp_commit();
        }

        // O = alpha O + P V over this warp's 64 features.
        {
            const float a_lo = sh.alpha[gid];
            const float a_hi = sh.alpha[gid + 8];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                acc[t][0] *= a_lo;
                acc[t][1] *= a_lo;
                acc[t][2] *= a_hi;
                acc[t][3] *= a_hi;
            }
#pragma unroll
            for (int step = 0; step < kKeys / 16; ++step) {
                unsigned af[4];
                ldmatrix_x4(
                    af[0], af[1], af[2], af[3],
                    smem_addr(
                        &sh.p_ptr()[(lane_id & 15) * kPStride + step * 16 + (lane_id >> 4) * 8]));
                const int key = step * 16 + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const int chunk = warp * 8 + pair * 2 + (lane_id >> 4);
                    unsigned b[4];
                    ldmatrix_x4_t(b[0], b[1], b[2], b[3],
                                  smem_addr(&sh.v[tile][swizzle(key, chunk)]));
                    mma_f16(acc[2 * pair][0], acc[2 * pair][1], acc[2 * pair][2], acc[2 * pair][3],
                            af[0], af[1], af[2], af[3], b[0], b[1]);
                    mma_f16(acc[2 * pair + 1][0], acc[2 * pair + 1][1], acc[2 * pair + 1][2],
                            acc[2 * pair + 1][3], af[0], af[1], af[2], af[3], b[2], b[3]);
                }
            }
        }
        // No barrier here: P, V and alpha are next written after the loop-top barrier of the next
        // tile. The codes buffer the next stage fills also holds the scores, which the softmax
        // reads: the stage is safe because it is issued after the post-softmax barrier, so it must
        // stay below that barrier. The BF16 path stages at the loop top into the buffer this
        // tile's MMAs read, so it keeps the barrier.
        if constexpr (!kQuantized) { __syncthreads(); }
    }
    cp_wait<0>();

    if constexpr (kRotatedValue) {
        // Gather O's unnormalized rows through the free K tile, then one warp per head rotates its
        // row back before normalizing or writing the partial state. The barrier: P of the last tile
        // lives in the K tile and the loop no longer ends with one.
        __syncthreads();
        auto* rows = reinterpret_cast<float*>(sh.k[0]);
        static_assert(sizeof(sh.k[0]) >= kRows * kHeadDim * sizeof(float));
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int d                        = warp * 64 + t * 8 + 2 * lid;
            rows[gid * kHeadDim + d]           = acc[t][0];
            rows[gid * kHeadDim + d + 1]       = acc[t][1];
            rows[(gid + 8) * kHeadDim + d]     = acc[t][2];
            rows[(gid + 8) * kHeadDim + d + 1] = acc[t][3];
        }
        __syncthreads();
        for (int h = warp; h < group; h += kThreads / 32) {
            float values[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { values[r] = rows[h * kHeadDim + lane_id + 32 * r]; }
            normalized_hadamard_d256_inplace(values, lane_id);
            if (partial != nullptr) {
                const std::int64_t state =
                    (static_cast<std::int64_t>(blockIdx.x) * gridDim.y + blockIdx.y) * group + h;
                float* p = partial + state * kPartialWords;
#pragma unroll
                for (int r = 0; r < 8; ++r) { p[lane_id + 32 * r] = values[r]; }
                if (lane_id == 0) {
                    p[kHeadDim]     = sh.m[h];
                    p[kHeadDim + 1] = sh.l[h];
                }
            } else {
                const float l    = sh.l[h];
                const float inv  = count > 0 && l > 0.0F ? 1.0F / l : 0.0F;
                __nv_bfloat16* o = a.out + (static_cast<std::int64_t>(column) * a.query_heads +
                                            kv_head * group + h) *
                                               kHeadDim;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    o[lane_id + 32 * r] = __float2bfloat16_rn(values[r] * inv);
                }
            }
        }
        return;
    }

    const float m_lo = sh.m[gid], m_hi = sh.m[gid + 8];
    const float l_lo = sh.l[gid], l_hi = sh.l[gid + 8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int d = warp * 64 + t * 8 + 2 * lid;
#pragma unroll
        for (int upper = 0; upper < 2; ++upper) {
            const int h = gid + upper * 8;
            if (h >= group) { continue; }
            const float x0 = acc[t][2 * upper], x1 = acc[t][2 * upper + 1];
            if (partial != nullptr) {
                const std::int64_t state =
                    (static_cast<std::int64_t>(blockIdx.x) * gridDim.y + blockIdx.y) * group + h;
                float* p = partial + state * kPartialWords;
                p[d]     = x0;
                p[d + 1] = x1;
                if (t == 0 && lid == 0) {
                    p[kHeadDim]     = upper == 0 ? m_lo : m_hi;
                    p[kHeadDim + 1] = upper == 0 ? l_lo : l_hi;
                }
            } else {
                const float l   = upper == 0 ? l_lo : l_hi;
                const float inv = count > 0 && l > 0.0F ? 1.0F / l : 0.0F;
                const std::int64_t index =
                    (static_cast<std::int64_t>(column) * a.query_heads + kv_head * group + h) *
                        kHeadDim +
                    d;
                *reinterpret_cast<__nv_bfloat162*>(&a.out[index]) =
                    __floats2bfloat162_rn(x0 * inv, x1 * inv);
            }
        }
    }
}

// One CTA per (column, KV head) and head of its group; thread d merges feature d over the splits.
__global__ void __launch_bounds__(256)
    qsa_attention_combine_kernel(const float* __restrict__ partial, std::int32_t splits,
                                 std::int32_t query_heads, std::int32_t kv_heads,
                                 __nv_bfloat16* __restrict__ out) {
    const int d       = static_cast<int>(threadIdx.x);
    const int kv_head = static_cast<int>(blockIdx.x) % kv_heads;
    const int column  = static_cast<int>(blockIdx.x) / kv_heads;
    const int group   = query_heads / kv_heads;
    const int h       = static_cast<int>(blockIdx.y);
    const float* p =
        partial + (static_cast<std::int64_t>(blockIdx.x) * splits * group + h) * kPartialWords;
    const std::int64_t stride = static_cast<std::int64_t>(group) * kPartialWords;
    float m                   = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) { m = fmaxf(m, p[s * stride + kHeadDim]); }
    float numerator = 0.0F, denominator = 0.0F;
    if (m > -CUDART_INF_F) {
        for (int s = 0; s < splits; ++s) {
            const float* state = p + s * stride;
            const float l      = state[kHeadDim + 1];
            if (l > 0.0F) {
                const float w = __expf(state[kHeadDim] - m);
                numerator += w * state[d];
                denominator += w * l;
            }
        }
    }
    const std::int64_t index =
        (static_cast<std::int64_t>(column) * query_heads + kv_head * group + h) * kHeadDim + d;
    out[index] = __float2bfloat16_rn(denominator > 0.0F ? numerator / denominator : 0.0F);
}

// Calls fn.template operator()<Storage>() for the cache profile.
template <typename Fn>
void dispatch_storage(KvCacheStorage storage, Fn&& fn) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        fn.template operator()<KvCacheStorage::BFloat16>();
        return;
    case KvCacheStorage::Int8Group64:
        fn.template operator()<KvCacheStorage::Int8Group64>();
        return;
    case KvCacheStorage::Fp8E4M3Row256:
        fn.template operator()<KvCacheStorage::Fp8E4M3Row256>();
        return;
    case KvCacheStorage::Nvfp4Group16:
        fn.template operator()<KvCacheStorage::Nvfp4Group16>();
        return;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        fn.template operator()<KvCacheStorage::Fp8KeyNvfp4Value>();
        return;
    }
    throw std::invalid_argument("Qwen Sparse Attention: unknown KV-cache profile");
}

const std::int32_t* optional_i32(const Tensor& t) {
    return t.data == nullptr ? nullptr : static_cast<const std::int32_t*>(t.data);
}

} // namespace

void qsa_append_launch(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                       const Tensor& rope_positions, const Tensor& cache_positions,
                       const Tensor& valid_columns, const Tensor& table_rows,
                       const Tensor& key_norm, const QsaGeometry& geometry,
                       const PagedKVBatchLayerView& kv, const QsaIndexPlanes& index,
                       cudaStream_t stream) {
    AppendArgs args{};
    args.k                  = static_cast<const __nv_bfloat16*>(k.data);
    args.v                  = static_cast<const __nv_bfloat16*>(v.data);
    args.index_keys         = static_cast<const __nv_bfloat16*>(index_keys.data);
    args.rope_positions     = static_cast<const std::int32_t*>(rope_positions.data);
    args.rope_axes          = rope_positions.ne[1];
    args.cache              = static_cast<const std::int32_t*>(cache_positions.data);
    args.valid              = optional_i32(valid_columns);
    args.table_rows         = static_cast<const std::int32_t*>(table_rows.data);
    args.kv_tables          = static_cast<const std::int32_t*>(kv.block_tables.data);
    args.kv_table_stride    = kv.block_tables.ne[0];
    args.index_tables       = static_cast<const std::int32_t*>(index.block_tables.data);
    args.index_table_stride = index.block_tables.ne[0];
    args.k_pages            = kv.k_pages.data;
    args.v_pages            = kv.v_pages.data;
    args.k_scales           = kv.k_scale_pages.data;
    args.v_scales           = kv.v_scale_pages.data;
    args.records            = static_cast<__nv_bfloat16*>(index.records.data);
    args.block_keys         = static_cast<__nv_bfloat16*>(index.block_keys.data);
    args.key_norm           = static_cast<const __nv_bfloat16*>(key_norm.data);
    args.kv_heads           = kv.num_kv_heads;
    args.width              = cache_positions.ne[0];
    args.lanes              = cache_positions.ne[1];
    const std::int64_t warps =
        static_cast<std::int64_t>(args.width) * args.lanes * (args.kv_heads + 1);
    const int blocks = static_cast<int>((warps * 32 + kAppendThreads - 1) / kAppendThreads);
    dispatch_storage(kv.storage, [&]<KvCacheStorage Storage>() {
        qsa_append_kernel<Storage><<<blocks, kAppendThreads, 0, stream>>>(args);
    });
    CUDA_CHECK(cudaGetLastError());
    qsa_publish_block_kernel<<<args.width * args.lanes, 32, 0, stream>>>(args,
                                                                         index_geometry(geometry));
    CUDA_CHECK(cudaGetLastError());
}

QsaSelectPlan qsa_select_plan(std::uint32_t max_visible_keys, std::int32_t ratio,
                              std::int32_t width, std::int32_t lanes) {
    QsaSelectPlan plan{};
    plan.tile_columns   = width <= 1 ? 1 : width <= 2 ? 2 : width <= 4 ? 4 : kTileColumns;
    plan.tiles_per_lane = (width + plan.tile_columns - 1) / plan.tile_columns;
    plan.pass_tiles     = std::min(plan.tiles_per_lane * lanes, kPassTiles);
    // Score rows are a multiple of four floats so long rows are read as aligned float4.
    plan.max_blocks = static_cast<std::int32_t>(
        (max_visible_keys / static_cast<std::uint32_t>(ratio) + 4) / 4 * 4);
    return plan;
}

std::size_t qsa_select_scratch_bytes(const QsaSelectPlan& plan) {
    return static_cast<std::size_t>(plan.pass_tiles) * kTileColumns * plan.max_blocks *
           sizeof(float);
}

void qsa_select_launch(const Tensor& index_query, const Tensor& query_rope_positions,
                       const Tensor& cache_positions, const Tensor& valid_columns,
                       const Tensor& table_rows, const Tensor& query_norm,
                       const QsaIndexPlanes& index, const QsaGeometry& geometry,
                       const QsaSelectPlan& plan, float* scores, Tensor& selected, Tensor& counts,
                       cudaStream_t stream) {
    SelectArgs args{};
    args.index_query      = static_cast<const __nv_bfloat16*>(index_query.data);
    args.query_positions  = static_cast<const std::int32_t*>(query_rope_positions.data);
    args.query_axes       = query_rope_positions.ne[1];
    args.cache            = static_cast<const std::int32_t*>(cache_positions.data);
    args.valid            = optional_i32(valid_columns);
    args.table_rows       = static_cast<const std::int32_t*>(table_rows.data);
    args.tables           = static_cast<const std::int32_t*>(index.block_tables.data);
    args.table_stride     = index.block_tables.ne[0];
    args.block_keys       = static_cast<const __nv_bfloat16*>(index.block_keys.data);
    args.query_norm       = static_cast<const __nv_bfloat16*>(query_norm.data);
    args.scores           = scores;
    args.selected         = static_cast<std::int32_t*>(selected.data);
    args.counts           = static_cast<std::int32_t*>(counts.data);
    args.width            = cache_positions.ne[0];
    args.lanes            = cache_positions.ne[1];
    args.max_selected     = selected.ne[0];
    args.tile_columns     = plan.tile_columns;
    args.tiles_per_lane   = plan.tiles_per_lane;
    args.max_blocks       = plan.max_blocks;
    const IndexGeometry g = index_geometry(geometry);

    // Kernel attributes are per device: both tensor-parallel ranks launch this kernel.
    static FuncAttrPerDevice topk_attribute;
    topk_attribute.ensure(
        qsa_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sizeof(SelectShared) + (kStagedWords + 1) * sizeof(std::uint32_t)));

    const std::int32_t tiles = plan.tiles_per_lane * args.lanes;
    for (std::int32_t begin = 0; begin < tiles; begin += plan.pass_tiles) {
        const std::int32_t pass = std::min(plan.pass_tiles, tiles - begin);
        args.tile_begin         = begin;
        args.pass_tiles         = pass;
        // Spread each tile's blocks over about kScoreTargetCtas CTAs, whole iterations apiece.
        const std::int32_t iterations = (plan.max_blocks + kScoreBlocks - 1) / kScoreBlocks;
        const std::int32_t wanted     = std::max(1, kScoreTargetCtas / pass);
        const std::int32_t per_cta    = std::max(1, (iterations + wanted - 1) / wanted);
        args.chunk_blocks             = per_cta * kScoreBlocks;
        const std::int32_t chunks = (plan.max_blocks + args.chunk_blocks - 1) / args.chunk_blocks;
        const std::size_t score_shared =
            kScoreKeyBytes +
            static_cast<std::size_t>(plan.tile_columns) * kMaxIndexHead * kIndexDim * sizeof(float);
        const dim3 score_grid(static_cast<unsigned>(chunks), static_cast<unsigned>(pass));
        if (score_reg_blocks(plan.tile_columns) == kScoreRegBlocksFull) {
            qsa_score_kernel<kScoreRegBlocksFull>
                <<<score_grid, score_threads(plan.tile_columns), score_shared, stream>>>(args, g);
        } else {
            qsa_score_kernel<1>
                <<<score_grid, score_threads(plan.tile_columns), score_shared, stream>>>(args, g);
        }
        CUDA_CHECK(cudaGetLastError());
        const std::size_t staged =
            static_cast<std::size_t>(std::min(plan.max_blocks, kStagedBlocks));
        const std::size_t shared =
            sizeof(SelectShared) + (staged + staged / 32 + 1) * sizeof(std::uint32_t);
        qsa_topk_kernel<<<static_cast<unsigned>(pass * kTileColumns), kSelectThreads, shared,
                          stream>>>(args, g);
        CUDA_CHECK(cudaGetLastError());
    }
}

QsaAttentionSplit qsa_attention_split(std::int32_t columns, std::int32_t kv_heads,
                                      std::int32_t max_selected) {
    const std::int64_t ctas    = static_cast<std::int64_t>(columns) * kv_heads;
    const std::int32_t by_keys = (max_selected + kMinSplitKeys - 1) / kMinSplitKeys;
    const auto wanted          = static_cast<std::int32_t>(
        std::min<std::int64_t>((kSplitTargetCtas + ctas - 1) / ctas, by_keys));
    if (wanted <= 1) { return {.splits = 1, .split_keys = max_selected}; }
    // Whole key tiles per split; the last split may be short or empty.
    const std::int32_t tiles = (max_selected + kKeyTile - 1) / kKeyTile;
    const std::int32_t keys  = (tiles + wanted - 1) / wanted * kKeyTile;
    return {.splits = (max_selected + keys - 1) / keys, .split_keys = keys};
}

std::size_t qsa_attention_partial_bytes(std::int32_t columns, std::int32_t query_heads,
                                        std::int32_t kv_heads, std::int32_t max_selected,
                                        std::int32_t split_columns) {
    const QsaAttentionSplit split = qsa_attention_split(split_columns, kv_heads, max_selected);
    if (split.splits == 1) { return 0; }
    return static_cast<std::size_t>(columns) * kv_heads * split.splits * (query_heads / kv_heads) *
           kPartialWords * sizeof(float);
}

void qsa_attention_launch(const Tensor& q, const Tensor& selected, const Tensor& counts,
                          const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                          float* partial, Tensor& out, cudaStream_t stream,
                          std::int32_t split_columns) {
    AttentionArgs args{};
    args.q                   = static_cast<const __nv_bfloat16*>(q.data);
    args.selected            = static_cast<const std::int32_t*>(selected.data);
    args.counts              = static_cast<const std::int32_t*>(counts.data);
    args.max_selected        = selected.ne[0];
    args.table_rows          = static_cast<const std::int32_t*>(table_rows.data);
    args.tables              = static_cast<const std::int32_t*>(kv.block_tables.data);
    args.table_stride        = kv.block_tables.ne[0];
    args.k_pages             = kv.k_pages.data;
    args.v_pages             = kv.v_pages.data;
    args.k_scales            = kv.k_scale_pages.data;
    args.v_scales            = kv.v_scale_pages.data;
    args.query_heads         = q.ne[1];
    args.kv_heads            = kv.num_kv_heads;
    args.width               = q.ne[2];
    args.scale               = scale;
    args.out                 = static_cast<__nv_bfloat16*>(out.data);
    const std::int32_t lanes = q.ne[3];
    const int blocks         = args.width * lanes * args.kv_heads;
    const QsaAttentionSplit split =
        qsa_attention_split(split_columns, args.kv_heads, args.max_selected);
    args.split_keys = split.split_keys;
    args.partial    = split.splits == 1 ? nullptr : partial;
    dispatch_storage(kv.storage, [&]<KvCacheStorage Storage>() {
        constexpr int kShared = sizeof(mma_attention::Shared<Storage != KvCacheStorage::BFloat16>);
        static FuncAttrPerDevice attribute, carveout;
        attribute.ensure(qsa_attention_mma_kernel<Storage>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
        carveout.ensure(qsa_attention_mma_kernel<Storage>,
                        cudaFuncAttributePreferredSharedMemoryCarveout,
                        cudaSharedmemCarveoutMaxShared);
        qsa_attention_mma_kernel<Storage>
            <<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(split.splits)),
               mma_attention::kThreads, kShared, stream>>>(args);
    });
    CUDA_CHECK(cudaGetLastError());
    if (args.partial != nullptr) {
        qsa_attention_combine_kernel<<<dim3(
                                           static_cast<unsigned>(blocks),
                                           static_cast<unsigned>(args.query_heads / args.kv_heads)),
                                       256, 0, stream>>>(args.partial, split.splits,
                                                         args.query_heads, args.kv_heads, args.out);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
