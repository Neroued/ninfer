// Implements: include/ninfer/ops/sparse_attention.h (QSA K/V staging: qsa_stage_*, qsa_staging_*,
// qsa_count_selected_blocks, qsa_stage_selected_union)
// Copies the K/V rows a QSA attention call reads from (Host-resident) paged planes into a device
// staging region: whole pages for a prefill call (or, beyond the region, the union of the call's
// selected blocks), selected 4-token blocks through a direct-mapped per-layer cache for a
// decode/verify call. Design: docs/maintainer/active-kv-in-ram.md.
#include "ninfer/ops/sparse_attention.h"

#include "core/arena.h"
#include "core/device.h" // CUDA_CHECK
#include "core/paged_kv_storage.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kRatio              = 4; // QSA block: 4 tokens (the only registered compress ratio)
constexpr int kBlocksPerPage      = kPagedKVPageSize / kRatio;
constexpr int kMaxPlanes          = 4; // K, V, K scales, V scales
constexpr int kBlockThreads       = 256;
constexpr int kPageThreads        = 256;
constexpr std::uint32_t kRowShift = 20; // tag = row << 20 | block

static_assert(kPagedKVPageSize % kRatio == 0);

struct Plane {
    const unsigned char* src; // Host plane (UVA)
    unsigned char* dst;       // region plane
    std::int64_t src_page;    // bytes per page (all heads)
    std::int64_t dst_page;
    std::int32_t row_bytes; // bytes of one token row of one head
};

struct Planes {
    Plane p[kMaxPlanes];
    std::int32_t count;
    std::int32_t heads;
};

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

// The Host and region planes of one layer, paired; both must share the storage geometry.
Planes pair_planes(const PagedKVBatchLayerView& kv, const QsaStagingRegion& region,
                   const char* op) {
    require(kv.storage == region.pages.storage && kv.num_kv_heads == region.pages.num_kv_heads &&
                kv.head_dim == region.pages.head_dim,
            op, "the region does not match the KV profile");
    Planes out{};
    out.heads      = kv.num_kv_heads;
    const auto add = [&](const Tensor& src, const Tensor& dst) {
        if (src.data == nullptr) { return; }
        require(dst.data != nullptr && src.dtype == dst.dtype && src.ne[0] == dst.ne[0] &&
                    src.ne[1] == kPagedKVPageSize && dst.ne[1] == kPagedKVPageSize &&
                    src.ne[2] == kv.num_kv_heads && dst.ne[2] == kv.num_kv_heads,
                op, "a region plane does not match its Host plane");
        Plane& p    = out.p[out.count++];
        p.src       = static_cast<const unsigned char*>(src.data);
        p.dst       = static_cast<unsigned char*>(dst.data);
        p.src_page  = static_cast<std::int64_t>(src.nb[3]);
        p.dst_page  = static_cast<std::int64_t>(dst.nb[3]);
        p.row_bytes = static_cast<std::int32_t>(src.nb[1]);
        require(p.row_bytes % 2 == 0 && (kRatio * p.row_bytes) % 8 == 0 &&
                    p.src_page ==
                        static_cast<std::int64_t>(p.row_bytes) * kPagedKVPageSize * kv.num_kv_heads,
                op, "plane rows are not page-major multiples of 8 bytes per block");
    };
    add(kv.k_pages, region.pages.k_pages);
    add(kv.v_pages, region.pages.v_pages);
    add(kv.k_scale_pages, region.pages.k_scale_pages);
    add(kv.v_scale_pages, region.pages.v_scale_pages);
    return out;
}

// Copies `bytes` (a multiple of 8, both ends 8-byte aligned; 16 when both allow) with `lanes`
// threads starting at `lane`.
__device__ __forceinline__ void copy_bytes(unsigned char* dst, const unsigned char* src,
                                           std::int32_t bytes, int lane, int lanes) {
    if ((bytes & 15) == 0 &&
        ((reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) & 15) ==
            0) {
        const auto* s = reinterpret_cast<const uint4*>(src);
        auto* d       = reinterpret_cast<uint4*>(dst);
        for (int i = lane; i < bytes / 16; i += lanes) { d[i] = __ldcs(s + i); }
    } else {
        const auto* s = reinterpret_cast<const uint2*>(src);
        auto* d       = reinterpret_cast<uint2*>(dst);
        for (int i = lane; i < bytes / 8; i += lanes) { d[i] = __ldcs(s + i); }
    }
}

struct PageArgs {
    Planes planes;
    const std::int32_t* cache; // [W]
    const std::int32_t* valid; // null or [1]
    std::int32_t width;
    const std::int32_t* table; // the block tables [stride, rows]
    const std::int32_t* table_rows;
    std::int32_t table_stride;
    long long* tags;
    std::int32_t tag_count;
    long long* stats; // null or [layers * kQsaStagingStatCount] (layer 0's row counts stage calls)
};

// CTA (page, plane): copies one whole page of one plane (every head) when the page is visible.
__global__ void __launch_bounds__(kPageThreads) stage_pages_kernel(PageArgs a) {
    __shared__ std::int32_t visible_pages;
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        const std::int32_t valid = a.valid == nullptr ? a.width : max(0, min(a.width, a.valid[0]));
        std::int32_t last        = -1;
        for (std::int32_t w = 0; w < valid; ++w) { last = max(last, a.cache[w]); }
        visible_pages = (last + kPagedKVPageSize) / kPagedKVPageSize; // pages of [0, last]
    }
    __syncthreads();
    // Every staged byte replaces cached blocks: forget them all (CTA-strided, any order).
    for (std::int32_t i =
             static_cast<std::int32_t>(blockIdx.y * gridDim.x + blockIdx.x) * kPageThreads + tid;
         i < a.tag_count; i += static_cast<std::int32_t>(gridDim.x * gridDim.y) * kPageThreads) {
        a.tags[i] = -1LL;
    }
    const std::int32_t page = static_cast<std::int32_t>(blockIdx.x);
    if (page >= visible_pages || static_cast<int>(blockIdx.y) >= a.planes.count) { return; }
    const Plane& p          = a.planes.p[blockIdx.y];
    const std::int32_t* row = a.table + static_cast<std::int64_t>(a.table_rows[0]) * a.table_stride;
    const std::int32_t physical = row[page];
    copy_bytes(p.dst + p.dst_page * page, p.src + p.src_page * physical,
               static_cast<std::int32_t>(p.src_page), tid, kPageThreads);
    if (a.stats != nullptr && blockIdx.y == 0 && tid == 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&a.stats[kQsaStagingPages]), 1ULL);
        if (page == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&a.stats[kQsaStagingPageCalls]), 1ULL);
        }
    }
}

struct BlockArgs {
    Planes planes;
    const std::int32_t* selected; // [max_selected, W, B]
    const std::int32_t* counts;   // [W, B]
    std::int32_t max_selected;
    std::int32_t width;        // W * B columns
    std::int32_t lane_width;   // W
    std::int32_t lanes;        // B
    std::int32_t groups;       // ceil(max_selected / 4)
    const std::int32_t* cache; // [W, B]
    const std::int32_t* tables;
    const std::int32_t* table_rows; // [B]
    std::int32_t table_stride;
    std::int32_t row_count;     // region.table_rows
    long long* tags;            // this layer's [cache_blocks]
    unsigned long long* claims; // this layer's [cache_blocks]
    std::int32_t* counters;     // this layer's [kQsaStagingCounterWords]
    std::int32_t* row_starts;   // this layer's [row_count]
    long long* stats;           // null or this layer's [kQsaStagingStatCount]
    std::int32_t cache_blocks;
    std::int32_t cache_base;    // global slot of this layer's first cache slot
    std::int32_t overflow_base; // global slot of the overflow area
    std::int32_t overflow_blocks;
    std::int32_t* out;
};

__device__ __forceinline__ void count(long long* stats, int which) {
    if (stats != nullptr) { atomicAdd(reinterpret_cast<unsigned long long*>(&stats[which]), 1ULL); }
}

// One warp per (column, 4-entry group of its selection).
__global__ void __launch_bounds__(kBlockThreads) stage_blocks_kernel(BlockArgs a) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t unit =
        static_cast<std::int32_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const std::int32_t column = unit / a.groups;
    const std::int32_t group  = unit - column * a.groups;
    const std::uint32_t epoch = static_cast<std::uint32_t>(a.counters[0]);
    // A row whose first position went back (a truncation without a new binding) may rewrite
    // positions cached earlier: a new generation forgets every cached block. Every CTA reads the
    // same previous values; only the last CTA updates them.
    std::uint32_t went_back = 0;
    for (std::int32_t b = 0; b < a.lanes; ++b) {
        const std::int32_t row = a.table_rows[b];
        if (row < 0 || row >= a.row_count) { __trap(); }
        const std::int32_t previous_first = a.row_starts[row] - 1;
        if (previous_first >= 0 && a.cache[b * a.lane_width] < previous_first) { went_back = 1; }
    }
    const auto generation = static_cast<std::uint32_t>(a.counters[4]) + went_back;
    if (column < a.width) {
        const std::int32_t first = group * kRatio;
        const std::int32_t n     = a.counts[column];
        // qsa_select never writes more than max_selected entries: a larger count is a caller bug.
        if (n > a.max_selected) { __trap(); }
        if (first < n) {
            const std::int32_t* list =
                a.selected + static_cast<std::int64_t>(column) * a.max_selected;
            const std::int32_t block          = list[first] >> 2;
            const std::int32_t owner          = column / a.lane_width;
            const std::int32_t row            = a.table_rows[owner];
            const std::int32_t first_position = a.cache[owner * a.lane_width];
            std::int32_t slot                 = 0;
            int copy                          = 0;
            if (lane == 0) {
                // Committed and complete: wholly before the lane's window.
                const bool cacheable = kRatio * block + (kRatio - 1) < first_position;
                int outcome          = kQsaStagingUncacheable;
                if (cacheable) {
                    const auto key = (static_cast<std::uint32_t>(row) << kRowShift) |
                                     static_cast<std::uint32_t>(block);
                    // Rows start at different points of the layer's slots (row 0 at block 0).
                    const std::int32_t h = static_cast<std::int32_t>(
                        (static_cast<std::int64_t>(block) +
                         static_cast<std::int64_t>(row) * (a.cache_blocks / 8)) %
                        a.cache_blocks);
                    const unsigned long long want =
                        (static_cast<unsigned long long>(epoch) << 32) | key;
                    unsigned long long* word = a.claims + h;
                    unsigned long long current =
                        *reinterpret_cast<volatile unsigned long long*>(word);
                    for (;;) {
                        if (static_cast<std::uint32_t>(current >> 32) == epoch) {
                            outcome = static_cast<std::uint32_t>(current) == key
                                          ? kQsaStagingShared
                                          : kQsaStagingCollisions;
                            break;
                        }
                        const unsigned long long previous = atomicCAS(word, current, want);
                        if (previous == current) {
                            const long long tag = static_cast<long long>(
                                (static_cast<unsigned long long>(generation) << 32) | key);
                            if (a.tags[h] == tag) {
                                outcome = kQsaStagingHits;
                            } else {
                                outcome   = kQsaStagingMisses;
                                a.tags[h] = tag;
                            }
                            break;
                        }
                        current = previous;
                    }
                    slot = a.cache_base + h;
                }
                if (outcome == kQsaStagingUncacheable || outcome == kQsaStagingCollisions) {
                    const std::int32_t k = atomicAdd(&a.counters[2], 1);
                    if (k >= a.overflow_blocks) { __trap(); } // sized for every group
                    slot = a.overflow_base + k;
                }
                copy = outcome == kQsaStagingMisses || outcome == kQsaStagingUncacheable ||
                       outcome == kQsaStagingCollisions;
                count(a.stats, outcome);
            }
            slot = __shfl_sync(0xffffffffU, slot, 0);
            copy = __shfl_sync(0xffffffffU, copy, 0);
            if (copy != 0) {
                const std::int32_t first_token = block * kRatio;
                const std::int32_t physical =
                    a.tables[static_cast<std::int64_t>(row) * a.table_stride + (first_token >> 6)];
                const std::int32_t src_row  = first_token & (kPagedKVPageSize - 1);
                const std::int32_t dst_page = slot / kBlocksPerPage;
                const std::int32_t dst_row  = (slot % kBlocksPerPage) * kRatio;
                for (int i = 0; i < a.planes.count; ++i) {
                    const Plane& p = a.planes.p[i];
                    for (int h = 0; h < a.planes.heads; ++h) {
                        const std::int64_t src =
                            p.src_page * physical +
                            static_cast<std::int64_t>(h * kPagedKVPageSize + src_row) * p.row_bytes;
                        const std::int64_t dst =
                            p.dst_page * dst_page +
                            static_cast<std::int64_t>(h * kPagedKVPageSize + dst_row) * p.row_bytes;
                        copy_bytes(p.dst + dst, p.src + src, kRatio * p.row_bytes, lane, 32);
                    }
                }
            }
            if (lane < kRatio && first + lane < n) {
                const std::int32_t index = list[first + lane];
                // qsa_select writes whole blocks, then the tail: a group is one block.
                if ((index >> 2) != block) { __trap(); }
                a.out[static_cast<std::int64_t>(column) * a.max_selected + first + lane] =
                    slot * kRatio + (index & (kRatio - 1));
            }
        }
    }
    // The last CTA ends the call: next epoch, overflow cursor back to zero.
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const int done = atomicAdd(&a.counters[1], 1);
        if (done == static_cast<int>(gridDim.x) - 1) {
            a.counters[1] = 0;
            a.counters[2] = 0;
            for (std::int32_t b = 0; b < a.lanes; ++b) {
                a.row_starts[a.table_rows[b]] = a.cache[b * a.lane_width] + 1;
            }
            a.counters[4] = static_cast<std::int32_t>(generation & 0x7FFFFFFFU);
            a.counters[0] = static_cast<std::int32_t>(epoch + 1U == 0U ? 1U : epoch + 1U);
            if (a.stats != nullptr) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&a.stats[kQsaStagingBlockCalls]),
                          1ULL);
            }
            __threadfence();
        }
    }
}

void require_region(const QsaStagingRegion& region, const char* op) {
    require(
        region.layers > 0 && region.cache_blocks > 0 && region.cache_blocks % 16 == 0 &&
            region.overflow_blocks > 0 && region.overflow_blocks % 16 == 0 &&
            region.tags.dtype == DType::I64 &&
            region.tags.numel() == static_cast<std::int64_t>(region.layers) * region.cache_blocks &&
            region.claims.dtype == DType::I64 && region.claims.numel() == region.tags.numel() &&
            region.counters.dtype == DType::I32 &&
            region.counters.numel() ==
                static_cast<std::int64_t>(region.layers) * kQsaStagingCounterWords &&
            region.stats.dtype == DType::I64 &&
            region.stats.numel() ==
                static_cast<std::int64_t>(region.layers) * kQsaStagingStatCount &&
            region.table_rows > 0 && region.table_rows <= (1 << (32 - kRowShift)) &&
            region.table_row.dtype == DType::I32 && region.table_row.numel() == region.table_rows &&
            region.row_starts.dtype == DType::I32 &&
            region.row_starts.numel() ==
                static_cast<std::int64_t>(region.layers) * region.table_rows &&
            region.pages.k_pages.ne[3] == region.total_pages() &&
            region.pages.block_tables.numel() == region.total_pages(),
        op, "staging region is inconsistent");
}

} // namespace

void qsa_stage_visible_pages(const PagedKVBatchLayerView& kv, const Tensor& cache_positions,
                             const Tensor& valid_columns, const Tensor& table_rows,
                             std::uint32_t max_visible_keys, const QsaStagingRegion& region,
                             bool stats, cudaStream_t stream) {
    constexpr const char* op = "qsa_stage_visible_pages";
    require_region(region, op);
    require(cache_positions.dtype == DType::I32 && cache_positions.ne[1] == 1 &&
                cache_positions.ne[0] > 0,
            op, "cache positions must be I32 [W,1]");
    require(table_rows.dtype == DType::I32 && table_rows.numel() == 1, op, "one lane only");
    const std::int64_t pages_bound =
        (static_cast<std::int64_t>(max_visible_keys) + kPagedKVPageSize - 1) / kPagedKVPageSize;
    require(max_visible_keys > 0 && pages_bound <= region.total_pages(), op,
            "the visible extent exceeds the staging region");
    PageArgs a{};
    a.planes       = pair_planes(kv, region, op);
    a.cache        = static_cast<const std::int32_t*>(cache_positions.data);
    a.valid        = valid_columns.data == nullptr ? nullptr
                                                   : static_cast<const std::int32_t*>(valid_columns.data);
    a.width        = cache_positions.ne[0];
    a.table        = static_cast<const std::int32_t*>(kv.block_tables.data);
    a.table_rows   = static_cast<const std::int32_t*>(table_rows.data);
    a.table_stride = kv.block_tables.ne[0];
    a.tags         = static_cast<long long*>(region.tags.data);
    a.tag_count    = static_cast<std::int32_t>(region.tags.numel());
    a.stats        = stats ? static_cast<long long*>(region.stats.data) : nullptr;
    stage_pages_kernel<<<dim3(static_cast<unsigned>(pages_bound),
                              static_cast<unsigned>(a.planes.count)),
                         kPageThreads, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

void qsa_stage_selected_blocks(const Tensor& selected, const Tensor& counts,
                               const Tensor& cache_positions, const Tensor& table_rows,
                               const PagedKVBatchLayerView& kv, const QsaStagingRegion& region,
                               std::int32_t layer, bool stats, Tensor& rewritten,
                               cudaStream_t stream) {
    constexpr const char* op = "qsa_stage_selected_blocks";
    require_region(region, op);
    require(layer >= 0 && layer < region.layers, op, "layer out of range");
    require(selected.dtype == DType::I32 && selected.ne[3] == 1 && selected.is_contiguous(), op,
            "selected must be I32 [max_selected,W,B]");
    const std::int32_t lane_width = selected.ne[1];
    const std::int32_t lanes      = selected.ne[2];
    const std::int32_t width      = lane_width * lanes;
    const std::int32_t groups     = (selected.ne[0] + kRatio - 1) / kRatio;
    require(lanes >= 1 && lanes <= region.table_rows, op, "more lanes than the region's rows");
    require(counts.dtype == DType::I32 && counts.numel() == width, op, "counts must be [W,B]");
    require(cache_positions.dtype == DType::I32 && cache_positions.numel() == width, op,
            "cache positions must be [W,B]");
    require(table_rows.dtype == DType::I32 && table_rows.numel() == lanes, op,
            "table rows must be [B]");
    require(rewritten.dtype == DType::I32 && rewritten.numel() == selected.numel() &&
                rewritten.data != selected.data && rewritten.is_contiguous(),
            op, "rewritten must be a distinct I32 tensor shaped like selected");
    require(static_cast<std::int64_t>(width) * groups <= region.overflow_blocks, op,
            "the overflow area cannot hold every group of the call");
    BlockArgs a{};
    a.planes       = pair_planes(kv, region, op);
    a.selected     = static_cast<const std::int32_t*>(selected.data);
    a.counts       = static_cast<const std::int32_t*>(counts.data);
    a.max_selected = selected.ne[0];
    a.width        = width;
    a.lane_width   = lane_width;
    a.lanes        = lanes;
    a.groups       = groups;
    a.cache        = static_cast<const std::int32_t*>(cache_positions.data);
    a.tables       = static_cast<const std::int32_t*>(kv.block_tables.data);
    a.table_rows   = static_cast<const std::int32_t*>(table_rows.data);
    a.table_stride = kv.block_tables.ne[0];
    a.row_count    = region.table_rows;
    a.tags         = static_cast<long long*>(region.tags.data) +
             static_cast<std::int64_t>(layer) * region.cache_blocks;
    a.claims = static_cast<unsigned long long*>(region.claims.data) +
               static_cast<std::int64_t>(layer) * region.cache_blocks;
    a.counters = static_cast<std::int32_t*>(region.counters.data) + kQsaStagingCounterWords * layer;
    a.row_starts = static_cast<std::int32_t*>(region.row_starts.data) +
                   static_cast<std::int64_t>(layer) * region.table_rows;
    a.stats                  = stats ? static_cast<long long*>(region.stats.data) +
                          static_cast<std::int64_t>(layer) * kQsaStagingStatCount
                                     : nullptr;
    a.cache_blocks           = region.cache_blocks;
    a.cache_base             = layer * region.cache_blocks;
    a.overflow_base          = region.layers * region.cache_blocks;
    a.overflow_blocks        = region.overflow_blocks;
    a.out                    = static_cast<std::int32_t*>(rewritten.data);
    const std::int64_t warps = static_cast<std::int64_t>(width) * groups;
    const auto blocks = static_cast<unsigned>((warps * 32 + kBlockThreads - 1) / kBlockThreads);
    stage_blocks_kernel<<<blocks, kBlockThreads, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// One thread per selection entry: marks the entry's block (complete blocks only).
__global__ void mark_blocks_kernel(const std::int32_t* selected, const std::int32_t* counts,
                                   const std::int32_t* cache, std::int32_t max_selected,
                                   std::int32_t width, std::uint32_t* bitmap,
                                   std::int32_t bitmap_blocks) {
    const std::int64_t i      = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t column = static_cast<std::int32_t>(i / max_selected);
    const std::int32_t entry  = static_cast<std::int32_t>(i % max_selected);
    if (column >= width || entry >= counts[column]) { return; }
    const std::int32_t block = selected[i] >> 2;
    if (kRatio * block + (kRatio - 1) > cache[column] || block >= bitmap_blocks) { return; }
    atomicOr(&bitmap[block >> 5], 1U << (block & 31));
}

__global__ void count_blocks_kernel(const std::uint32_t* bitmap, std::int32_t words,
                                    const std::int32_t* cache, std::int32_t width,
                                    long long* stats) {
    __shared__ unsigned long long total;
    if (threadIdx.x == 0) { total = 0; }
    __syncthreads();
    unsigned long long local = 0;
    for (std::int32_t w = static_cast<std::int32_t>(threadIdx.x); w < words; w += blockDim.x) {
        local += static_cast<unsigned long long>(__popc(bitmap[w]));
    }
    atomicAdd(&total, local);
    __syncthreads();
    if (threadIdx.x == 0) {
        std::int32_t last = -1;
        for (std::int32_t c = 0; c < width; ++c) { last = max(last, cache[c]); }
        atomicAdd(reinterpret_cast<unsigned long long*>(&stats[kQsaStagingUnionBlocks]), total);
        atomicAdd(reinterpret_cast<unsigned long long*>(&stats[kQsaStagingVisibleBlocks]),
                  static_cast<unsigned long long>((last + 1) / kRatio));
    }
}

} // namespace

std::size_t qsa_count_selected_blocks_workspace_bytes(std::uint32_t max_visible_keys) {
    if (max_visible_keys == 0) {
        throw std::invalid_argument("qsa_count_selected_blocks_workspace_bytes: empty envelope");
    }
    const std::size_t blocks = (static_cast<std::size_t>(max_visible_keys) + kRatio - 1) / kRatio;
    return (blocks + 31) / 32 * sizeof(std::uint32_t);
}

void qsa_count_selected_blocks(const Tensor& selected, const Tensor& counts,
                               const Tensor& cache_positions, std::uint32_t max_visible_keys,
                               const QsaStagingRegion& region, std::int32_t layer,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "qsa_count_selected_blocks";
    require_region(region, op);
    require(layer >= 0 && layer < region.layers && selected.dtype == DType::I32 &&
                selected.ne[2] == 1 && counts.numel() == selected.ne[1] &&
                cache_positions.numel() == selected.ne[1],
            op, "selected/counts/positions must describe one lane");
    const std::size_t bytes = qsa_count_selected_blocks_workspace_bytes(max_visible_keys);
    const auto blocks       = static_cast<std::int32_t>(bytes * 8);
    const auto words        = static_cast<std::int32_t>(bytes / sizeof(std::uint32_t));
    auto scope              = workspace.scope();
    auto* bitmap            = static_cast<std::uint32_t*>(workspace.alloc_bytes(bytes).data);
    CUDA_CHECK(cudaMemsetAsync(bitmap, 0, bytes, stream));
    const std::int64_t entries = static_cast<std::int64_t>(selected.ne[0]) * selected.ne[1];
    mark_blocks_kernel<<<static_cast<unsigned>((entries + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(selected.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(cache_positions.data), selected.ne[0], selected.ne[1],
        bitmap, blocks);
    CUDA_CHECK(cudaGetLastError());
    count_blocks_kernel<<<1, 1024, 0, stream>>>(
        bitmap, words, static_cast<const std::int32_t*>(cache_positions.data), selected.ne[1],
        static_cast<long long*>(region.stats.data) +
            static_cast<std::int64_t>(layer) * kQsaStagingStatCount);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

constexpr int kUnionScanThreads = 1024;

// One thread per selection entry of a prefill call: marks the entry's block in the bitmap.
__global__ void mark_union_kernel(const std::int32_t* selected, const std::int32_t* counts,
                                  std::int32_t max_selected, std::int32_t width,
                                  std::uint32_t* bitmap, std::int32_t bitmap_blocks) {
    const std::int64_t i      = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t column = static_cast<std::int32_t>(i / max_selected);
    const std::int32_t entry  = static_cast<std::int32_t>(i % max_selected);
    if (column >= width) { return; }
    const std::int32_t n = counts[column];
    if (n > max_selected) { __trap(); } // qsa_select's contract
    if (entry >= n) { return; }
    const std::int32_t block = selected[i] >> 2;
    if (block < 0 || block >= bitmap_blocks) { __trap(); } // beyond the launch envelope
    atomicOr(&bitmap[block >> 5], 1U << (block & 31));
}

// One CTA: exclusive prefix of the bitmap's popcounts (each block's slot is the marked blocks
// before it); traps when the union exceeds the region.
__global__ void __launch_bounds__(kUnionScanThreads)
    scan_union_kernel(const std::uint32_t* bitmap, std::int32_t words, std::uint32_t* prefix,
                      std::int64_t capacity, long long* stats) {
    __shared__ std::uint32_t partial[kUnionScanThreads];
    const int tid            = static_cast<int>(threadIdx.x);
    const std::int32_t per   = (words + kUnionScanThreads - 1) / kUnionScanThreads;
    const std::int32_t begin = tid * per;
    const std::int32_t end   = min(words, begin + per);
    std::uint32_t local      = 0;
    for (std::int32_t w = begin; w < end; ++w) { local += __popc(bitmap[w]); }
    partial[tid] = local;
    __syncthreads();
    for (int offset = 1; offset < kUnionScanThreads; offset <<= 1) {
        const std::uint32_t add = tid >= offset ? partial[tid - offset] : 0U;
        __syncthreads();
        partial[tid] += add;
        __syncthreads();
    }
    std::uint32_t running = partial[tid] - local; // exclusive
    for (std::int32_t w = begin; w < end; ++w) {
        prefix[w] = running;
        running += __popc(bitmap[w]);
    }
    if (tid == kUnionScanThreads - 1) {
        if (static_cast<std::int64_t>(partial[tid]) > capacity) { __trap(); }
        if (stats != nullptr) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&stats[kQsaStagingUnionCalls]), 1ULL);
            atomicAdd(reinterpret_cast<unsigned long long*>(&stats[kQsaStagingUnionStaged]),
                      static_cast<unsigned long long>(partial[tid]));
        }
    }
}

struct UnionCopyArgs {
    Planes planes;
    const std::uint32_t* bitmap;
    const std::uint32_t* prefix;
    std::int32_t words;
    const std::int32_t* tables;
    const std::int32_t* table_rows;
    std::int32_t table_stride;
};

// One warp per bitmap word: copies the word's marked blocks into their slots.
__global__ void __launch_bounds__(kBlockThreads) copy_union_kernel(UnionCopyArgs a) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t word =
        static_cast<std::int32_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (word >= a.words) { return; }
    std::uint32_t bits = a.bitmap[word];
    std::int32_t slot  = static_cast<std::int32_t>(a.prefix[word]);
    const std::int32_t* row =
        a.tables + static_cast<std::int64_t>(a.table_rows[0]) * a.table_stride;
    while (bits != 0U) {
        const int bit = __ffs(static_cast<int>(bits)) - 1;
        bits &= bits - 1U;
        const std::int32_t block       = word * 32 + bit;
        const std::int32_t first_token = block * kRatio;
        const std::int32_t physical    = row[first_token >> 6];
        const std::int32_t src_row     = first_token & (kPagedKVPageSize - 1);
        const std::int32_t dst_page    = slot / kBlocksPerPage;
        const std::int32_t dst_row     = (slot % kBlocksPerPage) * kRatio;
        for (int i = 0; i < a.planes.count; ++i) {
            const Plane& p = a.planes.p[i];
            for (int h = 0; h < a.planes.heads; ++h) {
                const std::int64_t src =
                    p.src_page * physical +
                    static_cast<std::int64_t>(h * kPagedKVPageSize + src_row) * p.row_bytes;
                const std::int64_t dst =
                    p.dst_page * dst_page +
                    static_cast<std::int64_t>(h * kPagedKVPageSize + dst_row) * p.row_bytes;
                copy_bytes(p.dst + dst, p.src + src, kRatio * p.row_bytes, lane, 32);
            }
        }
        ++slot;
    }
}

// One thread per selection entry: rewrites it onto its block's slot.
__global__ void rewrite_union_kernel(const std::int32_t* selected, const std::int32_t* counts,
                                     std::int32_t max_selected, std::int32_t width,
                                     const std::uint32_t* bitmap, const std::uint32_t* prefix,
                                     std::int32_t* out) {
    const std::int64_t i      = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t column = static_cast<std::int32_t>(i / max_selected);
    const std::int32_t entry  = static_cast<std::int32_t>(i % max_selected);
    if (column >= width || entry >= counts[column]) { return; }
    const std::int32_t index = selected[i];
    const std::int32_t block = index >> 2;
    const std::uint32_t below =
        bitmap[block >> 5] & ((1U << (block & 31)) - 1U); // marked blocks before it in its word
    const auto slot = static_cast<std::int32_t>(prefix[block >> 5] + __popc(below));
    out[i]          = slot * kRatio + (index & (kRatio - 1));
}

std::int32_t union_words(std::uint32_t max_visible_keys) {
    const std::uint64_t blocks =
        (static_cast<std::uint64_t>(max_visible_keys) + kRatio - 1) / kRatio;
    return static_cast<std::int32_t>((blocks + 31) / 32);
}

} // namespace

std::size_t qsa_stage_selected_union_workspace_bytes(std::uint32_t max_visible_keys) {
    if (max_visible_keys == 0) {
        throw std::invalid_argument("qsa_stage_selected_union_workspace_bytes: empty envelope");
    }
    // Bitmap and prefix, one 32-bit word per 32 blocks each, 256-byte aligned apart.
    const std::size_t words = static_cast<std::size_t>(union_words(max_visible_keys));
    return 2 * ((words * sizeof(std::uint32_t) + 255) / 256 * 256);
}

void qsa_stage_selected_union(const Tensor& selected, const Tensor& counts,
                              const Tensor& table_rows, const PagedKVBatchLayerView& kv,
                              std::uint32_t max_visible_keys, const QsaStagingRegion& region,
                              bool stats, WorkspaceArena& workspace, Tensor& rewritten,
                              cudaStream_t stream) {
    constexpr const char* op = "qsa_stage_selected_union";
    require_region(region, op);
    require(selected.dtype == DType::I32 && selected.ne[2] == 1 && selected.ne[3] == 1 &&
                selected.is_contiguous(),
            op, "selected must be I32 [max_selected,W,1]");
    const std::int32_t width  = selected.ne[1];
    const std::int32_t groups = (selected.ne[0] + kRatio - 1) / kRatio;
    require(counts.dtype == DType::I32 && counts.numel() == width, op, "counts must be [W,1]");
    require(table_rows.dtype == DType::I32 && table_rows.numel() == 1, op, "one lane only");
    require(rewritten.dtype == DType::I32 && rewritten.numel() == selected.numel() &&
                rewritten.data != selected.data && rewritten.is_contiguous(),
            op, "rewritten must be a distinct I32 tensor shaped like selected");
    require(max_visible_keys > 0, op, "empty envelope");
    const std::int64_t capacity = qsa_staging_region_blocks(region);
    require(static_cast<std::int64_t>(width) * groups <= capacity, op,
            "the region cannot hold every block the call may select");
    const std::int32_t words     = union_words(max_visible_keys);
    const std::size_t word_bytes = (static_cast<std::size_t>(words) * 4 + 255) / 256 * 256;
    auto scope                   = workspace.scope();
    auto* bitmap = static_cast<std::uint32_t*>(workspace.alloc_bytes(word_bytes).data);
    auto* prefix = static_cast<std::uint32_t*>(workspace.alloc_bytes(word_bytes).data);
    CUDA_CHECK(cudaMemsetAsync(bitmap, 0, word_bytes, stream));
    const std::int64_t entries = static_cast<std::int64_t>(selected.ne[0]) * width;
    const auto entry_blocks    = static_cast<unsigned>((entries + 255) / 256);
    const auto* sel            = static_cast<const std::int32_t*>(selected.data);
    const auto* cnt            = static_cast<const std::int32_t*>(counts.data);
    mark_union_kernel<<<entry_blocks, 256, 0, stream>>>(sel, cnt, selected.ne[0], width, bitmap,
                                                        words * 32);
    CUDA_CHECK(cudaGetLastError());
    long long* stats_layer0 = stats ? static_cast<long long*>(region.stats.data) : nullptr;
    scan_union_kernel<<<1, kUnionScanThreads, 0, stream>>>(bitmap, words, prefix, capacity,
                                                           stats_layer0);
    CUDA_CHECK(cudaGetLastError());
    // The copied blocks overwrite cache slots: forget them all.
    CUDA_CHECK(cudaMemsetAsync(region.tags.data, 0xFF, region.tags.bytes(), stream));
    UnionCopyArgs c{};
    c.planes               = pair_planes(kv, region, op);
    c.bitmap               = bitmap;
    c.prefix               = prefix;
    c.words                = words;
    c.tables               = static_cast<const std::int32_t*>(kv.block_tables.data);
    c.table_rows           = static_cast<const std::int32_t*>(table_rows.data);
    c.table_stride         = kv.block_tables.ne[0];
    const auto copy_blocks = static_cast<unsigned>(
        (static_cast<std::int64_t>(words) * 32 + kBlockThreads - 1) / kBlockThreads);
    copy_union_kernel<<<copy_blocks, kBlockThreads, 0, stream>>>(c);
    CUDA_CHECK(cudaGetLastError());
    rewrite_union_kernel<<<entry_blocks, 256, 0, stream>>>(
        sel, cnt, selected.ne[0], width, bitmap, prefix,
        static_cast<std::int32_t*>(rewritten.data));
    CUDA_CHECK(cudaGetLastError());
}

void qsa_staging_invalidate(const QsaStagingRegion& region, cudaStream_t stream) {
    require_region(region, "qsa_staging_invalidate");
    CUDA_CHECK(cudaMemsetAsync(region.tags.data, 0xFF, region.tags.bytes(), stream));
}

} // namespace ninfer::ops
