// Active KV in RAM (docs/maintainer/active-kv-in-ram.md): QSA attention over the device region
// filled by qsa_stage_selected_blocks (decode, MTP windows) or qsa_stage_visible_pages (prefill)
// from pinned, device-mapped Host planes must equal qsa_attention over the same pages in device
// memory bit for bit, for every cache profile and both head counts, across steps that hit, miss,
// collide and share blocks, after rewrites of uncommitted positions, a truncation, invalidation,
// and under CUDA Graph replay; batched calls of several lanes (concurrent requests, distinct table
// rows sharing the layer's slots) must keep each row's blocks apart; a prompt chunk staged as the
// union of its selections (beyond the region), whole or column range by column range.
#include "ninfer/ops/sparse_attention.h"

#include "core/arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim  = 256;
constexpr std::int32_t kPage     = kPagedKVPageSize;
const ops::QsaGeometry kGeometry = ops::qwen3_8_flash_next_qsa_geometry();
const std::int32_t kMaxSel       = kGeometry.max_selected();
const std::int32_t kGroups       = (kMaxSel + 3) / 4;

std::uint32_t lcg(std::uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return state >> 8;
}

// One plane in device memory and the same bytes in pinned, mapped Host memory.
struct TwinPlane {
    DeviceBuffer device;
    void* host           = nullptr;
    std::size_t bytes    = 0;
    DType dtype          = DType::U8;
    std::int32_t leading = 0;

    TwinPlane()                            = default;
    TwinPlane(const TwinPlane&)            = delete;
    TwinPlane& operator=(const TwinPlane&) = delete;

    ~TwinPlane() {
        if (host != nullptr) { (void)cudaFreeHost(host); }
    }
};

// Plane bytes that decode to finite values for the profile's codec.
std::uint8_t plane_byte(DType dtype, bool scale, std::uint32_t& state, std::size_t index) {
    const std::uint32_t r = lcg(state);
    if (dtype == DType::FP16 || (dtype == DType::BF16 && !scale)) {
        // 16-bit words: a finite value in [-2, 2) (scales: small and positive).
        const float x      = scale ? 0.004F + 0.004F * static_cast<float>(r % 1000) / 1000.0F
                                   : -2.0F + 4.0F * static_cast<float>(r % 65536) / 65536.0F;
        std::uint16_t bits = 0;
        if (dtype == DType::FP16) {
            const __half h = __float2half_rn(x);
            std::memcpy(&bits, &h, 2);
        } else {
            bits = f32_to_bf16(x);
        }
        return static_cast<std::uint8_t>(index % 2 == 0 ? bits & 0xFF : bits >> 8);
    }
    if (dtype == DType::FP8_E4M3FN) { return static_cast<std::uint8_t>(r & 0x77U); }
    if (dtype == DType::U8 && scale) { return static_cast<std::uint8_t>(0x20U | (r & 0x07U)); }
    return static_cast<std::uint8_t>(r & 0xFFU); // int8 codes, NVFP4 nibbles
}

struct Pool {
    std::int32_t kv_heads;
    std::int32_t pages;
    std::int32_t rows;
    KvCacheStorage storage;
    PagedKVStorageLayout layout;
    std::vector<int> table; // [rows][pages / rows]: logical -> physical
    DeviceBuffer d_table, d_row;
    TwinPlane k, v, ks, vs;
    PagedKVBatchLayerView device{}, host{};

    // `rows` table rows of pages / rows logical pages each; lane b of a call uses row rows - 1 - b.
    Pool(std::int32_t pages_, KvCacheStorage storage_, std::int32_t kv_heads_,
         std::int32_t rows_ = 1)
        : kv_heads(kv_heads_), pages(pages_), rows(rows_), storage(storage_),
          layout(paged_kv_storage_layout(storage_, kHeadDim)) {
        // A permutation of the physical pages (one logical page per physical page).
        table.resize(static_cast<std::size_t>(pages));
        std::iota(table.begin(), table.end(), 0);
        std::shuffle(table.begin(), table.end(), std::mt19937(0x5EED + pages));
        d_table = to_device_i32(table);
        std::vector<int> lane_rows(static_cast<std::size_t>(rows));
        for (int b = 0; b < rows; ++b) { lane_rows[static_cast<std::size_t>(b)] = rows - 1 - b; }
        d_row              = to_device_i32(lane_rows);
        std::uint32_t seed = 0x51U + static_cast<std::uint32_t>(storage) * 13U +
                             static_cast<std::uint32_t>(kv_heads);
        const auto make = [&](TwinPlane& plane, DType dtype, std::int32_t leading, bool scale,
                              Tensor& device_view, Tensor& host_view) {
            plane.dtype   = dtype;
            plane.leading = leading;
            plane.bytes =
                static_cast<std::size_t>(leading) * kPage * kv_heads * pages * dtype_size(dtype);
            std::vector<std::uint8_t> bytes(plane.bytes);
            for (std::size_t i = 0; i < bytes.size(); ++i) {
                bytes[i] = plane_byte(dtype, scale, seed, i);
            }
            plane.device = DeviceBuffer(plane.bytes);
            plane.device.copy_from_host(bytes.data(), bytes.size());
            cuda_check(cudaHostAlloc(&plane.host, plane.bytes,
                                     cudaHostAllocMapped | cudaHostAllocPortable),
                       "cudaHostAlloc");
            std::memcpy(plane.host, bytes.data(), bytes.size());
            device_view = Tensor(plane.device.p, dtype, {leading, kPage, kv_heads, pages});
            host_view   = Tensor(plane.host, dtype, {leading, kPage, kv_heads, pages});
        };
        make(k, layout.key.data_dtype, layout.key.data_leading_extent, false, device.k_pages,
             host.k_pages);
        make(v, layout.value.data_dtype, layout.value.data_leading_extent, false, device.v_pages,
             host.v_pages);
        if (layout.key.has_scale()) {
            make(ks, layout.key.scale_dtype, layout.key.scale_leading_extent, true,
                 device.k_scale_pages, host.k_scale_pages);
        }
        if (layout.value.has_scale()) {
            make(vs, layout.value.scale_dtype, layout.value.scale_leading_extent, true,
                 device.v_scale_pages, host.v_scale_pages);
        }
        for (PagedKVBatchLayerView* view : {&device, &host}) {
            view->block_tables = Tensor(d_table.p, DType::I32, {pages / rows, rows});
            view->head_dim     = kHeadDim;
            view->num_kv_heads = kv_heads;
            view->storage      = storage;
        }
    }

    // The table rows of lanes [0, lanes).
    Tensor row(int lanes = 1) const { return Tensor(d_row.p, DType::I32, {lanes}); }

    // Physical page of token t of table row `row`.
    [[nodiscard]] int physical(int row, int t) const {
        return table[static_cast<std::size_t>(row) * static_cast<std::size_t>(pages / rows) +
                     static_cast<std::size_t>(t / kPage)];
    }

    // Rewrites the K/V rows of tokens [first, first + count) of table row `row` in both copies (a
    // speculative rollback re-append, or a committed rewrite the caller then invalidates).
    void rewrite(int first, int count, std::uint32_t salt, int row = 0) {
        for (TwinPlane* plane : {&k, &v, &ks, &vs}) {
            if (plane->bytes == 0) { continue; }
            const std::size_t row_bytes =
                static_cast<std::size_t>(plane->leading) * dtype_size(plane->dtype);
            const bool scale   = plane == &ks || plane == &vs;
            std::uint32_t seed = salt + static_cast<std::uint32_t>(row_bytes);
            for (int t = first; t < first + count; ++t) {
                for (int h = 0; h < kv_heads; ++h) {
                    const std::size_t at =
                        ((static_cast<std::size_t>(physical(row, t)) * kv_heads + h) * kPage +
                         t % kPage) *
                        row_bytes;
                    auto* bytes = static_cast<std::uint8_t*>(plane->host) + at;
                    for (std::size_t i = 0; i < row_bytes; ++i) {
                        bytes[i] = plane_byte(plane->dtype, scale, seed, i);
                    }
                    plane->device.copy_from_host(bytes, row_bytes, at);
                }
            }
        }
    }
};

// The device region (ops::QsaStagingRegion) for `layers` attention layers.
struct Region {
    DeviceBuffer k, v, ks, vs, table, row, tags, claims, counters, row_starts, stats;
    ops::QsaStagingRegion region;

    Region(const Pool& pool, std::int32_t layers, std::int32_t cache_blocks, std::int32_t window) {
        region.layers            = layers;
        region.table_rows        = pool.rows;
        region.cache_blocks      = cache_blocks;
        region.overflow_blocks   = (window * pool.rows * kGroups + 15) / 16 * 16;
        const std::int32_t pages = region.total_pages();
        const auto plane         = [&](DeviceBuffer& buffer, const Tensor& like) {
            if (like.data == nullptr) { return Tensor(); }
            buffer = DeviceBuffer(static_cast<std::size_t>(like.ne[0]) * kPage * pool.kv_heads *
                                          pages * dtype_size(like.dtype));
            buffer.fill(0x7B); // stale garbage: every read slot must have been written
            return Tensor(buffer.p, like.dtype, {like.ne[0], kPage, pool.kv_heads, pages});
        };
        region.pages.k_pages       = plane(k, pool.host.k_pages);
        region.pages.v_pages       = plane(v, pool.host.v_pages);
        region.pages.k_scale_pages = plane(ks, pool.host.k_scale_pages);
        region.pages.v_scale_pages = plane(vs, pool.host.v_scale_pages);
        std::vector<int> identity(static_cast<std::size_t>(pages));
        std::iota(identity.begin(), identity.end(), 0);
        table = to_device_i32(identity);
        row   = to_device_i32(std::vector<int>(static_cast<std::size_t>(pool.rows), 0));
        region.pages.block_tables = Tensor(table.p, DType::I32, {pages, 1});
        region.pages.head_dim     = kHeadDim;
        region.pages.num_kv_heads = pool.kv_heads;
        region.pages.storage      = pool.storage;
        region.table_row          = Tensor(row.p, DType::I32, {pool.rows});
        const std::size_t slots   = static_cast<std::size_t>(layers) * cache_blocks;
        tags                      = DeviceBuffer(slots * 8);
        tags.fill(0xFF);
        claims = DeviceBuffer(slots * 8);
        claims.fill(0);
        constexpr int kWords = ops::kQsaStagingCounterWords;
        std::vector<int> counter_words(static_cast<std::size_t>(layers) * kWords, 0);
        for (int l = 0; l < layers; ++l) {
            counter_words[static_cast<std::size_t>(l) * kWords] = 1;
        }
        counters = to_device_i32(counter_words);
        row_starts =
            to_device_i32(std::vector<int>(static_cast<std::size_t>(layers) * pool.rows, 0));
        stats = DeviceBuffer(static_cast<std::size_t>(layers) * ops::kQsaStagingStatCount * 8);
        stats.fill(0);
        region.tags       = Tensor(tags.p, DType::I64, {static_cast<std::int32_t>(slots)});
        region.claims     = Tensor(claims.p, DType::I64, {static_cast<std::int32_t>(slots)});
        region.counters   = Tensor(counters.p, DType::I32, {layers * kWords});
        region.row_starts = Tensor(row_starts.p, DType::I32, {layers * pool.rows});
        region.stats      = Tensor(stats.p, DType::I64, {layers * ops::kQsaStagingStatCount});
    }

    // Whether region row `u` differs from source position `t` of table row `table_row` in any
    // plane, head or byte.
    bool row_differs(const Pool& pool, int t, int u, int table_row = 0) const {
        const TwinPlane* sources[]   = {&pool.k, &pool.v, &pool.ks, &pool.vs};
        const DeviceBuffer* staged[] = {&k, &v, &ks, &vs};
        for (int i = 0; i < 4; ++i) {
            if (sources[i]->bytes == 0) { continue; }
            const std::size_t row =
                static_cast<std::size_t>(sources[i]->leading) * dtype_size(sources[i]->dtype);
            for (int h = 0; h < pool.kv_heads; ++h) {
                const std::size_t src =
                    ((static_cast<std::size_t>(pool.physical(table_row, t)) * pool.kv_heads + h) *
                         kPage +
                     t % kPage) *
                    row;
                const std::size_t dst =
                    ((static_cast<std::size_t>(u / kPage) * pool.kv_heads + h) * kPage +
                     u % kPage) *
                    row;
                std::vector<unsigned char> got(row);
                staged[i]->copy_to_host(got.data(), row, dst);
                if (std::memcmp(got.data(),
                                static_cast<const unsigned char*>(sources[i]->host) + src,
                                row) != 0) {
                    return true;
                }
            }
        }
        return false;
    }

    // Exact oracle of qsa_stage_visible_pages: region page i < pages holds source page table[i]
    // (every plane), region page `pages` keeps its fill (not written).
    int check_pages(const Pool& pool, int pages, const std::string& label) const {
        const TwinPlane* sources[]   = {&pool.k, &pool.v, &pool.ks, &pool.vs};
        const DeviceBuffer* staged[] = {&k, &v, &ks, &vs};
        int bad                      = 0;
        for (int i = 0; i < 4; ++i) {
            if (sources[i]->bytes == 0) { continue; }
            const std::size_t page = sources[i]->bytes / static_cast<std::size_t>(pool.pages);
            std::vector<unsigned char> got(page);
            for (int g = 0; g <= pages && g < region.total_pages(); ++g) {
                staged[i]->copy_to_host(got.data(), page, page * static_cast<std::size_t>(g));
                if (g < pages) {
                    bad += std::memcmp(got.data(),
                                       static_cast<const unsigned char*>(sources[i]->host) +
                                           page * static_cast<std::size_t>(pool.table[g]),
                                       page) != 0
                               ? 1
                               : 0;
                } else {
                    bad += std::any_of(got.begin(), got.end(),
                                       [](unsigned char b) { return b != 0x7B; })
                               ? 1
                               : 0;
                }
            }
        }
        if (bad != 0) { std::cerr << label << ": " << bad << " staged pages differ\n"; }
        return bad != 0 ? 1 : 0;
    }

    std::vector<long long> totals() const {
        const auto all = from_device<long long>(stats, static_cast<std::size_t>(region.layers) *
                                                           ops::kQsaStagingStatCount);
        std::vector<long long> sum(ops::kQsaStagingStatCount, 0);
        for (int l = 0; l < region.layers; ++l) {
            for (int s = 0; s < ops::kQsaStagingStatCount; ++s) {
                sum[static_cast<std::size_t>(s)] +=
                    all[static_cast<std::size_t>(l) * ops::kQsaStagingStatCount + s];
            }
        }
        return sum;
    }
};

// qsa_select's contract for a column that sees [0, n): the 512 chosen complete blocks expanded
// in ascending order, then the incomplete tail. Overlap control: the first `common` blocks of
// `shared` are kept, the rest are drawn from `seed`.
std::vector<int> selection(int n, const std::vector<int>& shared, int common, std::uint32_t seed) {
    const int m    = n / 4;
    const int topk = kGeometry.block_topk();
    std::vector<int> blocks;
    if (m <= topk) {
        blocks.resize(static_cast<std::size_t>(m));
        std::iota(blocks.begin(), blocks.end(), 0);
    } else {
        std::vector<char> taken(static_cast<std::size_t>(m), 0);
        for (int i = 0; i < common && i < static_cast<int>(shared.size()); ++i) {
            if (shared[i] < m && taken[shared[i]] == 0) {
                taken[shared[i]] = 1;
                blocks.push_back(shared[i]);
            }
        }
        while (static_cast<int>(blocks.size()) < topk) {
            const int b = static_cast<int>(lcg(seed) % static_cast<std::uint32_t>(m));
            if (taken[b] == 0) {
                taken[b] = 1;
                blocks.push_back(b);
            }
        }
        std::sort(blocks.begin(), blocks.end());
    }
    std::vector<int> out;
    for (int b : blocks) {
        for (int j = 0; j < 4; ++j) { out.push_back(4 * b + j); }
    }
    for (int t = 4 * m; t < n; ++t) { out.push_back(t); }
    return out;
}

struct Window {
    std::vector<int> positions; // W * lanes cache positions, lane-major
    std::vector<int> sel;       // [max_sel, W, lanes]
    std::vector<int> cnt;       // [W, lanes]
    int lanes = 1;
};

Window window(const std::vector<int>& positions, const std::vector<int>& shared, int common,
              std::uint32_t seed) {
    Window w{
        positions, std::vector<int>(static_cast<std::size_t>(kMaxSel) * positions.size(), 0), {}};
    for (std::size_t c = 0; c < positions.size(); ++c) {
        const auto list =
            selection(positions[c] + 1, shared, common, seed + static_cast<std::uint32_t>(c));
        std::copy(list.begin(), list.end(),
                  w.sel.begin() + static_cast<std::ptrdiff_t>(c) * kMaxSel);
        w.cnt.push_back(static_cast<int>(list.size()));
    }
    return w;
}

// One batched call of several lanes (each a one-lane window of the same width).
Window lanes_window(const std::vector<Window>& lanes) {
    Window out;
    for (const Window& w : lanes) {
        out.positions.insert(out.positions.end(), w.positions.begin(), w.positions.end());
        out.sel.insert(out.sel.end(), w.sel.begin(), w.sel.end());
        out.cnt.insert(out.cnt.end(), w.cnt.begin(), w.cnt.end());
    }
    out.lanes = static_cast<int>(lanes.size());
    return out;
}

std::vector<int> random_blocks(int m, int count, std::uint32_t seed) {
    std::vector<int> out;
    for (int i = 0; i < count; ++i) {
        out.push_back(static_cast<int>(lcg(seed) % static_cast<std::uint32_t>(m)));
    }
    return out;
}

struct Attention {
    std::int32_t query_heads;
    std::int32_t width; // every column of every lane
    std::int32_t lanes;
    DeviceBuffer q, sel, cnt, cache, rewritten, out_ref, out_act;
    std::unique_ptr<WorkspaceArena> workspace;

    Attention(const Pool& pool, const Window& w, std::uint32_t seed)
        : query_heads(12 * pool.kv_heads), width(static_cast<std::int32_t>(w.positions.size())),
          lanes(w.lanes) {
        std::vector<std::uint16_t> qb(static_cast<std::size_t>(kHeadDim) * query_heads * width);
        std::uint32_t state = seed;
        for (auto& x : qb) {
            x = f32_to_bf16(-1.0F + 2.0F * static_cast<float>(lcg(state) % 4096) / 4096.0F);
        }
        q                  = to_device(qb);
        sel                = to_device_i32(w.sel);
        cnt                = to_device_i32(w.cnt);
        cache              = to_device_i32(w.positions);
        rewritten          = DeviceBuffer(w.sel.size() * 4);
        out_ref            = DeviceBuffer(qb.size() * 2);
        out_act            = DeviceBuffer(qb.size() * 2);
        ops::QsaGeometry g = kGeometry;
        g.query_heads      = query_heads;
        g.kv_heads         = pool.kv_heads;
        workspace = std::make_unique<WorkspaceArena>(ops::qsa_attention_workspace_bytes(g, width));
    }

    [[nodiscard]] std::int32_t W() const { return width / lanes; }

    Tensor tq() const { return Tensor(q.p, DType::BF16, {kHeadDim, query_heads, W(), lanes}); }

    Tensor tsel() const { return Tensor(sel.p, DType::I32, {kMaxSel, W(), lanes}); }

    Tensor tcnt() const { return Tensor(cnt.p, DType::I32, {W(), lanes}); }

    Tensor tcache() const { return Tensor(cache.p, DType::I32, {W(), lanes}); }

    Tensor rewritten_view() { return Tensor(rewritten.p, DType::I32, {kMaxSel, W(), lanes}); }

    // Lane b reads table row rows - 1 - b (Pool::row).
    void reference(const Pool& pool, cudaStream_t stream) {
        Tensor out(out_ref.p, DType::BF16, {kHeadDim, query_heads, W(), lanes});
        ops::qsa_attention(tq(), tsel(), tcnt(), pool.row(lanes), pool.device, 1.0F / 16.0F,
                           *workspace, out, stream);
    }

    void decode(const Pool& pool, Region& region, int layer, cudaStream_t stream) {
        Tensor rew = rewritten_view();
        // Entries past each count must stay untouched: start from -1.
        cuda_check(cudaMemsetAsync(rewritten.p, 0xFF, rewritten.bytes, stream), "memset");
        ops::qsa_stage_selected_blocks(tsel(), tcnt(), tcache(), pool.row(lanes), pool.host,
                                       region.region, layer, true, rew, stream);
        Tensor out(out_act.p, DType::BF16, {kHeadDim, query_heads, W(), lanes});
        ops::qsa_attention(tq(), rew, tcnt(), region.region.table_row.slice(0, 0, lanes),
                           region.region.pages, 1.0F / 16.0F, *workspace, out, stream);
    }

    void prefill(const Pool& pool, Region& region, std::uint32_t max_visible, cudaStream_t stream) {
        ops::qsa_stage_visible_pages(pool.host, tcache(), Tensor(), pool.row(), max_visible,
                                     region.region, true, stream);
        Tensor out(out_act.p, DType::BF16, {kHeadDim, query_heads, width, 1});
        ops::qsa_attention(tq(), tsel(), tcnt(), region.region.table_row.slice(0, 0, 1),
                           region.region.pages, 1.0F / 16.0F, *workspace, out, stream);
    }

    // Prompt chunk beyond the region: the union of the selections of `step` columns at a time,
    // then their attention (the composition of qsa.h).
    void union_prefill(const Pool& pool, Region& region, std::uint32_t max_visible,
                       std::int32_t step, cudaStream_t stream) {
        cuda_check(cudaMemsetAsync(rewritten.p, 0xFF, rewritten.bytes, stream), "memset");
        const std::size_t union_bytes = ops::qsa_stage_selected_union_workspace_bytes(max_visible);
        WorkspaceArena union_work(union_bytes);
        Tensor out(out_act.p, DType::BF16, {kHeadDim, query_heads, width, 1});
        Tensor rew = rewritten_view();
        for (std::int32_t w0 = 0; w0 < width; w0 += step) {
            const std::int32_t n = std::min(step, width - w0);
            Tensor part          = rew.slice(1, w0, n);
            Tensor out_part      = out.slice(2, w0, n);
            ops::qsa_stage_selected_union(tsel().slice(1, w0, n), tcnt().slice(0, w0, n),
                                          pool.row(), pool.host, max_visible, region.region, true,
                                          union_work, part, stream);
            ops::qsa_attention(tq().slice(2, w0, n), part, tcnt().slice(0, w0, n),
                               region.region.table_row.slice(0, 0, 1), region.region.pages,
                               1.0F / 16.0F, *workspace, out_part, stream, width);
        }
    }

    // Exact oracle of qsa_stage_selected_blocks: every rewritten entry addresses region rows
    // holding the source rows (every plane, head and byte) of its selected position, entries past
    // a count are untouched, and the low two bits are the position's offset in its block.
    int check_rows(const Pool& pool, const Region& region, const Window& w,
                   const std::string& label) const {
        const auto rew    = from_device<int>(rewritten, w.sel.size());
        const auto lane_w = w.positions.size() / static_cast<std::size_t>(w.lanes);
        int bad           = 0;
        for (std::size_t c = 0; c < w.positions.size(); ++c) {
            const int row = pool.rows - 1 - static_cast<int>(c / lane_w);
            for (int j = 0; j < kMaxSel; ++j) {
                const std::size_t at = c * kMaxSel + static_cast<std::size_t>(j);
                if (j >= w.cnt[c]) {
                    bad += rew[at] != -1 ? 1 : 0;
                    continue;
                }
                const int t = w.sel[at], u = rew[at];
                if ((u & 3) != (t & 3) || u < 0 || u / kPage >= region.region.total_pages()) {
                    ++bad;
                    continue;
                }
                bad += region.row_differs(pool, t, u, row) ? 1 : 0;
            }
        }
        if (bad != 0) {
            std::cerr << label << ": " << bad << " staged rows differ from the source\n";
        }
        return bad != 0 ? 1 : 0;
    }

    int compare(const std::string& label) const {
        const std::size_t n = static_cast<std::size_t>(kHeadDim) * query_heads * width;
        return verify_exact(label.c_str(), from_device<std::uint16_t>(out_act, n),
                            from_device<std::uint16_t>(out_ref, n));
    }
};

const char* profile_name(KvCacheStorage s) {
    switch (s) {
    case KvCacheStorage::BFloat16:
        return "BF16";
    case KvCacheStorage::Int8Group64:
        return "INT8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "FP8";
    case KvCacheStorage::Nvfp4Group16:
        return "NVFP4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "K8V4";
    }
    return "?";
}

// Decode/MTP steps over a deep context: hits, shared blocks, collisions, uncommitted tails.
int test_decode(KvCacheStorage storage, std::int32_t kv_heads) {
    int failures             = 0;
    constexpr int kTokens    = 12000;
    constexpr int kLayers    = 3;
    constexpr int kCache     = 1024; // blocks per layer: a deep context collides
    const std::string prefix = std::string("stage selected blocks ") + profile_name(storage) +
                               " heads " + std::to_string(kv_heads);
    Pool pool((kTokens + 64 + kPage - 1) / kPage, storage, kv_heads);
    Region region(pool, kLayers, kCache, 4);
    const auto shared = random_blocks((kTokens - 8) / 4, 512, 0x777U);
    int p0            = kTokens - 8;
    for (int step = 0; step < 6; ++step) {
        // MTP verify (W=4) on even steps, plain decode (W=1) on odd ones; 90 % of the blocks
        // repeat across steps, the columns of a window overlap.
        std::vector<int> positions;
        const int width = step % 2 == 0 ? 4 : 1;
        for (int w = 0; w < width; ++w) { positions.push_back(p0 + w); }
        const Window w = window(positions, shared, 460, 0x100U + static_cast<std::uint32_t>(step));
        Attention a(pool, w, 0x200U + static_cast<std::uint32_t>(step));
        // A rejected speculative row is re-appended with other bytes at the same positions.
        pool.rewrite(p0, width + 2, 0x300U + static_cast<std::uint32_t>(step));
        a.reference(pool, nullptr);
        a.decode(pool, region, step % kLayers, nullptr);
        cuda_synchronize();
        failures += a.compare(prefix + " step " + std::to_string(step));
        failures += a.check_rows(pool, region, w, prefix + " step " + std::to_string(step));
        p0 += width == 4 ? 2 : 1;
    }
    const auto totals = region.totals();
    std::cout << prefix << ": hits " << totals[ops::kQsaStagingHits] << " misses "
              << totals[ops::kQsaStagingMisses] << " shared " << totals[ops::kQsaStagingShared]
              << " uncacheable " << totals[ops::kQsaStagingUncacheable] << " collisions "
              << totals[ops::kQsaStagingCollisions] << '\n';
    if (totals[ops::kQsaStagingHits] == 0 || totals[ops::kQsaStagingShared] == 0 ||
        totals[ops::kQsaStagingUncacheable] == 0 || totals[ops::kQsaStagingCollisions] == 0 ||
        totals[ops::kQsaStagingBlockCalls] != 6) {
        std::cerr << prefix << ": a cache path was not exercised\n";
        ++failures;
    }
    // A truncation inside the same binding: the first position goes back to `cut`, the rows
    // after it are rewritten, then decode moves past them again. Without the generation bump the
    // blocks in [cut, p0) cached earlier would be served stale.
    {
        const int cut = p0 - 400;
        std::vector<int> around; // the blocks just below and after the cut
        for (int b = cut / 4 - 200; b < p0 / 4 - 2; ++b) { around.push_back(b); }
        const Window warm = window({p0}, around, 512, 0x450U);
        Attention a0(pool, warm, 0x451U);
        a0.decode(pool, region, 2, nullptr); // caches them
        pool.rewrite(cut, p0 + 8 - cut, 0x460U);
        for (const int position : {cut, p0 + 4}) {
            const Window w =
                window({position}, around, 512, 0x470U + static_cast<std::uint32_t>(position));
            Attention a(pool, w, 0x480U);
            a.reference(pool, nullptr);
            a.decode(pool, region, 2, nullptr);
            cuda_synchronize();
            failures +=
                a.compare(prefix + " after truncation, position " + std::to_string(position));
            failures +=
                a.check_rows(pool, region, w,
                             prefix + " after truncation, position " + std::to_string(position));
        }
    }
    // A committed rewrite (e.g. a page copy) followed by invalidation: no stale block survives.
    {
        const Window w = window({p0}, shared, 512, 0x400U);
        pool.rewrite(0, p0 - 16, 0x500U);
        ops::qsa_staging_invalidate(region.region, nullptr);
        Attention a(pool, w, 0x401U);
        a.reference(pool, nullptr);
        a.decode(pool, region, 0, nullptr);
        cuda_synchronize();
        failures += a.compare(prefix + " after invalidation");
        failures += a.check_rows(pool, region, w, prefix + " after invalidation");
    }
    return failures;
}

// Prefill staging: a 300-column chunk over 8000 visible tokens, then a decode reusing the region.
int test_stage(KvCacheStorage storage, std::int32_t kv_heads) {
    int failures             = 0;
    const std::string prefix = std::string("stage visible pages ") + profile_name(storage) +
                               " heads " + std::to_string(kv_heads);
    constexpr int kFirst = 7700, kWidth = 300, kMaxVisible = 8192;
    Pool pool(kMaxVisible / kPage, storage, kv_heads);
    Region region(pool, 3, 2048, 4); // 3*2048 cache blocks + overflow: more than 8192 tokens
    std::vector<int> positions;
    for (int w = 0; w < kWidth; ++w) { positions.push_back(kFirst + w); }
    const auto shared = random_blocks(kFirst / 4, 512, 0x88U);
    const Window w    = window(positions, shared, 300, 0x600U);
    Attention a(pool, w, 0x601U);
    a.reference(pool, nullptr);
    a.prefill(pool, region, kMaxVisible, nullptr);
    cuda_synchronize();
    failures += a.compare(prefix);
    failures += region.check_pages(pool, (kFirst + kWidth - 1) / kPage + 1, prefix);
    // The staging overwrote cache slots: a following decode must not hit stale blocks.
    const Window d = window({kFirst + kWidth}, shared, 400, 0x700U);
    Attention b(pool, d, 0x701U);
    b.reference(pool, nullptr);
    b.decode(pool, region, 1, nullptr);
    cuda_synchronize();
    failures += b.compare(prefix + " then decode");
    failures += b.check_rows(pool, region, d, prefix + " then decode");
    const auto totals = region.totals();
    if (totals[ops::kQsaStagingPageCalls] != 1 ||
        totals[ops::kQsaStagingPages] != (kFirst + kWidth - 1) / kPage + 1 ||
        totals[ops::kQsaStagingHits] != 0) {
        std::cerr << prefix << ": staging counters are wrong (pages "
                  << totals[ops::kQsaStagingPages] << ", hits " << totals[ops::kQsaStagingHits]
                  << ")\n";
        ++failures;
    }
    return failures;
}

// Concurrent requests: batched decode/verify calls of three lanes at different depths, whose
// table rows map the same logical blocks to different pages and share the layer's slots; a call of
// a subset of the lanes; a truncation of one lane.
int test_lanes(KvCacheStorage storage, std::int32_t kv_heads) {
    int failures          = 0;
    constexpr int kLanes  = 3;
    constexpr int kPerRow = 120; // logical pages per row (7,680 tokens)
    const std::string prefix =
        std::string("stage lanes ") + profile_name(storage) + " heads " + std::to_string(kv_heads);
    Pool pool(kLanes * kPerRow, storage, kv_heads, kLanes);
    Region region(pool, 2, 1024, 4);
    int p0[kLanes] = {7000, 3100, 5300};
    // The same block numbers in every lane: a key without the row would serve another row's
    // bytes.
    const auto shared = random_blocks(3000 / 4, 512, 0x1234U);
    for (int step = 0; step < 6; ++step) {
        const int width = step % 2 == 0 ? 4 : 1;
        // Step 3 is a call of the first two lanes only (a request between turns).
        const int lanes = step == 3 ? 2 : kLanes;
        std::vector<Window> per_lane;
        for (int b = 0; b < lanes; ++b) {
            std::vector<int> positions;
            for (int w = 0; w < width; ++w) { positions.push_back(p0[b] + w); }
            per_lane.push_back(
                window(positions, shared, 470, 0xA00U + static_cast<std::uint32_t>(step * 8 + b)));
            pool.rewrite(p0[b], width + 2, 0xB00U + static_cast<std::uint32_t>(step * 8 + b),
                         kLanes - 1 - b);
        }
        const Window w = lanes_window(per_lane);
        Attention a(pool, w, 0xC00U + static_cast<std::uint32_t>(step));
        a.reference(pool, nullptr);
        a.decode(pool, region, step % 2, nullptr);
        cuda_synchronize();
        failures += a.compare(prefix + " step " + std::to_string(step));
        failures += a.check_rows(pool, region, w, prefix + " step " + std::to_string(step));
        for (int b = 0; b < lanes; ++b) { p0[b] += width == 4 ? 2 : 1; }
    }
    const auto totals = region.totals();
    std::cout << prefix << ": hits " << totals[ops::kQsaStagingHits] << " misses "
              << totals[ops::kQsaStagingMisses] << " shared " << totals[ops::kQsaStagingShared]
              << " collisions " << totals[ops::kQsaStagingCollisions] << '\n';
    if (totals[ops::kQsaStagingHits] == 0 || totals[ops::kQsaStagingCollisions] == 0) {
        std::cerr << prefix << ": a cache path was not exercised\n";
        ++failures;
    }
    // Lane 1 truncates: its first position goes back and its rows after the cut change; lanes 0
    // and 2 keep decoding in the same calls.
    {
        const int cut = p0[1] - 300;
        std::vector<int> around;
        for (int b = cut / 4 - 150; b < p0[1] / 4 - 2; ++b) { around.push_back(b); }
        std::vector<Window> warm;
        for (int b = 0; b < kLanes; ++b) {
            warm.push_back(window({p0[b]}, around, 512, 0xD00U + static_cast<std::uint32_t>(b)));
        }
        Attention a0(pool, lanes_window(warm), 0xD10U);
        a0.decode(pool, region, 0, nullptr); // caches every lane's blocks
        pool.rewrite(cut, p0[1] + 8 - cut, 0xD20U, kLanes - 2);
        p0[1] = cut;
        std::vector<Window> next;
        for (int b = 0; b < kLanes; ++b) {
            next.push_back(window({p0[b]}, around, 512, 0xD30U + static_cast<std::uint32_t>(b)));
        }
        const Window w = lanes_window(next);
        Attention a(pool, w, 0xD40U);
        a.reference(pool, nullptr);
        a.decode(pool, region, 0, nullptr);
        cuda_synchronize();
        failures += a.compare(prefix + " after one lane's truncation");
        failures += a.check_rows(pool, region, w, prefix + " after one lane's truncation");
    }
    return failures;
}

// A prompt chunk whose visible context the region cannot hold: the union of the chunk's selections,
// whole and in column ranges, against the attention over device pages.
int test_union(KvCacheStorage storage, std::int32_t kv_heads) {
    int failures             = 0;
    const std::string prefix = std::string("stage selected union ") + profile_name(storage) +
                               " heads " + std::to_string(kv_heads);
    constexpr int kFirst = 15994, kWidth = 6, kMaxVisible = 16064;
    Pool pool(kMaxVisible / kPage, storage, kv_heads);
    // 1024 + 2064 blocks: below the 4016 visible ones, enough for 6 columns of 513 groups.
    Region region(pool, 1, 1024, 4);
    std::vector<int> positions;
    for (int w = 0; w < kWidth; ++w) { positions.push_back(kFirst + w); }
    const auto shared = random_blocks(kFirst / 4, 512, 0x9A1U);
    const Window w    = window(positions, shared, 380, 0x9B0U);
    for (const std::int32_t step : {kWidth, 2}) {
        Attention a(pool, w, 0x9C0U + static_cast<std::uint32_t>(step));
        a.reference(pool, nullptr);
        a.union_prefill(pool, region, kMaxVisible, step, nullptr);
        cuda_synchronize();
        const std::string label = prefix + " step " + std::to_string(step);
        failures += a.compare(label);
        // The last column range's rewrites address the region as it is now; earlier ranges'
        // slots were overwritten, so check the rows of the whole call only with one range.
        if (step == kWidth) { failures += a.check_rows(pool, region, w, label); }
    }
    const auto totals = region.totals();
    // The union of 6 overlapping columns is far below their 6 x 513 groups.
    if (totals[ops::kQsaStagingUnionCalls] != 1 + 3 || totals[ops::kQsaStagingUnionStaged] <= 512 ||
        totals[ops::kQsaStagingUnionStaged] >= 4 * 6 * 513) {
        std::cerr << prefix << ": union counters are wrong (calls "
                  << totals[ops::kQsaStagingUnionCalls] << ", blocks "
                  << totals[ops::kQsaStagingUnionStaged] << ")\n";
        ++failures;
    }
    return failures;
}

// The fetch + attention pair captured once and replayed over changing selections.
int test_graph() {
    int failures = 0;
    Pool pool(200, KvCacheStorage::Int8Group64, 1);
    Region region(pool, 2, 1024, 4);
    const auto shared = random_blocks(12000 / 4, 512, 0x99U);
    Window w          = window({12000, 12001, 12002, 12003}, shared, 480, 0x800U);
    Attention a(pool, w, 0x801U);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    a.decode(pool, region, 0, stream); // eager warm-up
    cuda_synchronize(stream);
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    a.decode(pool, region, 1, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
    for (int round = 0; round < 4; ++round) {
        w = window({12000 + round, 12001 + round, 12002 + round, 12003 + round}, shared, 470,
                   0x900U + static_cast<std::uint32_t>(round));
        a.sel.copy_from_host(w.sel.data(), w.sel.size() * 4);
        a.cnt.copy_from_host(w.cnt.data(), w.cnt.size() * 4);
        a.cache.copy_from_host(w.positions.data(), w.positions.size() * 4);
        a.reference(pool, stream);
        cuda_check(cudaGraphLaunch(executable, stream), "replay");
        cuda_synchronize(stream);
        failures += a.compare("staging graph replay round " + std::to_string(round));
        failures +=
            a.check_rows(pool, region, w, "staging graph replay round " + std::to_string(round));
    }
    cuda_check(cudaGraphExecDestroy(executable), "destroy executable");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        for (const KvCacheStorage storage :
             {KvCacheStorage::Int8Group64, KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256,
              KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
            for (const std::int32_t heads : {2, 1}) {
                failures += test_decode(storage, heads);
                failures += test_stage(storage, heads);
                failures += test_lanes(storage, heads);
                failures += test_union(storage, heads);
            }
        }
        failures += test_graph();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " qsa_staging\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "qsa_staging: " << error.what() << '\n';
        return 1;
    }
}
