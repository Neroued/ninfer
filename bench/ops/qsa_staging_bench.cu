// QSA K/V staging microbenchmark (active KV in RAM, docs/maintainer/active-kv-in-ram.md) at the
// registered geometry (head dim 256): INT8-G64 K/V of one attention layer in pinned, device-mapped
// Host memory (`--heads 1` = one tp 2 rank, `2` = one device), staged into a device region.
//
//   blocks: qsa_stage_selected_blocks for a window of W columns whose 513-block selections share
//           `--overlap` of their blocks with the call before it (0 = every block crosses PCIe,
//           1 = every block is cached); columns of one window share `--column-overlap` of their
//           blocks. The previous call runs untimed before each sample.
//   pages:  qsa_stage_visible_pages for a prefill call that sees `--visible` tokens.
//
// Each sample is one captured call; timings are medians of CUDA event intervals. "link" is the
// number of bytes that cross PCIe divided by the median.
#include "ninfer/ops/sparse_attention.h"
#include "ninfer_bench_common.h"

#include "core/arena.h"
#include "core/paged_kv_storage.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kPage       = kPagedKVPageSize;
constexpr int kHeadDim    = 256;
constexpr int kBlockBytes = 4 * (256 + 8) * 2; // K and V codes and scales of one block and head

std::vector<int> parse_list(const char* text) {
    std::vector<int> out;
    for (const char* p = text; *p != '\0';) {
        out.push_back(std::atoi(p));
        while (*p != '\0' && *p != ',') { ++p; }
        if (*p == ',') { ++p; }
    }
    return out;
}

std::vector<double> parse_doubles(const char* text) {
    std::vector<double> out;
    for (const char* p = text; *p != '\0';) {
        out.push_back(std::atof(p));
        while (*p != '\0' && *p != ',') { ++p; }
        if (*p == ',') { ++p; }
    }
    return out;
}

// One layer of Host planes and a device staging region for `pages` logical pages.
struct Fixture {
    int heads;
    int pages;
    PagedKVStorageLayout layout = paged_kv_storage_layout(KvCacheStorage::Int8Group64, kHeadDim);
    PinnedHostBuffer host;
    DeviceBuffer table, row, region_planes, identity, tags, claims, counters, row_starts, stats;
    PagedKVBatchLayerView kv{};
    ops::QsaStagingRegion region;

    static std::size_t host_bytes(int heads, int pages) {
        return static_cast<std::size_t>(pages) * kPage * heads * (256 + 8) * 2;
    }

    // `cache_blocks` per layer; `staging` raises it so the region holds every page (prefill).
    Fixture(int heads_, int pages_, int cache_blocks, int window, bool staging)
        : heads(heads_), pages(pages_),
          host(host_bytes(heads_, pages_), HostMapping::Yes, PinnedHostPages::Huge) {
        std::memset(host.data(), 0x11, host.size());
        std::vector<int> map(static_cast<std::size_t>(pages));
        std::iota(map.begin(), map.end(), 0);
        std::shuffle(map.begin(), map.end(), std::mt19937(7));
        table = DeviceBuffer(map.size() * 4);
        table.copy_from_host(map.data(), map.size() * 4);
        row              = make_zeros(4);
        auto* base       = static_cast<unsigned char*>(host.device_data());
        const auto plane = [&](unsigned char*& at, DType dtype, int leading) {
            Tensor t(at, dtype, {leading, kPage, heads, pages});
            at += t.bytes();
            return t;
        };
        kv.k_pages       = plane(base, DType::I8, 256);
        kv.v_pages       = plane(base, DType::I8, 256);
        kv.k_scale_pages = plane(base, DType::FP16, 4);
        kv.v_scale_pages = plane(base, DType::FP16, 4);
        kv.block_tables  = Tensor(table.p, DType::I32, {pages, 1});
        kv.head_dim      = kHeadDim;
        kv.num_kv_heads  = heads;
        kv.storage       = KvCacheStorage::Int8Group64;

        region.layers = 1;
        region.cache_blocks =
            staging ? std::max(cache_blocks, (pages * kPage / 4 + 15) / 16 * 16) : cache_blocks;
        region.overflow_blocks = (window * 513 + 15) / 16 * 16;
        const int total        = region.total_pages();
        const std::size_t per  = static_cast<std::size_t>(kPage) * heads;
        region_planes          = DeviceBuffer(per * total * (256 + 256 + 8 + 8));
        auto* r                = static_cast<unsigned char*>(region_planes.p);
        const auto rplane      = [&](DType dtype, int leading) {
            Tensor t(r, dtype, {leading, kPage, heads, total});
            r += t.bytes();
            return t;
        };
        region.pages.k_pages       = rplane(DType::I8, 256);
        region.pages.v_pages       = rplane(DType::I8, 256);
        region.pages.k_scale_pages = rplane(DType::FP16, 4);
        region.pages.v_scale_pages = rplane(DType::FP16, 4);
        std::vector<int> ids(static_cast<std::size_t>(total));
        std::iota(ids.begin(), ids.end(), 0);
        identity = DeviceBuffer(ids.size() * 4);
        identity.copy_from_host(ids.data(), ids.size() * 4);
        region.pages.block_tables = Tensor(identity.p, DType::I32, {total, 1});
        region.pages.head_dim     = kHeadDim;
        region.pages.num_kv_heads = heads;
        region.pages.storage      = KvCacheStorage::Int8Group64;
        region.table_row          = Tensor(row.p, DType::I32, {1});
        tags                      = DeviceBuffer(static_cast<std::size_t>(region.cache_blocks) * 8);
        tags.fill(0xFF);
        claims = make_zeros(static_cast<std::size_t>(region.cache_blocks) * 8);
        std::vector<int> words(ops::kQsaStagingCounterWords, 0);
        words[0] = 1;
        counters = DeviceBuffer(words.size() * 4);
        counters.copy_from_host(words.data(), words.size() * 4);
        row_starts        = make_zeros(4);
        stats             = make_zeros(ops::kQsaStagingStatCount * 8);
        region.row_starts = Tensor(row_starts.p, DType::I32, {1});
        region.tags       = Tensor(tags.p, DType::I64, {region.cache_blocks});
        region.claims     = Tensor(claims.p, DType::I64, {region.cache_blocks});
        region.counters   = Tensor(counters.p, DType::I32, {ops::kQsaStagingCounterWords});
        region.stats      = Tensor(stats.p, DType::I64, {ops::kQsaStagingStatCount});
    }
};

// Selections of W columns over `blocks` complete blocks: 512 sorted blocks per column, the first
// keeping `keep` of `previous`'s first column, the others `column_keep` of the first column.
std::vector<std::vector<int>> windows(int width, int blocks, const std::vector<int>& previous,
                                      double keep, double column_keep, std::mt19937& rng) {
    std::vector<std::vector<int>> out;
    for (int c = 0; c < width; ++c) {
        const std::vector<int>& base = c == 0 ? previous : out[0];
        const double share           = c == 0 ? keep : column_keep;
        std::set<int> chosen;
        std::vector<int> shuffled = base;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        for (int i = 0; i < static_cast<int>(share * 512) && i < static_cast<int>(shuffled.size());
             ++i) {
            chosen.insert(shuffled[static_cast<std::size_t>(i)]);
        }
        std::uniform_int_distribution<int> any(0, blocks - 1);
        while (chosen.size() < 512) { chosen.insert(any(rng)); }
        out.emplace_back(chosen.begin(), chosen.end());
    }
    return out;
}

// [max_selected, W] lists (blocks expanded to tokens, then the 1-3 token tail) and counts.
void expand(const std::vector<std::vector<int>>& blocks, int first_position,
            std::vector<int>& selected, std::vector<int>& counts) {
    const int max_sel = 2051;
    selected.assign(static_cast<std::size_t>(max_sel) * blocks.size(), 0);
    counts.clear();
    for (std::size_t c = 0; c < blocks.size(); ++c) {
        int n = 0;
        for (int b : blocks[c]) {
            for (int j = 0; j < 4; ++j) { selected[c * max_sel + n++] = 4 * b + j; }
        }
        const int position = first_position + static_cast<int>(c);
        // The incomplete tail [4m, n) of a column that sees n = position + 1 tokens (m = n / 4).
        for (int t = (position + 1) / 4 * 4; t <= position; ++t) {
            selected[c * max_sel + n++] = t;
        }
        counts.push_back(n);
    }
}

void bench_blocks(int heads, int width, double overlap, double column_overlap, int visible,
                  int warmup, int repeat) {
    const int pages = (visible + kPage - 1) / kPage + 1;
    // The production cache of a 262,144-token context: 5,472 blocks per layer.
    Fixture f(heads, pages, 5472, width, false);
    const int position = visible - 8;
    const int blocks   = position / 4 - 2;
    std::mt19937 rng(11);
    std::vector<int> seed_blocks(512);
    std::iota(seed_blocks.begin(), seed_blocks.end(), 0);
    const auto first  = windows(width, blocks, seed_blocks, 0.0, column_overlap, rng);
    const auto second = windows(width, blocks, first[0], overlap, column_overlap, rng);
    std::vector<int> sel_a, cnt_a, sel_b, cnt_b, pos(static_cast<std::size_t>(width));
    expand(first, position, sel_a, cnt_a);
    expand(second, position, sel_b, cnt_b);
    std::iota(pos.begin(), pos.end(), position);
    DeviceBuffer da(sel_a.size() * 4), db(sel_b.size() * 4), ca(cnt_a.size() * 4),
        cb(cnt_b.size() * 4), dp(pos.size() * 4), out(sel_a.size() * 4);
    da.copy_from_host(sel_a.data(), sel_a.size() * 4);
    db.copy_from_host(sel_b.data(), sel_b.size() * 4);
    ca.copy_from_host(cnt_a.data(), cnt_a.size() * 4);
    cb.copy_from_host(cnt_b.data(), cnt_b.size() * 4);
    dp.copy_from_host(pos.data(), pos.size() * 4);
    Tensor rewritten(out.p, DType::I32, {2051, width, 1});
    const Tensor positions(dp.p, DType::I32, {width, 1});
    const auto call = [&](const DeviceBuffer& s, const DeviceBuffer& c, cudaStream_t stream) {
        ops::qsa_stage_selected_blocks(Tensor(s.p, DType::I32, {2051, width, 1}),
                                       Tensor(c.p, DType::I32, {width, 1}), positions,
                                       Tensor(f.row.p, DType::I32, {1}), f.kv, f.region, 0, false,
                                       rewritten, stream);
    };
    // The bytes that cross PCIe: the distinct blocks of the timed window absent from the previous.
    std::set<int> before(first[0].begin(), first[0].end()), crossing;
    for (const auto& w : first) { before.insert(w.begin(), w.end()); }
    for (const auto& w : second) {
        for (int b : w) {
            if (before.count(b) == 0) { crossing.insert(b); }
        }
    }
    const double bytes = static_cast<double>(crossing.size() + width) * kBlockBytes * heads;
    const Result r     = bench_loop_prepared(
        [&](cudaStream_t stream) {
            ops::qsa_staging_invalidate(f.region, stream);
            call(da, ca, stream);
        },
        [&](cudaStream_t stream) { call(db, cb, stream); }, bytes, warmup, repeat);
    std::printf("blocks heads=%d W=%d overlap=%.2f column-overlap=%.2f visible=%d: "
                "median %.1f us (min %.1f, p95 %.1f), %zu new blocks, link %.1f GB/s\n",
                heads, width, overlap, column_overlap, visible, r.median_us, r.min_us, r.p95_us,
                crossing.size(), bytes / r.median_us / 1e3);
}

void bench_pages(int heads, int visible, int warmup, int repeat) {
    const int pages = (visible + kPage - 1) / kPage;
    Fixture f(heads, pages, 16, 1, true);
    std::vector<int> pos(2048);
    std::iota(pos.begin(), pos.end(), visible - 2048);
    DeviceBuffer dp(pos.size() * 4);
    dp.copy_from_host(pos.data(), pos.size() * 4);
    const Tensor positions(dp.p, DType::I32, {2048, 1});
    const double bytes = static_cast<double>(pages) * kPage * heads * (256 + 8) * 2;
    const Result r =
        bench_loop_prepared([](cudaStream_t) {},
                            [&](cudaStream_t stream) {
                                ops::qsa_stage_visible_pages(
                                    f.kv, positions, Tensor(), Tensor(f.row.p, DType::I32, {1}),
                                    static_cast<std::uint32_t>(visible), f.region, false, stream);
                            },
                            bytes, warmup, repeat);
    std::printf("pages heads=%d visible=%d: median %.1f us (min %.1f, p95 %.1f), %.1f MiB, link "
                "%.1f GB/s\n",
                heads, visible, r.median_us, r.min_us, r.p95_us, bytes / 1048576.0,
                bytes / r.median_us / 1e3);
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "all";
    std::vector<int> heads{1, 2}, widths{1, 4}, visible{16384, 65536, 131072, 262144};
    std::vector<double> overlap{0.0, 0.9, 1.0};
    double column_overlap = 0.9;
    int warmup = 5, repeat = 31;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const char* next      = i + 1 < argc ? argv[i + 1] : "";
        if (arg == "--mode") {
            mode = next;
            ++i;
        } else if (arg == "--heads") {
            heads = parse_list(next);
            ++i;
        } else if (arg == "--width") {
            widths = parse_list(next);
            ++i;
        } else if (arg == "--visible") {
            visible = parse_list(next);
            ++i;
        } else if (arg == "--overlap") {
            overlap = parse_doubles(next);
            ++i;
        } else if (arg == "--column-overlap") {
            column_overlap = std::atof(next);
            ++i;
        } else if (arg == "--warmup") {
            warmup = std::atoi(next);
            ++i;
        } else if (arg == "--repeat") {
            repeat = std::atoi(next);
            ++i;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--mode all|blocks|pages] [--heads 1,2] [--width 1,4] "
                         "[--visible 16384,...] [--overlap 0,0.9,1] [--column-overlap F] "
                         "[--warmup N] [--repeat N]\n",
                         argv[0]);
            return 2;
        }
    }
    for (const int h : heads) {
        if (mode == "all" || mode == "blocks") {
            for (const int w : widths) {
                for (const double o : overlap) {
                    bench_blocks(h, w, o, column_overlap, 131072, warmup, repeat);
                }
            }
        }
        if (mode == "all" || mode == "pages") {
            for (const int v : visible) { bench_pages(h, v, warmup, repeat); }
        }
    }
    return 0;
}
