// Qwen Sparse Attention Ops at the Qwen3.8-Flash-Next geometry (24/2 heads of 256, indexer 4x128,
// budget 2048 in blocks of 4) over permuted physical pages, plus the two-device shard (12/1).
//
// Ported from giveen/ninfer-ext tests/ops/test_sparse_attention.cpp (Apache-2.0) and extended:
// published pooled block keys, cross-call and speculative-rollback publication, selection over
// deep contexts (staged and unstaged top-k), the prefill multi-pass route, the 12/1 shard, and a
// CUDA Graph replay of the decode chain.
//
// Oracles (all FP64 from represented inputs):
// - qsa_append K/V: byte-identical to kv_cache_append of the same rows (every cache profile);
//   records exact; block keys against the FP64 pooled key, within its BF16 storage rounding.
// - qsa_select: FP64 scores over the contract's pooled keys (BF16 mean, ideal FP64 norm/RoPE,
//   then the documented BF16 storage rounding). A selected block may differ from the oracle only
//   when its oracle score lies within kNearTie (relative to the largest score) of the cut.
// - qsa_attention: FP64 softmax over the device selection, reading K/V at their storage boundary,
//   judged by the registered causal-attention criterion of the storage profile.
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sparse_attention.h"

#include "core/arena.h"
#include "core/paged_kv_storage.h"
#include "ops/attention_criteria.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kIndexHeads = 4;
constexpr std::int32_t kIndexDim   = 128;
constexpr std::int32_t kRatio      = 4;
constexpr std::int32_t kRotary     = 64;
constexpr double kEps              = 1.0e-6;
constexpr std::int32_t kPage       = kPagedKVPageSize;
// A selection may differ from the oracle only at a near tie of the cut. The device scores use
// FP32 prepared queries and FP32 accumulation over the same BF16 block keys the oracle rounds to;
// a one-ulp BF16 disagreement of a key element moves a score by ~4e-4 of its scale.
constexpr double kNearTie = 2.0e-3; // relative to the largest block score

const ops::QsaGeometry kGeometry = ops::qwen3_8_flash_next_qsa_geometry();

std::uint16_t bf16_bits(double x) { return f32_to_bf16(static_cast<float>(x)); }

double bf16_round(double x) { return bf16_to_f32(bf16_bits(x)); }

// Deterministic BF16 test value in [lo, hi) for any 64-bit key.
float hashed(std::uint64_t key, float lo, float hi) {
    key += 0x9e3779b97f4a7c15ULL;
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    key ^= key >> 31;
    const double unit = static_cast<double>(key >> 11) * 0x1.0p-53;
    return bf16_to_f32(f32_to_bf16(static_cast<float>(lo + (hi - lo) * unit)));
}

std::uint64_t mix(std::uint64_t a, std::uint64_t b, std::uint64_t c = 0, std::uint64_t d = 0) {
    return (((a * 1000003ULL + b) * 1000033ULL + c) * 1000037ULL) + d;
}

// Three-axis positions of a token: distinct per axis so the pair-to-axis mapping matters.
std::int32_t axis_position(std::int32_t token, int axis) { return token / (axis + 1) + 3 * axis; }

double half_value(std::uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, 2);
    return static_cast<double>(__half2float(h));
}

double e4m3_value(std::uint8_t code) {
    const int e    = (code >> 3) & 0xF;
    const int m    = code & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return (code & 0x80) != 0 ? -v : v;
}

double e2m1_value(std::uint8_t nibble) {
    static constexpr double kMagnitude[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double v                        = kMagnitude[nibble & 7];
    return (nibble & 8) != 0 ? -v : v;
}

// Normalized natural-order Sylvester transform, H[i][j] = (-1)^popcount(i & j) / 16.
std::vector<double> hadamard256(std::vector<double> v) {
    for (std::size_t span = 1; span < v.size(); span <<= 1) {
        for (std::size_t base = 0; base < v.size(); base += 2 * span) {
            for (std::size_t i = base; i < base + span; ++i) {
                const double low = v[i], high = v[i + span];
                v[i]        = low + high;
                v[i + span] = low - high;
            }
        }
    }
    for (double& x : v) { x /= 16.0; }
    return v;
}

std::vector<double> offset_norm(const std::vector<double>& x, const std::vector<float>& weight) {
    double sum = 0.0;
    for (const double v : x) { sum += v * v; }
    const double inv = 1.0 / std::sqrt(sum / static_cast<double>(x.size()) + kEps);
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { out[i] = x[i] * inv * (1.0 + double(weight[i])); }
    return out;
}

// The contract's FP32 inverse-frequency table and FP32 angle, then FP64 rotation.
std::vector<double> rotate(std::vector<double> v, const std::int32_t position[3]) {
    for (int i = 0; i < kRotary / 2; ++i) {
        const float inv_freq =
            1.0F / std::pow(1.0e7F, static_cast<float>(2 * i) / static_cast<float>(kRotary));
        // The contract's angle is the FP32 product (Hugging Face); the rotation is FP64.
        const double angle = static_cast<double>(
            static_cast<float>(position[kGeometry.pair_axes[static_cast<std::size_t>(i)]]) *
            inv_freq);
        const double x0 = v[i], x1 = v[i + kRotary / 2];
        v[i]               = x0 * std::cos(angle) - x1 * std::sin(angle);
        v[i + kRotary / 2] = x1 * std::cos(angle) + x0 * std::sin(angle);
    }
    return v;
}

// ---------------------------------------------------------------------------------------------
// A paged QSA layer of `lanes` sequences with `capacity` tokens each, over one KV profile.

struct Layer {
    std::int32_t lanes;
    std::int32_t capacity;
    std::int32_t kv_heads;
    std::int32_t pages_per_lane;
    std::int32_t pages;
    KvCacheStorage storage;
    PagedKVStorageLayout layout;
    std::vector<int> table_row;         // per lane
    std::vector<std::vector<int>> page; // per lane: logical -> physical
    std::vector<int> tables;            // I32 [pages_per_lane, lanes]
    DeviceBuffer d_tables, d_rows;
    DeviceBuffer k_pages, v_pages, k_scales, v_scales, records, block_keys;
    PagedKVBatchLayerView kv{};
    ops::QsaIndexPlanes index{};
    // Host copy of the raw index keys as stored, per lane [capacity][128].
    std::vector<std::vector<float>> raw;

    Layer(std::int32_t lanes_, std::int32_t capacity_, KvCacheStorage storage_,
          std::int32_t kv_heads_ = 2)
        : lanes(lanes_), capacity(capacity_), kv_heads(kv_heads_),
          pages_per_lane((capacity_ + kPage - 1) / kPage), pages(pages_per_lane * lanes_),
          storage(storage_), layout(paged_kv_storage_layout(storage_, kHeadDim)) {
        // Lane b uses table row lanes-1-b; physical pages interleave across lanes, reversed.
        table_row.resize(lanes);
        page.resize(lanes);
        tables.assign(static_cast<std::size_t>(pages_per_lane) * lanes, -1);
        for (int b = 0; b < lanes; ++b) {
            table_row[b] = lanes - 1 - b;
            for (int p = 0; p < pages_per_lane; ++p) {
                const int physical = (pages_per_lane - 1 - p) * lanes + b;
                page[b].push_back(physical);
                tables[static_cast<std::size_t>(table_row[b]) * pages_per_lane + p] = physical;
            }
        }
        d_tables         = to_device_i32(tables);
        d_rows           = to_device_i32(table_row);
        const auto plane = [&](DeviceBuffer& buffer, DType dtype, std::int32_t extent) {
            const std::size_t bytes =
                static_cast<std::size_t>(extent) * kPage * kv_heads * pages * dtype_size(dtype);
            buffer = DeviceBuffer(std::max<std::size_t>(bytes, 256));
            buffer.fill(0);
            return Tensor(buffer.p, dtype, {extent, kPage, kv_heads, pages});
        };
        kv.k_pages = plane(k_pages, layout.key.data_dtype, layout.key.data_leading_extent);
        kv.v_pages = plane(v_pages, layout.value.data_dtype, layout.value.data_leading_extent);
        if (layout.key.has_scale()) {
            kv.k_scale_pages =
                plane(k_scales, layout.key.scale_dtype, layout.key.scale_leading_extent);
        }
        if (layout.value.has_scale()) {
            kv.v_scale_pages =
                plane(v_scales, layout.value.scale_dtype, layout.value.scale_leading_extent);
        }
        kv.block_tables = Tensor(d_tables.p, DType::I32, {pages_per_lane, lanes});
        kv.head_dim     = kHeadDim;
        kv.num_kv_heads = kv_heads;
        kv.storage      = storage;
        records =
            DeviceBuffer(static_cast<std::size_t>(ops::kQsaIndexRecordWords) * kPage * pages * 2);
        records.fill(0);
        block_keys =
            DeviceBuffer(static_cast<std::size_t>(ops::kQsaBlockKeyWords) * kPage * pages * 2);
        block_keys.fill(0);
        index.records =
            Tensor(records.p, DType::BF16, {ops::kQsaIndexRecordWords, kPage, 1, pages});
        index.block_keys =
            Tensor(block_keys.p, DType::BF16, {ops::kQsaBlockKeyWords, kPage, 1, pages});
        index.block_tables = kv.block_tables;
        raw.assign(lanes, std::vector<float>(static_cast<std::size_t>(capacity) * kIndexDim, 0.0F));
    }

    Tensor rows() const { return Tensor(d_rows.p, DType::I32, {lanes}); }

    // Physical row (page group, head, offset) of (lane, token, head) in a page-major plane.
    std::size_t row_of(int lane, int token, int head) const {
        return (static_cast<std::size_t>(page[lane][token / kPage]) * kv_heads + head) * kPage +
               token % kPage;
    }
};

// Token values. `salt` != 0 produces a different (garbage) version of the same token.
float k_value(int lane, int token, int head, int d, int salt) {
    return hashed(mix(1, lane, token, head * 1000 + d) + salt * 7919ULL, -2.0F, 2.0F);
}

float v_value(int lane, int token, int head, int d, int salt) {
    return hashed(mix(2, lane, token, head * 1000 + d) + salt * 7919ULL, -2.0F, 2.0F);
}

float index_key_value(int lane, int token, int d, int salt) {
    return hashed(mix(3, lane, token, d) + salt * 7919ULL, -2.0F, 2.0F);
}

std::vector<float> norm_weight(std::uint64_t seed) {
    std::vector<float> w(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) { w[d] = hashed(mix(seed, d), -0.3F, 0.3F); }
    return w;
}

const std::vector<float> kKeyNorm   = norm_weight(0x23);
const std::vector<float> kQueryNorm = norm_weight(0x22);

struct Window {
    std::int32_t width;
    std::vector<std::vector<int>> positions; // per lane, W positions (tail repeats the last)
    std::vector<int> valid;                  // per lane
    int salt = 0;
};

// Appends one [W,B] window through qsa_append and mirrors the stored raw keys on the host.
void append(Layer& layer, const Window& window, const DeviceBuffer& key_norm, cudaStream_t stream) {
    const int W = window.width, B = layer.lanes;
    const std::size_t columns = static_cast<std::size_t>(W) * B;
    std::vector<std::uint16_t> k(columns * layer.kv_heads * kHeadDim),
        v(columns * layer.kv_heads * kHeadDim), idx(columns * kIndexDim);
    std::vector<int> cache(columns), rope(columns * 3);
    for (int b = 0; b < B; ++b) {
        for (int w = 0; w < W; ++w) {
            const std::size_t c = static_cast<std::size_t>(b) * W + w;
            const int t         = window.positions[b][w];
            cache[c]            = t;
            for (int a = 0; a < 3; ++a) { rope[a * columns + c] = axis_position(t, a); }
            for (int h = 0; h < layer.kv_heads; ++h) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const std::size_t at = (c * layer.kv_heads + h) * kHeadDim + d;
                    k[at]                = f32_to_bf16(k_value(b, t, h, d, window.salt));
                    v[at]                = f32_to_bf16(v_value(b, t, h, d, window.salt));
                }
            }
            for (int d = 0; d < kIndexDim; ++d) {
                const float x          = index_key_value(b, t, d, window.salt);
                idx[c * kIndexDim + d] = f32_to_bf16(x);
                if (w < window.valid[b]) {
                    layer.raw[b][static_cast<std::size_t>(t) * kIndexDim + d] = x;
                }
            }
        }
    }
    DeviceBuffer dk = to_device(k), dv = to_device(v), di = to_device(idx);
    DeviceBuffer dc = to_device_i32(cache), dr = to_device_i32(rope),
                 dvalid = to_device_i32(window.valid);
    ops::qsa_append(Tensor(dk.p, DType::BF16, {kHeadDim, layer.kv_heads, W, B}),
                    Tensor(dv.p, DType::BF16, {kHeadDim, layer.kv_heads, W, B}),
                    Tensor(di.p, DType::BF16, {kIndexDim, W, B}),
                    Tensor(dr.p, DType::I32, {static_cast<std::int32_t>(columns), 3}),
                    Tensor(dc.p, DType::I32, {W, B}), Tensor(dvalid.p, DType::I32, {B}),
                    layer.rows(), Tensor(key_norm.p, DType::BF16, {kIndexDim}), kGeometry, layer.kv,
                    layer.index, stream);
    cuda_synchronize(stream);
}

// One dense window per lane covering tokens [first, first+count).
Window bulk(int lanes, int first, int count) {
    Window w{count, std::vector<std::vector<int>>(lanes), std::vector<int>(lanes, count)};
    for (int b = 0; b < lanes; ++b) {
        for (int t = 0; t < count; ++t) { w.positions[b].push_back(first + t); }
    }
    return w;
}

// FP64 pooled key of block `blk` of a lane, rounded once to BF16 as the contract publishes it.
std::vector<double> pooled_key(const Layer& layer, int lane, int blk, bool round_storage) {
    std::vector<double> mean(kIndexDim, 0.0);
    for (int t = 0; t < kRatio; ++t) {
        for (int d = 0; d < kIndexDim; ++d) {
            mean[d] +=
                layer.raw[lane][(static_cast<std::size_t>(blk) * kRatio + t) * kIndexDim + d];
        }
    }
    for (double& m : mean) { m = bf16_round(m / kRatio); } // upstream pools in BF16
    std::int32_t pos[3];
    for (int a = 0; a < 3; ++a) { pos[a] = axis_position(blk * kRatio, a); }
    auto key = rotate(offset_norm(mean, kKeyNorm), pos);
    if (round_storage) {
        for (double& x : key) { x = bf16_round(x); }
    }
    return key;
}

int check_block_keys(const Layer& layer, int lane, int tokens, const std::string& label) {
    const auto stored = from_device<std::uint16_t>(
        layer.block_keys, static_cast<std::size_t>(ops::kQsaBlockKeyWords) * kPage * layer.pages);
    std::vector<double> got, ref;
    for (int blk = 0; blk < tokens / kRatio; ++blk) {
        const std::size_t base =
            (static_cast<std::size_t>(layer.page[lane][blk * kRatio / kPage]) * kPage +
             (blk * kRatio) % kPage) *
            ops::kQsaBlockKeyWords;
        const auto key = pooled_key(layer, lane, blk, false);
        for (int d = 0; d < kIndexDim; ++d) {
            got.push_back(bf16_to_f32(stored[base + d]));
            ref.push_back(key[d]);
        }
    }
    // BF16 storage rounding of an FP32 evaluation: at most one BF16 ulp (2^-8 relative).
    return verify_pointwise(label, got, ref, {1.0e-5, 0x1.0p-8});
}

// ---------------------------------------------------------------------------------------------
// Decoding a stored cache row for the attention oracle.

struct HostCache {
    const Layer* layer;
    std::vector<std::uint8_t> k, v, ks, vs;

    explicit HostCache(const Layer& l) : layer(&l) {
        const auto get = [&](const Tensor& t, const DeviceBuffer& buffer) {
            if (t.data == nullptr) { return std::vector<std::uint8_t>{}; }
            return from_device<std::uint8_t>(buffer, static_cast<std::size_t>(t.bytes()));
        };
        k  = get(l.kv.k_pages, l.k_pages);
        v  = get(l.kv.v_pages, l.v_pages);
        ks = get(l.kv.k_scale_pages, l.k_scales);
        vs = get(l.kv.v_scale_pages, l.v_scales);
    }

    static double decode(DType dtype, const std::vector<std::uint8_t>& codes,
                         const std::vector<std::uint8_t>& scales, std::size_t row, int d) {
        const auto half_at = [&](std::size_t i) {
            return half_value(static_cast<std::uint16_t>(scales[2 * i] | (scales[2 * i + 1] << 8)));
        };
        const auto u16 = [&](std::size_t i) {
            return static_cast<std::uint16_t>(codes[2 * i] | (codes[2 * i + 1] << 8));
        };
        switch (dtype) {
        case DType::BF16:
            return bf16_to_f32(u16(row * kHeadDim + d));
        case DType::FP16:
            return half_value(u16(row * kHeadDim + d));
        case DType::FP8_E4M3FN:
            return e4m3_value(codes[row * kHeadDim + d]) * half_at(row);
        case DType::I8:
            return double(static_cast<std::int8_t>(codes[row * kHeadDim + d])) *
                   half_at(row * 4 + d / 64);
        default: {
            const std::uint8_t byte = codes[row * (kHeadDim / 2) + d / 2];
            return e2m1_value(d % 2 != 0 ? byte >> 4 : byte & 0xF) *
                   e4m3_value(scales[row * 16 + d / 16]);
        }
        }
    }

    double key(std::size_t row, int d) const {
        return decode(layer->layout.key.data_dtype, k, ks, row, d);
    }

    double value(std::size_t row, int d) const {
        return decode(layer->layout.value.data_dtype, v, vs, row, d);
    }
};

bool quantized(KvCacheStorage s) { return s != KvCacheStorage::BFloat16; }

bool v_rotated(KvCacheStorage s) {
    return s == KvCacheStorage::Nvfp4Group16 || s == KvCacheStorage::Fp8KeyNvfp4Value;
}

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

// qsa_attention over `sel`/`cnt` ([max_selected, W, B] / [W, B]) against the FP64 oracle.
int attend(const Layer& layer, const std::string& label, std::int32_t width,
           std::int32_t query_heads, const std::vector<int>& sel, const std::vector<int>& cnt,
           std::uint32_t seed) {
    const int B       = layer.lanes;
    const int max_sel = kGeometry.max_selected();
    const HostCache host(layer);
    std::vector<float> q(static_cast<std::size_t>(kHeadDim) * query_heads * width * B);
    for (std::size_t i = 0; i < q.size(); ++i) { q[i] = hashed(mix(seed, i), -1.0F, 1.0F); }
    std::vector<std::uint16_t> qb(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) { qb[i] = f32_to_bf16(q[i]); }
    DeviceBuffer dq = to_device(qb), dsel = to_device_i32(sel), dcnt = to_device_i32(cnt);
    GuardedDeviceBuffer out(q.size() * 2);
    Tensor tq(dq.p, DType::BF16, {kHeadDim, query_heads, width, B});
    Tensor ts(dsel.p, DType::I32, {max_sel, width, B});
    Tensor tc(dcnt.p, DType::I32, {width, B});
    Tensor tout(out.data(), DType::BF16, {kHeadDim, query_heads, width, B});
    const float scale  = 1.0F / 16.0F;
    ops::QsaGeometry g = kGeometry;
    g.query_heads      = query_heads;
    g.kv_heads         = layer.kv_heads;
    WorkspaceArena workspace(ops::qsa_attention_workspace_bytes(g, width * B));
    ops::qsa_attention(tq, ts, tc, layer.rows(), layer.kv, scale, workspace, tout, nullptr);
    cuda_synchronize();
    if (const char* dir = std::getenv("NINFER_TEST_DUMP_DIR"); dir != nullptr && *dir != '\0') {
        // The raw output of every call, numbered in call order, for bit comparisons between builds.
        static int dump_index = 0;
        const auto host       = from_device<std::uint16_t>(out.data(), q.size());
        const std::string path =
            std::string(dir) + "/attend_" + std::to_string(dump_index++) + ".bin";
        if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(host.data(), 2, host.size(), f);
            std::fclose(f);
        }
    }
    std::vector<double> reference(q.size(), 0.0);
    const int group = query_heads / layer.kv_heads;
    for (int b = 0; b < B; ++b) {
        for (int w = 0; w < width; ++w) {
            const int c     = b * width + w;
            const int count = cnt[c];
            const int* list = sel.data() + static_cast<std::ptrdiff_t>(c) * max_sel;
            for (int h = 0; h < query_heads; ++h) {
                const int kvh = h / group;
                const float* qrow =
                    q.data() + (static_cast<std::size_t>(c) * query_heads + h) * kHeadDim;
                std::vector<double> query(qrow, qrow + kHeadDim);
                if (quantized(layer.storage)) { query = hadamard256(std::move(query)); }
                std::vector<double> logits(count);
                double max_logit = -1e300;
                for (int j = 0; j < count; ++j) {
                    const std::size_t row = layer.row_of(b, list[j], kvh);
                    double dot            = 0.0;
                    for (int d = 0; d < kHeadDim; ++d) { dot += query[d] * host.key(row, d); }
                    logits[j] = dot * scale;
                    max_logit = std::max(max_logit, logits[j]);
                }
                double denominator = 0.0;
                for (double& l : logits) {
                    l = std::exp(l - max_logit);
                    denominator += l;
                }
                std::vector<double> row(kHeadDim, 0.0);
                for (int j = 0; j < count; ++j) {
                    const std::size_t r = layer.row_of(b, list[j], kvh);
                    for (int d = 0; d < kHeadDim; ++d) {
                        row[d] += logits[j] / denominator * host.value(r, d);
                    }
                }
                if (v_rotated(layer.storage)) { row = hadamard256(std::move(row)); }
                std::copy(row.begin(), row.end(),
                          reference.begin() +
                              static_cast<std::ptrdiff_t>(
                                  (static_cast<std::size_t>(c) * query_heads + h) * kHeadDim));
            }
        }
    }
    return verify_reduction(label, from_device_bf16(out.data(), q.size()), reference,
                            attention_criterion(layer.storage)) +
           out.verify_guards(label);
}

// A sorted, duplicate-free selection of `count` of the lane's first `tokens` tokens.
std::vector<int> pick(int tokens, int count, std::uint32_t seed) {
    std::vector<int> all(tokens);
    std::iota(all.begin(), all.end(), 0);
    std::uint32_t state = seed;
    for (int i = tokens - 1; i > 0; --i) {
        state = state * 1664525U + 1013904223U;
        std::swap(all[i], all[state % static_cast<std::uint32_t>(i + 1)]);
    }
    std::vector<int> out(all.begin(), all.begin() + count);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> synthetic(int width, int lanes, int tokens, const std::vector<int>& counts) {
    const int max_sel = kGeometry.max_selected();
    std::vector<int> sel(static_cast<std::size_t>(max_sel) * width * lanes, 0);
    for (std::size_t c = 0; c < counts.size(); ++c) {
        const auto list = pick(tokens, counts[c], 0x40U + static_cast<std::uint32_t>(c));
        std::copy(list.begin(), list.end(), sel.begin() + static_cast<std::ptrdiff_t>(c) * max_sel);
    }
    return sel;
}

// ---------------------------------------------------------------------------------------------
// Selection

struct SelectResult {
    std::vector<int> selected;
    std::vector<int> counts;
};

struct QueryWindow {
    std::int32_t width;
    std::vector<std::vector<int>> positions; // per lane
    std::vector<int> valid;
    std::uint32_t seed;
};

std::vector<float> index_query(const QueryWindow& qw, int lanes) {
    std::vector<float> q(static_cast<std::size_t>(kIndexHeads) * kIndexDim * qw.width * lanes);
    for (std::size_t i = 0; i < q.size(); ++i) { q[i] = hashed(mix(qw.seed, i), -2.0F, 2.0F); }
    return q;
}

SelectResult run_select(const Layer& layer, const QueryWindow& qw, std::uint32_t max_visible,
                        cudaStream_t stream) {
    const int W = qw.width, B = layer.lanes;
    const std::size_t columns = static_cast<std::size_t>(W) * B;
    const auto q              = index_query(qw, B);
    std::vector<std::uint16_t> qb(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) { qb[i] = f32_to_bf16(q[i]); }
    std::vector<int> cache(columns), rope(columns * 3);
    for (int b = 0; b < B; ++b) {
        for (int w = 0; w < W; ++w) {
            const std::size_t c = static_cast<std::size_t>(b) * W + w;
            cache[c]            = qw.positions[b][w];
            for (int a = 0; a < 3; ++a) { rope[a * columns + c] = axis_position(cache[c], a); }
        }
    }
    std::vector<std::uint16_t> qn(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) { qn[d] = f32_to_bf16(kQueryNorm[d]); }
    DeviceBuffer dq = to_device(qb), dc = to_device_i32(cache), dr = to_device_i32(rope),
                 dvalid = to_device_i32(qw.valid), dqn = to_device(qn);
    const int max_sel = kGeometry.max_selected();
    GuardedDeviceBuffer selected(static_cast<std::size_t>(max_sel) * columns * 4);
    GuardedDeviceBuffer counts(columns * 4);
    Tensor tsel(selected.data(), DType::I32, {max_sel, W, B});
    Tensor tcnt(counts.data(), DType::I32, {W, B});
    WorkspaceArena workspace(ops::qsa_select_workspace_bytes(kGeometry, max_visible, W, B));
    ops::qsa_select(Tensor(dq.p, DType::BF16, {kIndexHeads * kIndexDim, W, B}),
                    Tensor(dr.p, DType::I32, {static_cast<std::int32_t>(columns), 3}),
                    Tensor(dc.p, DType::I32, {W, B}), Tensor(dvalid.p, DType::I32, {B}),
                    layer.rows(), Tensor(dqn.p, DType::BF16, {kIndexDim}), layer.index, kGeometry,
                    max_visible, workspace, tsel, tcnt, stream);
    cuda_synchronize(stream);
    if (const char* dir = std::getenv("NINFER_TEST_DUMP_DIR"); dir != nullptr && *dir != '\0') {
        // Every selection (ids, then counts), numbered in call order, for bit comparisons.
        static int select_index = 0;
        const auto ids =
            from_device<int>(selected.data(), static_cast<std::size_t>(max_sel) * columns);
        const auto cnt = from_device<int>(counts.data(), columns);
        const std::string path =
            std::string(dir) + "/select_" + std::to_string(select_index++) + ".bin";
        if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(ids.data(), 4, ids.size(), f);
            std::fwrite(cnt.data(), 4, cnt.size(), f);
            std::fclose(f);
        }
    }
    if (selected.verify_guards("qsa_select selected") + counts.verify_guards("qsa_select counts") !=
        0) {
        throw std::runtime_error("qsa_select wrote outside its outputs");
    }
    return {from_device<int>(selected.data(), static_cast<std::size_t>(max_sel) * columns),
            from_device<int>(counts.data(), columns)};
}

// Compares every valid column with the FP64 oracle (near ties at the cut allowed).
int check_select(const Layer& layer, const QueryWindow& qw, const SelectResult& r,
                 const std::string& name, int* differing_blocks = nullptr) {
    int failures      = 0;
    const int W       = qw.width;
    const int max_sel = kGeometry.max_selected();
    const auto q      = index_query(qw, layer.lanes);
    std::map<std::pair<int, int>, std::vector<double>> keys; // (lane, block) -> key
    for (int b = 0; b < layer.lanes; ++b) {
        for (int w = 0; w < W; ++w) {
            const int c = b * W + w;
            const std::string label =
                name + " lane=" + std::to_string(b) + " column=" + std::to_string(w);
            if (w >= qw.valid[b]) {
                if (r.counts[c] != 0) {
                    std::cerr << label << ": invalid column has a selection\n";
                    ++failures;
                }
                continue;
            }
            const int n      = qw.positions[b][w] + 1;
            const int blocks = n / kRatio;
            std::vector<int> expected;
            std::vector<double> scores(blocks, 0.0);
            if (blocks <= kGeometry.block_topk()) {
                expected.resize(n);
                std::iota(expected.begin(), expected.end(), 0);
            } else {
                std::int32_t qpos[3];
                for (int a = 0; a < 3; ++a) { qpos[a] = axis_position(qw.positions[b][w], a); }
                std::vector<std::vector<double>> queries(kIndexHeads);
                for (int h = 0; h < kIndexHeads; ++h) {
                    std::vector<double> raw(kIndexDim);
                    for (int d = 0; d < kIndexDim; ++d) {
                        raw[d] = q[(static_cast<std::size_t>(c) * kIndexHeads + h) * kIndexDim + d];
                    }
                    queries[h] = rotate(offset_norm(raw, kQueryNorm), qpos);
                }
                for (int blk = 0; blk < blocks; ++blk) {
                    auto& key = keys[{b, blk}];
                    if (key.empty()) { key = pooled_key(layer, b, blk, true); }
                    double score = 0.0;
                    for (const auto& qh : queries) {
                        double dot = 0.0;
                        for (int d = 0; d < kIndexDim; ++d) { dot += qh[d] * key[d]; }
                        score += std::max(dot, 0.0);
                    }
                    scores[blk] = score / std::sqrt(double(kIndexDim));
                }
                std::vector<int> order(blocks);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                                 [&](int x, int y) { return scores[x] > scores[y]; });
                std::vector<int> kept(order.begin(), order.begin() + kGeometry.block_topk());
                std::sort(kept.begin(), kept.end());
                for (const int blk : kept) {
                    for (int t = 0; t < kRatio; ++t) { expected.push_back(blk * kRatio + t); }
                }
                for (int t = blocks * kRatio; t < n; ++t) { expected.push_back(t); }
            }
            const std::vector<int> got(
                r.selected.begin() + static_cast<std::ptrdiff_t>(c) * max_sel,
                r.selected.begin() + static_cast<std::ptrdiff_t>(c) * max_sel + r.counts[c]);
            if (got.size() != expected.size() || !std::is_sorted(got.begin(), got.end()) ||
                std::adjacent_find(got.begin(), got.end()) != got.end()) {
                std::cerr << label << ": count " << got.size() << " expected " << expected.size()
                          << " (or unsorted / duplicated)\n";
                ++failures;
                continue;
            }
            if (got != expected) {
                std::vector<double> sorted = scores;
                std::sort(sorted.rbegin(), sorted.rend());
                const double cut   = sorted[kGeometry.block_topk() - 1];
                const double scale = sorted.front();
                std::set<int> a(got.begin(), got.end()), e(expected.begin(), expected.end());
                bool near  = true;
                int differ = 0;
                for (int t = 0; t < n; ++t) {
                    if (a.count(t) != e.count(t)) {
                        const int blk = t / kRatio;
                        near =
                            near && blk < blocks && std::abs(scores[blk] - cut) <= kNearTie * scale;
                        differ += t % kRatio == 0;
                    }
                }
                if (differing_blocks != nullptr) { *differing_blocks += differ; }
                if (!near) {
                    std::cerr << label << ": selection differs beyond near ties\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------

int test_append_profiles() {
    int failures          = 0;
    constexpr int kLanes  = 2;
    constexpr int kTokens = 300;
    std::vector<std::uint16_t> kn(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) { kn[d] = f32_to_bf16(kKeyNorm[d]); }
    DeviceBuffer dkn = to_device(kn);
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        Layer layer(kLanes, kTokens, storage);
        append(layer, bulk(kLanes, 0, kTokens), dkn, nullptr);
        // The same rows through kv_cache_append, lane by lane, into a second cache.
        Layer reference(kLanes, kTokens, storage);
        for (int b = 0; b < kLanes; ++b) {
            std::vector<std::uint16_t> k(static_cast<std::size_t>(kTokens) * 2 * kHeadDim),
                v(k.size());
            std::vector<int> positions(kTokens);
            for (int t = 0; t < kTokens; ++t) {
                positions[t] = t;
                for (int h = 0; h < 2; ++h) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        const std::size_t at = (static_cast<std::size_t>(t) * 2 + h) * kHeadDim + d;
                        k[at]                = f32_to_bf16(k_value(b, t, h, d, 0));
                        v[at]                = f32_to_bf16(v_value(b, t, h, d, 0));
                    }
                }
            }
            DeviceBuffer dk = to_device(k), dv = to_device(v), dp = to_device_i32(positions);
            DeviceBuffer row_table = to_device_i32(reference.page[b]);
            PagedKVLayerView view{.k_pages       = reference.kv.k_pages,
                                  .v_pages       = reference.kv.v_pages,
                                  .k_scale_pages = reference.kv.k_scale_pages,
                                  .v_scale_pages = reference.kv.v_scale_pages,
                                  .block_table =
                                      Tensor(row_table.p, DType::I32, {reference.pages_per_lane}),
                                  .head_dim     = kHeadDim,
                                  .num_kv_heads = 2,
                                  .storage      = storage};
            ops::kv_cache_append(Tensor(dk.p, DType::BF16, {kHeadDim, 2, kTokens}),
                                 Tensor(dv.p, DType::BF16, {kHeadDim, 2, kTokens}),
                                 Tensor(dp.p, DType::I32, {kTokens}), view, nullptr);
            cuda_synchronize();
        }
        const std::string name = std::string("qsa_append ") + profile_name(storage);
        const auto bytes_of    = [](const Tensor& t, const DeviceBuffer& b) {
            return t.data == nullptr
                          ? std::vector<std::uint8_t>{}
                          : from_device<std::uint8_t>(b, static_cast<std::size_t>(t.bytes()));
        };
        failures += verify_exact((name + " K").c_str(), bytes_of(layer.kv.k_pages, layer.k_pages),
                                 bytes_of(reference.kv.k_pages, reference.k_pages));
        failures += verify_exact((name + " V").c_str(), bytes_of(layer.kv.v_pages, layer.v_pages),
                                 bytes_of(reference.kv.v_pages, reference.v_pages));
        failures += verify_exact((name + " K scales").c_str(),
                                 bytes_of(layer.kv.k_scale_pages, layer.k_scales),
                                 bytes_of(reference.kv.k_scale_pages, reference.k_scales));
        failures += verify_exact((name + " V scales").c_str(),
                                 bytes_of(layer.kv.v_scale_pages, layer.v_scales),
                                 bytes_of(reference.kv.v_scale_pages, reference.v_scales));
        // Records: raw keys and the three positions, exactly.
        const auto records = from_device<std::uint16_t>(
            layer.records,
            static_cast<std::size_t>(ops::kQsaIndexRecordWords) * kPage * layer.pages);
        int mismatches = 0;
        for (int b = 0; b < kLanes; ++b) {
            for (int t = 0; t < kTokens; ++t) {
                const std::size_t base =
                    (static_cast<std::size_t>(layer.page[b][t / kPage]) * kPage + t % kPage) *
                    ops::kQsaIndexRecordWords;
                for (int d = 0; d < kIndexDim; ++d) {
                    mismatches += records[base + d] != f32_to_bf16(index_key_value(b, t, d, 0));
                }
                for (int a = 0; a < 3; ++a) {
                    const auto p = static_cast<std::uint32_t>(axis_position(t, a));
                    mismatches += records[base + kIndexDim + 2 * a] != (p & 0xFFFFU);
                    mismatches += records[base + kIndexDim + 2 * a + 1] != (p >> 16);
                }
            }
        }
        if (mismatches != 0) {
            std::cerr << name << ": index records differ from their inputs\n";
            ++failures;
        }
        if (storage == KvCacheStorage::BFloat16) {
            for (int b = 0; b < kLanes; ++b) {
                failures += check_block_keys(layer, b, kTokens,
                                             name + " block keys lane " + std::to_string(b));
            }
        }
    }
    return failures;
}

// Blocks published across calls and after a speculative rollback equal the one-call publication.
int test_incremental_publication() {
    int failures         = 0;
    constexpr int kLanes = 2;
    constexpr int kEnd   = 403;
    std::vector<std::uint16_t> kn(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) { kn[d] = f32_to_bf16(kKeyNorm[d]); }
    DeviceBuffer dkn = to_device(kn);
    Layer once(kLanes, kEnd + 8, KvCacheStorage::BFloat16);
    append(once, bulk(kLanes, 0, kEnd), dkn, nullptr);

    Layer steps(kLanes, kEnd + 8, KvCacheStorage::BFloat16);
    append(steps, bulk(kLanes, 0, 101), dkn, nullptr);
    std::vector<int> next(kLanes, 101);
    std::uint32_t state = 7;
    constexpr int W     = 4;
    while (next[0] < kEnd || next[1] < kEnd) {
        // A rejected speculative window writes different keys at next..next+3 (publishing any
        // block it completes); the accepted prefix is then re-appended with the real keys.
        Window garbage{W, std::vector<std::vector<int>>(kLanes), std::vector<int>(kLanes, 0)};
        Window real{W, std::vector<std::vector<int>>(kLanes), std::vector<int>(kLanes, 0)};
        garbage.salt = 1;
        for (int b = 0; b < kLanes; ++b) {
            state            = state * 1664525U + 1013904223U;
            const int accept = std::min(1 + static_cast<int>(state >> 30), kEnd - next[b]);
            for (int w = 0; w < W; ++w) {
                garbage.positions[b].push_back(next[b] + w);
                real.positions[b].push_back(accept > 0 ? next[b] + std::min(w, accept - 1) : 0);
            }
            garbage.valid[b] = accept > 0 ? W : 0;
            real.valid[b]    = accept;
            next[b] += accept;
        }
        append(steps, garbage, dkn, nullptr);
        append(steps, real, dkn, nullptr);
    }
    const auto blocks_of = [&](const Layer& l) {
        return from_device<std::uint16_t>(
            l.block_keys, static_cast<std::size_t>(ops::kQsaBlockKeyWords) * kPage * l.pages);
    };
    const auto a = blocks_of(once), s = blocks_of(steps);
    int mismatches = 0;
    for (int b = 0; b < kLanes; ++b) {
        for (int blk = 0; blk < kEnd / kRatio; ++blk) {
            const std::size_t base =
                (static_cast<std::size_t>(once.page[b][blk * kRatio / kPage]) * kPage +
                 (blk * kRatio) % kPage) *
                ops::kQsaBlockKeyWords;
            for (int d = 0; d < kIndexDim; ++d) { mismatches += a[base + d] != s[base + d]; }
        }
    }
    if (mismatches != 0) {
        std::cerr << "qsa_append incremental/rollback publication differs from one call ("
                  << mismatches << " words)\n";
        ++failures;
    }
    return failures;
}

int test_select_and_attention() {
    int failures         = 0;
    constexpr int kLanes = 2;
    std::vector<std::uint16_t> kn(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) { kn[d] = f32_to_bf16(kKeyNorm[d]); }
    DeviceBuffer dkn = to_device(kn);

    // Lane 0 holds 20000 tokens (5000 blocks), lane 1 2300 (575 blocks, barely sparse).
    constexpr int kTokens = 20000;
    std::vector<std::unique_ptr<Layer>> layers;
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        auto layer = std::make_unique<Layer>(kLanes, kTokens, storage);
        append(*layer, bulk(kLanes, 0, kTokens), dkn, nullptr);
        layers.push_back(std::move(layer));
    }
    const Layer& bf16 = *layers.front();

    // MTP-verify window W=4: lane 0 sparse, lane 1 sparse at 2300 with its last column invalid.
    const QueryWindow verify{
        4, {{19990, 19991, 19992, 19993}, {2296, 2297, 2298, 2298}}, {4, 3}, 0x21U};
    const auto r = run_select(bf16, verify, kTokens, nullptr);
    failures += check_select(bf16, verify, r, "qsa_select verify W=4");
    // Dense-equivalent: n <= 2051 selects everything in order.
    const QueryWindow dense{2, {{1500, 2050}, {7, 2048}}, {2, 2}, 0x24U};
    failures += check_select(bf16, dense, run_select(bf16, dense, kTokens, nullptr),
                             "qsa_select dense-equivalent");
    // Decode W=1 at a shallow envelope over a deep cache must still be exact for its live n.
    const QueryWindow decode{1, {{19999}, {2051}}, {1, 1}, 0x25U};
    failures += check_select(bf16, decode, run_select(bf16, decode, kTokens, nullptr),
                             "qsa_select decode W=1");
    // Prefill: 300 consecutive columns per lane run as multiple tile passes.
    {
        QueryWindow prefill{300, std::vector<std::vector<int>>(kLanes), {300, 297}, 0x26U};
        for (int b = 0; b < kLanes; ++b) {
            for (int w = 0; w < 300; ++w) {
                prefill.positions[b].push_back((b == 0 ? 6000 : 1900) +
                                               std::min(w, b == 0 ? 299 : 296));
            }
        }
        failures += check_select(bf16, prefill, run_select(bf16, prefill, kTokens, nullptr),
                                 "qsa_select prefill W=300");
    }

    for (const auto& layer : layers) {
        const std::string profile = std::string("qsa_attention ") + profile_name(layer->storage);
        failures += attend(*layer, profile + " select W=4", 4, 24, r.selected, r.counts, 0x31U);
        const std::vector<int> decode_counts = {kGeometry.max_selected(), 1};
        failures += attend(*layer, profile + " decode", 1, 24,
                           synthetic(1, kLanes, 2300, decode_counts), decode_counts, 0x32U);
        // 64 columns x 2 lanes x 2 KV heads fill the GPU: one CTA per pair, counts 0..max.
        constexpr int kWide = 64;
        std::vector<int> counts(static_cast<std::size_t>(kWide) * kLanes);
        for (std::size_t c = 0; c < counts.size(); ++c) {
            counts[c] = static_cast<int>((c * 977U) %
                                         static_cast<std::size_t>(kGeometry.max_selected() + 1));
        }
        counts[1] = 0;
        counts[2] = kGeometry.max_selected();
        failures += attend(*layer, profile + " wide", kWide, 24,
                           synthetic(kWide, kLanes, 2300, counts), counts, 0x33U);
    }
    return failures;
}

// Deep decode: 90001 visible tokens (22500 blocks) exceed the staged top-k capacity (20480).
int test_deep_select() {
    std::vector<std::uint16_t> kn(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) { kn[d] = f32_to_bf16(kKeyNorm[d]); }
    DeviceBuffer dkn      = to_device(kn);
    constexpr int kTokens = 90004;
    Layer layer(1, kTokens, KvCacheStorage::Fp8E4M3Row256);
    append(layer, bulk(1, 0, kTokens), dkn, nullptr);
    int failures = check_block_keys(layer, 0, kTokens, "qsa_append deep block keys");
    for (const int width : {1, 4}) {
        QueryWindow qw{width, {{}}, {width}, 0x27U + static_cast<std::uint32_t>(width)};
        for (int w = 0; w < width; ++w) { qw.positions[0].push_back(90000 - width + w + 1); }
        int differ = 0;
        failures += check_select(layer, qw, run_select(layer, qw, kTokens, nullptr),
                                 "qsa_select deep W=" + std::to_string(width), &differ);
        std::cout << "qsa_select deep W=" << width << ": " << differ
                  << " near-tie blocks differ from the oracle\n";
    }
    return failures;
}

// The two-device shard: one KV head and its 12 query heads per GPU.
int test_shard_attention() {
    int failures = 0;
    std::vector<std::uint16_t> kn(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) { kn[d] = f32_to_bf16(kKeyNorm[d]); }
    DeviceBuffer dkn = to_device(kn);
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Int8Group64,
          KvCacheStorage::Nvfp4Group16}) {
        Layer layer(2, 2400, storage, /*kv_heads=*/1);
        append(layer, bulk(2, 0, 2400), dkn, nullptr);
        const std::vector<int> counts = {kGeometry.max_selected(), 777, 5,
                                         kGeometry.max_selected()};
        failures += attend(layer, std::string("qsa_attention shard 12/1 ") + profile_name(storage),
                           2, 12, synthetic(2, 2, 2400, counts), counts, 0x34U);
    }
    return failures;
}

// The decode chain (append, select, attention) captured once and replayed: no host sync and
// replays bit-identical to eager calls over the same device inputs.
int test_graph_replay() {
    int failures         = 0;
    constexpr int kLanes = 2, W = 4, kTokens = 9000, kMaxVisible = 9100;
    std::vector<std::uint16_t> kn(kIndexDim), qn(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) {
        kn[d] = f32_to_bf16(kKeyNorm[d]);
        qn[d] = f32_to_bf16(kQueryNorm[d]);
    }
    DeviceBuffer dkn = to_device(kn), dqn = to_device(qn);
    Layer layer(kLanes, kMaxVisible, KvCacheStorage::BFloat16);
    append(layer, bulk(kLanes, 0, kTokens), dkn, nullptr);

    const std::size_t columns = static_cast<std::size_t>(W) * kLanes;
    const int max_sel         = kGeometry.max_selected();
    DeviceBuffer dk(columns * 2 * kHeadDim * 2), dv(columns * 2 * kHeadDim * 2),
        dik(columns * kIndexDim * 2), diq(columns * kIndexHeads * kIndexDim * 2),
        dq(columns * 24 * kHeadDim * 2), dcache(columns * 4), drope(columns * 3 * 4),
        dvalid(kLanes * 4), dsel(static_cast<std::size_t>(max_sel) * columns * 4),
        dcnt(columns * 4), dout(columns * 24 * kHeadDim * 2);
    const Tensor tk(dk.p, DType::BF16, {kHeadDim, 2, W, kLanes});
    const Tensor tv(dv.p, DType::BF16, {kHeadDim, 2, W, kLanes});
    const Tensor tik(dik.p, DType::BF16, {kIndexDim, W, kLanes});
    const Tensor tiq(diq.p, DType::BF16, {kIndexHeads * kIndexDim, W, kLanes});
    const Tensor tq(dq.p, DType::BF16, {kHeadDim, 24, W, kLanes});
    const Tensor tcache(dcache.p, DType::I32, {W, kLanes});
    const Tensor trope(drope.p, DType::I32, {static_cast<std::int32_t>(columns), 3});
    const Tensor tvalid(dvalid.p, DType::I32, {kLanes});
    Tensor tsel(dsel.p, DType::I32, {max_sel, W, kLanes});
    Tensor tcnt(dcnt.p, DType::I32, {W, kLanes});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, 24, W, kLanes});
    WorkspaceArena workspace(ops::qsa_select_workspace_bytes(kGeometry, kMaxVisible, W, kLanes) +
                             ops::qsa_attention_workspace_bytes(kGeometry, W * kLanes));
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    const auto chain = [&] {
        ops::qsa_append(tk, tv, tik, trope, tcache, tvalid, layer.rows(),
                        Tensor(dkn.p, DType::BF16, {kIndexDim}), kGeometry, layer.kv, layer.index,
                        stream);
        ops::qsa_select(tiq, trope, tcache, tvalid, layer.rows(),
                        Tensor(dqn.p, DType::BF16, {kIndexDim}), layer.index, kGeometry,
                        kMaxVisible, workspace, tsel, tcnt, stream);
        ops::qsa_attention(tq, tsel, tcnt, layer.rows(), layer.kv, 1.0F / 16.0F, workspace, tout,
                           stream);
    };
    // Round inputs: round r appends positions kTokens + 4r .. +3 (lane 1 accepts three).
    const auto upload = [&](int round) {
        std::vector<std::uint16_t> k(columns * 2 * kHeadDim), v(k.size()), ik(columns * kIndexDim),
            iq(columns * kIndexHeads * kIndexDim), q(columns * 24 * kHeadDim);
        std::vector<int> cache(columns), rope(columns * 3);
        for (std::size_t c = 0; c < columns; ++c) {
            const int b = static_cast<int>(c) / W, w = static_cast<int>(c) % W;
            const int t = kTokens + 4 * round + std::min(w, b == 1 ? 2 : 3);
            cache[c]    = t;
            for (int a = 0; a < 3; ++a) { rope[a * columns + c] = axis_position(t, a); }
            for (int i = 0; i < 2 * kHeadDim; ++i) {
                k[c * 2 * kHeadDim + i] = f32_to_bf16(k_value(b, t, i / kHeadDim, i % kHeadDim, 3));
                v[c * 2 * kHeadDim + i] = f32_to_bf16(v_value(b, t, i / kHeadDim, i % kHeadDim, 3));
            }
            for (int d = 0; d < kIndexDim; ++d) {
                ik[c * kIndexDim + d] = f32_to_bf16(index_key_value(b, t, d, 3));
            }
            for (int i = 0; i < kIndexHeads * kIndexDim; ++i) {
                iq[c * kIndexHeads * kIndexDim + i] =
                    f32_to_bf16(hashed(mix(9, round, c, i), -2.0F, 2.0F));
            }
            for (int i = 0; i < 24 * kHeadDim; ++i) {
                q[c * 24 * kHeadDim + i] = f32_to_bf16(hashed(mix(10, round, c, i), -1.0F, 1.0F));
            }
        }
        const std::vector<int> valid = {4, 3};
        dk.copy_from_host(k.data(), k.size() * 2);
        dv.copy_from_host(v.data(), v.size() * 2);
        dik.copy_from_host(ik.data(), ik.size() * 2);
        diq.copy_from_host(iq.data(), iq.size() * 2);
        dq.copy_from_host(q.data(), q.size() * 2);
        dcache.copy_from_host(cache.data(), cache.size() * 4);
        drope.copy_from_host(rope.data(), rope.size() * 4);
        dvalid.copy_from_host(valid.data(), valid.size() * 4);
    };
    const auto outputs = [&] {
        auto out = from_device<std::uint16_t>(dout, columns * 24 * kHeadDim);
        auto cnt = from_device<int>(dcnt, columns);
        auto sel = from_device<int>(dsel, static_cast<std::size_t>(max_sel) * columns);
        std::vector<std::int64_t> all(out.begin(), out.end());
        all.insert(all.end(), cnt.begin(), cnt.end());
        // Selected entries past each count are unspecified.
        for (std::size_t c = 0; c < columns; ++c) {
            all.insert(all.end(), sel.begin() + static_cast<std::ptrdiff_t>(c * max_sel),
                       sel.begin() + static_cast<std::ptrdiff_t>(c * max_sel + cnt[c]));
        }
        return all;
    };
    upload(0);
    chain(); // eager warm-up also configures kernel attributes
    cuda_synchronize(stream);
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    chain();
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
    for (int round = 0; round < 3; ++round) {
        upload(round);
        chain();
        cuda_synchronize(stream);
        const auto eager = outputs();
        upload(round); // the append is idempotent for identical rows
        cuda_check(cudaGraphLaunch(executable, stream), "replay");
        cuda_synchronize(stream);
        failures += verify_exact(("qsa graph replay round " + std::to_string(round)).c_str(),
                                 outputs(), eager);
    }
    cuda_check(cudaGraphExecDestroy(executable), "destroy executable");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    return failures;
}

// --bench [tokens] [columns] [reps]: times qsa_attention on the int8 G64 cache of the
// Flash-Next profile with 12 query heads and one KV head (one rank at tp 2): `columns` query
// columns at the end of a `tokens`-token cache, each selecting the budget - the last 64 blocks
// before its position and 448 random earlier blocks (sorted), 2048 keys - as a prefill chunk does
// at long context. Prints ms per call; --dump OUT writes the output for bit comparisons between
// kernel builds.
int bench_attention(int tokens, int columns, int reps, const char* dump) {
    Layer layer(1, tokens, KvCacheStorage::Int8Group64, /*kv_heads=*/1);
    std::vector<std::uint16_t> kn(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) kn[d] = f32_to_bf16(kKeyNorm[d]);
    DeviceBuffer dkn = to_device(kn);
    for (int first = 0; first < tokens; first += 8192) {
        append(layer, bulk(1, first, std::min(8192, tokens - first)), dkn, nullptr);
    }
    const int max_sel = kGeometry.max_selected();
    std::vector<int> sel(static_cast<std::size_t>(max_sel) * columns, 0), cnt(columns, 0);
    std::uint64_t rs = 12345;
    auto rnd         = [&]() {
        rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(rs >> 33);
    };
    for (int c = 0; c < columns; ++c) {
        const int pos = tokens - columns + c, last = pos / kRatio;
        std::set<int> blocks;
        for (int b = std::max(0, last - 63); b <= last; ++b) blocks.insert(b);
        const int older = std::max(0, last - 63);
        while (static_cast<int>(blocks.size()) < kGeometry.block_topk() &&
               static_cast<int>(blocks.size()) < last + 1) {
            blocks.insert(static_cast<int>(rnd() % static_cast<std::uint32_t>(std::max(older, 1))));
        }
        int n = 0;
        for (const int b : blocks) {
            for (int t = b * kRatio; t < b * kRatio + kRatio && t <= pos; ++t) {
                sel[static_cast<std::size_t>(c) * max_sel + n++] = t;
            }
        }
        cnt[c] = n;
    }
    const int qh = 12;
    std::vector<std::uint16_t> qb(static_cast<std::size_t>(kHeadDim) * qh * columns);
    for (std::size_t i = 0; i < qb.size(); ++i)
        qb[i] = f32_to_bf16(hashed(mix(77, i), -1.0F, 1.0F));
    DeviceBuffer dq = to_device(qb), dsel = to_device_i32(sel), dcnt = to_device_i32(cnt);
    DeviceBuffer out(qb.size() * 2);
    Tensor tq(dq.p, DType::BF16, {kHeadDim, qh, columns, 1});
    Tensor ts(dsel.p, DType::I32, {max_sel, columns, 1});
    Tensor tc(dcnt.p, DType::I32, {columns, 1});
    Tensor tout(out.p, DType::BF16, {kHeadDim, qh, columns, 1});
    ops::QsaGeometry g = kGeometry;
    g.query_heads      = qh;
    g.kv_heads         = 1;
    WorkspaceArena workspace(ops::qsa_attention_workspace_bytes(g, columns));
    const auto run = [&] {
        workspace.reset();
        ops::qsa_attention(tq, ts, tc, layer.rows(), layer.kv, 1.0F / 16.0F, workspace, tout,
                           nullptr);
    };
    run();
    cuda_synchronize();
    cudaEvent_t e0, e1;
    cuda_check(cudaEventCreate(&e0), "event");
    cuda_check(cudaEventCreate(&e1), "event");
    cuda_check(cudaEventRecord(e0, nullptr), "record");
    for (int r = 0; r < reps; ++r) run();
    cuda_check(cudaEventRecord(e1, nullptr), "record");
    cuda_check(cudaEventSynchronize(e1), "sync");
    float ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    std::cout << "qsa_attention bench: tokens " << tokens << " columns " << columns
              << " keys/column " << cnt[columns / 2] << ": " << ms / reps << " ms per call\n";
    if (dump != nullptr) {
        const auto host = from_device<std::uint16_t>(out.p, qb.size());
        std::FILE* f    = std::fopen(dump, "wb");
        if (f == nullptr) throw std::runtime_error(std::string("cannot write ") + dump);
        std::fwrite(host.data(), 2, host.size(), f);
        std::fclose(f);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
            const char* dump = nullptr;
            int nums[3] = {100352, 24576, 5}, k = 0;
            for (int i = 2; i < argc; ++i) {
                if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
                    dump = argv[++i];
                } else if (k < 3) {
                    nums[k++] = std::atoi(argv[i]);
                }
            }
            return bench_attention(nums[0], nums[1], nums[2], dump);
        }
        int failures = 0;
        failures += test_append_profiles();
        failures += test_incremental_publication();
        failures += test_select_and_attention();
        failures += test_deep_select();
        failures += test_shard_attention();
        failures += test_graph_replay();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_attention\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "sparse_attention: " << error.what() << '\n';
        return 1;
    }
}
