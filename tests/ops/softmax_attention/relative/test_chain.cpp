#include "ninfer/ops/relative_attention.h"
#include "ops/op_tester.h"
#include "ops/ragged_fixture.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

constexpr AttentionHeadGeometry kGeometry{128, 24, 8};
constexpr int kDim = 128, kRel = 16, kQH = 24, kKH = 8, kChannels = kDim * kKH;
constexpr int kHistory = 3, kWindow = 1024, kPageSize = 64;
constexpr float kScale = 0.08838834764831845f; // 1 / sqrt(128)
constexpr ReductionCriterion kCriterion{0.006, 0.0005, 0.015};

using Word  = std::uint16_t;
using Words = std::vector<Word>;

std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

Words random_bf16(std::size_t n, std::uint32_t seed, float amp = 0.65f) {
    Words out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float u = float(int(mix(unsigned(i) + seed) % 2001) - 1000) / 1000.0f;
        out[i]        = f32_to_bf16(u * amp);
    }
    return out;
}

double round_bf16(double x) {
    return bf16_to_f32(f32_to_bf16(static_cast<float>(x)));
}

double round_fp16(double x) {
    return __half2float(__float2half_rn(static_cast<float>(x)));
}

double e4m3_val(unsigned code) {
    const double m = code & 7u;
    const int e    = int((code >> 3) & 15u);
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m, -9);
    return (code & 128u) ? -v : v;
}

std::uint8_t encode_e4m3(double x) {
    static const auto grid = [] {
        std::array<double, 127> g{};
        for (unsigned i = 0; i < g.size(); ++i) g[i] = e4m3_val(i);
        return g;
    }();
    const bool neg = std::signbit(x);
    x              = std::abs(x);
    const auto it  = std::lower_bound(grid.begin(), grid.end(), x);
    unsigned code  = static_cast<unsigned>(it - grid.begin());
    if (it == grid.end()) {
        code = 126;
    } else if (it != grid.begin()) {
        const double lo = x - *(it - 1), hi = *it - x;
        if (lo < hi || (lo == hi && (code & 1u))) --code;
    }
    return static_cast<std::uint8_t>(code | (neg ? 128u : 0u));
}

void encode_decode_vector(const std::array<double, kDim>& src, bool is_v, KvCacheStorage storage,
                          std::array<double, kDim>& dst) {
    if (storage == KvCacheStorage::BFloat16) {
        for (int d = 0; d < kDim; ++d) dst[d] = is_v ? round_fp16(src[d]) : round_bf16(src[d]);
        return;
    }
    double max_abs = 0.0;
    for (double x : src) max_abs = std::max(max_abs, std::abs(x));
    if (max_abs == 0.0) {
        dst.fill(0.0);
        return;
    }
    int exp       = std::ilogb(max_abs / 448.0);
    double scale  = std::ldexp(1.0, exp);
    if (max_abs / scale > 448.0) {
        ++exp;
        scale = std::ldexp(1.0, exp);
    }
    exp   = std::clamp(exp, -127, 127);
    scale = std::ldexp(1.0, exp);
    for (int d = 0; d < kDim; ++d) {
        const auto code = encode_e4m3(src[d] / scale);
        dst[d]          = e4m3_val(code) * scale;
    }
}

struct ChunkBuffers {
    RelativeAttentionChunkKV view;
    std::array<DeviceBuffer, 4> bufs;

    ChunkBuffers(Device dev, KvCacheStorage storage, std::initializer_list<int> trailing) {
        view.storage      = storage;
        const auto schema = paged_kv_storage_layout(storage, kDim);
        std::vector<int> k_dims{kDim, kKH}, v_dims{kDim, kKH}, s_dims{1, kKH};
        for (int x : trailing) {
            k_dims.push_back(x);
            v_dims.push_back(x);
            s_dims.push_back(x);
        }
        auto elems = [](const std::vector<int>& dims) {
            std::size_t n = 1;
            for (int d : dims) n *= static_cast<std::size_t>(d);
            return n;
        };
        auto alloc = [&](int idx, DType dt, const std::vector<int>& dims, Tensor& t) {
            bufs[idx] = DeviceBuffer(dev, elems(dims) * dtype_size(dt));
            CUDA_CHECK(cudaMemset(bufs[idx].data(), 0xa5, bufs[idx].size()));
            if (dims.size() == 3)
                t = Tensor(bufs[idx].span(), dt, {dims[0], dims[1], dims[2]});
            else
                t = Tensor(bufs[idx].span(), dt, {dims[0], dims[1], dims[2], dims[3]});
        };
        alloc(0, schema.key.data_dtype, k_dims, view.k);
        alloc(1, schema.value.data_dtype, v_dims, view.v);
        if (schema.key.has_scale()) {
            alloc(2, DType::U8, s_dims, view.k_scales);
            alloc(3, DType::U8, s_dims, view.v_scales);
        }
    }
};

struct HostSeqOracle {
    std::vector<std::array<double, kDim>> k_hist[kKH];
    std::vector<std::array<double, kDim>> v_hist[kKH];
    std::array<Words, kHistory> k_conv_state;
    std::array<Words, kHistory> v_conv_state;

    HostSeqOracle() {
        for (int t = 0; t < kHistory; ++t) {
            k_conv_state[t].assign(kChannels, 0);
            v_conv_state[t].assign(kChannels, 0);
        }
    }
};

struct StepOracleResult {
    Words q_out, r_out, sink_out;
    std::vector<std::array<double, kDim>> k_encoded; // [token * kKH + h]
    std::vector<std::array<double, kDim>> v_encoded;
    std::vector<std::array<Words, kHistory>> k_snaps; // [token]
    std::vector<std::array<Words, kHistory>> v_snaps;
    std::vector<double> attn_out; // [token * kQH * kDim]
};

StepOracleResult run_host_step(
    bool swa, KvCacheStorage storage, const RelativeAttentionPrepareParams& params, bool dynamic,
    int sink_h, const Words& q_in, const Words& r_in, const Words& k_in, const Words& v_in,
    const Words& qs, const Words& rs, const Words& kw, const Words& vw, const Words& kdyn,
    const Words& vdyn, const Words& sink_x, const Words& sink_w, const Words& rel_proj,
    const Words& gate, const Words& explicit_sink, const std::vector<int>& lengths,
    const std::vector<int>& starts, bool ragged, int width, std::vector<HostSeqOracle>& seqs) {
    const int b = static_cast<int>(lengths.size());
    const int n = ragged ? std::accumulate(lengths.begin(), lengths.end(), 0) : b * width;
    StepOracleResult res;
    res.q_out.assign(n * kQH * kDim, 0);
    res.r_out.assign(n * kQH * kRel, 0);
    if (sink_h > 0) res.sink_out.assign(n * kQH, 0);
    res.k_encoded.resize(n * kKH);
    res.v_encoded.resize(n * kKH);
    res.k_snaps.resize(n);
    res.v_snaps.resize(n);
    res.attn_out.assign(n * kQH * kDim, 0.0);

    int flat = 0;
    for (int s = 0; s < b; ++s) {
        const int base = ragged ? flat : s * width;
        flat += lengths[s];
        auto cur_k_state = seqs[s].k_conv_state;
        auto cur_v_state = seqs[s].v_conv_state;
        const int L      = starts[s];

        // 1. Prepare all tokens in this sequence chunk
        for (int col = 0; col < lengths[s]; ++col) {
            const int tok = base + col;
            const int pos = L + col;
            double tau    = 1.0;
            if (params.use_log_scaling && pos + 1 > params.log_scaling_floor) {
                tau = 1.0 + double(params.log_scaling_alpha) *
                                std::log(double(pos + 1) / double(params.log_scaling_floor));
            }
            for (int h = 0; h < kQH; ++h) {
                double qv[kDim], qss = 0.0;
                for (int d = 0; d < kDim; ++d) {
                    qv[d] = double(bf16_to_f32(q_in[(tok * kQH + h) * kDim + d])) *
                            double(bf16_to_f32(qs[h * kDim + d]));
                    qss += qv[d] * qv[d];
                }
                const double qf = std::min(
                    double(params.qk_clip_threshold) / (std::sqrt(qss) + double(params.clip_eps)),
                    1.0);
                for (int d = 0; d < kDim; ++d) {
                    double qc = round_bf16(qv[d] * qf);
                    if (params.use_log_scaling && tau != 1.0) qc = round_bf16(qc * tau);
                    res.q_out[(tok * kQH + h) * kDim + d] = f32_to_bf16(float(qc));
                }
                double rv[kRel], rss = 0.0;
                for (int d = 0; d < kRel; ++d) {
                    rv[d] = double(bf16_to_f32(r_in[(tok * kQH + h) * kRel + d])) *
                            double(bf16_to_f32(rs[h * kRel + d]));
                    rss += rv[d] * rv[d];
                }
                const double rf = std::min(
                    double(params.rel_clip_threshold) / (std::sqrt(rss) + double(params.clip_eps)),
                    1.0);
                for (int d = 0; d < kRel; ++d) {
                    double rc = round_bf16(rv[d] * rf);
                    if (params.use_log_scaling && tau != 1.0) rc = round_bf16(rc * tau);
                    res.r_out[(tok * kQH + h) * kRel + d] = f32_to_bf16(float(rc));
                }
                if (sink_h > 0) {
                    double acc = 0.0;
                    for (int c = 0; c < sink_h; ++c) {
                        acc += double(bf16_to_f32(sink_x[tok * sink_h + c])) *
                               double(bf16_to_f32(sink_w[h * sink_h + c]));
                    }
                    res.sink_out[tok * kQH + h] = f32_to_bf16(float(acc));
                }
            }

            // Causal width-4 conv + normalize + codec for K and V
            auto step_conv = [&](const Words& raw_in, const Words& w, const Words& dyn,
                                 std::array<Words, kHistory>& st, bool is_v,
                                 std::vector<std::array<double, kDim>>& enc_out,
                                 std::vector<std::array<Words, kHistory>>& snaps) {
                const double cscale = dynamic ? (1.0 / 3.0) : (1.0 / std::sqrt(5.0));
                std::vector<double> y(kChannels);
                for (int c = 0; c < kChannels; ++c) {
                    const int h      = c / kDim;
                    const double cur = bf16_to_f32(raw_in[tok * kChannels + c]);
                    double acc       = 0.0;
                    for (int tap = 0; tap < 4; ++tap) {
                        const int lag = dynamic ? tap : (3 - tap);
                        const double u =
                            (lag == 0) ? cur : double(bf16_to_f32(st[kHistory - lag][c]));
                        double wt = bf16_to_f32(w[tap * kChannels + c]);
                        if (dynamic) wt += bf16_to_f32(dyn[(tok * kKH + h) * 4 + tap]);
                        acc += wt * u;
                    }
                    y[c] = round_bf16((acc + cur) * cscale);
                }
                for (int j = 0; j + 1 < kHistory; ++j) st[j] = st[j + 1];
                st[kHistory - 1].assign(raw_in.begin() + tok * kChannels,
                                        raw_in.begin() + (tok + 1) * kChannels);
                snaps[tok] = st;

                for (int h = 0; h < kKH; ++h) {
                    double ss = 0.0;
                    for (int d = 0; d < kDim; ++d) ss += y[h * kDim + d] * y[h * kDim + d];
                    const double fac =
                        is_v ? (1.0 / std::sqrt(ss / kDim + double(params.v_norm_eps)))
                             : std::min(double(params.qk_clip_threshold) /
                                            (std::sqrt(ss) + double(params.clip_eps)),
                                        1.0);
                    std::array<double, kDim> normed{};
                    for (int d = 0; d < kDim; ++d) normed[d] = y[h * kDim + d] * fac;
                    encode_decode_vector(normed, is_v, storage, enc_out[tok * kKH + h]);
                }
            };
            step_conv(k_in, kw, kdyn, cur_k_state, false, res.k_encoded, res.k_snaps);
            step_conv(v_in, vw, vdyn, cur_v_state, true, res.v_encoded, res.v_snaps);
        }

        // 2. Attention over committed history [0, L) + current chunk [L, L + lengths[s])
        for (int col = 0; col < lengths[s]; ++col) {
            const int tok    = base + col;
            const int p_i    = L + col;
            const int k_lo   = swa ? std::max(0, p_i - (kWindow - 1)) : 0;
            const int k_cnt  = p_i - k_lo + 1;
            const auto& sink = (sink_h > 0) ? res.sink_out : explicit_sink;

            for (int h = 0; h < kQH; ++h) {
                const int kvh = h / 3;
                std::vector<double> scores(k_cnt);
                double max_s = -std::numeric_limits<double>::infinity();
                for (int idx = 0; idx < k_cnt; ++idx) {
                    const int p_j = k_lo + idx;
                    const auto& kv =
                        (p_j < L) ? seqs[s].k_hist[kvh][p_j] : res.k_encoded[(base + p_j - L) * kKH + kvh];
                    double dot = 0.0;
                    for (int d = 0; d < kDim; ++d)
                        dot += double(bf16_to_f32(res.q_out[(tok * kQH + h) * kDim + d])) * kv[d];
                    double bias    = 0.0;
                    const int dist = p_i - p_j;
                    if (dist < kWindow) {
                        for (int u = 0; u < kRel; ++u)
                            bias += double(bf16_to_f32(res.r_out[(tok * kQH + h) * kRel + u])) *
                                    double(bf16_to_f32(rel_proj[dist * kRel + u]));
                    }
                    const double sc = double(kScale) * dot + bias;
                    scores[idx]     = sc;
                    max_s           = std::max(max_s, sc);
                }
                double sum_e = 0.0;
                std::vector<double> probs(k_cnt);
                for (int idx = 0; idx < k_cnt; ++idx) {
                    probs[idx] = std::exp(scores[idx] - max_s);
                    sum_e += probs[idx];
                }
                const double lse     = max_s + std::log(sum_e);
                const double sink_v  = bf16_to_f32(sink[tok * kQH + h]);
                const double sink_s  = 1.0 / (1.0 + std::exp(-(lse - sink_v)));
                std::array<double, kDim> acc{};
                for (int idx = 0; idx < k_cnt; ++idx) {
                    const int p_j = k_lo + idx;
                    const auto& vv =
                        (p_j < L) ? seqs[s].v_hist[kvh][p_j] : res.v_encoded[(base + p_j - L) * kKH + kvh];
                    const double w = probs[idx] / sum_e;
                    for (int d = 0; d < kDim; ++d) acc[d] += w * vv[d];
                }
                for (int d = 0; d < kDim; ++d) {
                    const double gv = bf16_to_f32(gate[(tok * kQH + h) * kDim + d]);
                    const double gs = 1.0 / (1.0 + std::exp(-gv));
                    res.attn_out[(tok * kQH + h) * kDim + d] = acc[d] * gs * sink_s;
                }
            }
        }
    }
    return res;
}

void commit_host_seqs(const StepOracleResult& step, const std::vector<int>& lengths,
                      const std::vector<int>& accepted, const std::vector<int>& starts, bool ragged,
                      int width, std::vector<HostSeqOracle>& seqs) {
    int flat = 0;
    for (std::size_t s = 0; s < lengths.size(); ++s) {
        const int base = ragged ? flat : int(s) * width;
        flat += lengths[s];
        const int acc = accepted[s];
        if (acc <= 0) continue;
        for (int h = 0; h < kKH; ++h) {
            if (int(seqs[s].k_hist[h].size()) < starts[s] + acc) {
                seqs[s].k_hist[h].resize(starts[s] + acc);
                seqs[s].v_hist[h].resize(starts[s] + acc);
            }
            for (int col = 0; col < acc; ++col) {
                seqs[s].k_hist[h][starts[s] + col] = step.k_encoded[(base + col) * kKH + h];
                seqs[s].v_hist[h][starts[s] + col] = step.v_encoded[(base + col) * kKH + h];
            }
        }
        seqs[s].k_conv_state = step.k_snaps[base + acc - 1];
        seqs[s].v_conv_state = step.v_snaps[base + acc - 1];
    }
}

int check_outputs(const DeviceBuffer& gpu_out, const std::vector<double>& ref,
                  const std::string& label) {
    Words got(ref.size());
    gpu_out.copy_to_host(got.data(), got.size() * sizeof(Word));
    std::vector<double> got_d(ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) got_d[i] = bf16_to_f32(got[i]);
    return verify_reduction(label, got_d, ref, kCriterion);
}

int run_swa_lifecycle(KvCacheStorage storage) {
    const auto& env   = cuda_environment();
    const auto dev    = env.context.device();
    const auto exec   = env.context.execution_view();
    const auto schema = paged_kv_storage_layout(storage, kDim);
    constexpr int B   = 2;

    // 1. Allocate cyclic KV cache (2 records, capacity 1024)
    KVPageGeometry kv_geom{
        kWindow, {{schema.key.data_dtype, kDim, kKH}, {schema.value.data_dtype, kDim, kKH}}};
    if (schema.key.has_scale()) {
        kv_geom.planes.push_back({DType::U8, 1, kKH});
        kv_geom.planes.push_back({DType::U8, 1, kKH});
    }
    tool::PackedKVFixture kv_fixture(dev, kv_geom, B);
    const std::vector<int> rec_ids{0, 1};
    const auto kv_off_host = kv_fixture.table_offsets(rec_ids);
    auto kv_off_buf        = to_device(kv_off_host);
    Tensor kv_off_t(kv_off_buf.span(), DType::I64, {B});
    auto make_kv_field = [&](int p) -> StateFieldView {
        const auto reg = kv_fixture.layout().planes[p].tensor;
        return {{kv_fixture.storage(), kv_off_t}, {reg, 1, reg.region.bytes}};
    };
    CyclicKVCacheLayerView cyclic{.k               = make_kv_field(0),
                                  .v               = make_kv_field(1),
                                  .k_scales        = schema.key.has_scale() ? make_kv_field(2) : StateFieldView{},
                                  .v_scales        = schema.key.has_scale() ? make_kv_field(3) : StateFieldView{},
                                  .capacity        = kWindow,
                                  .padded_capacity = kWindow,
                                  .num_kv_heads    = kKH,
                                  .head_dim        = kDim,
                                  .storage         = storage};

    // 2. Allocate conv state storage (2 records, K and V fields)
    const std::size_t field_bytes = std::size_t(kChannels) * kHistory * sizeof(Word);
    KVPageGeometry st_geom{kHistory, {{DType::BF16, kChannels, 1}, {DType::BF16, kChannels, 1}}};
    tool::PackedKVFixture st_fixture(dev, st_geom, B);
    CUDA_CHECK(cudaMemset(st_fixture.storage().data, 0, st_fixture.storage().bytes));
    auto st_off_buf = to_device(st_fixture.table_offsets(rec_ids));
    Tensor st_off_t(st_off_buf.span(), DType::I64, {B});
    auto make_st_field = [&](int p) -> StateFieldView {
        const auto reg = st_fixture.layout().planes[p].tensor;
        return {{st_fixture.storage(), st_off_t}, {reg, 1, reg.region.bytes}};
    };
    StateFieldView k_state = make_st_field(0);
    StateFieldView v_state = make_st_field(1);

    // Shared layer weights
    const auto qs_h       = random_bf16(kDim * kQH, 101, 0.9f);
    const auto rs_h       = random_bf16(kRel * kQH, 102, 0.9f);
    const auto kw_h       = random_bf16(kChannels * 4, 103, 0.25f);
    const auto vw_h       = random_bf16(kChannels * 4, 104, 0.25f);
    const auto rel_proj_h = random_bf16(kRel * kWindow, 105, 0.35f);
    auto qs_d             = to_device(qs_h);
    auto rs_d             = to_device(rs_h);
    auto kw_d             = to_device(kw_h);
    auto vw_d             = to_device(vw_h);
    auto rel_proj_d       = to_device(rel_proj_h);

    RelativeAttentionPrepareParams params{};
    params.use_log_scaling = false;

    const std::size_t ws_cap = std::max(
        relative_softmax_attention_ragged_workspace_capacity_bytes(
            kGeometry, storage, true, {ContextBucket::K2, ContextBucket::K8}, 1, 4096, 1, B,
            exec.multiprocessor_count()),
        relative_softmax_attention_swa_workspace_capacity_bytes(
            kGeometry, storage, kWindow, {ContextBucket::K2, ContextBucket::K8}, B, 1, 8,
            exec.multiprocessor_count()));
    DeviceBuffer ws_buf(dev, std::max<std::size_t>(ws_cap, 64));
    MemoryArena arena(ws_buf.span());

    std::vector<HostSeqOracle> seqs(B);
    int errors = 0;

    // --- Stage 1: Ragged SWA Prefill exceeding 1024 ring ({1023, 1029}, N=2052) ---
    {
        const std::vector<int> lens{1023, 1029};
        const std::vector<int> starts{0, 0};
        const int N = 2052;
        RaggedFixture ragged(lens);
        std::vector<int> pos_h(N);
        std::iota(pos_h.begin(), pos_h.begin() + 1023, 0);
        std::iota(pos_h.begin() + 1023, pos_h.end(), 0);
        auto pos_d  = to_device(pos_h);
        auto q_h    = random_bf16(N * kQH * kDim, 201);
        auto r_h    = random_bf16(N * kQH * kRel, 202);
        auto k_h    = random_bf16(N * kChannels, 203);
        auto v_h    = random_bf16(N * kChannels, 204);
        auto gate_h = random_bf16(N * kQH * kDim, 205);
        auto sink_h = random_bf16(N * kQH, 206);
        auto q_d    = to_device(q_h);
        auto r_d    = to_device(r_h);
        auto k_d    = to_device(k_h);
        auto v_d    = to_device(v_h);
        auto gate_d = to_device(gate_h);
        auto sink_d = to_device(sink_h);

        DeviceBuffer qo_d(dev, q_d.size()), ro_d(dev, r_d.size()), out_d(dev, q_d.size());
        ChunkBuffers chunk(dev, storage, {N});
        RelativeAttentionPrepareInputs pin{
            Tensor(q_d.span(), DType::BF16, {kDim, kQH, N}),
            Tensor(r_d.span(), DType::BF16, {kRel, kQH, N}),
            Tensor(k_d.span(), DType::BF16, {kDim, kKH, N}),
            Tensor(v_d.span(), DType::BF16, {kDim, kKH, N}),
            Tensor(qs_d.span(), DType::BF16, {kDim, kQH}),
            Tensor(rs_d.span(), DType::BF16, {kRel, kQH}),
            Tensor(kw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(vw_d.span(), DType::BF16, {kChannels, 4}),
            {}, {},
            Tensor(pos_d.span(), DType::I32, {N}),
            {}, {}};
        RelativeAttentionPrepareOutputs pout{
            Tensor(qo_d.span(), DType::BF16, {kDim, kQH, N}),
            Tensor(ro_d.span(), DType::BF16, {kRel, kQH, N}),
            {}};
        relative_attention_prepare(pin, ragged.view, k_state, k_state, v_state, v_state,
                                   chunk.view, params, pout, exec);

        RelativeAttentionQueryInputs qin{
            pout.q_out, pout.r_out, Tensor(rel_proj_d.span(), DType::BF16, {kRel, kWindow}),
            Tensor(gate_d.span(), DType::BF16, {kDim, kQH, N}),
            Tensor(sink_d.span(), DType::BF16, {kQH, N}), pin.positions};
        Tensor out_t(out_d.span(), DType::BF16, {kDim, kQH, N});
        relative_softmax_attention(qin, ragged.view, kGeometry, kWindow, kScale, cyclic,
                                   chunk.view, ContextBucket::K2, arena, out_t, exec);
        relative_attention_commit_cyclic_kv(chunk.view, ragged.view, pin.positions, cyclic, exec);
        CUDA_CHECK(cudaStreamSynchronize(exec.stream()));

        const auto ref = run_host_step(true, storage, params, false, 0, q_h, r_h, k_h, v_h, qs_h,
                                       rs_h, kw_h, vw_h, {}, {}, {}, {}, rel_proj_h, gate_h, sink_h,
                                       lens, starts, true, N, seqs);
        errors += check_outputs(out_d, ref.attn_out, "swa_chain/stage1_ragged_prefill");
        commit_host_seqs(ref, lens, lens, starts, true, N, seqs);
    }

    // --- Stage 2: SWA Verify across 1023->1024 wrap (W=4, B=2) + partial accept {2, 1} ---
    {
        constexpr int W = 4;
        const std::vector<int> lens{W, W};
        const std::vector<int> starts{1023, 1029};
        const std::vector<int> accept{2, 1};
        const int N = B * W;
        std::vector<int> pos_h(N);
        for (int s = 0; s < B; ++s)
            for (int j = 0; j < W; ++j) pos_h[s * W + j] = starts[s] + j;
        auto pos_d   = to_device(pos_h);
        auto valid_d = to_device(lens);
        auto acc_d   = to_device(accept);
        const std::vector<int> base_slots{0, W};
        auto slots_d = to_device(base_slots);
        auto q_h     = random_bf16(N * kQH * kDim, 301);
        auto r_h     = random_bf16(N * kQH * kRel, 302);
        auto k_h     = random_bf16(N * kChannels, 303);
        auto v_h     = random_bf16(N * kChannels, 304);
        auto gate_h  = random_bf16(N * kQH * kDim, 305);
        auto sink_h  = random_bf16(N * kQH, 306);
        auto q_d     = to_device(q_h);
        auto r_d     = to_device(r_h);
        auto k_d     = to_device(k_h);
        auto v_d     = to_device(v_h);
        auto gate_d  = to_device(gate_h);
        auto sink_d  = to_device(sink_h);

        DeviceBuffer qo_d(dev, q_d.size()), ro_d(dev, r_d.size()), out_d(dev, q_d.size());
        DeviceBuffer k_snaps_d(dev, B * W * field_bytes), v_snaps_d(dev, B * W * field_bytes);
        Tensor k_snaps_t(k_snaps_d.span(), DType::BF16, {kChannels, kHistory, B * W});
        Tensor v_snaps_t(v_snaps_d.span(), DType::BF16, {kChannels, kHistory, B * W});
        ChunkBuffers chunk(dev, storage, {W, B});

        RelativeAttentionPrepareInputs pin{
            Tensor(q_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(r_d.span(), DType::BF16, {kRel, kQH, W, B}),
            Tensor(k_d.span(), DType::BF16, {kDim, kKH, W, B}),
            Tensor(v_d.span(), DType::BF16, {kDim, kKH, W, B}),
            Tensor(qs_d.span(), DType::BF16, {kDim, kQH}),
            Tensor(rs_d.span(), DType::BF16, {kRel, kQH}),
            Tensor(kw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(vw_d.span(), DType::BF16, {kChannels, 4}),
            {}, {},
            Tensor(pos_d.span(), DType::I32, {W, B}),
            {}, {}};
        RelativeAttentionPrepareOutputs pout{
            Tensor(qo_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(ro_d.span(), DType::BF16, {kRel, kQH, W, B}),
            {}};
        Tensor valid_t(valid_d.span(), DType::I32, {B});
        Tensor slots_t(slots_d.span(), DType::I32, {B});
        relative_attention_prepare_snapshot(pin, valid_t, slots_t, k_state, k_snaps_t, v_state,
                                            v_snaps_t, chunk.view, params, pout, exec);

        RelativeAttentionQueryInputs qin{
            pout.q_out, pout.r_out, Tensor(rel_proj_d.span(), DType::BF16, {kRel, kWindow}),
            Tensor(gate_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(sink_d.span(), DType::BF16, {kQH, W, B}), pin.positions};
        Tensor out_t(out_d.span(), DType::BF16, {kDim, kQH, W, B});
        relative_softmax_attention(qin, valid_t, kGeometry, kWindow, kScale, cyclic, chunk.view,
                                   ContextBucket::K2, arena, out_t, exec);

        // Commit ONLY accepted prefix {2, 1} into cyclic KV & conv state
        Tensor acc_t(acc_d.span(), DType::I32, {B});
        relative_attention_commit_cyclic_kv(chunk.view, pin.positions, acc_t, cyclic, exec);
        for (int s = 0; s < B; ++s) {
            const std::size_t snap_off = std::size_t(base_slots[s] + accept[s] - 1) * field_bytes;
            const auto rec_off         = st_fixture.physical_offset(s);
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char*>(st_fixture.storage().data) + rec_off +
                    st_fixture.layout().planes[0].tensor.region.offset,
                static_cast<const char*>(k_snaps_d.data()) + snap_off, field_bytes,
                cudaMemcpyDeviceToDevice, exec.stream()));
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char*>(st_fixture.storage().data) + rec_off +
                    st_fixture.layout().planes[1].tensor.region.offset,
                static_cast<const char*>(v_snaps_d.data()) + snap_off, field_bytes,
                cudaMemcpyDeviceToDevice, exec.stream()));
        }
        CUDA_CHECK(cudaStreamSynchronize(exec.stream()));

        const auto ref = run_host_step(true, storage, params, false, 0, q_h, r_h, k_h, v_h, qs_h,
                                       rs_h, kw_h, vw_h, {}, {}, {}, {}, rel_proj_h, gate_h, sink_h,
                                       lens, starts, false, W, seqs);
        errors += check_outputs(out_d, ref.attn_out, "swa_chain/stage2_verify_wrap");
        commit_host_seqs(ref, lens, accept, starts, false, W, seqs);
    }

    // --- Stage 3: SWA Direct Decode W=1 (positions 1025 & 1030; verifies rejected slots untouched) ---
    {
        constexpr int W = 1;
        const std::vector<int> lens{1, 1};
        const std::vector<int> starts{1025, 1030};
        auto pos_d  = to_device(starts);
        auto q_h    = random_bf16(B * kQH * kDim, 401);
        auto r_h    = random_bf16(B * kQH * kRel, 402);
        auto k_h    = random_bf16(B * kChannels, 403);
        auto v_h    = random_bf16(B * kChannels, 404);
        auto gate_h = random_bf16(B * kQH * kDim, 405);
        auto sink_h = random_bf16(B * kQH, 406);
        auto q_d    = to_device(q_h);
        auto r_d    = to_device(r_h);
        auto k_d    = to_device(k_h);
        auto v_d    = to_device(v_h);
        auto gate_d = to_device(gate_h);
        auto sink_d = to_device(sink_h);
        DeviceBuffer qo_d(dev, q_d.size()), ro_d(dev, r_d.size()), out_d(dev, q_d.size());

        RelativeAttentionPrepareInputs pin{
            Tensor(q_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(r_d.span(), DType::BF16, {kRel, kQH, W, B}),
            Tensor(k_d.span(), DType::BF16, {kDim, kKH, W, B}),
            Tensor(v_d.span(), DType::BF16, {kDim, kKH, W, B}),
            Tensor(qs_d.span(), DType::BF16, {kDim, kQH}),
            Tensor(rs_d.span(), DType::BF16, {kRel, kQH}),
            Tensor(kw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(vw_d.span(), DType::BF16, {kChannels, 4}),
            {}, {},
            Tensor(pos_d.span(), DType::I32, {W, B}),
            {}, {}};
        RelativeAttentionPrepareOutputs pout{
            Tensor(qo_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(ro_d.span(), DType::BF16, {kRel, kQH, W, B}),
            {}};
        relative_attention_prepare(pin, Tensor{}, k_state, k_state, v_state, v_state, cyclic,
                                   params, pout, exec);

        RelativeAttentionQueryInputs qin{
            pout.q_out, pout.r_out, Tensor(rel_proj_d.span(), DType::BF16, {kRel, kWindow}),
            Tensor(gate_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(sink_d.span(), DType::BF16, {kQH, W, B}), pin.positions};
        Tensor out_t(out_d.span(), DType::BF16, {kDim, kQH, W, B});
        relative_softmax_attention(qin, Tensor{}, kGeometry, kWindow, kScale, cyclic,
                                   RelativeAttentionChunkKV{}, ContextBucket::K2, arena, out_t,
                                   exec);
        CUDA_CHECK(cudaStreamSynchronize(exec.stream()));

        const auto ref = run_host_step(true, storage, params, false, 0, q_h, r_h, k_h, v_h, qs_h,
                                       rs_h, kw_h, vw_h, {}, {}, {}, {}, rel_proj_h, gate_h, sink_h,
                                       lens, starts, false, W, seqs);
        errors += check_outputs(out_d, ref.attn_out, "swa_chain/stage3_direct_decode");
    }
    return errors;
}

int run_full_lifecycle(KvCacheStorage storage) {
    const auto& env      = cuda_environment();
    const auto dev       = env.context.device();
    const auto exec      = env.context.execution_view();
    const auto schema    = paged_kv_storage_layout(storage, kDim);
    constexpr int B      = 2;
    constexpr int Pages  = 4; // 256 tokens per sequence
    constexpr int SinkH  = 64;

    KVPageGeometry kv_geom{
        kPageSize, {{schema.key.data_dtype, kDim, kKH}, {schema.value.data_dtype, kDim, kKH}}};
    if (schema.key.has_scale()) {
        kv_geom.planes.push_back({DType::U8, 1, kKH});
        kv_geom.planes.push_back({DType::U8, 1, kKH});
    }
    tool::PackedKVFixture kv_fixture(dev, kv_geom, B * Pages);
    std::vector<int> map_ids(B * Pages);
    std::iota(map_ids.begin(), map_ids.end(), 0);
    auto page_off_buf = to_device(kv_fixture.table_offsets(map_ids));
    Tensor page_table(page_off_buf.span(), DType::I64, {Pages, B});
    PagedKVBatchLayerView paged{kv_fixture.plane(0, page_table),
                                kv_fixture.plane(1, page_table),
                                {},
                                {},
                                kDim,
                                kKH,
                                storage};
    if (schema.key.has_scale()) {
        paged.k_scale_pages = kv_fixture.plane(2, page_table);
        paged.v_scale_pages = kv_fixture.plane(3, page_table);
    }
    auto rows_d = to_device(std::vector<int>{0, 1});
    Tensor rows_t(rows_d.span(), DType::I32, {B});

    const std::size_t field_bytes = std::size_t(kChannels) * kHistory * sizeof(Word);
    KVPageGeometry st_geom{kHistory, {{DType::BF16, kChannels, 1}, {DType::BF16, kChannels, 1}}};
    tool::PackedKVFixture st_fixture(dev, st_geom, B);
    CUDA_CHECK(cudaMemset(st_fixture.storage().data, 0, st_fixture.storage().bytes));
    auto st_off_buf = to_device(st_fixture.table_offsets(std::vector<int>{0, 1}));
    Tensor st_off_t(st_off_buf.span(), DType::I64, {B});
    auto make_st_field = [&](int p) -> StateFieldView {
        const auto reg = st_fixture.layout().planes[p].tensor;
        return {{st_fixture.storage(), st_off_t}, {reg, 1, reg.region.bytes}};
    };
    StateFieldView k_state = make_st_field(0);
    StateFieldView v_state = make_st_field(1);

    const auto qs_h       = random_bf16(kDim * kQH, 501, 0.9f);
    const auto rs_h       = random_bf16(kRel * kQH, 502, 0.9f);
    const auto kw_h       = random_bf16(kChannels * 4, 503, 0.25f);
    const auto vw_h       = random_bf16(kChannels * 4, 504, 0.25f);
    const auto sink_w_h   = random_bf16(SinkH * kQH, 505, 0.35f);
    const auto rel_proj_h = random_bf16(kRel * kWindow, 506, 0.35f);
    auto qs_d             = to_device(qs_h);
    auto rs_d             = to_device(rs_h);
    auto kw_d             = to_device(kw_h);
    auto vw_d             = to_device(vw_h);
    auto sink_w_d         = to_device(sink_w_h);
    auto rel_proj_d       = to_device(rel_proj_h);

    RelativeAttentionPrepareParams params{};
    params.use_log_scaling   = true;
    params.log_scaling_floor = 64;
    params.log_scaling_alpha = 0.25f;

    const std::size_t ws_cap = std::max(
        relative_softmax_attention_ragged_workspace_capacity_bytes(
            kGeometry, storage, false, {ContextBucket::K2, ContextBucket::K8}, 1, 512, 1, B,
            exec.multiprocessor_count()),
        relative_softmax_attention_full_workspace_capacity_bytes(
            kGeometry, storage, {ContextBucket::K2, ContextBucket::K8}, B, 1, 8,
            exec.multiprocessor_count()));
    DeviceBuffer ws_buf(dev, std::max<std::size_t>(ws_cap, 64));
    MemoryArena arena(ws_buf.span());

    std::vector<HostSeqOracle> seqs(B);
    int errors = 0;

    // --- Stage 1: Ragged Full Prefill (dynamic conv + fused sink + log-scaling tau) ---
    {
        const std::vector<int> lens{130, 70};
        const std::vector<int> starts{0, 0};
        const int N = 200;
        RaggedFixture ragged(lens);
        std::vector<int> pos_h(N);
        std::iota(pos_h.begin(), pos_h.begin() + 130, 0);
        std::iota(pos_h.begin() + 130, pos_h.end(), 0);
        auto pos_d    = to_device(pos_h);
        auto q_h      = random_bf16(N * kQH * kDim, 601);
        auto r_h      = random_bf16(N * kQH * kRel, 602);
        auto k_h      = random_bf16(N * kChannels, 603);
        auto v_h      = random_bf16(N * kChannels, 604);
        auto kdyn_h   = random_bf16(N * kKH * 4, 605, 0.2f);
        auto vdyn_h   = random_bf16(N * kKH * 4, 606, 0.2f);
        auto sink_x_h = random_bf16(N * SinkH, 607);
        auto gate_h   = random_bf16(N * kQH * kDim, 608);
        auto q_d      = to_device(q_h);
        auto r_d      = to_device(r_h);
        auto k_d      = to_device(k_h);
        auto v_d      = to_device(v_h);
        auto kdyn_d   = to_device(kdyn_h);
        auto vdyn_d   = to_device(vdyn_h);
        auto sink_x_d = to_device(sink_x_h);
        auto gate_d   = to_device(gate_h);

        DeviceBuffer qo_d(dev, q_d.size()), ro_d(dev, r_d.size()),
            so_d(dev, std::size_t(N) * kQH * sizeof(Word)), out_d(dev, q_d.size());
        RelativeAttentionPrepareInputs pin{
            Tensor(q_d.span(), DType::BF16, {kDim, kQH, N}),
            Tensor(r_d.span(), DType::BF16, {kRel, kQH, N}),
            Tensor(k_d.span(), DType::BF16, {kDim, kKH, N}),
            Tensor(v_d.span(), DType::BF16, {kDim, kKH, N}),
            Tensor(qs_d.span(), DType::BF16, {kDim, kQH}),
            Tensor(rs_d.span(), DType::BF16, {kRel, kQH}),
            Tensor(kw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(vw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(kdyn_d.span(), DType::BF16, {4, kKH, N}),
            Tensor(vdyn_d.span(), DType::BF16, {4, kKH, N}),
            Tensor(pos_d.span(), DType::I32, {N}),
            Tensor(sink_x_d.span(), DType::BF16, {SinkH, N}),
            Tensor(sink_w_d.span(), DType::BF16, {SinkH, kQH})};
        RelativeAttentionPrepareOutputs pout{
            Tensor(qo_d.span(), DType::BF16, {kDim, kQH, N}),
            Tensor(ro_d.span(), DType::BF16, {kRel, kQH, N}),
            Tensor(so_d.span(), DType::BF16, {kQH, N})};
        relative_attention_prepare(pin, ragged.view, rows_t, k_state, k_state, v_state, v_state,
                                   paged, params, pout, exec);

        RelativeAttentionQueryInputs qin{
            pout.q_out, pout.r_out, Tensor(rel_proj_d.span(), DType::BF16, {kRel, kWindow}),
            Tensor(gate_d.span(), DType::BF16, {kDim, kQH, N}), pout.sink_out, pin.positions};
        Tensor out_t(out_d.span(), DType::BF16, {kDim, kQH, N});
        relative_softmax_attention(qin, ragged.view, rows_t, kGeometry, kScale, paged,
                                   ContextBucket::K2, arena, out_t, exec);
        CUDA_CHECK(cudaStreamSynchronize(exec.stream()));

        const auto ref = run_host_step(false, storage, params, true, SinkH, q_h, r_h, k_h, v_h,
                                       qs_h, rs_h, kw_h, vw_h, kdyn_h, vdyn_h, sink_x_h, sink_w_h,
                                       rel_proj_h, gate_h, {}, lens, starts, true, N, seqs);
        errors += check_outputs(out_d, ref.attn_out, "full_chain/stage1_ragged_prefill");
        commit_host_seqs(ref, lens, lens, starts, true, N, seqs);
    }

    // --- Stage 2: Full Snapshot Verify (W=4, B=2) + partial accept {2, 3} ---
    {
        constexpr int W = 4;
        const std::vector<int> lens{W, W};
        const std::vector<int> starts{130, 70};
        const std::vector<int> accept{2, 3};
        const int N = B * W;
        std::vector<int> pos_h(N);
        for (int s = 0; s < B; ++s)
            for (int j = 0; j < W; ++j) pos_h[s * W + j] = starts[s] + j;
        auto pos_d   = to_device(pos_h);
        auto valid_d = to_device(lens);
        const std::vector<int> base_slots{0, W};
        auto slots_d  = to_device(base_slots);
        auto q_h      = random_bf16(N * kQH * kDim, 701);
        auto r_h      = random_bf16(N * kQH * kRel, 702);
        auto k_h      = random_bf16(N * kChannels, 703);
        auto v_h      = random_bf16(N * kChannels, 704);
        auto kdyn_h   = random_bf16(N * kKH * 4, 705, 0.2f);
        auto vdyn_h   = random_bf16(N * kKH * 4, 706, 0.2f);
        auto sink_x_h = random_bf16(N * SinkH, 707);
        auto gate_h   = random_bf16(N * kQH * kDim, 708);
        auto q_d      = to_device(q_h);
        auto r_d      = to_device(r_h);
        auto k_d      = to_device(k_h);
        auto v_d      = to_device(v_h);
        auto kdyn_d   = to_device(kdyn_h);
        auto vdyn_d   = to_device(vdyn_h);
        auto sink_x_d = to_device(sink_x_h);
        auto gate_d   = to_device(gate_h);

        DeviceBuffer qo_d(dev, q_d.size()), ro_d(dev, r_d.size()),
            so_d(dev, std::size_t(N) * kQH * sizeof(Word)), out_d(dev, q_d.size());
        DeviceBuffer k_snaps_d(dev, B * W * field_bytes), v_snaps_d(dev, B * W * field_bytes);
        Tensor k_snaps_t(k_snaps_d.span(), DType::BF16, {kChannels, kHistory, B * W});
        Tensor v_snaps_t(v_snaps_d.span(), DType::BF16, {kChannels, kHistory, B * W});

        RelativeAttentionPrepareInputs pin{
            Tensor(q_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(r_d.span(), DType::BF16, {kRel, kQH, W, B}),
            Tensor(k_d.span(), DType::BF16, {kDim, kKH, W, B}),
            Tensor(v_d.span(), DType::BF16, {kDim, kKH, W, B}),
            Tensor(qs_d.span(), DType::BF16, {kDim, kQH}),
            Tensor(rs_d.span(), DType::BF16, {kRel, kQH}),
            Tensor(kw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(vw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(kdyn_d.span(), DType::BF16, {4, kKH, W, B}),
            Tensor(vdyn_d.span(), DType::BF16, {4, kKH, W, B}),
            Tensor(pos_d.span(), DType::I32, {W, B}),
            Tensor(sink_x_d.span(), DType::BF16, {SinkH, W, B}),
            Tensor(sink_w_d.span(), DType::BF16, {SinkH, kQH})};
        RelativeAttentionPrepareOutputs pout{
            Tensor(qo_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(ro_d.span(), DType::BF16, {kRel, kQH, W, B}),
            Tensor(so_d.span(), DType::BF16, {kQH, W, B})};
        Tensor valid_t(valid_d.span(), DType::I32, {B});
        Tensor slots_t(slots_d.span(), DType::I32, {B});
        relative_attention_prepare_snapshot(pin, valid_t, rows_t, slots_t, k_state, k_snaps_t,
                                            v_state, v_snaps_t, paged, params, pout, exec);

        RelativeAttentionQueryInputs qin{
            pout.q_out, pout.r_out, Tensor(rel_proj_d.span(), DType::BF16, {kRel, kWindow}),
            Tensor(gate_d.span(), DType::BF16, {kDim, kQH, W, B}), pout.sink_out, pin.positions};
        Tensor out_t(out_d.span(), DType::BF16, {kDim, kQH, W, B});
        relative_softmax_attention(qin, valid_t, rows_t, kGeometry, kScale, paged,
                                   ContextBucket::K2, arena, out_t, exec);

        for (int s = 0; s < B; ++s) {
            const std::size_t snap_off = std::size_t(base_slots[s] + accept[s] - 1) * field_bytes;
            const auto rec_off         = st_fixture.physical_offset(s);
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char*>(st_fixture.storage().data) + rec_off +
                    st_fixture.layout().planes[0].tensor.region.offset,
                static_cast<const char*>(k_snaps_d.data()) + snap_off, field_bytes,
                cudaMemcpyDeviceToDevice, exec.stream()));
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char*>(st_fixture.storage().data) + rec_off +
                    st_fixture.layout().planes[1].tensor.region.offset,
                static_cast<const char*>(v_snaps_d.data()) + snap_off, field_bytes,
                cudaMemcpyDeviceToDevice, exec.stream()));
        }
        CUDA_CHECK(cudaStreamSynchronize(exec.stream()));

        const auto ref = run_host_step(false, storage, params, true, SinkH, q_h, r_h, k_h, v_h,
                                       qs_h, rs_h, kw_h, vw_h, kdyn_h, vdyn_h, sink_x_h, sink_w_h,
                                       rel_proj_h, gate_h, {}, lens, starts, false, W, seqs);
        errors += check_outputs(out_d, ref.attn_out, "full_chain/stage2_verify");
        commit_host_seqs(ref, lens, accept, starts, false, W, seqs);
    }

    // --- Stage 3: Full Decode W=1 at positions {132, 73} overwriting rejected tail ---
    {
        constexpr int W = 1;
        const std::vector<int> lens{1, 1};
        const std::vector<int> starts{132, 73};
        auto pos_d    = to_device(starts);
        auto q_h      = random_bf16(B * kQH * kDim, 801);
        auto r_h      = random_bf16(B * kQH * kRel, 802);
        auto k_h      = random_bf16(B * kChannels, 803);
        auto v_h      = random_bf16(B * kChannels, 804);
        auto kdyn_h   = random_bf16(B * kKH * 4, 805, 0.2f);
        auto vdyn_h   = random_bf16(B * kKH * 4, 806, 0.2f);
        auto sink_x_h = random_bf16(B * SinkH, 807);
        auto gate_h   = random_bf16(B * kQH * kDim, 808);
        auto q_d      = to_device(q_h);
        auto r_d      = to_device(r_h);
        auto k_d      = to_device(k_h);
        auto v_d      = to_device(v_h);
        auto kdyn_d   = to_device(kdyn_h);
        auto vdyn_d   = to_device(vdyn_h);
        auto sink_x_d = to_device(sink_x_h);
        auto gate_d   = to_device(gate_h);

        DeviceBuffer qo_d(dev, q_d.size()), ro_d(dev, r_d.size()),
            so_d(dev, std::size_t(B) * kQH * sizeof(Word)), out_d(dev, q_d.size());
        RelativeAttentionPrepareInputs pin{
            Tensor(q_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(r_d.span(), DType::BF16, {kRel, kQH, W, B}),
            Tensor(k_d.span(), DType::BF16, {kDim, kKH, W, B}),
            Tensor(v_d.span(), DType::BF16, {kDim, kKH, W, B}),
            Tensor(qs_d.span(), DType::BF16, {kDim, kQH}),
            Tensor(rs_d.span(), DType::BF16, {kRel, kQH}),
            Tensor(kw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(vw_d.span(), DType::BF16, {kChannels, 4}),
            Tensor(kdyn_d.span(), DType::BF16, {4, kKH, W, B}),
            Tensor(vdyn_d.span(), DType::BF16, {4, kKH, W, B}),
            Tensor(pos_d.span(), DType::I32, {W, B}),
            Tensor(sink_x_d.span(), DType::BF16, {SinkH, W, B}),
            Tensor(sink_w_d.span(), DType::BF16, {SinkH, kQH})};
        RelativeAttentionPrepareOutputs pout{
            Tensor(qo_d.span(), DType::BF16, {kDim, kQH, W, B}),
            Tensor(ro_d.span(), DType::BF16, {kRel, kQH, W, B}),
            Tensor(so_d.span(), DType::BF16, {kQH, W, B})};
        relative_attention_prepare(pin, Tensor{}, rows_t, k_state, k_state, v_state, v_state, paged,
                                   params, pout, exec);

        RelativeAttentionQueryInputs qin{
            pout.q_out, pout.r_out, Tensor(rel_proj_d.span(), DType::BF16, {kRel, kWindow}),
            Tensor(gate_d.span(), DType::BF16, {kDim, kQH, W, B}), pout.sink_out, pin.positions};
        Tensor out_t(out_d.span(), DType::BF16, {kDim, kQH, W, B});
        relative_softmax_attention(qin, Tensor{}, rows_t, kGeometry, kScale, paged,
                                   ContextBucket::K2, arena, out_t, exec);
        CUDA_CHECK(cudaStreamSynchronize(exec.stream()));

        const auto ref = run_host_step(false, storage, params, true, SinkH, q_h, r_h, k_h, v_h,
                                       qs_h, rs_h, kw_h, vw_h, kdyn_h, vdyn_h, sink_x_h, sink_w_h,
                                       rel_proj_h, gate_h, {}, lens, starts, false, W, seqs);
        errors += check_outputs(out_d, ref.attn_out, "full_chain/stage3_decode");
    }
    return errors;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite("relative_attention_chain");
    for (const auto [format, storage] :
         {std::pair{"bf16", KvCacheStorage::BFloat16},
          std::pair{"fp8", KvCacheStorage::Fp8E4m3fnGroup128Ue8m0}}) {
        suite.add(std::string("swa/") + format + "/prefill_verify_accept_decode",
                  [=] { return run_swa_lifecycle(storage); }, {{"format", format}});
        suite.add(std::string("full/") + format + "/prefill_verify_accept_decode",
                  [=] { return run_full_lifecycle(storage); }, {{"format", format}});
    }
    return suite.main(argc, argv);
}
