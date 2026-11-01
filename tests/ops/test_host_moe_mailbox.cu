// GPU <-> CPU handshake of the host expert path inside a CUDA graph (HostMoeMailbox).
//
// The executor's numerics are qualified against the FP64 oracle in test_host_moe; this test
// qualifies the transport, so its oracle is exact: for every layer and replay, y must equal
// bf16(y0 + p) (FP32 accumulator: y0 + p) bit for bit, where p is the partial a separate
// HostMoeExecutor computes on the host from the same x, the ids the device does not hold
// (device_slot < 0) and the same weights. Also checked: a layer whose experts are all on the
// device leaves y untouched (no partial read), graph replays with new routing between launches,
// one trace record per served layer, and the failure path (an executor exception) releases every
// wait so the graph completes, with failed() set and y untouched.
// Miss split (T18, split_shares): for every HostMoeSplit mode the device's share equals an
// independent host implementation of the rule (reads, pick list, scratch table, split counters:
// exact), the host computes exactly the complement (y = y0 + device partial + host partial bit for
// bit), the staged share (streaming copy into scratch slots, then moe_slots) is bit-identical to
// the zero-copy one, and against the FP64 oracle (fixture decode, unquantized activations) the
// device share stays within moe_slots' W4A4 bound (relative L2 0.18, test_moe_slots kW4A4), the
// host share within 0.05 (W4A8: T16 measured 0.011-0.013; 4x margin) and their sum within 0.18.
#include "host_moe_fixture.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_slots.h"
#include "ops/sparse_moe/host/host_moe_mailbox.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace ninfer::ops::host;
using namespace ninfer::test::host_moe;

namespace {

// The shares are checked against W4A8 references; A4 (the GPU activation codec) is the default
// since v0.2.1.
HostMoeExecutor::Options w4a8_options() {
    HostMoeExecutor::Options o;
    o.a4 = false;
    return o;
}

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

__global__ void busy_kernel(long long ns) {
    long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (long long t = t0; t - t0 < ns;) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
}

struct Routing {
    std::vector<std::int32_t> ids, slot;
    std::vector<float> alpha;
};

Routing route(int L, int T, int K, int E, std::mt19937_64& rng, bool all_device_layer0) {
    Routing r;
    r.ids.resize(static_cast<std::size_t>(L) * T * K);
    r.slot.resize(r.ids.size());
    r.alpha.resize(r.ids.size());
    std::uniform_real_distribution<float> w(0.02F, 0.3F);
    std::uniform_int_distribution<int> host(0, 3);
    for (int l = 0; l < L; ++l) {
        for (int t = 0; t < T; ++t) {
            std::vector<int> all(E);
            for (int e = 0; e < E; ++e) all[e] = e;
            std::shuffle(all.begin(), all.end(), rng);
            for (int s = 0; s < K; ++s) {
                const std::size_t i = (static_cast<std::size_t>(l) * T + t) * K + s;
                r.ids[i]            = all[s];
                r.slot[i] =
                    (all_device_layer0 && l == 0) || host(rng) != 0 ? s : -1; // ~1 in 4 on the host
                r.alpha[i] = w(rng);
            }
        }
    }
    return r;
}

// publish_table: residency by a per-expert slot table, the routing log (all routed ids, flag and
// token count per layer), and a layer called in two slices of 4 tokens (log offsets 0 and 4; the
// service must take the second slice of the same layer at once, not after its 1 ms resync).
void table_and_slices(HostMoeMailboxConfig cfg, const std::vector<HostExpertLayer>& layers,
                      const std::vector<std::uint16_t>& x, const std::vector<float>& y0f,
                      const std::vector<int>& ref_cpus, std::mt19937_64& rng) {
    constexpr int T = 4, S = 2, K = 10, H = 2560, I = 640;
    const int L = cfg.layers, E = layers[0].experts;
    cfg.route_log_tokens = T * S;
    cfg.trace_capacity   = 4 * L * S; // cleared before every replay
    const std::size_t nx = static_cast<std::size_t>(L) * T * S * H,
                      nr = static_cast<std::size_t>(L) * T * S * K;
    std::vector<std::uint16_t> xs(nx);
    for (std::size_t i = 0; i < nx; ++i) xs[i] = x[i % x.size()];
    __nv_bfloat16* dx = nullptr;
    float *dyf = nullptr, *dalpha = nullptr;
    std::int32_t *dids = nullptr, *dtable = nullptr;
    CK(cudaMalloc(&dx, nx * 2));
    CK(cudaMalloc(&dyf, nx * 4));
    CK(cudaMalloc(&dalpha, nr * 4));
    CK(cudaMalloc(&dids, nr * 4));
    CK(cudaMalloc(&dtable, static_cast<std::size_t>(L) * E * 4));
    CK(cudaMemcpy(dx, xs.data(), nx * 2, cudaMemcpyHostToDevice));
    HostMoeMailbox mailbox(cfg, layers);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int l = 0; l < L; ++l) {
        for (int slice = 0; slice < S; ++slice) {
            const std::size_t t0 = static_cast<std::size_t>(l * S + slice) * T;
            mailbox.publish_table(s, l, T, dx + t0 * H, dids + t0 * K, dalpha + t0 * K,
                                  dtable + l * E, E, slice * T);
            busy_kernel<<<1, 32, 0, s>>>(5000);
            mailbox.accumulate(s, l, T, dyf + t0 * H);
        }
    }
    CK(cudaStreamEndCapture(s, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<float> yf(nx), p(static_cast<std::size_t>(T) * H);
    std::vector<float> y0s(nx);
    for (std::size_t i = 0; i < nx; ++i) y0s[i] = y0f[i % y0f.size()];
    double max_gap_us = 0;
    int compared      = 0;
    for (int replay = 0; replay < 4; ++replay) {
        std::vector<std::int32_t> ids(nr), table(static_cast<std::size_t>(L) * E);
        std::vector<float> alpha(nr);
        std::uniform_real_distribution<float> w(0.02F, 0.3F);
        std::uniform_int_distribution<int> coin(0, 2);
        for (auto& v : table) v = coin(rng) == 0 ? -1 : 7; // ~1 in 3 experts on the host
        for (std::size_t r = 0; r < nr / K; ++r) {
            std::vector<int> all(E);
            for (int e = 0; e < E; ++e) all[e] = e;
            std::shuffle(all.begin(), all.end(), rng);
            for (int k = 0; k < K; ++k) {
                ids[r * K + k]   = all[k];
                alpha[r * K + k] = w(rng);
            }
        }
        CK(cudaMemcpy(dids, ids.data(), nr * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dalpha, alpha.data(), nr * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dtable, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dyf, y0s.data(), nx * 4, cudaMemcpyHostToDevice));
        // A cudaMemcpy from pageable memory may return before its DMA lands, and the non-blocking
        // stream s does not wait for the default stream: settle the uploads before the launch.
        CK(cudaDeviceSynchronize());
        mailbox.clear_route_flags();
        mailbox.clear_trace();
        CK(cudaGraphLaunch(exec, s));
        CK(cudaStreamSynchronize(s));
        expect(!mailbox.failed(), "table: mailbox failed: " + mailbox.error());
        const auto records = mailbox.trace();
        expect(records.size() == static_cast<std::size_t>(L * S),
               "table: one trace record per slice");
        // Host gap from a layer's first slice done to its second slice seen: the device needs a few
        // us in between (accumulate, publish); a resync scan would add a millisecond.
        for (std::size_t r = 1; r < records.size(); ++r) {
            if (records[r].layer == records[r - 1].layer) {
                max_gap_us =
                    std::max(max_gap_us, 1e-3 * static_cast<double>(records[r].seen_ns -
                                                                    records[r - 1].done_ns));
            }
        }
        CK(cudaMemcpy(yf.data(), dyf, nx * 4, cudaMemcpyDeviceToHost));
        HostCpuPool pool(ref_cpus);
        HostMoeExecutor ex(pool, H, I, 8, K, w4a8_options());
        for (int l = 0; l < L; ++l) {
            const HostMoeRouteLog log = mailbox.route_log(l);
            expect(*log.published == 1 && *log.tokens == static_cast<std::uint32_t>(T * S),
                   "table: layer " + std::to_string(l) + " log flag and token count");
            int log_bad = 0;
            for (int i = 0; i < T * S * K; ++i)
                log_bad += log.ids[i] != ids[static_cast<std::size_t>(l) * T * S * K + i];
            expect(log_bad == 0, "table: layer " + std::to_string(l) + " logged ids (" +
                                     std::to_string(log_bad) + " wrong)");
            for (int slice = 0; slice < S; ++slice) {
                const std::size_t t0 = static_cast<std::size_t>(l * S + slice) * T;
                std::vector<std::int32_t> host_ids(ids.begin() + t0 * K,
                                                   ids.begin() + (t0 + T) * K);
                bool any = false;
                for (auto& e : host_ids) {
                    if (table[static_cast<std::size_t>(l) * E + e] >= 0) e = -1;
                    any = any || e >= 0;
                }
                const HostMoeJob job{
                    T, K, xs.data() + t0 * H, host_ids.data(), alpha.data() + t0 * K, p.data()};
                ex.run(layers[l], job);
                int bad = 0;
                for (int i = 0; i < T * H; ++i)
                    bad += yf[t0 * H + i] != (any ? y0s[t0 * H + i] + p[i] : y0s[t0 * H + i]);
                compared += T * H;
                expect(bad == 0, "table: replay " + std::to_string(replay) + " layer " +
                                     std::to_string(l) + " slice " + std::to_string(slice) + ": " +
                                     std::to_string(bad) + " mismatches");
            }
        }
    }
    expect(max_gap_us < 300.0,
           "table: second slice of a layer served without a resync stall (largest gap " +
               std::to_string(max_gap_us) + " us)");
    std::printf("table + log + slices: %d outputs bit-exact, largest slice-to-slice host gap %.1f "
                "us (%d layers x %d "
                "slices)\n",
                compared, max_gap_us, L, S);
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(dx));
    CK(cudaFree(dyf));
    CK(cudaFree(dalpha));
    CK(cudaFree(dids));
    CK(cudaFree(dtable));
}

// ---- miss split -------------------------------------------------------------------------------

// The rule of HostMoeSplit, written independently: `cols` = columns of the misses in split order.
int reference_share(const HostMoeSplit& s, const std::vector<int>& cols) {
    const int n = static_cast<int>(cols.size());
    if (n == 0) return 0;
    const int cap = std::min({n, s.max_reads, HostMoeSplit::kMaxReads});
    switch (s.mode) {
    case HostMoeSplit::Mode::Off:
        return 0;
    case HostMoeSplit::Mode::Count:
        return std::max(0, std::min(s.count, cap));
    case HostMoeSplit::Mode::Fraction:
        return std::max(
            0,
            std::min(cap, static_cast<int>(std::floor(s.fraction * static_cast<float>(n) + 0.5F))));
    case HostMoeSplit::Mode::Cost: {
        if (n <= s.min_misses) return 0;
        auto host = [&](int from, float slow) {
            if (from >= n) return 0.0F;
            float t = 0.0F;
            for (int i = from; i < n; ++i)
                t += s.host_expert_us + s.host_column_us * static_cast<float>(cols[i] - 1);
            return s.host_fixed_us + slow * t;
        };
        auto dev = [&](int g) {
            float t = s.device_fixed_us;
            for (int i = 0; i < g; ++i)
                t += s.device_expert_us + s.device_column_us * static_cast<float>(cols[i] - 1);
            return t;
        };
        // The test's costs are small multiples of 1/4: every sum and product is exact in FP32, so
        // ties are exact ties on both sides (strict <: the smallest g wins).
        int best_g = 0;
        float best = host(0, 1.0F);
        for (int g = 1; g <= cap; ++g) {
            const float t = std::max(host(g, s.host_read_slowdown), dev(g));
            if (t < best) best = t, best_g = g;
        }
        return best_g;
    }
    }
    return 0;
}

// FP64 routed partial over the experts in `which`, unquantized activations (fixture decode).
struct Oracle {
    const LayerData& d;
    std::map<int, std::vector<float>>
        decoded; // expert -> gate|up rows [2I][H], then down rows [H][I]

    explicit Oracle(const LayerData& data) : d(data) {}

    const std::vector<float>& expert(int e) {
        auto it = decoded.find(e);
        if (it != decoded.end()) return it->second;
        const int H = d.hidden, I = d.intermediate;
        std::vector<float> w(static_cast<std::size_t>(2 * I) * H + static_cast<std::size_t>(H) * I);
        for (int r = 0; r < 2 * I; ++r)
            for (int c = 0; c < H; ++c)
                w[static_cast<std::size_t>(r) * H + c] = static_cast<float>(
                    d.gate_up.weight(static_cast<std::int64_t>(e) * 2 * I + r, c));
        float* dn = w.data() + static_cast<std::size_t>(2 * I) * H;
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < I; ++c)
                dn[static_cast<std::size_t>(r) * I + c] =
                    static_cast<float>(d.down.weight(static_cast<std::int64_t>(e) * H + r, c));
        return decoded.emplace(e, std::move(w)).first->second;
    }

    double hmax = 0; // largest |SwiGLU output| seen by partial()

    std::vector<double> partial(const std::vector<std::uint16_t>& x,
                                const std::vector<std::int32_t>& ids,
                                const std::vector<float>& alpha, const std::vector<int>& which,
                                int T, int K) {
        const int H = d.hidden, I = d.intermediate;
        std::vector<double> y(static_cast<std::size_t>(T) * H, 0.0);
        for (const int e : which) {
            const std::vector<float>& w = expert(e);
            const float* dn             = w.data() + static_cast<std::size_t>(2 * I) * H;
            for (int t = 0; t < T; ++t) {
                double a = 0.0;
                for (int k = 0; k < K; ++k) a += ids[t * K + k] == e ? alpha[t * K + k] : 0.0;
                if (a == 0.0) continue;
                std::vector<double> h(I);
                for (int r = 0; r < I; ++r) {
                    double g = 0, u = 0;
                    for (int c = 0; c < H; ++c) {
                        const double xv = from_bf16(x[static_cast<std::size_t>(t) * H + c]);
                        g += w[static_cast<std::size_t>(r) * H + c] * xv;
                        u += w[static_cast<std::size_t>(I + r) * H + c] * xv;
                    }
                    h[r] = g / (1.0 + std::exp(-g)) * u;
                    hmax = std::max(hmax, std::abs(h[r]));
                }
                for (int o = 0; o < H; ++o) {
                    double acc = 0;
                    for (int r = 0; r < I; ++r)
                        acc += dn[static_cast<std::size_t>(o) * I + r] * h[r];
                    y[static_cast<std::size_t>(t) * H + o] += a * acc;
                }
            }
        }
        return y;
    }
};

double rel_l2(const std::vector<double>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
        num += (got[i] - ref[i]) * (got[i] - ref[i]), den += ref[i] * ref[i];
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

void split_shares(HostMoeMailboxConfig cfg, const LayerData& data, const std::vector<int>& ref_cpus,
                  std::mt19937_64& rng) {
    namespace nops  = ninfer::ops;
    constexpr int T = 4, K = 10, H = 2560, I = 640;
    const int E = data.experts;
    // The bank in mapped pinned memory (the engine reads its bank in place from such memory).
    void *gu = nullptr, *dn = nullptr;
    CK(cudaHostAlloc(&gu, data.gate_up.payload.size(),
                     cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&dn, data.down.payload.size(), cudaHostAllocMapped | cudaHostAllocPortable));
    std::memcpy(gu, data.gate_up.payload.data(), data.gate_up.payload.size());
    std::memcpy(dn, data.down.payload.data(), data.down.payload.size());
    const HostExpertLayer layer{Nvfp4Plane::from_payload(gu, data.gate_up.rows, H, 2 * E),
                                Nvfp4Plane::from_payload(dn, data.down.rows, I, E), E, H, I};
    cfg.layers           = 1;
    cfg.route_log_tokens = T;
    cfg.trace_capacity   = 0;
    HostMoeMailbox mailbox(cfg, {layer});
    const nops::MoeSlotPool bank{
        .gate_up_codes  = {layer.gate_up.codes, nops::kMoeSlotsGateUpCodeBytes},
        .gate_up_scales = {layer.gate_up.scales, nops::kMoeSlotsGateUpScaleBytes},
        .down_codes     = {layer.down.codes, nops::kMoeSlotsDownCodeBytes},
        .down_scales    = {layer.down.scales, nops::kMoeSlotsDownScaleBytes},
        .slots          = E,
    };
    const std::array<std::int64_t, 4> plane_bytes{
        nops::kMoeSlotsGateUpCodeBytes, nops::kMoeSlotsGateUpScaleBytes,
        nops::kMoeSlotsDownCodeBytes, nops::kMoeSlotsDownScaleBytes};
    std::array<std::byte*, 4> staging{}, staging_narrow{};
    for (int q = 0; q < 4; ++q)
        CK(cudaMalloc(&staging[q], plane_bytes[q] * HostMoeSplit::kMaxReads));
    for (int q = 0; q < 4; ++q)
        CK(cudaMalloc(&staging_narrow[q], plane_bytes[q] * HostMoeSplit::kMaxReads));
    // The scale planes again at base + 4 (only 4-byte aligned): the copy kernel's 4-byte path.
    std::byte* narrow_host      = nullptr;
    const std::size_t gu_scales = static_cast<std::size_t>(plane_bytes[1]) * E,
                      dn_scales = static_cast<std::size_t>(plane_bytes[3]) * E;
    CK(cudaHostAlloc(reinterpret_cast<void**>(&narrow_host), gu_scales + dn_scales + 64,
                     cudaHostAllocMapped | cudaHostAllocPortable));
    std::memcpy(narrow_host + 4, layer.gate_up.scales, gu_scales);
    std::memcpy(narrow_host + 32 + 4 + gu_scales, layer.down.scales, dn_scales);
    const std::array<const std::byte*, 4> narrow_bank{
        reinterpret_cast<const std::byte*>(layer.gate_up.codes), narrow_host + 4,
        reinterpret_cast<const std::byte*>(layer.down.codes), narrow_host + 32 + 4 + gu_scales};
    const nops::MoeSlotPool scratch_pool{
        .gate_up_codes  = {staging[0], plane_bytes[0]},
        .gate_up_scales = {staging[1], plane_bytes[1]},
        .down_codes     = {staging[2], plane_bytes[2]},
        .down_scales    = {staging[3], plane_bytes[3]},
        .slots          = HostMoeSplit::kMaxReads,
    };
    float *d13 = nullptr, *d2 = nullptr, *dyf = nullptr, *dz = nullptr, *ds = nullptr,
          *dalpha      = nullptr;
    std::int32_t *dids = nullptr, *dtable = nullptr, *dreads = nullptr, *dpicks = nullptr,
                 *dstage = nullptr, *dident = nullptr;
    __nv_bfloat16* dx = nullptr;
    CK(cudaMalloc(&d13, 4 * 2 * E));
    CK(cudaMalloc(&d2, 4 * E));
    CK(cudaMemcpy(d13, layer.gate_up.divisors, 4 * 2 * E, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d2, layer.down.divisors, 4 * E, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dyf, 4 * T * H));
    CK(cudaMalloc(&dz, 4 * T * H));
    CK(cudaMalloc(&ds, 4 * T * H));
    CK(cudaMalloc(&dalpha, 4 * T * K));
    CK(cudaMalloc(&dids, 4 * T * K));
    CK(cudaMalloc(&dtable, 4 * E));
    CK(cudaMalloc(&dreads, 4 * T * K));
    CK(cudaMalloc(&dpicks, 4 * (HostMoeSplit::kMaxReads + 1)));
    CK(cudaMalloc(&dstage, 4 * E));
    CK(cudaMalloc(&dident, 4 * E));
    CK(cudaMalloc(&dx, 2 * T * H));
    std::vector<std::int32_t> ident(E);
    for (int e = 0; e < E; ++e) ident[e] = e;
    CK(cudaMemcpy(dident, ident.data(), 4 * E, cudaMemcpyHostToDevice));
    ninfer::WorkspaceArena work(nops::moe_slots_nvfp4_workspace_bytes(T));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    HostCpuPool pool(ref_cpus);
    HostMoeExecutor ex(pool, H, I, 8, K, w4a8_options());
    Oracle oracle(data);

    // Activation divisors as a checkpoint calibrates them (448 * 6 / amax): x from the call's
    // input, h from the oracle's SwiGLU outputs of the call's missing experts.
    float x_divisor = 600.0F, h_divisor = 2000.0F;
    auto slots_layer = [&](std::int32_t* table) {
        nops::MoeSlotsLayer l;
        l.slot_table            = ninfer::Tensor(table, ninfer::DType::I32, {E});
        l.gate_up_divisors      = ninfer::Tensor(d13, ninfer::DType::FP32, {2 * E});
        l.down_divisors         = ninfer::Tensor(d2, ninfer::DType::FP32, {E});
        l.gate_up_input_divisor = x_divisor;
        l.down_input_divisor    = h_divisor;
        return l;
    };

    struct Case {
        std::string name;
        HostMoeSplit split;
    };

    std::vector<Case> cases;
    {
        HostMoeSplit c;
        c.mode  = HostMoeSplit::Mode::Count;
        c.count = 3;
        cases.push_back({"count3", c});
        HostMoeSplit f;
        f.mode     = HostMoeSplit::Mode::Fraction;
        f.fraction = 0.4F;
        cases.push_back({"fraction0.4", f});
        HostMoeSplit
            a; // cost: device 2.5x host per expert, host columns cost a lot (ordering matters)
        a.mode               = HostMoeSplit::Mode::Cost;
        a.host_fixed_us      = 13.0F;
        a.host_expert_us     = 56.0F;
        a.host_column_us     = 9.0F;
        a.host_read_slowdown = 1.25F;
        a.device_fixed_us    = 10.0F;
        a.device_expert_us   = 140.0F;
        cases.push_back({"cost", a});
        HostMoeSplit a2     = a;
        a2.max_reads        = 2;
        a2.device_expert_us = 60.0F;
        cases.push_back({"cost-max2", a2});
        HostMoeSplit m = a;
        m.min_misses   = 64; // never: every miss stays on the host
        cases.push_back({"cost-min64", m});
        HostMoeSplit all;
        all.mode     = HostMoeSplit::Mode::Fraction;
        all.fraction = 1.0F;
        cases.push_back({"fraction1", all});
    }
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> w(0.02F, 0.3F);
    double worst_dev = 0, worst_host = 0, worst_sum = 0;
    int checked = 0, device_experts = 0, host_experts = 0;
    for (const Case& c : cases) {
        for (int replay = 0; replay < 3; ++replay) {
            std::vector<std::uint16_t> x(static_cast<std::size_t>(T) * H);
            for (auto& v : x) v = to_bf16(normal(rng));
            std::vector<float> y0(static_cast<std::size_t>(T) * H);
            for (auto& v : y0) v = normal(rng);
            std::vector<std::int32_t> ids(T * K), table(E);
            std::vector<float> alpha(T * K);
            // tokens route 3..10 of the E experts (columns per expert vary), ~half the experts
            // resident
            for (int t = 0; t < T; ++t) {
                std::vector<int> all(E);
                for (int e = 0; e < E; ++e) all[e] = e;
                std::shuffle(all.begin(), all.end(), rng);
                const int used = 3 + static_cast<int>(rng() % 8);
                for (int k = 0; k < K; ++k)
                    ids[t * K + k] = k < used ? all[k] : -1, alpha[t * K + k] = w(rng);
            }
            for (int e = 0; e < E; ++e) table[e] = (rng() % 2) ? 7 : -1;
            // FP64 contribution of every missing expert (before the launch: it sets the divisors).
            std::map<int, std::vector<double>> ref_of;
            oracle.hmax = 0;
            for (int j = 0; j < T * K; ++j) {
                const int e = ids[j];
                if (e >= 0 && table[e] < 0 && !ref_of.count(e))
                    ref_of[e] = oracle.partial(x, ids, alpha, {e}, T, K);
            }
            double xmax = 0;
            for (auto v : x) xmax = std::max(xmax, std::abs(static_cast<double>(from_bf16(v))));
            x_divisor = static_cast<float>(448.0 * 6.0 / xmax);
            h_divisor = oracle.hmax > 0 ? static_cast<float>(448.0 * 6.0 / oracle.hmax) : 2000.0F;
            CK(cudaMemcpy(dx, x.data(), 2 * T * H, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dids, ids.data(), 4 * T * K, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dalpha, alpha.data(), 4 * T * K, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dtable, table.data(), 4 * E, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dyf, y0.data(), 4 * T * H, cudaMemcpyHostToDevice));
            // Pageable uploads may still be in flight; s does not wait for the default stream.
            CK(cudaDeviceSynchronize());
            mailbox.clear_route_flags();
            mailbox.publish_table(s, 0, T, dx, dids, dalpha, dtable, E, 0, c.split,
                                  HostMoeDeviceShare{dreads, dpicks, dstage});
            ninfer::Tensor tx(dx, ninfer::DType::BF16, {H, T}),
                tr(dreads, ninfer::DType::I32, {K, T}), tw(dalpha, ninfer::DType::FP32, {K, T}),
                tz(dz, ninfer::DType::FP32, {H, T}), ts(ds, ninfer::DType::FP32, {H, T});
            nops::moe_slots_nvfp4(tx, tr, tw, slots_layer(dident), bank,
                                  nops::MoeSlotsOutput::Store, tz, work, s);
            stage_device_share(s, dpicks,
                               {static_cast<const std::byte*>(bank.gate_up_codes.base),
                                static_cast<const std::byte*>(bank.gate_up_scales.base),
                                static_cast<const std::byte*>(bank.down_codes.base),
                                static_cast<const std::byte*>(bank.down_scales.base)},
                               staging, plane_bytes, 8);
            stage_device_share(s, dpicks, narrow_bank, staging_narrow, plane_bytes, 8);
            nops::moe_slots_nvfp4(tx, tr, tw, slots_layer(dstage), scratch_pool,
                                  nops::MoeSlotsOutput::Store, ts, work, s);
            mailbox.accumulate(s, 0, T, dyf, ds);
            CK(cudaStreamSynchronize(s));
            expect(!mailbox.failed(), "split: mailbox failed: " + mailbox.error());

            // Reference selection: distinct misses with first route and columns, in split order.
            struct Miss {
                int id, first, cols;
            };

            std::vector<Miss> miss;
            for (int j = 0; j < T * K; ++j) {
                const int e = ids[j];
                if (e < 0 || table[e] >= 0) continue;
                auto it = std::find_if(miss.begin(), miss.end(),
                                       [&](const Miss& m) { return m.id == e; });
                if (it == miss.end())
                    miss.push_back({e, j, 1});
                else
                    ++it->cols;
            }
            if (c.split.mode != HostMoeSplit::Mode::Count) {
                std::stable_sort(miss.begin(), miss.end(), [](const Miss& a, const Miss& b) {
                    return a.cols != b.cols ? a.cols > b.cols : a.first < b.first;
                });
            }
            std::vector<int> cols;
            for (const auto& m : miss) cols.push_back(m.cols);
            const int g = reference_share(c.split, cols);
            std::vector<int> picked, rest;
            int picked_routes = 0;
            for (int i = 0; i < static_cast<int>(miss.size()); ++i) {
                (i < g ? picked : rest).push_back(miss[i].id);
                if (i < g) picked_routes += miss[i].cols;
            }
            const std::string tag = "split " + c.name + " replay " + std::to_string(replay);
            std::vector<std::int32_t> reads(T * K), picks(HostMoeSplit::kMaxReads + 1), stage(E);
            CK(cudaMemcpy(reads.data(), dreads, 4 * T * K, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(picks.data(), dpicks, 4 * picks.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(stage.data(), dstage, 4 * E, cudaMemcpyDeviceToHost));
            int bad_sel = picks[HostMoeSplit::kMaxReads] != g;
            for (int k = 0; k < HostMoeSplit::kMaxReads; ++k)
                bad_sel += picks[k] != (k < g ? picked[k] : -1);
            for (int j = 0; j < T * K; ++j) {
                const bool dev = std::find(picked.begin(), picked.end(), ids[j]) != picked.end();
                bad_sel += reads[j] != (dev ? ids[j] : -1);
            }
            for (int e = 0; e < E; ++e) {
                const auto it = std::find(picked.begin(), picked.end(), e);
                bad_sel +=
                    stage[e] != (it == picked.end() ? -1 : static_cast<int>(it - picked.begin()));
            }
            bad_sel += mailbox.device_share_experts(0) != g;
            bad_sel += mailbox.device_share_routes(0) != picked_routes;
            expect(bad_sel == 0, tag + ": device share differs from the reference rule (" +
                                     std::to_string(bad_sel) + " mismatches, g " +
                                     std::to_string(g) + " of " + std::to_string(miss.size()) +
                                     ")");

            std::vector<float> zf(T * H), sf(T * H), yf(T * H), p(T * H, 0.0F);
            CK(cudaMemcpy(zf.data(), dz, 4 * T * H, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sf.data(), ds, 4 * T * H, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(yf.data(), dyf, 4 * T * H, cudaMemcpyDeviceToHost));
            int bad_narrow = 0;
            for (int q = 0; q < 4; ++q) {
                const std::size_t bytes = static_cast<std::size_t>(plane_bytes[q]) * g;
                std::vector<std::uint8_t> wide_copy(bytes), narrow_copy(bytes);
                if (bytes == 0) continue;
                CK(cudaMemcpy(wide_copy.data(), staging[q], bytes, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(narrow_copy.data(), staging_narrow[q], bytes,
                              cudaMemcpyDeviceToHost));
                bad_narrow += wide_copy != narrow_copy;
                // and both equal the bank's bytes of the picked experts
                const auto* src = reinterpret_cast<const std::uint8_t*>(
                    q == 0   ? static_cast<const void*>(layer.gate_up.codes)
                    : q == 1 ? static_cast<const void*>(layer.gate_up.scales)
                    : q == 2 ? static_cast<const void*>(layer.down.codes)
                             : static_cast<const void*>(layer.down.scales));
                for (int k = 0; k < g; ++k)
                    bad_narrow +=
                        std::memcmp(wide_copy.data() + static_cast<std::size_t>(k) * plane_bytes[q],
                                    src + static_cast<std::size_t>(picked[k]) * plane_bytes[q],
                                    plane_bytes[q]) != 0;
            }
            expect(bad_narrow == 0,
                   tag +
                       ": staged scratch slots differ from the bank (16-byte or 4-byte copy path)");
            int bad_stage = 0;
            for (int i = 0; i < T * H; ++i) bad_stage += std::memcmp(&zf[i], &sf[i], 4) != 0;
            expect(bad_stage == 0, tag + ": staged share differs from zero-copy in " +
                                       std::to_string(bad_stage) + " outputs");
            std::vector<std::int32_t> host_ids(ids);
            for (auto& e : host_ids)
                if (e >= 0 &&
                    (table[e] >= 0 || std::find(picked.begin(), picked.end(), e) != picked.end()))
                    e = -1;
            const bool any =
                std::any_of(host_ids.begin(), host_ids.end(), [](int e) { return e >= 0; });
            if (any) {
                const HostMoeJob job{T, K, x.data(), host_ids.data(), alpha.data(), p.data()};
                ex.run(layer, job);
            }
            int bad_sum = 0;
            for (int i = 0; i < T * H; ++i) {
                float v = y0[i];
                v += sf[i];
                if (any) v += p[i];
                bad_sum += std::memcmp(&v, &yf[i], 4) != 0;
            }
            expect(bad_sum == 0, tag + ": y != y0 + device + host partial in " +
                                     std::to_string(bad_sum) + " outputs");

            // FP64 oracle per side and for the sum of both.
            std::vector<double> ref_dev(static_cast<std::size_t>(T) * H, 0.0), ref_host(ref_dev);
            for (const int e : picked)
                for (int i = 0; i < T * H; ++i) ref_dev[i] += ref_of[e][i];
            for (const int e : rest)
                for (int i = 0; i < T * H; ++i) ref_host[i] += ref_of[e][i];
            std::vector<double> got_dev(T * H), got_host(T * H), got_sum(T * H), ref_sum(T * H);
            for (int i = 0; i < T * H; ++i) {
                got_dev[i]  = sf[i];
                got_host[i] = p[i];
                got_sum[i]  = static_cast<double>(sf[i]) + p[i];
                ref_sum[i]  = ref_dev[i] + ref_host[i];
            }
            if (!picked.empty()) {
                const double e = rel_l2(got_dev, ref_dev);
                worst_dev      = std::max(worst_dev, e);
                expect(e <= 0.18,
                       tag + ": device share rel L2 vs FP64 " + std::to_string(e) + " > 0.18");
            }
            if (!rest.empty()) {
                const double e = rel_l2(got_host, ref_host);
                worst_host     = std::max(worst_host, e);
                expect(e <= 0.05,
                       tag + ": host share rel L2 vs FP64 " + std::to_string(e) + " > 0.05");
            }
            if (!miss.empty()) {
                const double e = rel_l2(got_sum, ref_sum);
                worst_sum      = std::max(worst_sum, e);
                expect(e <= 0.18,
                       tag + ": split sum rel L2 vs FP64 " + std::to_string(e) + " > 0.18");
            }
            ++checked;
            device_experts += g;
            host_experts += static_cast<int>(rest.size());
            std::printf("  %-12s replay %d: misses %zu (columns", c.name.c_str(), replay,
                        miss.size());
            for (int v : cols) std::printf(" %d", v);
            std::printf(") -> device %d, host %zu\n", g, rest.size());
        }
    }
    std::printf("miss split: %d calls, device %d / host %d experts; selection, complement and "
                "staged==zero-copy "
                "exact (16- and 4-byte copy paths); rel L2 vs FP64 worst device %.4f (<= 0.18), "
                "host %.4f (<= 0.05), sum %.4f (<= 0.18)\n",
                checked, device_experts, host_experts, worst_dev, worst_host, worst_sum);
    CK(cudaStreamDestroy(s));
    for (void* q : {static_cast<void*>(d13), static_cast<void*>(d2), static_cast<void*>(dyf),
                    static_cast<void*>(dz), static_cast<void*>(ds), static_cast<void*>(dalpha),
                    static_cast<void*>(dids), static_cast<void*>(dtable),
                    static_cast<void*>(dreads), static_cast<void*>(dpicks),
                    static_cast<void*>(dstage), static_cast<void*>(dident), static_cast<void*>(dx)})
        CK(cudaFree(q));
    for (auto* q : staging) CK(cudaFree(q));
    for (auto* q : staging_narrow) CK(cudaFree(q));
    CK(cudaFreeHost(narrow_host));
    CK(cudaFreeHost(gu));
    CK(cudaFreeHost(dn));
}

// fit_host_costs (--miss-split calibrate): recovers a known cost line from noisy samples with
// outliers, and keeps the given column cost when every call has one column per expert.
void cost_fit() {
    std::mt19937_64 rng(11);
    std::normal_distribution<float> noise(0.0F, 2.0F);
    std::vector<HostMoeCostSample> s;
    for (int i = 0; i < 4000; ++i) {
        const int e     = 1 + static_cast<int>(rng() % 12),
                  extra = static_cast<int>(rng() % (3 * e + 1));
        float us        = 20.0F + 50.0F * e + 3.0F * extra + noise(rng);
        if (i % 97 == 0) us += 2000.0F; // wake-up after idle
        s.push_back({e, e + extra, us});
    }
    const HostMoeCostFit f = fit_host_costs(s, 1.9F);
    expect(f.samples > 3800 && std::fabs(f.fixed_us - 20.0F) < 1.5F &&
               std::fabs(f.expert_us - 50.0F) < 0.5F && std::fabs(f.column_us - 3.0F) < 0.3F &&
               f.rms_us < 3.0F,
           "cost fit recovers 20 + 50/expert + 3/column");
    std::vector<HostMoeCostSample> one;
    for (int i = 0; i < 500; ++i) {
        const int e = 1 + i % 9;
        one.push_back({e, e, 10.0F + 40.0F * e + noise(rng)});
    }
    const HostMoeCostFit g = fit_host_costs(one, 1.9F);
    expect(g.samples > 450 && std::fabs(g.expert_us - 40.0F) < 0.5F && g.column_us == 1.9F,
           "cost fit with one column per expert keeps the given column cost");
    expect(fit_host_costs(std::span(s).first(8), 1.9F).samples == 0, "cost fit needs samples");
}

} // namespace

int main() {
    cost_fit();
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0 || !host_experts_supported()) {
        std::printf("SKIP: needs a CUDA device and AVX2 + FMA + F16C\n");
        return 77;
    }
    CK(cudaSetDevice(0));
    constexpr int L = 6, T = 4, K = 10, E = 12, H = 2560, I = 640;
    std::mt19937_64 rng(7);
    LayerData a(E, H, I, rng), b(E, H, I, rng);
    std::vector<HostExpertLayer> layers;
    for (int l = 0; l < L; ++l) layers.push_back(l % 2 ? b.layer() : a.layer());

    HostMoeMailboxConfig cfg;
    cfg.device = 0, cfg.layers = L, cfg.max_tokens = 8, cfg.top_k = K, cfg.hidden = H,
    cfg.intermediate   = I;
    cfg.cpus           = spread_cpus(6);
    cfg.trace_capacity = 1024;
    cfg.executor.a4 =
        false; // the shares are checked against the W4A8 reference (A4 is the default since v0.2.1)

    // Host reference executor on its own pool (the mailbox's service thread owns the first cpus).
    const auto ref_cpus = std::vector<int>(cfg.cpus.begin() + 3, cfg.cpus.end());

    const std::size_t nx = static_cast<std::size_t>(L) * T * H,
                      nr = static_cast<std::size_t>(L) * T * K;
    std::vector<std::uint16_t> x(nx), y0(nx);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) v = to_bf16(normal(rng));
    for (auto& v : y0) v = to_bf16(normal(rng));
    std::vector<float> y0f(nx);
    for (std::size_t i = 0; i < nx; ++i) y0f[i] = normal(rng);

    __nv_bfloat16 *dx = nullptr, *dy = nullptr;
    float *dyf = nullptr, *dalpha = nullptr;
    std::int32_t *dids = nullptr, *dslot = nullptr;
    CK(cudaMalloc(&dx, nx * 2));
    CK(cudaMalloc(&dy, nx * 2));
    CK(cudaMalloc(&dyf, nx * 4));
    CK(cudaMalloc(&dalpha, nr * 4));
    CK(cudaMalloc(&dids, nr * 4));
    CK(cudaMalloc(&dslot, nr * 4));
    CK(cudaMemcpy(dx, x.data(), nx * 2, cudaMemcpyHostToDevice));

    {
        HostMoeMailbox mailbox(cfg, layers);
        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < L; ++l) {
            const std::size_t ox  = static_cast<std::size_t>(l) * T * H,
                              orr = static_cast<std::size_t>(l) * T * K;
            mailbox.publish(s, l, T, dx + ox, dids + orr, dalpha + orr, dslot + orr);
            busy_kernel<<<1, 32, 0, s>>>(20000);
            mailbox.accumulate(s, l, T, dy + ox);
            // Same partial into an FP32 accumulator: a second wait on a completed layer passes.
            mailbox.accumulate(s, l, T, dyf + ox);
        }
        CK(cudaStreamEndCapture(s, &graph));
        CK(cudaGraphInstantiate(&exec, graph, 0));

        std::vector<std::uint16_t> y(nx);
        std::vector<float> yf(nx), p(static_cast<std::size_t>(T) * H);
        int compared = 0;
        for (int replay = 0; replay < 4; ++replay) {
            const Routing r = route(L, T, K, E, rng, true);
            CK(cudaMemcpy(dids, r.ids.data(), nr * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dslot, r.slot.data(), nr * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dalpha, r.alpha.data(), nr * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dy, y0.data(), nx * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dyf, y0f.data(), nx * 4, cudaMemcpyHostToDevice));
            // Pageable uploads may still be in flight; s does not wait for the default stream.
            CK(cudaDeviceSynchronize());
            mailbox.clear_trace();
            CK(cudaGraphLaunch(exec, s));
            CK(cudaStreamSynchronize(s));
            expect(!mailbox.failed(), "mailbox failed: " + mailbox.error());
            CK(cudaMemcpy(y.data(), dy, nx * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(yf.data(), dyf, nx * 4, cudaMemcpyDeviceToHost));
            expect(mailbox.trace().size() == static_cast<std::size_t>(L),
                   "one trace record per layer");

            // Reference partials (pool built on this thread for the comparison only).
            HostCpuPool pool(ref_cpus);
            HostMoeExecutor ex(pool, H, I, 8, K, w4a8_options());
            for (int l = 0; l < L; ++l) {
                const std::size_t ox  = static_cast<std::size_t>(l) * T * H,
                                  orr = static_cast<std::size_t>(l) * T * K;
                std::vector<std::int32_t> host_ids(r.ids.begin() + orr,
                                                   r.ids.begin() + orr + T * K);
                bool any = false;
                for (int i = 0; i < T * K; ++i) {
                    if (r.slot[orr + i] >= 0) host_ids[i] = -1;
                    any = any || host_ids[i] >= 0;
                }
                expect(l != 0 || !any, "layer 0 is all-device");
                const HostMoeJob job{
                    T, K, x.data() + ox, host_ids.data(), r.alpha.data() + orr, p.data()};
                ex.run(layers[l], job);
                int bad = 0, badf = 0;
                for (int i = 0; i < T * H; ++i) {
                    const std::uint16_t want =
                        any ? to_bf16(from_bf16(y0[ox + i]) + p[i]) : y0[ox + i];
                    const float wantf = any ? y0f[ox + i] + p[i] : y0f[ox + i];
                    bad += y[ox + i] != want;
                    badf += yf[ox + i] != wantf;
                }
                compared += T * H;
                expect(bad == 0, "replay " + std::to_string(replay) + " layer " +
                                     std::to_string(l) + ": " + std::to_string(bad) +
                                     " bf16 mismatches");
                expect(badf == 0, "replay " + std::to_string(replay) + " layer " +
                                      std::to_string(l) + ": " + std::to_string(badf) +
                                      " fp32 mismatches");
            }
        }
        std::printf("graph: %d layers x 4 replays, %d outputs compared bit-exact (bf16 and fp32)\n",
                    L, compared);

        // Failure path: an out-of-range id makes the executor throw inside the service.
        Routing r = route(L, T, K, E, rng, false);
        for (int t = 0; t < T; ++t) r.slot[(2 * T + t) * K] = -1;
        r.ids[2 * T * K] = E + 5; // layer 2, token 0
        CK(cudaMemcpy(dids, r.ids.data(), nr * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dslot, r.slot.data(), nr * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dy, y0.data(), nx * 2, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        CK(cudaGraphLaunch(exec, s));
        CK(cudaStreamSynchronize(s)); // must not hang
        expect(mailbox.failed(), "failure reported");
        CK(cudaMemcpy(y.data(), dy, nx * 2, cudaMemcpyDeviceToHost));
        int touched = 0;
        for (std::size_t i = 2 * T * H; i < nx; ++i) touched += y[i] != y0[i];
        expect(touched == 0,
               "after the failure no partial is added (" + std::to_string(touched) + " changed)");
        CK(cudaGraphLaunch(exec, s)); // a failed mailbox keeps answering
        CK(cudaStreamSynchronize(s));
        std::printf("failure path: graph completed, failed() = %d (%s)\n", mailbox.failed(),
                    mailbox.error().c_str());
        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
        CK(cudaStreamDestroy(s));
    }
    // After the first mailbox stopped (its service thread spins on the same cpus).
    table_and_slices(cfg, layers, x, y0f, ref_cpus, rng);
    split_shares(cfg, a, ref_cpus, rng);
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
