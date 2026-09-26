// Public KDA timing and production-stage attribution. Head counts/grouping remain runtime data.
#include "ninfer/ops/kimi_delta_attention.h"
#include "ops/linear_attention/kimi_delta_attention/launch.h"
#include "ninfer_bench_common.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;
namespace kda = ninfer::ops::detail::kimi_delta_attention;

namespace {
constexpr int D       = 128;
constexpr float scale = 1.0F / 11.313708498984761F;

struct Profile {
    const char* name;
    int qk, value;
};

constexpr Profile profiles[] = {{"qwen3.5-27b", 16, 48},
                                {"qwen3.5-35b-a3b", 16, 32},
                                {"kimi-k3", 96, 96},
                                {"glm5.3-flash", 64, 64},
                                {"kimi-linear", 32, 32}};

struct Options {
    std::string profile = "kimi-k3";
    int qk = 96, value = 96, tokens = 1024, batch = 1, warmup = 20, repeat = 100;
    bool custom = false, sweep = false, breakdown = false;
    bool batch_update = false, batch_sweep = false, csv = false, help = false;
    std::size_t flush = 256ULL << 20;
};

int integer(const char* flag, const char* value, int minimum) {
    char* end = nullptr;
    errno     = 0;
    long n    = std::strtol(value, &end, 10);
    if (errno || end == value || *end || n < minimum || n > INT32_MAX)
        throw std::invalid_argument(std::string("invalid ") + flag);
    return static_cast<int>(n);
}

Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        const auto take = [&]() {
            if (++i >= argc) throw std::invalid_argument("missing value");
            return argv[i];
        };
        if (a == "--profile")
            o.profile = take();
        else if (a == "--qk-heads") {
            o.qk     = integer(a.data(), take(), 1);
            o.custom = true;
        } else if (a == "--value-heads") {
            o.value  = integer(a.data(), take(), 1);
            o.custom = true;
        } else if (a == "--tokens")
            o.tokens = integer(a.data(), take(), 1);
        else if (a == "--batch")
            o.batch = integer(a.data(), take(), 1);
        else if (a == "--warmup")
            o.warmup = integer(a.data(), take(), 0);
        else if (a == "--repeat")
            o.repeat = integer(a.data(), take(), 1);
        else if (a == "--flush-mib")
            o.flush = static_cast<std::size_t>(integer(a.data(), take(), 1)) << 20;
        else if (a == "--sweep")
            o.sweep = true;
        else if (a == "--batch-sweep")
            o.batch_sweep = true;
        else if (a == "--breakdown")
            o.breakdown = true;
        else if (a == "--batch-update")
            o.batch_update = true;
        else if (a == "--csv")
            o.csv = true;
        else if (a == "--help" || a == "-h")
            o.help = true;
        else
            throw std::invalid_argument("unknown argument: " + std::string(a));
    }
    if (o.batch > 8) throw std::invalid_argument("batch must be in [1,8]");
    if (o.batch_update && (o.sweep || o.breakdown))
        throw std::invalid_argument("batch update has T=1 and no stages");
    if (!o.batch_update && (o.batch != 1 || o.batch_sweep))
        throw std::invalid_argument("batch requires --batch-update");
    if (o.custom && o.profile == "all")
        throw std::invalid_argument("custom heads cannot use --profile all");
    return o;
}

DeviceBuffer bf16(std::size_t count, unsigned seed, float low, float high) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<__nv_bfloat16> values(count);
    for (auto& v : values) v = __float2bfloat16(dist(rng));
    DeviceBuffer out(count * 2);
    out.copy_from_host(values.data(), out.bytes);
    return out;
}

DeviceBuffer fp32(std::size_t count, unsigned seed, float low, float high) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> values(count);
    for (auto& v : values) v = dist(rng);
    DeviceBuffer out(count * 4);
    out.copy_from_host(values.data(), out.bytes);
    return out;
}

struct Fixture {
    int hq, hv, t, b;
    DeviceBuffer q, k, v, g, beta, alog, bias, initial, final, out, slots;

    Fixture(Profile p, int tokens, int batch)
        : hq(p.qk), hv(p.value), t(tokens), b(batch),
          q(bf16(std::size_t(D) * hq * t * b, 11, -1, 1)),
          k(bf16(std::size_t(D) * hq * t * b, 12, -1, 1)),
          v(bf16(std::size_t(D) * hv * t * b, 13, -0.5F, 0.5F)),
          g(bf16(std::size_t(D) * hv * t * b, 14, -4, 4)),
          beta(bf16(std::size_t(hv) * t * b, 15, -5, 5)), alog(fp32(hv, 16, -0.5F, 0.5F)),
          bias(fp32(std::size_t(D) * hv, 17, -2, 2)),
          initial(fp32(std::size_t(D) * D * hv * b, 18, -0.02F, 0.02F)), final(initial.bytes),
          out(v.bytes), slots(std::size_t(b) * sizeof(std::int32_t)) {
        std::vector<std::int32_t> selectors(b);
        for (int i = 0; i < b; ++i) selectors[i] = b - 1 - i;
        slots.copy_from_host(selectors.data(), slots.bytes);
    }

    kda::Arguments args() const {
        return {static_cast<const __nv_bfloat16*>(q.p),
                static_cast<const __nv_bfloat16*>(k.p),
                static_cast<const __nv_bfloat16*>(v.p),
                static_cast<const __nv_bfloat16*>(g.p),
                static_cast<const __nv_bfloat16*>(beta.p),
                static_cast<const float*>(alog.p),
                static_cast<const float*>(bias.p),
                static_cast<const float*>(initial.p),
                static_cast<float*>(final.p),
                static_cast<__nv_bfloat16*>(out.p),
                hq,
                hv,
                t,
                -5.0F,
                scale};
    }

    void public_call(WorkspaceArena& ws, DeviceExecutionView ex, bool batch_mode) {
        Tensor qt(q.p, DType::BF16, {D, hq, t, b}), kt(k.p, DType::BF16, {D, hq, t, b});
        Tensor vt(v.p, DType::BF16, {D, hv, t, b}), gt(g.p, DType::BF16, {D, hv, t, b});
        Tensor bt(beta.p, DType::BF16, {hv, t, b}), at(alog.p, DType::FP32, {hv}),
            dt(bias.p, DType::FP32, {D, hv});
        Tensor si(initial.p, DType::FP32, {D, D, hv, b}), so(final.p, DType::FP32, {D, D, hv, b});
        Tensor ot(out.p, DType::BF16, {D, hv, t, b}), st(slots.p, DType::I32, {b});
        if (batch_mode)
            ops::kimi_delta_attention_batch_update(qt, kt, vt, gt, bt, at, dt, -5, scale, so, st,
                                                   ot, ex.stream);
        else
            ops::kimi_delta_attention(qt, kt, vt, gt, bt, at, dt, -5, scale, ws, si, so, ot, ex);
    }
};

struct Traffic {
    double logical = 0, packet_write = 0, packet_read = 0, requests = 0, bf16_flops = 0,
           tf32_flops = 0;
};

Traffic traffic(Profile p, int t, int b, std::size_t ws, int slices, std::string_view stage) {
    const double hv = p.value, hq = p.qk, nt = static_cast<double>(t), nc = kda::chunk_count(t);
    const double state = 8.0 * D * D * hv;
    const double logical =
        (4.0 * D * hq * nt + (6.0 * D + 2) * hv * nt + state) * b + 4.0 * (D + 1) * hv;
    const double prepare    = (6.0 * D + 2) * nt * hv + 4.0 * (D + 1) * hv * nc + ws;
    const double recurrence = slices * static_cast<double>(ws) + 4.0 * D * nt * hv + state;
    if (stage == "prepare") return {0, double(ws), 0, prepare, 4.0 * 16 * 16 * D * hv * nc, 0};
    if (stage == "recurrence")
        return {0,          0, slices * double(ws),
                recurrence, 0, (6.0 * 16 * D * D + 4.0 * 16 * 16 * D) * hv * nc};
    if (ws == 0) return {logical, 0, 0, logical, 0, 0};
    return {logical,
            double(ws),
            slices * double(ws),
            prepare + recurrence,
            4.0 * 16 * 16 * D * hv * nc,
            (6.0 * 16 * D * D + 4.0 * 16 * 16 * D) * hv * nc};
}

void print(Profile p, int t, int b, const char* stage, int tile, std::size_t ws, std::size_t nodes,
           Traffic io, ColdTiming time, const Options& o) {
    if (o.csv) {
        std::printf("%s,%d,%d,%d,%d,%s,%d,%zu,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%zu,cold_l2,%.3f,%.3f,%"
                    ".3f,%.3f,%.3f\n",
                    p.name, p.qk, p.value, t, b, stage, tile, ws, io.logical, io.packet_write,
                    io.packet_read, io.requests, io.bf16_flops, io.tf32_flops, nodes,
                    time.median_us, time.min_us, time.p95_us, io.logical / (time.median_us * 1e3),
                    io.requests / (time.median_us * 1e3));
    } else {
        std::printf("%-17s Hqk=%3d Hv=%3d T=%5d B=%d %-18s tile=%2d median=%9.3f us min=%9.3f "
                    "p95=%9.3f | ws=%7.2f MiB tensor_requests=%7.1f GB/s nodes=%zu\n",
                    p.name, p.qk, p.value, t, b, stage, tile, time.median_us, time.min_us,
                    time.p95_us, ws / double(1ULL << 20), io.requests / (time.median_us * 1e3),
                    nodes);
    }
}

void run(Profile p, int t, int b, const Options& o, DeviceContext& ctx, DeviceBuffer& flush) {
    if (!kda::valid_heads(p.qk, p.value))
        throw std::invalid_argument("require Hqk>0, Hv>=Hqk, Hv%Hqk=0");
    const auto ws_bytes = ops::kimi_delta_attention_workspace_capacity_bytes(p.qk, p.value, t, t);
    const bool chunked  = ws_bytes != 0;
    Fixture f(p, t, b);
    WorkspaceArena ws(std::max<std::size_t>(256, ws_bytes));
    auto* packets  = static_cast<kda::Chunk*>(ws.base());
    const auto a   = f.args();
    const auto ex  = ctx.execution_view();
    const int tile = chunked ? kda::chunk_value_tile(p.value, ex.multiprocessor_count) : 0;
    auto reset     = [&](cudaStream_t s) {
        if (o.batch_update)
            CUDA_CHECK(cudaMemcpyAsync(f.final.p, f.initial.p, f.initial.bytes,
                                           cudaMemcpyDeviceToDevice, s));
    };
    auto measure = [&](const char* name, std::string_view kind, auto&& body) {
        reset(ex.stream);
        body(ex.stream);
        ctx.synchronize();
        TimedGraph graph;
        graph.capture(ex.stream, body);
        const auto result =
            measure_cold_graph_prepared(reset, graph, flush, ex.stream, o.warmup, o.repeat);
        print(p, t, b, name, tile, ws_bytes, graph.nodes(),
              traffic(p, t, b, ws_bytes, tile ? D / tile : 0, kind), result, o);
    };
    auto total = [&](cudaStream_t stream) {
        f.public_call(ws, {stream, ex.multiprocessor_count}, o.batch_update);
    };
    measure("public.total", "total", total);
    if (o.breakdown && chunked) {
        measure("chunked.prepare", "prepare",
                [&](cudaStream_t stream) { kda::launch_prepare(a, packets, stream); });
        measure("chunked.recurrence", "recurrence", [&](cudaStream_t stream) {
            kda::launch_chunk_recurrence(a, packets, {stream, ex.multiprocessor_count});
        });
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = options(argc, argv);
        if (o.help) {
            std::printf(
                "KDA public Op and production-stage benchmark\n"
                "  --profile NAME      qwen3.5-27b, qwen3.5-35b-a3b, kimi-k3 (default), "
                "glm5.3-flash, kimi-linear, all\n"
                "  --qk-heads H --value-heads V   custom runtime geometry\n"
                "  --tokens T          default 1024\n"
                "  --sweep             T=1,8,11,12,13,16,32,64,128,256,512,1024,2048,4096,8192\n"
                "  --breakdown         also time the two chunked stages separately\n"
                "  --batch-update --batch B | --batch-sweep   selected-slot T=1, B=1..8\n"
                "  --warmup N --repeat N --flush-mib M        defaults 20,100,256\n"
                "  --csv               report timing and algorithm/CTA-request traffic (not "
                "measured DRAM bytes)\n");
            return 0;
        }
        DeviceContext ctx;
        DeviceBuffer flush(o.flush);
        if (o.csv)
            std::puts("profile,qk_heads,value_heads,tokens,batch,stage,value_tile,workspace_bytes,"
                      "logical_bytes,workspace_write_bytes,workspace_read_request_bytes,tensor_io_"
                      "request_bytes,bf16_flops,tf32_flops,graph_nodes,cache,median_us,min_us,p95_"
                      "us,logical_gbps,tensor_io_request_gbps");
        else
            std::printf("%s; %d SMs; cold-L2 graph; flush outside timer; workspace requests "
                        "include CTA replication\n",
                        ctx.props.name, ctx.multiprocessor_count());
        std::vector<Profile> selected;
        if (o.custom)
            selected.push_back({"custom", o.qk, o.value});
        else if (o.profile == "all")
            selected.assign(profiles, profiles + 4);
        else {
            for (auto p : profiles)
                if (o.profile == p.name) selected.push_back(p);
            if (selected.empty()) throw std::invalid_argument("unknown profile");
        }
        const std::vector<int> lengths =
            o.batch_update ? std::vector<int>{1}
            : o.sweep      ? std::vector<int>{1,   8,   11,  12,   13,   16,   32,  64,
                                              128, 256, 512, 1024, 2048, 4096, 8192}
                           : std::vector<int>{o.tokens};
        const std::vector<int> batches =
            o.batch_sweep ? std::vector<int>{1, 2, 4, 8} : std::vector<int>{o.batch};
        for (auto p : selected)
            for (int t : lengths)
                for (int b : batches) run(p, t, b, o, ctx, flush);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "KDA benchmark: %s\n", e.what());
        return 1;
    }
}
