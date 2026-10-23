// Public KDA timing. Head counts/grouping remain runtime data.
#include "ninfer/ops/kimi_delta_attention.h"
#include "core/buffer.h"
#include "op_measurement.cuh"

#include <algorithm>
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
    bool custom = false, sweep = false;
    bool batch_update = false, batch_sweep = false, csv = false, help = false;
    std::size_t flush = 256ULL << 20;
};

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
            o.qk     = measurement::integer(take(), 1);
            o.custom = true;
        } else if (a == "--value-heads") {
            o.value  = measurement::integer(take(), 1);
            o.custom = true;
        } else if (a == "--tokens")
            o.tokens = measurement::integer(take(), 1);
        else if (a == "--batch")
            o.batch = measurement::integer(take(), 1);
        else if (a == "--warmup")
            o.warmup = measurement::integer(take(), 0);
        else if (a == "--repeat")
            o.repeat = measurement::integer(take(), 1);
        else if (a == "--flush-mib")
            o.flush = static_cast<std::size_t>(measurement::integer(take(), 1)) << 20;
        else if (a == "--sweep")
            o.sweep = true;
        else if (a == "--batch-sweep")
            o.batch_sweep = true;
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
    if (o.batch_update && o.sweep) throw std::invalid_argument("batch update has T=1");
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
    DeviceBuffer out(ninfer::bench::device_context().device(), count * 2);
    out.copy_from_host(values.data(), out.size());
    return out;
}

DeviceBuffer fp32(std::size_t count, unsigned seed, float low, float high) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> values(count);
    for (auto& v : values) v = dist(rng);
    DeviceBuffer out(ninfer::bench::device_context().device(), count * 4);
    out.copy_from_host(values.data(), out.size());
    return out;
}

struct Fixture {
    int hq, hv, t, b;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
    DeviceBuffer alog;
    DeviceBuffer bias;
    DeviceBuffer initial;
    DeviceBuffer final;
    DeviceBuffer out;
    DeviceBuffer state_offsets;

    Fixture(Profile p, int tokens, int batch)
        : hq(p.qk), hv(p.value), t(tokens), b(batch),
          q(bf16(std::size_t(D) * hq * t * b, 11, -1, 1)),
          k(bf16(std::size_t(D) * hq * t * b, 12, -1, 1)),
          v(bf16(std::size_t(D) * hv * t * b, 13, -0.5F, 0.5F)),
          g(bf16(std::size_t(D) * hv * t * b, 14, -4, 4)),
          beta(bf16(std::size_t(hv) * t * b, 15, -5, 5)), alog(fp32(hv, 16, -0.5F, 0.5F)),
          bias(fp32(std::size_t(D) * hv, 17, -2, 2)),
          initial(fp32(std::size_t(D) * D * hv * b, 18, -0.02F, 0.02F)),
          final(ninfer::bench::device_context().device(), initial.size()),
          out(ninfer::bench::device_context().device(), v.size()),
          state_offsets(ninfer::bench::device_context().device(),
                        std::size_t(b) * sizeof(std::int64_t)) {
        std::vector<std::int64_t> offsets(b);
        for (int i = 0; i < b; ++i)
            offsets[i] = std::int64_t(b - 1 - i) * D * D * hv * sizeof(float);
        state_offsets.copy_from_host(offsets.data(), state_offsets.size());
    }

    void public_call(MemoryArena& ws, CudaExecutionView ex, bool batch_mode) {
        Tensor qt(q.span(), DType::BF16, {D, hq, t, b}), kt(k.span(), DType::BF16, {D, hq, t, b});
        Tensor vt(v.span(), DType::BF16, {D, hv, t, b}), gt(g.span(), DType::BF16, {D, hv, t, b});
        Tensor bt(beta.span(), DType::BF16, {hv, t, b}), at(alog.span(), DType::FP32, {hv}),
            dt(bias.span(), DType::FP32, {D, hv});
        Tensor si(initial.span(), DType::FP32, {D, D, hv, b}),
            so(final.span(), DType::FP32, {D, D, hv, b});
        Tensor ot(out.span(), DType::BF16, {D, hv, t, b}),
            st(state_offsets.span(), DType::I64, {b});
        if (batch_mode) {
            TensorRegion region{
                {0, std::size_t(D) * D * hv * sizeof(float), 16}, DType::FP32, {D, D, hv, 1}};
            StateFieldView state{{final.span(), st}, StateFieldLayout{region, 1, 0}};
            ops::kimi_delta_attention_batch_update(qt, kt, vt, gt, bt, at, dt, -5, scale, state, ot,
                                                   ex);
        } else
            ops::kimi_delta_attention(qt, kt, vt, gt, bt, at, dt, -5, scale, ws, si, so, ot, ex);
    }
};

double logical_bytes(Profile p, int t, int b) {
    const double hv = p.value, hq = p.qk, nt = static_cast<double>(t);
    const double state = 8.0 * D * D * hv;
    return (4.0 * D * hq * nt + (6.0 * D + 2) * hv * nt + state) * b + 4.0 * (D + 1) * hv;
}

void print(Profile p, int t, int b, std::size_t ws, std::size_t nodes, ColdTiming time,
           const Options& o) {
    const char* op = o.batch_update ? "kimi_delta_attention_batch_update" : "kimi_delta_attention";
    const double bytes = logical_bytes(p, t, b) + (o.batch_update ? b * sizeof(std::int64_t) : 0);
    const double gbps  = bytes / (time.median_us * 1e3);
    if (o.csv) {
        std::printf("%s,%d,%d,%d,%d,%s,%zu,%.0f,%zu,cold_l2,%.3f,%.3f,%.3f,%.3f\n", p.name, p.qk,
                    p.value, t, b, op, ws, bytes, nodes, time.median_us, time.min_us, time.p95_us,
                    gbps);
    } else {
        std::printf("%-17s Hqk=%3d Hv=%3d T=%5d B=%d %s median=%9.3f us min=%9.3f "
                    "p95=%9.3f | ws=%7.2f MiB logical=%7.1f GB/s nodes=%zu\n",
                    p.name, p.qk, p.value, t, b, op, time.median_us, time.min_us, time.p95_us,
                    ws / double(1ULL << 20), gbps, nodes);
    }
}

void run(Profile p, int t, int b, const Options& o, DeviceContext& ctx,
         bench::L2FlushBuffer& flush) {
    const auto ws_bytes = ops::kimi_delta_attention_workspace_capacity_bytes(p.qk, p.value, t, t);
    Fixture f(p, t, b);
    ninfer::DeviceBuffer ws_storage(ninfer::bench::device_context().device(),
                                    std::max<std::size_t>(256, ws_bytes));
    MemoryArena ws(ws_storage.span());
    const auto ex = ctx.execution_view();
    auto reset    = [&](CudaExecutionView cuda) {
        const auto s = cuda.stream();
        if (o.batch_update)
            CUDA_CHECK(cudaMemcpyAsync(f.final.data(), f.initial.data(), f.initial.size(),
                                          cudaMemcpyDeviceToDevice, s));
    };
    auto measure = [&](auto&& body) {
        reset(ex);
        body(ex);
        ctx.synchronize();
        TimedGraph graph;
        graph.capture(ex, body);
        const auto result =
            measure_cold_graph_prepared(reset, graph, flush, ex, o.warmup, o.repeat);
        print(p, t, b, ws_bytes, graph.nodes(), result, o);
    };
    auto total = [&](CudaExecutionView cuda) { f.public_call(ws, cuda, o.batch_update); };
    measure(total);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = options(argc, argv);
        if (o.help) {
            std::printf(
                "KDA public Op benchmark\n"
                "  --profile NAME      qwen3.5-27b, qwen3.5-35b-a3b, kimi-k3 (default), "
                "glm5.3-flash, kimi-linear, all\n"
                "  --qk-heads H --value-heads V   custom runtime geometry\n"
                "  --tokens T          default 1024\n"
                "  --sweep             T=1,8,11,12,13,16,32,64,128,256,512,1024,2048,4096,8192\n"
                "  --batch-update --batch B | --batch-sweep   selected-slot T=1, B=1..8\n"
                "  --warmup N --repeat N --flush-mib M        defaults 20,100,256\n"
                "  --csv               report timing and logical bytes (not measured DRAM "
                "bytes)\n");
            return 0;
        }
        auto& ctx = ninfer::bench::device_context();
        bench::L2FlushBuffer flush(o.flush);
        if (o.csv)
            std::puts("profile,qk_heads,value_heads,tokens,batch,op,workspace_bytes,"
                      "logical_bytes,graph_nodes,cache,median_us,min_us,p95_us,logical_gbps");
        else
            std::printf("%s; %d SMs; cold-L2 graph; flush outside timer\n", ctx.properties().name,
                        ctx.multiprocessor_count());
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
