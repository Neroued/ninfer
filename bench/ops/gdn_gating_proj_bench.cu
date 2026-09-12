#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    bool geometry35 = false, parent = true, norm = true, profile = false;
    std::vector<int> tokens{1, 2, 4, 6, 8, 9, 16, 32, 64, 128};
    std::string execution = "graph", cache = "cold", csv;
    int warmup = 10, repeat = 61, graph_calls = 1;
};

void help() {
    std::cout
        << "usage: ninfer_gdn_gating_proj_bench [--geometry 27b|35b] [--weights parent|split] "
           "[--op norm|control] [--tokens T,...] [--execution eager|graph|both] [--cache "
           "cold|warm|both] "
           "[--graph-calls N] [--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n";
}

int integer(std::string_view raw, int low, int high) {
    std::size_t end = 0;
    int n           = std::stoi(std::string(raw), &end);
    if (end != raw.size() || n < low || n > high) throw std::invalid_argument("invalid integer");
    return n;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        const auto next    = [&]() -> std::string {
            if (++i == argc) throw std::invalid_argument("missing argument value");
            return argv[i];
        };
        if (a == "--help") {
            help();
            std::exit(0);
        } else if (a == "--geometry") {
            auto v = next();
            if (v != "27b" && v != "35b") throw std::invalid_argument("invalid geometry");
            o.geometry35 = v == "35b";
        } else if (a == "--weights") {
            auto v = next();
            if (v != "parent" && v != "split") throw std::invalid_argument("invalid weight form");
            o.parent = v == "parent";
        } else if (a == "--op") {
            auto v = next();
            if (v != "norm" && v != "control") throw std::invalid_argument("invalid Op");
            o.norm = v == "norm";
        } else if (a == "--tokens") {
            auto v = next();
            o.tokens.clear();
            std::size_t begin = 0;
            while (begin < v.size()) {
                auto end = v.find(',', begin);
                if (end == std::string::npos) end = v.size();
                o.tokens.push_back(
                    integer(std::string_view(v).substr(begin, end - begin), 1, 1000000));
                begin = end + 1;
            }
            if (o.tokens.empty()) throw std::invalid_argument("empty tokens");
        } else if (a == "--execution")
            o.execution = next();
        else if (a == "--cache")
            o.cache = next();
        else if (a == "--warmup")
            o.warmup = integer(next(), 0, 10000);
        else if (a == "--repeat")
            o.repeat = integer(next(), 1, 10000);
        else if (a == "--graph-calls")
            o.graph_calls = integer(next(), 1, 128);
        else if (a == "--csv-out")
            o.csv = next();
        else if (a == "--profile")
            o.profile = true;
        else
            throw std::invalid_argument("unknown argument: " + std::string(a));
    }
    if (o.execution != "eager" && o.execution != "graph" && o.execution != "both")
        throw std::invalid_argument("invalid execution");
    if (o.cache != "cold" && o.cache != "warm" && o.cache != "both")
        throw std::invalid_argument("invalid cache state");
    if (o.geometry35 && !o.parent)
        throw std::invalid_argument("35b requires the parent weight form");
    if (o.graph_calls > 1 && (o.execution != "graph" || o.cache != "warm"))
        throw std::invalid_argument("repeated graphs require --execution graph --cache warm");
    if (o.profile && (o.tokens.size() != 1 || o.execution == "both" || o.cache == "both"))
        throw std::invalid_argument(
            "profile requires one token extent, execution, and cache state");
    return o;
}

DeviceBuffer bf16_values(std::size_t count, std::uint32_t seed, float extent) {
    std::vector<std::uint16_t> values(count);
    for (auto& v : values) {
        seed = 1664525u * seed + 1013904223u;
        v    = f32_to_bf16(extent * (2.0f * float(seed >> 8) / 16777216.0f - 1.0f));
    }
    DeviceBuffer out(count * 2);
    out.copy_from_host(values.data(), out.bytes);
    return out;
}

DeviceBuffer fp32_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    for (auto& v : values) {
        seed = 1664525u * seed + 1013904223u;
        v    = low + (high - low) * float(seed >> 8) / 16777216.0f;
    }
    DeviceBuffer out(count * 4);
    out.copy_from_host(values.data(), out.bytes);
    return out;
}

Weight weight(const void* data, int rows, int hidden) {
    Weight w{};
    w.qtype   = QType::BF16_CTRL;
    w.layout  = QuantLayout::Contiguous;
    w.payload = w.qdata = data;
    w.payload_bytes     = std::uint64_t(rows) * hidden * 2;
    w.ndim              = 2;
    w.shape[0] = w.padded_shape[0] = w.n = rows;
    w.shape[1] = w.padded_shape[1] = w.k = hidden;
    return w;
}

void run(const Options& o, int t, DeviceExecutionView execution, DeviceBuffer& flush,
         std::ostream* csv) {
    const int heads = o.geometry35 ? 32 : 48, hidden = o.geometry35 ? 2048 : 5120;
    auto x    = bf16_values(std::size_t(hidden) * t, 13u, 1.0f);
    auto nw   = bf16_values(hidden, 29u, 0.2f);
    auto w    = bf16_values(std::size_t(2 * heads) * hidden, 47u, 0.015f);
    auto alog = fp32_values(heads, 59u, -2.0f, 1.0f);
    auto bias = fp32_values(heads, 61u, -1.0f, 1.0f);
    DeviceBuffer h(std::size_t(hidden) * t * 2), g(std::size_t(heads) * t * 4),
        beta(std::size_t(heads) * t * 4);
    Tensor tx(x.p, DType::BF16, {hidden, t}), tn(nw.p, DType::BF16, {hidden}),
        th(h.p, DType::BF16, {hidden, t});
    Tensor ta(alog.p, DType::FP32, {heads}), td(bias.p, DType::FP32, {heads}),
        tg(g.p, DType::FP32, {heads, t}), tb(beta.p, DType::FP32, {heads, t});
    const auto wp = weight(w.p, 2 * heads, hidden), wa = weight(w.p, heads, hidden),
               wb = weight(static_cast<const std::uint16_t*>(w.p) + std::size_t(heads) * hidden,
                           heads, hidden);
    const auto capacity =
        o.norm ? ops::gdn_norm_gating_proj_workspace_capacity_bytes(heads, hidden, t, t)
               : ops::gdn_gating_proj_workspace_capacity_bytes(heads, hidden, t, t);
    WorkspaceArena ws(std::max<std::size_t>(capacity, 256));
    auto launch = [&](cudaStream_t stream) {
        const DeviceExecutionView e{stream, execution.multiprocessor_count};
        if (o.norm) {
            if (o.parent)
                ops::gdn_norm_gating_proj(tx, tn, 1.0e-6f, wp, ta, td, ws, th, tg, tb, e);
            else
                ops::gdn_norm_gating_proj(tx, tn, 1.0e-6f, wa, wb, ta, td, ws, th, tg, tb, e);
        } else {
            if (o.parent)
                ops::gdn_gating_proj(tx, wp, ta, td, ws, tg, tb, e);
            else
                ops::gdn_gating_proj(tx, wa, wb, ta, td, ws, tg, tb, e);
        }
    };
    TimedGraph graph;
    if (o.execution != "eager") {
        launch(execution.stream);
        CUDA_CHECK(cudaStreamSynchronize(execution.stream));
        graph.capture(execution.stream, [&](cudaStream_t s) {
            for (int i = 0; i < o.graph_calls; ++i) launch(s);
        });
    }
    for (std::string_view mode : {"eager", "graph"}) {
        if (o.execution != "both" && o.execution != mode) continue;
        for (std::string_view cache : {"cold", "warm"}) {
            if (o.cache != "both" && o.cache != cache) continue;
            const bool gr = mode == "graph", cold = cache == "cold";
            if (o.profile) {
                for (int i = 0; i < o.warmup; ++i) {
                    if (gr)
                        graph.launch(execution.stream);
                    else
                        launch(execution.stream);
                }
                if (cold) flush_l2(flush, execution.stream);
                CUDA_CHECK(cudaStreamSynchronize(execution.stream));
                std::cout << "PROFILE public gdn " << (o.norm ? "norm-control" : "control")
                          << " T=" << t << " graph_calls=" << o.graph_calls << std::endl;
                CUDA_CHECK(cudaProfilerStart());
                if (gr)
                    graph.launch(execution.stream);
                else
                    launch(execution.stream);
                CUDA_CHECK(cudaStreamSynchronize(execution.stream));
                CUDA_CHECK(cudaProfilerStop());
                continue;
            }
            ColdTiming time =
                gr ? (cold ? measure_cold_graph(graph, flush, execution.stream, o.warmup, o.repeat)
                           : measure_graph(graph, execution.stream, o.warmup, o.repeat))
                   : (cold
                          ? measure_cold_launch(launch, flush, execution.stream, o.warmup, o.repeat)
                          : measure_launch(launch, execution.stream, o.warmup, o.repeat));
            const int calls = gr ? o.graph_calls : 1;
            time.median_us /= calls;
            time.min_us /= calls;
            time.p95_us /= calls;
            if (ws.used() != 0 || ws.peak_used() != capacity)
                throw std::runtime_error("workspace query/peak mismatch");
            std::cout << (o.geometry35 ? "35b" : "27b") << ' ' << (o.norm ? "norm" : "control")
                      << " T=" << t << ' ' << mode << ' ' << cache << " median=" << time.median_us
                      << " min=" << time.min_us << " p95=" << time.p95_us
                      << " us workspace=" << capacity << " nodes=" << (gr ? graph.nodes() : 0)
                      << " calls=" << calls << '\n';
            if (csv)
                *csv << (o.geometry35 ? "35b" : "27b") << ',' << (o.norm ? "norm" : "control")
                     << ',' << (o.parent ? "parent" : "split") << ',' << t << ',' << mode << ','
                     << cache << ',' << calls << ',' << (gr ? graph.nodes() : 0) << ',' << capacity
                     << ',' << ws.peak_used() << ',' << time.median_us << ',' << time.min_us << ','
                     << time.p95_us << '\n';
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        int count       = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(256ULL << 20);
        std::ofstream csv;
        if (!o.csv.empty()) {
            csv.open(o.csv);
            if (!csv) throw std::runtime_error("cannot open CSV");
            csv << "geometry,op,weights,T,execution,cache,graph_calls,graph_nodes,workspace_bytes,"
                   "workspace_peak_bytes,median_us,min_us,p95_us\n";
        }
        for (int t : o.tokens)
            run(o, t, {stream, props.multiProcessorCount}, flush, csv.is_open() ? &csv : nullptr);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
