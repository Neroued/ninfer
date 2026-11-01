// Measures the public hyper-connection mix Ops at the Qwen3.8-Flash-Next geometry (4 x 2560,
// low rank 320) with Q8_G32_FP16 Down(+inject)/Up weights, as CUDA Graph replays.
#include "ninfer/ops/hyper_connection.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kStreams = 4, kHidden = 2560, kWide = kStreams * kHidden, kRank = 320;

struct Options {
    int warmup = 20, repeat = 200, chain = 8;
    bool separate = false;
    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 16, 64};
    std::string form = "combine_mix", cache = "cold", format = "q8", csv;
};

std::vector<int> list(const std::string& text) {
    std::vector<int> out;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        out.push_back(std::stoi(text.substr(start, end - start)));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--separate") {
            o.separate = true;
            continue;
        }
        if (arg == "--help") {
            std::puts("usage: ninfer_hyper_connection_bench [--form mix|combine_mix|head] "
                      "[--format q8|nvfp4] [--tokens T,...] [--cache cold|warm] [--chain N] "
                      "[--warmup N] [--repeat N] [--separate] [--csv-out PATH]\n"
                      "One graph replays N calls over N distinct weight sets (N x 7 MB exceeds L2 "
                      "for N >= 8); times are per call. --separate puts an empty kernel between "
                      "calls (no programmatic overlap of one call with the next) and reports the "
                      "time per call minus that of the empty kernels alone.");
            std::exit(0);
        }
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (arg == "--form")
            o.form = value;
        else if (arg == "--tokens")
            o.tokens = list(value);
        else if (arg == "--cache")
            o.cache = value;
        else if (arg == "--format")
            o.format = value;
        else if (arg == "--warmup")
            o.warmup = std::stoi(value);
        else if (arg == "--repeat")
            o.repeat = std::stoi(value);
        else if (arg == "--chain")
            o.chain = std::stoi(value);
        else if (arg == "--csv-out")
            o.csv = value;
        else
            throw std::invalid_argument("unknown option: " + arg);
    }
    if ((o.form != "mix" && o.form != "combine_mix" && o.form != "head") ||
        (o.cache != "cold" && o.cache != "warm") || (o.format != "q8" && o.format != "nvfp4"))
        throw std::invalid_argument("invalid form or cache");
    return o;
}

__global__ void separator_kernel() {}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        int devices     = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) return 77;
        DeviceContext device;
        bench::L2FlushBuffer flush(std::size_t{256} << 20);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU=%s SMs=%d CUDART=%d form=%s format=%s cache=%s separate=%d\n", prop.name,
                    prop.multiProcessorCount, CUDART_VERSION, o.form.c_str(), o.format.c_str(),
                    o.cache.c_str(), o.separate ? 1 : 0);
        const bool inject = o.form != "head";
        const int rows    = kRank + (inject ? kStreams : 0);
        std::vector<bench::PackedQuantizedWeight> downs, ups;
        std::vector<DeviceBuffer> norms;
        std::vector<ops::HyperConnectionWeights> weights;
        const bool fp4 = o.format == "nvfp4";
        for (int c = 0; c < o.chain; ++c) {
            if (fp4) { // NVFP4 Down is stored padded to whole 128-row tiles
                downs.push_back(bench::make_nvfp4_weight((rows + 127) / 128 * 128, kWide,
                                                         0x71U + 16 * c, 0.04F));
                ups.push_back(bench::make_nvfp4_weight(kWide, kRank, 0x72U + 16 * c, 0.25F));
            } else {
                downs.push_back(bench::make_row_split_weight(QType::Q8_G32_FP16, rows, kWide, kWide,
                                                             0x71U + 16 * c, 0.04F));
                ups.push_back(bench::make_row_split_weight(QType::Q8_G32_FP16, kWide, kRank, 384,
                                                           0x72U + 16 * c, 0.25F));
            }
            norms.push_back(bench::make_bf16(kWide, 0x73U + 16 * c, -0.1F, 0.1F));
        }
        for (int c = 0; c < o.chain; ++c)
            weights.push_back(
                {Tensor(norms[c].p, DType::BF16, {kWide}), downs[c].weight, ups[c].weight});
        // Bytes the small-T route reads per call: Down rows (codes + scales), the touched K of Up
        // (Q8: 320 of 384 codes, whole 24-byte scale rows), the norm.
        const double weight_bytes = fp4 ? double(rows) * (kWide / 2 + kWide / 16) +
                                              double(kWide) * (kRank / 2 + kRank / 16) + kWide * 2.0
                                        : double(rows) * (kWide + kWide / 32 * 2) +
                                              double(kWide) * (kRank + 24) + kWide * 2.0;
        std::ofstream csv;
        if (!o.csv.empty()) {
            csv.open(o.csv);
            csv << "form,format,T,cache,median_us,min_us,p95_us,touched_MB,touched_GBps\n";
        }
        for (const int t : o.tokens) {
            DeviceBuffer residual = bench::make_bf16(std::size_t(kWide) * t, 0x74U, -0.75F, 0.75F);
            DeviceBuffer y = bench::make_bf16(std::size_t(kHidden) * t, 0x75U, -0.01F, 0.01F);
            DeviceBuffer previous = bench::make_f32(std::size_t(kStreams) * t, 0x76U, 0.0F, 0.0F);
            DeviceBuffer mixed(std::size_t(kHidden) * t * 2), gates(std::size_t(kStreams) * t * 4);
            Tensor tr(residual.p, DType::BF16, {kWide, t});
            Tensor ty(y.p, DType::BF16, {kHidden, t});
            Tensor tp(previous.p, DType::FP32, {kStreams, t});
            Tensor tm(mixed.p, DType::BF16, {kHidden, t});
            Tensor ti(gates.p, DType::FP32, {kStreams, t});
            WorkspaceArena workspace(
                ops::hyper_connection_mix_workspace_bytes(t, t, inject, kHidden, kRank));
            // combine_mix commits y with a zero gate so repeated replays keep the residual fixed.
            const auto launch = [&](cudaStream_t stream) {
                for (const auto& w : weights) {
                    if (o.separate) separator_kernel<<<1, 32, 0, stream>>>();
                    if (o.form == "combine_mix")
                        ops::hyper_connection_combine_mix(tr, ty, tp, w, 1e-6F, tm, &ti, workspace,
                                                          stream);
                    else
                        ops::hyper_connection_mix(tr, w, 1e-6F, tm, inject ? &ti : nullptr,
                                                  workspace, stream);
                }
            };
            launch(device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            bench::TimedGraph graph;
            graph.capture(device.stream, launch);
            bench::ColdTiming time =
                o.cache == "cold"
                    ? bench::measure_cold_graph(graph, flush, device.stream, o.warmup, o.repeat)
                    : bench::measure_graph(graph, device.stream, o.warmup, o.repeat);
            if (o.separate) { // subtract the empty kernels' own time
                bench::TimedGraph empty;
                empty.capture(device.stream, [&](cudaStream_t stream) {
                    for (int c = 0; c < o.chain; ++c) separator_kernel<<<1, 32, 0, stream>>>();
                });
                const bench::ColdTiming alone =
                    bench::measure_graph(empty, device.stream, o.warmup, o.repeat);
                time.median_us -= alone.median_us;
                time.min_us -= alone.median_us;
                time.p95_us -= alone.median_us;
            }
            time.median_us /= o.chain;
            time.min_us /= o.chain;
            time.p95_us /= o.chain;
            const double gbps = weight_bytes / (time.median_us * 1e3);
            std::printf("%s T=%d cache=%s chain=%d nodes=%zu median=%.2f min=%.2f p95=%.2f us "
                        "touched=%.2f MB -> %.0f GB/s\n",
                        o.form.c_str(), t, o.cache.c_str(), o.chain, graph.nodes(), time.median_us,
                        time.min_us, time.p95_us, weight_bytes / 1e6, gbps);
            if (csv)
                csv << o.form << ',' << o.format << ',' << t << ',' << o.cache << ','
                    << time.median_us << ',' << time.min_us << ',' << time.p95_us << ','
                    << weight_bytes / 1e6 << ',' << gbps << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hyper_connection_bench: %s\n", error.what());
        return 1;
    }
}
