// Public warm-cache Graph benchmark for the fused tail-pack RMSNorm contract.
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer_bench_common.h"
#include <cstdlib>
#include <stdexcept>
#include <string>
using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    int width = 0, batch = 0, warmup = 8, repeat = 60;
};

Options parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") {
            std::printf("usage: %s [--width W] [--batch B] [--warmup N] [--repeat N]\n", argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing argument");
        const int value = std::atoi(argv[++i]);
        if (flag == "--width")
            out.width = value;
        else if (flag == "--batch")
            out.batch = value;
        else if (flag == "--warmup")
            out.warmup = value;
        else if (flag == "--repeat")
            out.repeat = value;
        else
            throw std::invalid_argument("unknown flag: " + flag);
    }
    if ((out.width && (out.width < 2 || out.width > 16)) || out.batch < 0 || out.batch > 8 ||
        out.warmup < 0 || out.repeat < 1)
        throw std::invalid_argument("invalid width/batch/timing domain");
    return out;
}

void run(int width, int batch, const Options& options, cudaStream_t stream) {
    const int columns  = (width - 1) * batch;
    DeviceBuffer input = make_bf16(5120ULL * width * batch), weight = make_bf16(5120),
                 output = make_zeros(5120ULL * columns * 2);
    Tensor x(input.p, DType::BF16, {5120, width, batch}), w(weight.p, DType::BF16, {5120}),
        y(output.p, DType::BF16, {5120, columns});
    TimedGraph graph;
    constexpr int repeats = 32;
    graph.capture(stream, [&](cudaStream_t s) {
        for (int i = 0; i < repeats; ++i) { ops::rmsnorm_pack_tail(x, w, y, s); }
    });
    const auto timing = measure_graph(graph, stream, options.warmup, options.repeat);
    std::printf("%d,%d,%d,%d,%.3f,%.3f,%.3f,%zu\n", width, batch, columns, 512,
                timing.median_us / repeats, timing.min_us / repeats, timing.p95_us / repeats,
                graph.nodes());
}
} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::puts("SKIP: no usable CUDA device");
        return 0;
    }
    try {
        const auto options  = parse(argc, argv);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::puts("W,B,U,threads,median_us,min_us,p95_us,graph_repetitions");
        for (int width = 2; width <= 16; ++width)
            for (int batch = 1; batch <= 8; ++batch) {
                if ((options.width && width != options.width) ||
                    (options.batch && batch != options.batch))
                    continue;
                run(width, batch, options, stream);
            }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rmsnorm_pack_tail: %s\n", e.what());
        return 1;
    }
}
