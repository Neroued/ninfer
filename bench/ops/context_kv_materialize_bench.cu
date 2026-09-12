// Cold-cache benchmark for the complete DFlash2 five-layer context state transition.

#include "ninfer/ops/context_kv_materialize.h"

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kLayers       = static_cast<int>(ops::kContextKVMaterializeLayers);
constexpr int kParentRows   = 6144;
constexpr int kKeyRow       = 4096;
constexpr int kValueRow     = 5120;
constexpr int kHidden       = 5120;
constexpr int kRows         = 1024;
constexpr int kHeadDim      = 128;
constexpr int kHeads        = 8;
constexpr int kCapacity     = 2048;
constexpr int kLaneCapacity = 8;

enum class Execution : std::uint8_t { Eager, Graph };

struct Options {
    std::vector<int> widths{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::string counts = "full";
    std::vector<int> batches{1, 2, 3, 4, 5, 6, 7, 8};
    Execution execution     = Execution::Graph;
    int warmup              = 10;
    int repeat              = 100;
    std::size_t flush_bytes = 256ULL << 20;
};

[[noreturn]] void usage(const char* program, const char* error) {
    std::fprintf(stderr,
                 "error: %s\nusage: %s [--widths W,...] [--batches B,...] [--counts "
                 "full|one|ragged|zero] [--execution eager|graph] "
                 "[--warmup N] [--repeat N] [--flush-mib N]\n",
                 error, program);
    std::exit(2);
}

int parse_i32(std::string_view text, int minimum, int maximum, const char* flag) {
    const std::string value(text);
    char* end         = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::invalid_argument(std::string("invalid ") + flag);
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_list(std::string_view text, int maximum) {
    std::vector<int> result;
    while (!text.empty()) {
        const std::size_t comma     = text.find(',');
        const std::string_view item = text.substr(0, comma);
        result.push_back(parse_i32(item, 1, maximum, "list"));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.empty()) throw std::invalid_argument("empty --batches");
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&]() -> std::string_view {
            if (++index >= argc) usage(argv[0], "missing option value");
            return argv[index];
        };
        if (argument == "--batches") {
            options.batches = parse_list(next(), 8);
        } else if (argument == "--widths") {
            options.widths = parse_list(next(), 2048);
        } else if (argument == "--counts") {
            options.counts = next();
            if (options.counts != "full" && options.counts != "one" && options.counts != "ragged" &&
                options.counts != "zero")
                usage(argv[0], "invalid --counts");
        } else if (argument == "--execution") {
            const std::string_view value = next();
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else
                usage(argv[0], "--execution expects eager or graph");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next(), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next(), 1, 10000, "--repeat");
        } else if (argument == "--flush-mib") {
            const int mib       = parse_i32(next(), 1, 4096, "--flush-mib");
            options.flush_bytes = static_cast<std::size_t>(mib) << 20;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0], "help");
        } else {
            usage(argv[0], "unknown option");
        }
    }
    return options;
}

struct Fixture {
    std::array<bench::PackedQuantizedWeight, kLayers> parents;
    std::array<DeviceBuffer, kLayers> norms;
    std::array<DeviceBuffer, kLayers> cache_k;
    std::array<DeviceBuffer, kLayers> cache_v;
    std::array<ops::ContextKVMaterializeLayerView, kLayers> layers;
    DeviceBuffer context = bench::make_bf16(static_cast<std::size_t>(kHidden) * 2048);
    DeviceBuffer positions;
    DeviceBuffer counts;
    DeviceBuffer slots;
    DeviceBuffer direct_workspace;
    WorkspaceArena direct_arena;
    ops::ContextKVMaterializeExecutionEnvelope envelope{};

    Fixture()
        : positions(sizeof(std::int32_t) * 2048), counts(sizeof(std::int32_t) * 8),
          slots(sizeof(std::int32_t) * 8),
          direct_workspace(ops::context_kv_materialize_workspace_capacity_bytes(1, 1, 2048)),
          direct_arena(DeviceSpan{direct_workspace.p, direct_workspace.bytes}) {
        const std::size_t cache_bytes = static_cast<std::size_t>(kHeadDim) * kCapacity * kHeads *
                                        kLaneCapacity * sizeof(std::uint16_t);
        for (int layer = 0; layer < kLayers; ++layer) {
            parents[static_cast<std::size_t>(layer)] =
                bench::make_row_split_weight(QType::W8G32_F16S, kParentRows, kHidden, kHidden,
                                             {static_cast<std::uint8_t>(0x31 + layer), 0, 0x2800});
            norms[static_cast<std::size_t>(layer)]   = bench::make_bf16(kHeadDim);
            cache_k[static_cast<std::size_t>(layer)] = DeviceBuffer(cache_bytes);
            cache_v[static_cast<std::size_t>(layer)] = DeviceBuffer(cache_bytes);
            layers[static_cast<std::size_t>(layer)]  = {
                bench::row_view(parents[static_cast<std::size_t>(layer)].weight, kKeyRow, kRows),
                bench::row_view(parents[static_cast<std::size_t>(layer)].weight, kValueRow, kRows),
                Tensor(norms[static_cast<std::size_t>(layer)].p, DType::BF16, {kHeadDim}),
                CyclicKVCacheLayerView{
                     .k        = Tensor(cache_k[static_cast<std::size_t>(layer)].p, DType::BF16,
                                        {kHeadDim, kCapacity, kHeads, kLaneCapacity}),
                     .v        = Tensor(cache_v[static_cast<std::size_t>(layer)].p, DType::FP16,
                                        {kHeadDim, kCapacity, kHeads, kLaneCapacity}),
                     .capacity = kCapacity,
                     .padded_capacity = kCapacity,
                     .num_kv_heads    = kHeads,
                     .head_dim        = kHeadDim,
                     .lane_capacity   = kLaneCapacity,
                },
            };
        }
    }

    void prepare(int width, int batch, const std::string& mode) {
        direct_arena.reset_peak();
        std::vector<int> host_positions(width * batch), host_counts(batch), host_slots(batch);
        for (int b = 0; b < batch; ++b) {
            host_counts[b] = mode == "zero"     ? 0
                             : mode == "one"    ? 1
                             : mode == "ragged" ? (b * 7 + width) % (width + 1)
                                                : width;
            host_slots[b]  = 7 - b;
            for (int i = 0; i < width; ++i) host_positions[b * width + i] = 262140 + 8192 * b + i;
        }
        envelope = {
            static_cast<std::uint32_t>(*std::min_element(host_counts.begin(), host_counts.end())),
            static_cast<std::uint32_t>(*std::max_element(host_counts.begin(), host_counts.end()))};
        positions.copy_from_host(host_positions.data(), host_positions.size() * 4);
        counts.copy_from_host(host_counts.data(), host_counts.size() * 4);
        slots.copy_from_host(host_slots.data(), host_slots.size() * 4);
    }

    void launch(int width, int batch, cudaStream_t stream) {
        Tensor x(context.p, DType::BF16, {kHidden, width, batch});
        Tensor pos(positions.p, DType::I32, {width, batch});
        Tensor count(counts.p, DType::I32, {batch});
        Tensor lane(slots.p, DType::I32, {batch});
        ops::context_kv_materialize(x, pos, count, lane, layers, envelope, direct_arena, stream);
    }
};

template <class Launch>
bench::ColdTiming measure(const Options& options, Launch&& launch, DeviceBuffer& flush,
                          cudaStream_t stream, std::size_t* graph_nodes) {
    if (options.execution == Execution::Eager) {
        *graph_nodes = 0;
        return bench::measure_cold_launch(launch, flush, stream, options.warmup, options.repeat);
    }
    bench::TimedGraph graph;
    graph.capture(stream, launch);
    *graph_nodes = graph.nodes();
    return bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        Fixture fixture;
        DeviceBuffer flush(options.flush_bytes);
        std::printf("# gpu=%s public=context_kv_materialize geometry=L5_K5120_N1024 cache=cold "
                    "flush_mib=%zu execution=%s\n",
                    properties.name, options.flush_bytes >> 20,
                    options.execution == Execution::Graph ? "graph" : "eager");
        std::printf("width,batch,columns,counts,min_count,max_count,median_us,min_us,p95_us,nodes,"
                    "workspace_bytes,workspace_peak_bytes\n");
        for (const int width : options.widths)
            for (const int batch : options.batches) {
                if (width > 16 && batch != 1) continue;
                fixture.prepare(width, batch, options.counts);
                std::size_t nodes = 0;
                if (fixture.envelope.max_count == 0) {
                    // No GPU interval exists to time. Verify the public no-op produces no
                    // capture nodes or scratch and report latency as not applicable.
                    cudaGraph_t empty = nullptr;
                    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                    fixture.launch(width, batch, stream);
                    CUDA_CHECK(cudaStreamEndCapture(stream, &empty));
                    CUDA_CHECK(cudaGraphGetNodes(empty, nullptr, &nodes));
                    CUDA_CHECK(cudaGraphDestroy(empty));
                    if (nodes != 0 || fixture.direct_arena.peak_used() != 0)
                        throw std::runtime_error("zero-count materialization is not empty");
                    std::printf(
                        "%d,%d,%d,%s,0,0,nan,nan,nan,0,%zu,0\n", width, batch, width * batch,
                        options.counts.c_str(),
                        ops::context_kv_materialize_workspace_capacity_bytes(batch, width, width));
                    continue;
                }
                const auto timing = measure(
                    options, [&](cudaStream_t s) { fixture.launch(width, batch, s); }, flush,
                    stream, &nodes);
                std::printf(
                    "%d,%d,%d,%s,%u,%u,%.3f,%.3f,%.3f,%zu,%zu,%zu\n", width, batch, width * batch,
                    options.counts.c_str(), fixture.envelope.min_count, fixture.envelope.max_count,
                    timing.median_us, timing.min_us, timing.p95_us, nodes,
                    ops::context_kv_materialize_workspace_capacity_bytes(batch, width, width),
                    fixture.direct_arena.peak_used());
            }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_context_kv_materialize_bench: %s\n", error.what());
        return 1;
    }
}
