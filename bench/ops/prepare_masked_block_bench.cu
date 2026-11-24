// Public-Op benchmark for the exact anchor-and-mask block transform.

#include "common/roofline.h"
#include "ninfer/ops/sequence.h"
#include "core/buffer.h"

#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kMaskId    = 248077;
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;

enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    std::vector<std::int32_t> block_sizes{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Execution execution = Execution::Graph;
    CacheMode cache     = CacheMode::Cold;
    int warmup          = 20;
    int repeat          = 101;
    bool profile        = false;
    std::string csv_out;
};

struct Result {
    std::int32_t block_size;
    Execution execution;
    CacheState cache;
    double useful_bytes;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_prepare_masked_block_bench [--block-sizes 2,...,16] "
                 "[--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::int32_t> parse_list(const char* text, const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_i32(item, 2, 16, flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--block-sizes") {
            options.block_sizes =
                parse_list(next("--block-sizes requires a value"), "--block-sizes");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else if (value == "both")
                options.execution = Execution::Both;
            else
                usage("--execution expects eager, graph, or both");
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else if (value == "both")
                options.cache = CacheMode::Both;
            else
                usage("--cache expects cold, warm, or both");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out requires a path");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    if (options.profile &&
        (options.block_sizes.size() != 1 || options.execution == Execution::Both ||
         options.cache == CacheMode::Both)) {
        usage("--profile requires one block size, one execution, and one cache state");
    }
    return options;
}

class Case {
public:
    explicit Case(std::int32_t block_size)
        : block_size_(block_size),
          anchor_(ninfer::bench::device_context().device(), sizeof(std::int32_t)),
          length_(ninfer::bench::device_context().device(), sizeof(std::int32_t)),
          valid_(ninfer::bench::device_context().device(), sizeof(std::int32_t)),
          ids_(ninfer::bench::device_context().device(),
               static_cast<std::size_t>(block_size) * sizeof(std::int32_t)),
          positions_(ninfer::bench::device_context().device(),
                     static_cast<std::size_t>(block_size) * sizeof(std::int32_t)),
          anchor_tensor_(anchor_.span(), DType::I32, {1}),
          length_tensor_(length_.span(), DType::I32, {1}),
          valid_tensor_(valid_.span(), DType::I32, {1}),
          ids_tensor_(ids_.span(), DType::I32, {block_size, 1}),
          positions_tensor_(positions_.span(), DType::I32, {block_size, 1}) {
        const std::int32_t anchor = 42;
        const std::int32_t length = 4096;
        const std::int32_t valid  = block_size;
        CUDA_CHECK(cudaMemcpy(anchor_.data(), &anchor, sizeof(anchor), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(length_.data(), &length, sizeof(length), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(valid_.data(), &valid, sizeof(valid), cudaMemcpyHostToDevice));
    }

    void launch(CudaExecutionView cuda) {
        ops::prepare_masked_block(anchor_tensor_, length_tensor_, valid_tensor_, kMaskId,
                                  ids_tensor_, positions_tensor_, cuda);
    }

private:
    std::int32_t block_size_;
    DeviceBuffer anchor_;
    DeviceBuffer length_;
    DeviceBuffer valid_;
    DeviceBuffer ids_;
    DeviceBuffer positions_;
    Tensor anchor_tensor_;
    Tensor length_tensor_;
    Tensor valid_tensor_;
    Tensor ids_tensor_;
    Tensor positions_tensor_;
};

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

bench::ColdTiming measure(Case& data, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, bench::L2FlushBuffer& flush,
                          CudaExecutionView cuda, int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](CudaExecutionView cuda) { data.launch(cuda); };
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, cuda, warmup, repeat)
                   : bench::measure_launch(launch, cuda, warmup, repeat);
    }
    return cache == CacheState::Cold
               ? bench::measure_cold_graph(*graph, flush, cuda, warmup, repeat)
               : bench::measure_graph(*graph, cuda, warmup, repeat);
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = result.useful_bytes / seconds / 1.0e9;
    std::printf("entry=prepare_masked_block execution=%-5s cache=%-4s B=%2d "
                "median=%7.3f us min=%7.3f us p95=%7.3f us useful=%7.4f GB/s "
                "(%7.4f%% of %.0f)\n",
                execution_name(result.execution), cache_name(result.cache), result.block_size,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us, gbps,
                gbps / bench::kRoofline.memory_gbs * 100.0, bench::kRoofline.memory_gbs);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,execution,cache,block_size,useful_bytes,median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << "prepare_masked_block," << execution_name(result.execution) << ','
               << cache_name(result.cache) << ',' << result.block_size << ',' << result.useful_bytes
               << ',' << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << '\n';
    }
}

void profile(Case& data, const Options& options, bench::L2FlushBuffer& flush,
             CudaExecutionView cuda) {
    const auto stream         = cuda.stream();
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(cuda, [&](CudaExecutionView cuda) { data.launch(cuda); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(cuda); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(cuda); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=prepare_masked_block dispatch=public execution=%s cache=%s\n",
                execution_name(execution), cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(cuda);
    else
        data.launch(cuda);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
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
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        bench::L2FlushBuffer flush(kFlushBytes);

        if (options.profile) {
            Case data(options.block_sizes.front());
            profile(data, options, flush, cuda);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const std::int32_t block_size : options.block_sizes) {
            Case data(block_size);
            bench::TimedGraph graph;
            if (options.execution != Execution::Eager) {
                data.launch(cuda);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                graph.capture(cuda, [&](CudaExecutionView cuda) { data.launch(cuda); });
            }
            for (const Execution execution : {Execution::Eager, Execution::Graph}) {
                if ((options.execution == Execution::Eager && execution != Execution::Eager) ||
                    (options.execution == Execution::Graph && execution != Execution::Graph)) {
                    continue;
                }
                for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
                    if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                        (options.cache == CacheMode::Warm && cache != CacheState::Warm)) {
                        continue;
                    }
                    Result result{block_size, execution, cache,
                                  static_cast<double>(2 + 2 * block_size) * sizeof(std::int32_t),
                                  measure(data, execution, cache, &graph, flush, cuda,
                                          options.warmup, options.repeat)};
                    report(result);
                    results.push_back(result);
                }
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_prepare_masked_block_bench: %s\n", error.what());
        return 1;
    }
}
