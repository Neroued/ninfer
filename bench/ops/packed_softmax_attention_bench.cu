// Public-Op benchmark for plain/uniform and packed dense Softmax Attention.
// Tile selection and launch decomposition remain private to packed_softmax_attention().

#include "ninfer/ops/softmax_attention.h"

#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim = 72;
constexpr std::int32_t kHeads   = 16;
constexpr float kScale          = 0.11785113019775792073F;
constexpr ops::AttentionHeadGeometry kGeometry{kHeadDim, kHeads, kHeads};
constexpr std::size_t kFlushBytes   = std::size_t{256} << 20;
constexpr double kDenseBf16TcTflops = 209.5;
constexpr double kRtx5090DramGBs    = 1792.0;

enum class Entry : std::uint8_t { Uniform, Packed, Both };
enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    Entry entry           = Entry::Both;
    Execution execution   = Execution::Graph;
    CacheMode cache       = CacheMode::Cold;
    std::int32_t segments = 1;
    std::int32_t length   = 256;
    std::vector<std::int32_t> segment_lengths;
    bool explicit_lengths = false;
    int warmup            = 3;
    int repeat            = 30;
    bool profile          = false;
    std::string csv_out;
};

struct Result {
    Entry entry;
    Execution execution;
    CacheState cache;
    std::int32_t tokens;
    std::int32_t segments;
    std::size_t workspace_bytes;
    double useful_bytes;
    double useful_flops;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_packed_softmax_attention_bench "
                 "[--entry uniform|packed|both] [--segments S --length L | "
                 "--segment-lengths L1,L2,...] "
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
        result.push_back(parse_i32(item, 1, std::numeric_limits<std::int32_t>::max(), flag));
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
        if (argument == "--entry") {
            const std::string_view value(next("--entry requires a value"));
            if (value == "uniform")
                options.entry = Entry::Uniform;
            else if (value == "packed")
                options.entry = Entry::Packed;
            else if (value == "both")
                options.entry = Entry::Both;
            else
                usage("--entry expects uniform, packed, or both");
        } else if (argument == "--segments") {
            options.segments = parse_i32(next("--segments requires a value"), 1,
                                         std::numeric_limits<std::int32_t>::max(), "--segments");
        } else if (argument == "--length") {
            options.length = parse_i32(next("--length requires a value"), 1,
                                       std::numeric_limits<std::int32_t>::max(), "--length");
        } else if (argument == "--segment-lengths") {
            options.segment_lengths =
                parse_list(next("--segment-lengths requires a value"), "--segment-lengths");
            options.explicit_lengths = true;
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

    if (!options.explicit_lengths) {
        const std::int64_t tokens = static_cast<std::int64_t>(options.segments) * options.length;
        if (tokens > std::numeric_limits<std::int32_t>::max()) {
            usage("segments * length exceeds int32");
        }
        options.segment_lengths.assign(static_cast<std::size_t>(options.segments), options.length);
    }
    const std::int64_t total = std::accumulate(options.segment_lengths.begin(),
                                               options.segment_lengths.end(), std::int64_t{0});
    if (total > std::numeric_limits<std::int32_t>::max()) {
        usage("sum of segment lengths exceeds int32");
    }
    const bool uniform =
        std::adjacent_find(options.segment_lengths.begin(), options.segment_lengths.end(),
                           std::not_equal_to<>()) == options.segment_lengths.end();
    if (!uniform && options.entry != Entry::Packed) {
        usage("nonuniform segment lengths require --entry packed");
    }
    if (options.profile && (options.entry == Entry::Both || options.execution == Execution::Both ||
                            options.cache == CacheMode::Both)) {
        usage("--profile requires one entry, one execution, and one cache state");
    }
    return options;
}

class Case {
public:
    explicit Case(std::vector<std::int32_t> segment_lengths)
        : segment_lengths_(std::move(segment_lengths)),
          tokens_(static_cast<std::int32_t>(
              std::accumulate(segment_lengths_.begin(), segment_lengths_.end(), std::int64_t{0}))),
          q_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kHeads * tokens_)),
          k_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kHeads * tokens_)),
          v_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kHeads * tokens_)),
          cu_seqlens_((segment_lengths_.size() + 1) * sizeof(std::int32_t)),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * kHeads * tokens_ * 2)),
          workspace_bytes_(ops::packed_softmax_attention_workspace_capacity_bytes(
              kGeometry, tokens_, tokens_, static_cast<std::int32_t>(segment_lengths_.size()),
              static_cast<std::int32_t>(segment_lengths_.size()))),
          workspace_(std::max<std::size_t>(workspace_bytes_, 1)),
          q_tensor_(q_.p, DType::BF16, {kHeadDim, kHeads, tokens_}),
          k_tensor_(k_.p, DType::BF16, {kHeadDim, kHeads, tokens_}),
          v_tensor_(v_.p, DType::BF16, {kHeadDim, kHeads, tokens_}),
          cu_tensor_(cu_seqlens_.p, DType::I32,
                     {static_cast<std::int32_t>(segment_lengths_.size() + 1)}),
          output_tensor_(output_.p, DType::BF16, {kHeadDim, kHeads, tokens_}) {
        std::vector<std::int32_t> cumulative(segment_lengths_.size() + 1, 0);
        for (std::size_t index = 0; index < segment_lengths_.size(); ++index) {
            cumulative[index + 1] = cumulative[index] + segment_lengths_[index];
        }
        CUDA_CHECK(cudaMemcpy(cu_seqlens_.p, cumulative.data(), cu_seqlens_.bytes,
                              cudaMemcpyHostToDevice));
    }

    void launch(Entry entry, cudaStream_t stream) {
        if (entry == Entry::Uniform) {
            if (segment_lengths_.size() == 1) {
                ops::softmax_attention(q_tensor_, k_tensor_, v_tensor_, kGeometry, kScale,
                                       workspace_, output_tensor_, stream);
            } else {
                ops::packed_softmax_attention(q_tensor_, k_tensor_, v_tensor_, kGeometry, kScale,
                                              segment_lengths_.front(), output_tensor_, stream);
            }
        } else {
            ops::packed_softmax_attention(q_tensor_, k_tensor_, v_tensor_, kGeometry, kScale,
                                          cu_tensor_, workspace_, output_tensor_, stream);
        }
    }

    [[nodiscard]] std::int32_t tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::int32_t segments() const noexcept {
        return static_cast<std::int32_t>(segment_lengths_.size());
    }

    [[nodiscard]] std::size_t workspace_bytes(Entry entry) const noexcept {
        return entry == Entry::Packed ? workspace_bytes_ : 0;
    }

    [[nodiscard]] double useful_flops() const noexcept {
        double squared_lengths = 0.0;
        for (const std::int32_t length : segment_lengths_) {
            squared_lengths += static_cast<double>(length) * length;
        }
        return 4.0 * kHeadDim * kHeads * squared_lengths;
    }

    [[nodiscard]] double useful_bytes() const noexcept {
        return 8.0 * static_cast<double>(kHeadDim) * kHeads * tokens_;
    }

private:
    std::vector<std::int32_t> segment_lengths_;
    std::int32_t tokens_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer cu_seqlens_;
    DeviceBuffer output_;
    std::size_t workspace_bytes_;
    WorkspaceArena workspace_;
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor cu_tensor_;
    Tensor output_tensor_;
};

const char* entry_name(Entry entry) { return entry == Entry::Uniform ? "uniform" : "packed"; }

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

bench::ColdTiming measure(Case& data, Entry entry, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, DeviceBuffer& flush, cudaStream_t stream,
                          int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); };
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, stream, warmup, repeat)
                   : bench::measure_launch(launch, stream, warmup, repeat);
    }
    return cache == CacheState::Cold
               ? bench::measure_cold_graph(*graph, flush, stream, warmup, repeat)
               : bench::measure_graph(*graph, stream, warmup, repeat);
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = result.useful_bytes / seconds / 1.0e9;
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=%-7s execution=%-5s cache=%-4s P=%6d S=%4d workspace=%8zu "
                "median=%9.3f us min=%9.3f us p95=%9.3f us useful=%8.1f GB/s "
                "(%5.1f%% of %.0f) math=%7.2f TFLOP/s (%5.1f%% of %.1f)\n",
                entry_name(result.entry), execution_name(result.execution),
                cache_name(result.cache), result.tokens, result.segments, result.workspace_bytes,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us, gbps,
                gbps / kRtx5090DramGBs * 100.0, kRtx5090DramGBs, tflops,
                tflops / kDenseBf16TcTflops * 100.0, kDenseBf16TcTflops);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,execution,cache,tokens,segments,workspace_bytes,useful_bytes,useful_flops,"
              "median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << entry_name(result.entry) << ',' << execution_name(result.execution) << ','
               << cache_name(result.cache) << ',' << result.tokens << ',' << result.segments << ','
               << result.workspace_bytes << ',' << result.useful_bytes << ',' << result.useful_flops
               << ',' << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << '\n';
    }
}

void profile(Case& data, Entry entry, const Options& options, DeviceBuffer& flush,
             cudaStream_t stream) {
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(entry, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream,
                      [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(stream); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(entry, stream); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=%s dispatch=public execution=%s cache=%s\n", entry_name(entry),
                execution_name(execution), cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(stream);
    else
        data.launch(entry, stream);
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
        DeviceBuffer flush(kFlushBytes);
        Case data(options.segment_lengths);

        if (options.profile) {
            profile(data, options.entry, options, flush, stream);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const Entry entry : {Entry::Uniform, Entry::Packed}) {
            if ((options.entry == Entry::Uniform && entry != Entry::Uniform) ||
                (options.entry == Entry::Packed && entry != Entry::Packed)) {
                continue;
            }
            bench::TimedGraph graph;
            if (options.execution != Execution::Eager) {
                data.launch(entry, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                graph.capture(
                    stream, [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); });
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
                    Result result{entry,
                                  execution,
                                  cache,
                                  data.tokens(),
                                  data.segments(),
                                  data.workspace_bytes(entry),
                                  data.useful_bytes(),
                                  data.useful_flops(),
                                  measure(data, entry, execution, cache, &graph, flush, stream,
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
        std::fprintf(stderr, "ninfer_packed_softmax_attention_bench: %s\n", error.what());
        return 1;
    }
}
