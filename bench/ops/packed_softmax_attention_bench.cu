// Public-Op benchmark for uniform and packed dense Softmax Attention.
// Tile selection and launch decomposition remain private to packed_softmax_attention().

#include "common/roofline.h"
#include "ninfer/ops/softmax_attention.h"
#include "core/buffer.h"

#include "core/device.h"
#include "op_measurement.cuh"
#include "common/gpu_activity.cuh"

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

constexpr std::int32_t kHeads     = 16;
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;

enum class Entry : std::uint8_t { Uniform, Packed, Plain, Both };
enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    int head_dim          = 72;
    Entry entry           = Entry::Both;
    Execution execution   = Execution::Graph;
    CacheMode cache       = CacheMode::Cold;
    std::int32_t segments = 1;
    std::int32_t length   = 256;
    std::vector<std::int32_t> segment_lengths;
    bool explicit_lengths = false;
    bool interleaved_qkv  = true;
    int calls             = 1;
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
    std::fprintf(
        stderr,
        "error: %s\n"
        "usage: ninfer_packed_softmax_attention_bench "
        "[--dim 64|72] [--entry plain|uniform|packed|both] [--segments S --length L | "
        "--segment-lengths L1,L2,...] "
        "[--layout qkv|contiguous] [--execution eager|graph|both] [--cache cold|warm|both] "
        "[--warmup N] [--repeat N] [--calls N] [--profile] [--csv-out PATH]\n",
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
        if (argument == "--dim") {
            options.head_dim = parse_i32(next("--dim requires 64 or 72"), 64, 72, "--dim");
            if (options.head_dim != 64 && options.head_dim != 72) usage("--dim requires 64 or 72");
        } else if (argument == "--entry") {
            const std::string_view value(next("--entry requires a value"));
            if (value == "uniform")
                options.entry = Entry::Uniform;
            else if (value == "packed")
                options.entry = Entry::Packed;
            else if (value == "both")
                options.entry = Entry::Both;
            else if (value == "plain")
                options.entry = Entry::Plain;
            else
                usage("--entry expects plain, uniform, packed, or both");
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
        } else if (argument == "--layout") {
            const std::string_view value(next("--layout requires a value"));
            if (value != "qkv" && value != "contiguous")
                usage("--layout expects qkv or contiguous");
            options.interleaved_qkv = value == "qkv";
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
        } else if (argument == "--calls") {
            options.calls = parse_i32(next("--calls requires a value"), 1, 10000, "--calls");
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

    if (options.calls > 1 && options.cache != CacheMode::Warm)
        usage("--calls > 1 requires --cache warm");
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
    if (options.entry == Entry::Plain && options.segment_lengths.size() != 1)
        usage("plain requires one segment");
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
    Case(std::vector<std::int32_t> segment_lengths, bool interleaved_qkv, int head_dim)
        : geometry_{head_dim, kHeads, kHeads}, scale_(1.F / std::sqrt(float(head_dim))),
          segment_lengths_(std::move(segment_lengths)),
          tokens_(static_cast<std::int32_t>(
              std::accumulate(segment_lengths_.begin(), segment_lengths_.end(), std::int64_t{0}))),
          qkv_(bench::make_bf16(static_cast<std::size_t>(3 * geometry_.head_dim) * kHeads * tokens_,
                                101U)),
          cu_seqlens_(ninfer::bench::device_context().device(),
                      (segment_lengths_.size() + 1) * sizeof(std::int32_t)),
          output_(bench::make_zeros(static_cast<std::size_t>(geometry_.head_dim) * kHeads *
                                    tokens_ * 2)),
          workspace_bytes_(ops::packed_softmax_attention_workspace_capacity_bytes(
              geometry_, tokens_, tokens_, static_cast<std::int32_t>(segment_lengths_.size()),
              static_cast<std::int32_t>(segment_lengths_.size()))),
          workspace_storage_(bench::make_buffer(std::max<std::size_t>(workspace_bytes_, 1))),
          workspace_(workspace_storage_.span()),
          q_tensor_(qkv_.span(), DType::BF16, {geometry_.head_dim, kHeads, tokens_}),
          k_tensor_(qkv_.span(), DType::BF16, {geometry_.head_dim, kHeads, tokens_}),
          v_tensor_(qkv_.span(), DType::BF16, {geometry_.head_dim, kHeads, tokens_}),
          cu_tensor_(cu_seqlens_.span(), DType::I32,
                     {static_cast<std::int32_t>(segment_lengths_.size() + 1)}),
          output_tensor_(output_.span(), DType::BF16, {geometry_.head_dim, kHeads, tokens_}) {
        const std::size_t plane = geometry_.head_dim * kHeads * sizeof(std::uint16_t);
        const auto offset       = interleaved_qkv ? plane : plane * tokens_;
        k_tensor_.data          = static_cast<std::byte*>(qkv_.data()) + offset;
        v_tensor_.data          = static_cast<std::byte*>(qkv_.data()) + 2 * offset;
        if (interleaved_qkv) {
            for (auto* tensor : {&q_tensor_, &k_tensor_, &v_tensor_}) {
                tensor->nb[2] = 3 * plane;
                tensor->nb[3] = 3 * plane * tokens_;
            }
        }
        std::vector<std::int32_t> cumulative(segment_lengths_.size() + 1, 0);
        for (std::size_t index = 0; index < segment_lengths_.size(); ++index) {
            cumulative[index + 1] = cumulative[index] + segment_lengths_[index];
        }
        CUDA_CHECK(cudaMemcpy(cu_seqlens_.data(), cumulative.data(), cu_seqlens_.size(),
                              cudaMemcpyHostToDevice));
    }

    void launch(Entry entry, CudaExecutionView cuda) {
        if (entry == Entry::Uniform) {
            ops::packed_softmax_attention(q_tensor_, k_tensor_, v_tensor_, geometry_, scale_,
                                          segment_lengths_.front(), output_tensor_, cuda);
        } else if (entry == Entry::Plain) {
            ops::softmax_attention(q_tensor_, k_tensor_, v_tensor_, geometry_, scale_, workspace_,
                                   output_tensor_, cuda);
        } else {
            ops::packed_softmax_attention(q_tensor_, k_tensor_, v_tensor_, geometry_, scale_,
                                          cu_tensor_, workspace_, output_tensor_, cuda);
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
            // A one-key segment is the identity on V and needs no contractions.
            if (length > 1) squared_lengths += static_cast<double>(length) * length;
        }
        return 4.0 * geometry_.head_dim * kHeads * squared_lengths;
    }

    [[nodiscard]] double useful_bytes() const noexcept {
        double bytes = 0;
        for (const std::int32_t length : segment_lengths_)
            bytes += (length == 1 ? 4.0 : 8.0) * geometry_.head_dim * kHeads * length;
        return bytes;
    }

private:
    ops::AttentionHeadGeometry geometry_;
    float scale_;
    std::vector<std::int32_t> segment_lengths_;
    std::int32_t tokens_;
    DeviceBuffer qkv_;
    DeviceBuffer cu_seqlens_;
    DeviceBuffer output_;
    std::size_t workspace_bytes_;
    DeviceBuffer workspace_storage_;
    MemoryArena workspace_;
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor cu_tensor_;
    Tensor output_tensor_;
};

const char* entry_name(Entry entry) {
    return entry == Entry::Uniform ? "uniform" : entry == Entry::Plain ? "plain" : "packed";
}

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

bench::ColdTiming measure(Case& data, Entry entry, Execution execution, CacheState cache,
                          bench::L2FlushBuffer& flush, CudaExecutionView cuda,
                          const Options& options) {
    bench::measurement::Options timing;
    timing.warmup = options.warmup;
    timing.repeat = options.repeat;
    auto result   = bench::measurement::measure(
                      [&](auto e) {
                          for (int i = 0; i < options.calls; ++i) data.launch(entry, e);
                      },
                      [](auto e) { bench::condition_gpu(e, 200); }, flush, cuda, timing,
                      execution_name(execution), cache_name(cache))
                      .gpu;
    result.median_us /= options.calls;
    result.min_us /= options.calls;
    result.p95_us /= options.calls;
    return result;
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = result.useful_bytes / seconds / 1.0e9;
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=%-7s execution=%-5s cache=%-4s P=%6d S=%4d workspace=%8zu "
                "median=%9.3f us min=%9.3f us p95=%9.3f us useful=%8.1f GB/s "
                "(%5.1f%% of %.0f) math=%7.2f TFLOP/s (%5s%% of %s)\n",
                entry_name(result.entry), execution_name(result.execution),
                cache_name(result.cache), result.tokens, result.segments, result.workspace_bytes,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us, gbps,
                gbps / bench::kRoofline.memory_gbs * 100.0, bench::kRoofline.memory_gbs, tflops,
                bench::format_metric(tflops / bench::kRoofline.bf16_f32_tflops * 100.0, 1).c_str(),
                bench::format_metric(bench::kRoofline.bf16_f32_tflops, 1).c_str());
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output
        << "entry,execution,cache,tokens,segments,workspace_bytes,useful_bytes,useful_flops,"
           "median_us,min_us,p95_us,layout,memory_reference_us,math_reference_us,head_dim,calls\n";
    for (const Result& result : results) {
        output << entry_name(result.entry) << ',' << execution_name(result.execution) << ','
               << cache_name(result.cache) << ',' << result.tokens << ',' << result.segments << ','
               << result.workspace_bytes << ',' << result.useful_bytes << ',' << result.useful_flops
               << ',' << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << ',' << (options.interleaved_qkv ? "qkv" : "contiguous")
               << ',' << result.useful_bytes / (bench::kRoofline.memory_gbs * 1000.) << ','
               << result.useful_flops / (bench::kRoofline.bf16_f32_tflops * 1.e6) << ','
               << options.head_dim << ',' << options.calls << '\n';
    }
}

void profile(Case& data, Entry entry, const Options& options, bench::L2FlushBuffer& flush,
             CudaExecutionView cuda) {
    const auto stream         = cuda.stream();
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(entry, cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(cuda, [&](CudaExecutionView cuda) { data.launch(entry, cuda); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(cuda); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(entry, cuda); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=%s dispatch=public execution=%s cache=%s\n", entry_name(entry),
                execution_name(execution), cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(cuda);
    else
        data.launch(entry, cuda);
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
        Case data(options.segment_lengths, options.interleaved_qkv, options.head_dim);

        if (options.profile) {
            profile(data, options.entry, options, flush, cuda);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const Entry entry : {Entry::Uniform, Entry::Packed, Entry::Plain}) {
            if ((options.entry != Entry::Both && options.entry != entry) ||
                (options.entry == Entry::Both && entry == Entry::Plain)) {
                continue;
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
                                  measure(data, entry, execution, cache, flush, cuda, options)};
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
