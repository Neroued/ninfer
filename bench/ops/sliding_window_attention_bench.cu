// Public-Op benchmark for symmetric sliding-window Softmax Attention.
// Each measured eager launch or graph-bundle entry calls public sliding_window_attention().

#include "common/roofline.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "core/buffer.h"

#include "core/device.h"
#include "core/cyclic_kv_cache.h"
#include "op_measurement.cuh"

#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kQueryHeads = 32;
constexpr std::int32_t kKvHeads    = 8;
constexpr float kScale             = 0.08838834764831844055F;
constexpr ops::AttentionHeadGeometry kGeometry{kHeadDim, kQueryHeads, kKvHeads};
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;

enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 12, 16};
    std::vector<std::int32_t> contexts{0, 128, 2048, 4095, 4096, 8192, 262144};
    std::vector<std::int32_t> batches{1};
    std::int32_t window       = 4096;
    std::int32_t envelope_max = -1;
    Execution execution       = Execution::Graph;
    CacheMode cache           = CacheMode::Cold;
    int warmup                = 5;
    int repeat                = 50;
    int graph_calls           = 1;
    bool profile              = false;
    std::string csv_out;
};

struct Result {
    std::int32_t window;
    std::int32_t tokens;
    std::int32_t batch;
    std::int32_t context;
    std::int32_t envelope_max;
    std::size_t graph_nodes;
    int graph_calls;
    Execution execution;
    CacheState cache;
    std::size_t workspace_bytes;
    double useful_bytes;
    double useful_flops;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_sliding_window_attention_bench "
                 "[--window 2048|4096] [--tokens 1,...,16] [--batches 1,...,8] "
                 "[--context 0,...,262144] [--envelope-max N] "
                 "[--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--warmup N] [--repeat N] [--graph-calls N] [--profile] [--csv-out PATH]\n",
                 message);
    std::exit(2);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--tokens") {
            options.tokens = bench::measurement::integers(next("--tokens requires a value"), 1, 16);
        } else if (argument == "--batches") {
            options.batches =
                bench::measurement::integers(next("--batches requires a value"), 1, 8);
        } else if (argument == "--window") {
            options.window =
                bench::measurement::integer(next("--window requires a value"), 2048, 4096);
            if (options.window != 2048 && options.window != 4096) {
                usage("--window expects 2048 or 4096");
            }
        } else if (argument == "--context") {
            options.contexts =
                bench::measurement::integers(next("--context requires a value"), 0, 262144);
        } else if (argument == "--envelope-max") {
            options.envelope_max =
                bench::measurement::integer(next("--envelope-max requires a value"), 0, 262144);
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
            options.warmup =
                bench::measurement::integer(next("--warmup requires a value"), 0, 10000);
        } else if (argument == "--repeat") {
            options.repeat =
                bench::measurement::integer(next("--repeat requires a value"), 1, 10000);
        } else if (argument == "--graph-calls") {
            options.graph_calls =
                bench::measurement::integer(next("--graph-calls requires a value"), 1, 64);
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
    if (options.profile && (options.tokens.size() != 1 || options.contexts.size() != 1 ||
                            options.batches.size() != 1 || options.execution == Execution::Both ||
                            options.cache == CacheMode::Both)) {
        usage("--profile requires one T, one B, one context, one execution, and one cache state");
    }
    if (options.graph_calls > 1 && (options.execution != Execution::Graph || options.profile))
        usage("graph bundles require --execution graph and no --profile");
    for (const int context : options.contexts) {
        if (options.envelope_max >= 0 && options.envelope_max < context)
            usage("envelope max must cover every context");
    }
    return options;
}

CyclicKVCacheLayerView make_context_view(DeviceBuffer& k, DeviceBuffer& v, std::int32_t window,
                                         std::int32_t batch) {
    return {
        .k               = Tensor(k.span(), DType::BF16, {kHeadDim, window, kKvHeads, batch}),
        .v               = Tensor(v.span(), DType::FP16, {kHeadDim, window, kKvHeads, batch}),
        .capacity        = static_cast<std::uint32_t>(window),
        .padded_capacity = static_cast<std::uint32_t>(window),
        .num_kv_heads    = kKvHeads,
        .head_dim        = kHeadDim,
        .slot_count      = batch,
    };
}

std::size_t workspace_capacity(std::int32_t window, std::int32_t tokens, std::int32_t batch,
                               std::int32_t maximum) {
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{
        0, static_cast<std::uint32_t>(maximum)};
    return ops::sliding_window_attention_workspace_capacity_bytes(
        kGeometry, static_cast<std::uint32_t>(window), envelope, tokens, tokens, batch);
}

DeviceBuffer make_context_values(std::size_t elements) {
    DeviceBuffer result(ninfer::bench::device_context().device(), elements * sizeof(__half));
    CUDA_CHECK(bench::fixture::fill_values(static_cast<__half*>(result.data()), elements, 311U,
                                           -1.F, 1.F));
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

class Case {
public:
    Case(std::int32_t window, std::int32_t tokens, std::int32_t batch, std::int32_t context,
         std::int32_t maximum)
        : window_(window), tokens_(tokens), batch_(batch), context_(context),
          q_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens * batch,
                              101U)),
          query_k_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * tokens * batch,
                                    103U)),
          query_v_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * tokens * batch,
                                    105U)),
          positions_(ninfer::bench::device_context().device(),
                     static_cast<std::size_t>(tokens) * batch * sizeof(std::int32_t)),
          valid_(ninfer::bench::device_context().device(),
                 static_cast<std::size_t>(batch) * sizeof(std::int32_t)),
          lane_(ninfer::bench::device_context().device(),
                static_cast<std::size_t>(batch) * sizeof(std::int32_t)),
          context_k_(bench::make_bf16(
              static_cast<std::size_t>(kHeadDim) * window * kKvHeads * batch, 107U)),
          context_v_(
              make_context_values(static_cast<std::size_t>(kHeadDim) * window * kKvHeads * batch)),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens *
                                    batch * 2)),
          workspace_bytes_(workspace_capacity(window, tokens, batch, maximum)),
          workspace_storage_(bench::make_buffer(std::max<std::size_t>(workspace_bytes_, 1))),
          workspace_(workspace_storage_.span()),
          q_tensor_(q_.span(), DType::BF16, {kHeadDim, kQueryHeads, tokens, batch}),
          query_k_tensor_(query_k_.span(), DType::BF16, {kHeadDim, kKvHeads, tokens, batch}),
          query_v_tensor_(query_v_.span(), DType::BF16, {kHeadDim, kKvHeads, tokens, batch}),
          positions_tensor_(positions_.span(), DType::I32, {tokens, batch}),
          valid_tensor_(valid_.span(), DType::I32, {batch}),
          lane_tensor_(lane_.span(), DType::I32, {batch}),
          output_tensor_(output_.span(), DType::BF16, {kHeadDim, kQueryHeads, tokens, batch}),
          context_view_(make_context_view(context_k_, context_v_, window, batch)),
          envelope_{0, static_cast<std::uint32_t>(maximum)} {
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens) * batch);
        std::vector<std::int32_t> host_valid(static_cast<std::size_t>(batch), tokens);
        std::vector<std::int32_t> host_lanes(static_cast<std::size_t>(batch));
        for (std::int32_t row = 0; row < batch; ++row) {
            host_lanes[static_cast<std::size_t>(row)] = row;
            for (std::int32_t token = 0; token < tokens; ++token) {
                host_positions[static_cast<std::size_t>(row * tokens + token)] = context + token;
            }
        }
        CUDA_CHECK(cudaMemcpy(positions_.data(), host_positions.data(), positions_.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(valid_.data(), host_valid.data(), valid_.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(lane_.data(), host_lanes.data(), lane_.size(), cudaMemcpyHostToDevice));
    }

    void launch(CudaExecutionView cuda) {
        ops::sliding_window_attention(q_tensor_, query_k_tensor_, query_v_tensor_,
                                      positions_tensor_, valid_tensor_, lane_tensor_, kGeometry,
                                      static_cast<std::uint32_t>(window_), kScale, context_view_,
                                      envelope_, workspace_, output_tensor_, cuda);
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

private:
    std::int32_t window_;
    std::int32_t tokens_;
    std::int32_t batch_;
    std::int32_t context_;
    DeviceBuffer q_;
    DeviceBuffer query_k_;
    DeviceBuffer query_v_;
    DeviceBuffer positions_;
    DeviceBuffer valid_;
    DeviceBuffer lane_;
    DeviceBuffer context_k_;
    DeviceBuffer context_v_;
    DeviceBuffer output_;
    std::size_t workspace_bytes_;
    DeviceBuffer workspace_storage_;
    MemoryArena workspace_;
    Tensor q_tensor_;
    Tensor query_k_tensor_;
    Tensor query_v_tensor_;
    Tensor positions_tensor_;
    Tensor valid_tensor_;
    Tensor lane_tensor_;
    Tensor output_tensor_;
    CyclicKVCacheLayerView context_view_;
    ops::SlidingWindowAttentionExecutionEnvelope envelope_;
};

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache, int graph_calls = 1) {
    if (cache == CacheState::Warm) return "warm";
    return graph_calls == 1 ? "cold" : "cold-before-graph-bundle";
}

double useful_bytes(std::int32_t window, std::int32_t tokens, std::int32_t batch,
                    std::int32_t context) {
    const std::int32_t context_union = std::min(context, window - 1);
    return static_cast<double>(batch) *
           (static_cast<double>(context_union) * 4096.0 + static_cast<double>(tokens) * 20480.0);
}

double useful_flops(std::int32_t window, std::int32_t tokens, std::int32_t batch,
                    std::int32_t context) {
    std::int64_t visible_keys = 0;
    for (std::int32_t token = 0; token < tokens; ++token) {
        visible_keys += std::min(context, std::max(0, window - 1 - token)) + tokens;
    }
    return static_cast<double>(batch) * 4.0 * kQueryHeads * kHeadDim *
           static_cast<double>(visible_keys);
}

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
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=sliding_window execution=%-5s cache=%-4s W=%d T=%2d B=%d L=%6d "
                "max_L=%d graph_nodes=%zu "
                "graph_calls=%d workspace=%8zu "
                "median=%9.3f us min=%9.3f us p95=%9.3f us "
                "useful=%8.1f GB/s ",
                execution_name(result.execution), cache_name(result.cache, result.graph_calls),
                result.window, result.tokens, result.batch, result.context, result.envelope_max,
                result.graph_nodes, result.graph_calls, result.workspace_bytes,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us, gbps);
    if (result.cache == CacheState::Cold && result.graph_calls == 1) {
        std::printf("DRAM=%5.1f%% of %.0f GB/s ", gbps / bench::kRoofline.memory_gbs * 100.0,
                    bench::kRoofline.memory_gbs);
    } else {
        std::printf("DRAM=n/a ");
    }
    std::printf("math=%7.2f TFLOP/s (%5s%% of %s)\n", tflops,
                bench::format_metric(tflops / bench::kRoofline.bf16_f32_tflops * 100.0, 1).c_str(),
                bench::format_metric(bench::kRoofline.bf16_f32_tflops, 1).c_str());
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,execution,cache,window,T,B,context,envelope_max,"
              "graph_nodes,graph_calls,"
              "workspace_bytes,useful_bytes,useful_flops,median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << "sliding_window," << execution_name(result.execution) << ','
               << cache_name(result.cache, result.graph_calls) << ',' << result.window << ','
               << result.tokens << ',' << result.batch << ',' << result.context << ','
               << result.envelope_max << ',' << result.graph_nodes << ',' << result.graph_calls
               << ',' << result.workspace_bytes << ',' << result.useful_bytes << ','
               << result.useful_flops << ',' << result.timing.median_us << ','
               << result.timing.min_us << ',' << result.timing.p95_us << '\n';
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

    std::printf("PROFILE entry=sliding_window dispatch=public W=%d execution=%s cache=%s\n",
                options.window, execution_name(execution), cache_name(cache));
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
            Case data(options.window, options.tokens.front(), options.batches.front(),
                      options.contexts.front(),
                      options.envelope_max < 0 ? options.contexts.front() : options.envelope_max);
            profile(data, options, flush, cuda);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const std::int32_t context : options.contexts) {
            for (const std::int32_t tokens : options.tokens) {
                for (const std::int32_t batch : options.batches) {
                    const int maximum = options.envelope_max < 0 ? context : options.envelope_max;
                    Case data(options.window, tokens, batch, context, maximum);
                    bench::TimedGraph graph;
                    if (options.execution != Execution::Eager) {
                        data.launch(cuda);
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        graph.capture(cuda, [&](CudaExecutionView cuda) {
                            for (int call = 0; call < options.graph_calls; ++call)
                                data.launch(cuda);
                        });
                    }
                    for (const Execution execution : {Execution::Eager, Execution::Graph}) {
                        if ((options.execution == Execution::Eager &&
                             execution != Execution::Eager) ||
                            (options.execution == Execution::Graph &&
                             execution != Execution::Graph)) {
                            continue;
                        }
                        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
                            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                                (options.cache == CacheMode::Warm && cache != CacheState::Warm)) {
                                continue;
                            }
                            Result result{
                                options.window,
                                tokens,
                                batch,
                                context,
                                maximum,
                                execution == Execution::Graph ? graph.nodes() : 0,
                                options.graph_calls,
                                execution,
                                cache,
                                data.workspace_bytes(),
                                useful_bytes(options.window, tokens, batch, context),
                                useful_flops(options.window, tokens, batch, context),
                                measure(data, execution, cache, &graph, flush, cuda, options.warmup,
                                        options.repeat),
                            };
                            result.timing.median_us /= options.graph_calls;
                            result.timing.min_us /= options.graph_calls;
                            result.timing.p95_us /= options.graph_calls;
                            report(result);
                            results.push_back(result);
                        }
                    }
                }
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_sliding_window_attention_bench: %s\n", error.what());
        return 1;
    }
}
