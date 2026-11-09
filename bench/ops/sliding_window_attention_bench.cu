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

constexpr std::int32_t kHeadDim   = 128;
constexpr float kScale            = 0.08838834764831844055F;
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;

enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 12, 16};
    std::vector<std::int32_t> contexts{0, 128, 2048, 4095, 4096, 8192, 262144};
    std::vector<std::int32_t> batches{1};
    std::int32_t kv_heads = 8;
    std::string valid = "full", lengths = "uniform";
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
    std::int32_t kv_heads;
    std::string valid, lengths;
    int live_columns;
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
                 "[--kv-heads 4|8] [--window 2048|4096] [--tokens 1,...,16] [--batches 1,...,8] "
                 "[--context 0,...,1048576] [--envelope-max N] [--valid full|half|mixed|zero] "
                 "[--lengths uniform|mixed] "
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
        } else if (argument == "--kv-heads") {
            options.kv_heads =
                bench::measurement::integer(next("--kv-heads requires a value"), 4, 8);
            if (options.kv_heads != 4 && options.kv_heads != 8) usage("--kv-heads expects 4 or 8");
        } else if (argument == "--valid") {
            options.valid = next("--valid requires a value");
            if (options.valid != "full" && options.valid != "half" && options.valid != "mixed" &&
                options.valid != "zero")
                usage("valid mode");
        } else if (argument == "--lengths") {
            options.lengths = next("--lengths requires a value");
            if (options.lengths != "uniform" && options.lengths != "mixed") usage("lengths mode");
        } else if (argument == "--window") {
            options.window =
                bench::measurement::integer(next("--window requires a value"), 2048, 4096);
            if (options.window != 2048 && options.window != 4096) {
                usage("--window expects 2048 or 4096");
            }
        } else if (argument == "--context") {
            options.contexts = bench::measurement::integers(next("--context requires a value"), 0,
                                                            std::numeric_limits<int>::max() - 16);
        } else if (argument == "--envelope-max") {
            options.envelope_max = bench::measurement::integer(
                next("--envelope-max requires a value"), 0, std::numeric_limits<int>::max() - 16);
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
                                         const Tensor& offsets, int kv_heads) {
    const std::size_t bytes = std::size_t(kHeadDim) * window * kv_heads * 2;
    return {
        .k               = {{k.span(), offsets},
                            {{{0, bytes, 16}, DType::BF16, {kHeadDim, window, kv_heads, 1}}, 1, bytes}},
        .v               = {{v.span(), offsets},
                            {{{0, bytes, 16}, DType::FP16, {kHeadDim, window, kv_heads, 1}}, 1, bytes}},
        .capacity        = static_cast<std::uint32_t>(window),
        .padded_capacity = static_cast<std::uint32_t>(window),
        .num_kv_heads    = kv_heads,
        .head_dim        = kHeadDim,
    };
}

std::size_t workspace_capacity(std::int32_t window, std::int32_t tokens, std::int32_t batch,
                               std::int32_t maximum, ops::AttentionHeadGeometry geometry) {
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{
        0, static_cast<std::uint32_t>(maximum)};
    return ops::sliding_window_attention_workspace_capacity_bytes(
        geometry, static_cast<std::uint32_t>(window), envelope, tokens, tokens, batch);
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
         std::int32_t maximum, const Options& options)
        : geometry_{128, options.kv_heads * 4, options.kv_heads}, window_(window), tokens_(tokens),
          batch_(batch), context_(context),
          q_(bench::make_bf16(
              static_cast<std::size_t>(kHeadDim) * geometry_.query_heads * tokens * batch, 101U)),
          query_k_(bench::make_bf16(
              static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads * tokens * batch, 103U)),
          query_v_(bench::make_bf16(
              static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads * tokens * batch, 105U)),
          positions_(ninfer::bench::device_context().device(),
                     static_cast<std::size_t>(tokens) * batch * sizeof(std::int32_t)),
          valid_(ninfer::bench::device_context().device(),
                 static_cast<std::size_t>(batch) * sizeof(std::int32_t)),
          lane_(ninfer::bench::device_context().device(),
                static_cast<std::size_t>(batch) * sizeof(std::int64_t)),
          context_k_(bench::make_bf16(
              static_cast<std::size_t>(kHeadDim) * window * geometry_.kv_heads * batch, 107U)),
          context_v_(make_context_values(static_cast<std::size_t>(kHeadDim) * window *
                                         geometry_.kv_heads * batch)),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * geometry_.query_heads *
                                    tokens * batch * 2)),
          workspace_bytes_(workspace_capacity(window, tokens, batch, maximum, geometry_)),
          workspace_storage_(bench::make_buffer(std::max<std::size_t>(workspace_bytes_, 1))),
          workspace_(workspace_storage_.span()),
          q_tensor_(q_.span(), DType::BF16, {kHeadDim, geometry_.query_heads, tokens, batch}),
          query_k_tensor_(query_k_.span(), DType::BF16,
                          {kHeadDim, geometry_.kv_heads, tokens, batch}),
          query_v_tensor_(query_v_.span(), DType::BF16,
                          {kHeadDim, geometry_.kv_heads, tokens, batch}),
          positions_tensor_(positions_.span(), DType::I32, {tokens, batch}),
          valid_tensor_(valid_.span(), DType::I32, {batch}),
          lane_tensor_(lane_.span(), DType::I64, {batch}),
          output_tensor_(output_.span(), DType::BF16,
                         {kHeadDim, geometry_.query_heads, tokens, batch}),
          context_view_(
              make_context_view(context_k_, context_v_, window, lane_tensor_, geometry_.kv_heads)),
          envelope_{0, static_cast<std::uint32_t>(maximum)} {
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens) * batch);
        valids_.resize(batch);
        lengths_.resize(batch);
        std::vector<std::int64_t> host_lanes(static_cast<std::size_t>(batch));
        for (std::int32_t row = 0; row < batch; ++row) {
            valids_[row]  = options.valid == "zero"    ? 0
                            : options.valid == "half"  ? (tokens + 1) / 2
                            : options.valid == "mixed" ? (row * 7 + tokens) % (tokens + 1)
                                                       : tokens;
            lengths_[row] = options.lengths == "uniform" || row % 3 == 0 ? context
                            : row % 3 == 1                               ? std::max(0, context - 63)
                                                                         : std::min(context, 128);
            live_columns_ += valids_[row];
            host_lanes[static_cast<std::size_t>(row)] =
                valids_[row] ? std::int64_t(row) * kHeadDim * window * geometry_.kv_heads * 2 : -1;
            for (std::int32_t token = 0; token < tokens; ++token) {
                host_positions[static_cast<std::size_t>(row * tokens + token)] =
                    token < valids_[row] ? lengths_[row] + token : -1234567;
            }
        }
        CUDA_CHECK(cudaMemcpy(positions_.data(), host_positions.data(), positions_.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(valid_.data(), valids_.data(), valid_.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(lane_.data(), host_lanes.data(), lane_.size(), cudaMemcpyHostToDevice));
    }

    void launch(CudaExecutionView cuda) {
        ops::sliding_window_attention(q_tensor_, query_k_tensor_, query_v_tensor_,
                                      positions_tensor_, valid_tensor_, geometry_,
                                      static_cast<std::uint32_t>(window_), kScale, context_view_,
                                      envelope_, workspace_, output_tensor_, cuda);
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

    int live_columns() const { return live_columns_; }

    double useful_bytes() const {
        double bytes = 2. * 128 * geometry_.query_heads * tokens_ * batch_ + 4. * batch_;
        for (int b = 0; b < batch_; ++b)
            if (valids_[b]) {
                bytes += 4. * 128 * geometry_.kv_heads * std::min(lengths_[b], window_ - 1) +
                         (2. * 128 * geometry_.query_heads + 4. * 128 * geometry_.kv_heads + 4.) *
                             valids_[b] +
                         8;
            }
        return bytes;
    }

    double useful_flops() const {
        std::int64_t keys = 0;
        for (int b = 0; b < batch_; ++b)
            for (int t = 0; t < valids_[b]; ++t)
                keys += std::min(lengths_[b], std::max(0, window_ - 1 - t)) + valids_[b];
        return 4. * 128 * geometry_.query_heads * keys;
    }
private:
    ops::AttentionHeadGeometry geometry_;
    std::vector<int> lengths_, valids_;
    int live_columns_ = 0;
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
    std::printf("entry=sliding_window execution=%-5s cache=%-4s W=%d Hkv=%d T=%2d B=%d L=%6d "
                "max_L=%d graph_nodes=%zu "
                "graph_calls=%d workspace=%8zu "
                "median=%9.3f us min=%9.3f us p95=%9.3f us "
                "useful=%8.1f GB/s ",
                execution_name(result.execution), cache_name(result.cache, result.graph_calls),
                result.window, result.kv_heads, result.tokens, result.batch, result.context,
                result.envelope_max, result.graph_nodes, result.graph_calls, result.workspace_bytes,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us, gbps);
    if (result.cache == CacheState::Cold && result.graph_calls == 1) {
        std::printf("logical/DRAM-roof=%5.1f%% of %.0f GB/s ",
                    gbps / bench::kRoofline.memory_gbs * 100.0, bench::kRoofline.memory_gbs);
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
    output << "entry,execution,cache,window,kv_heads,valid,lengths,live_columns,T,B,context,"
              "envelope_max,"
              "graph_nodes,graph_calls,"
              "workspace_bytes,useful_bytes,useful_flops,median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << "sliding_window," << execution_name(result.execution) << ','
               << cache_name(result.cache, result.graph_calls) << ',' << result.window << ','
               << result.kv_heads << ',' << result.valid << ',' << result.lengths << ','
               << result.live_columns << ',' << result.tokens << ',' << result.batch << ','
               << result.context << ',' << result.envelope_max << ',' << result.graph_nodes << ','
               << result.graph_calls << ',' << result.workspace_bytes << ',' << result.useful_bytes
               << ',' << result.useful_flops << ',' << result.timing.median_us << ','
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
                      options.envelope_max < 0 ? options.contexts.front() : options.envelope_max,
                      options);
            profile(data, options, flush, cuda);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const std::int32_t context : options.contexts) {
            for (const std::int32_t tokens : options.tokens) {
                for (const std::int32_t batch : options.batches) {
                    const int maximum = options.envelope_max < 0 ? context : options.envelope_max;
                    Case data(options.window, tokens, batch, context, maximum, options);
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
                                options.kv_heads,
                                options.valid,
                                options.lengths,
                                data.live_columns(),
                                tokens,
                                batch,
                                context,
                                maximum,
                                execution == Execution::Graph ? graph.nodes() : 0,
                                options.graph_calls,
                                execution,
                                cache,
                                data.workspace_bytes(),
                                data.useful_bytes(),
                                data.useful_flops(),
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
