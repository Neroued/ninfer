#include "core/weight.h"
#include "core/weight_view.h"
#include "core/buffer.h"
#include "ninfer/ops/linear_split.h"
#include "op_measurement.cuh"
#include <span>

#include "core/device.h"
#include "direct_bf16_weight.cuh"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::size_t kFlushBytes = std::size_t{256} << 20;


enum class Format : std::uint8_t {
    Q4Q5,
    Q8Qgkv,
    Q8Qkv,
    Q8DFlash2Qkv,
    Q8Segmented,
    Bf16Segmented,
    Bf16,
    Nvfp4,
    Fp8,
    Fp8Segmented,
    All
};
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    Format format = Format::All;
    int n = 0, k = 0;
    int parent_rows = 0;
    std::vector<int> rows;
    ops::LinearPolicy nvfp4_policy = ops::LinearPolicy::AllowA4;
    ops::LinearPolicy fp8_policy   = ops::LinearPolicy::AllowA8;
    CacheMode cache                = CacheMode::Cold;
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024};
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
    bool graph   = false;
    std::string csv_out;
};

struct Result {
    std::string format;
    const char* policy;
    std::int32_t tokens;
    CacheState cache;
    std::size_t workspace_bytes;
    std::uint64_t logical_bytes;
    double useful_flops;
    bench::ColdTiming timing;
    std::size_t graph_nodes              = 0;
    std::size_t workspace_capacity_bytes = 0;
    std::size_t workspace_peak_bytes     = 0;
};

struct Measurement {
    bench::ColdTiming timing;
    std::size_t graph_nodes = 0;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(
        stderr,
        "error: %s\n"
        "usage: ninfer_linear_split_multi_output_bench "
        "[--format "
        "q4q5|q8-qgkv|q8-qkv|q8-dflash2-qkv|q8-segmented|bf16-segmented|bf16|nvfp4|fp8|fp8-segmented|all] "
        "[--n N --k K --rows R1,R2,... (segmented; fp8-segmented also accepts --parent-rows N)] "
        "[--nvfp4-policy a16|a4] [--fp8-policy a16|a8] "
        "[--tokens T,...] [--cache cold|warm|both] [--execution eager|graph] "
        "Graph measurements time the captured GPU body. "
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
        if (argument == "--format") {
            const std::string_view value(next("--format requires a value"));
            if (value == "q4q5")
                options.format = Format::Q4Q5;
            else if (value == "q8-qgkv")
                options.format = Format::Q8Qgkv;
            else if (value == "q8-qkv")
                options.format = Format::Q8Qkv;
            else if (value == "q8-dflash2-qkv")
                options.format = Format::Q8DFlash2Qkv;
            else if (value == "q8-segmented")
                options.format = Format::Q8Segmented;
            else if (value == "bf16-segmented")
                options.format = Format::Bf16Segmented;
            else if (value == "bf16")
                options.format = Format::Bf16;
            else if (value == "nvfp4")
                options.format = Format::Nvfp4;
            else if (value == "fp8")
                options.format = Format::Fp8;
            else if (value == "fp8-segmented")
                options.format = Format::Fp8Segmented;
            else if (value == "all")
                options.format = Format::All;
            else
                usage("--format expects q4q5, q8-qgkv, q8-qkv, q8-dflash2-qkv, q8-segmented, bf16, "
                      "nvfp4, "
                      "fp8, or all");
        } else if (argument == "--nvfp4-policy") {
            const std::string_view value(next("--nvfp4-policy requires a value"));
            if (value == "a16")
                options.nvfp4_policy = ops::LinearPolicy::A16Only;
            else if (value == "a4")
                options.nvfp4_policy = ops::LinearPolicy::AllowA4;
            else
                usage("--nvfp4-policy expects a16 or a4");
        } else if (argument == "--fp8-policy") {
            const std::string_view value(next("--fp8-policy requires a value"));
            if (value == "a16")
                options.fp8_policy = ops::LinearPolicy::A16Only;
            else if (value == "a8")
                options.fp8_policy = ops::LinearPolicy::AllowA8;
            else
                usage("--fp8-policy expects a16 or a8");
        } else if (argument == "--n") {
            options.n = parse_i32(next("--n"), 1, INT32_MAX, "--n");
        } else if (argument == "--parent-rows") {
            options.parent_rows = parse_i32(next("--parent-rows"), 1, INT32_MAX, "--parent-rows");
        } else if (argument == "--k") {
            options.k = parse_i32(next("--k"), 1, INT32_MAX, "--k");
        } else if (argument == "--rows") {
            options.rows = parse_list(next("--rows"), "--rows");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), "--tokens");
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
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value != "eager" && value != "graph") usage("--execution expects eager or graph");
            options.graph = value == "graph";
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
    if (options.profile && (options.format == Format::All || options.tokens.size() != 1 ||
                            options.cache == CacheMode::Both)) {
        usage("--profile requires one format, one T, and one cache state");
    }
    if ((options.format == Format::Q8Segmented || options.format == Format::Bf16Segmented ||
         options.format == Format::Fp8Segmented) &&
        (!options.n || !options.k || options.rows.empty()))
        usage("segmented requires --n, --k and --rows");
    return options;
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

const char* policy_name(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return "a16";
    case ops::LinearPolicy::AllowA8:
        return "a8";
    case ops::LinearPolicy::AllowA4:
        return "a4";
    }
    throw std::invalid_argument("unknown linear policy");
}

template <class Launch>
Measurement measure_public(Launch&& launch, CacheState cache, bench::L2FlushBuffer& flush,
                           CudaExecutionView cuda, int warmup, int repeat, bool graph) {
    bench::measurement::Options options;
    options.warmup    = warmup;
    options.repeat    = repeat;
    const auto result = bench::measurement::measure(
        std::forward<Launch>(launch), [](auto) {}, flush, cuda, options, graph ? "graph" : "eager",
        cache_name(cache));
    return {result.gpu, graph ? result.nodes : 0};
}

template <class Launch>
void profile_public(Launch&& launch, const char* format, const char* policy, CacheState cache,
                    bench::L2FlushBuffer& flush, CudaExecutionView cuda, int warmup, bool graph) {
    const auto stream = cuda.stream();
    bench::TimedGraph captured;
    if (graph) captured.capture(cuda, launch);
    const auto invoke = [&] {
        if (graph)
            captured.launch(cuda);
        else
            launch(cuda);
    };
    for (int index = 0; index < warmup; ++index) { invoke(); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=linear_split format=%s policy=%s dispatch=public execution=%s "
                "graph_nodes=%zu cache=%s\n",
                format, policy, graph ? "graph" : "eager", captured.nodes(), cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    invoke();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = static_cast<double>(result.logical_bytes) / seconds / 1.0e9;
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=linear_split format=%-8s policy=%-3s cache=%-4s execution=%-5s "
                "graph_nodes=%zu T=%4d "
                "workspace=%9zu capacity=%9zu peak=%9zu median=%9.3f us min=%9.3f us p95=%9.3f us "
                "logical=%8.1f GB/s math=%8.2f TFLOP/s\n",
                result.format.c_str(), result.policy, cache_name(result.cache),
                result.graph_nodes ? "graph" : "eager", result.graph_nodes, result.tokens,
                result.workspace_bytes, result.workspace_capacity_bytes,
                result.workspace_peak_bytes, result.timing.median_us, result.timing.min_us,
                result.timing.p95_us, gbps, tflops);
}

void append_result(std::vector<Result>& results, const char* format, const char* policy,
                   std::int32_t tokens, CacheState cache, std::size_t workspace_bytes,
                   std::uint64_t logical_bytes, double useful_flops, Measurement measurement,
                   std::size_t capacity = 0, std::size_t peak = 0) {
    Result result{format,
                  policy,
                  tokens,
                  cache,
                  workspace_bytes,
                  logical_bytes,
                  useful_flops,
                  measurement.timing,
                  measurement.graph_nodes,
                  capacity,
                  peak};
    report(result);
    results.push_back(result);
}

std::uint64_t tensor_bytes(std::int32_t rows, std::int32_t tokens) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(tokens) * 2ULL;
}

void run_q4q5(const Options& options, bench::L2FlushBuffer& flush, CudaExecutionView cuda,
              std::vector<Result>& results) {
    ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(), 256);
    MemoryArena workspace(workspace_storage.span());
    constexpr std::int32_t hidden      = 5120;
    constexpr std::int32_t q_rows      = 6144;
    constexpr std::int32_t kv_rows     = 1024;
    constexpr std::int32_t parent_rows = q_rows + kv_rows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight qk =
        bench::make_row_split_weight(QType::Q4_G64_FP16, parent_rows, hidden, hidden, 501U);
    bench::PackedQuantizedWeight gv =
        bench::make_row_split_weight(QType::Q5_G64_FP16, parent_rows, hidden, hidden, 503U);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens, 199U);
    DeviceBuffer q(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer gate(ninfer::bench::device_context().device(),
                      static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.span(), DType::BF16, {hidden, tokens});
        Tensor tq(q.span(), DType::BF16, {q_rows, tokens});
        Tensor tg(gate.span(), DType::BF16, {q_rows, tokens});
        Tensor tk(k.span(), DType::BF16, {kv_rows, tokens});
        Tensor tv(v.span(), DType::BF16, {kv_rows, tokens});
        const auto launch = [&](CudaExecutionView cuda) {
            ops::linear_split(x, ops::PairedProjectionWeights{qk.weight, gv.weight},
                              std::array{tq, tk, tg, tv}, workspace, cuda);
        };
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, "q4q5", "a16", profile_cache, flush, cuda, options.warmup,
                           options.graph);
            continue;
        }
        const std::uint64_t logical = qk.model_weight_bytes() + gv.model_weight_bytes() +
                                      tensor_bytes(hidden, tokens) +
                                      tensor_bytes(2 * q_rows + 2 * kv_rows, tokens);
        const double flops = 4.0 * parent_rows * hidden * static_cast<double>(tokens);
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm))
                continue;
            append_result(results, "q4q5", "a16", tokens, cache, 0, logical, flops,
                          measure_public(launch, cache, flush, cuda, options.warmup, options.repeat,
                                         options.graph));
        }
    }
}

template <class WeightFixture>
void run_four_output(const Options& options, const char* format, ops::LinearPolicy policy,
                     bool implicit_a16_entry, std::int32_t hidden, std::int32_t q_rows,
                     std::int32_t kv_rows, std::int32_t parent_rows, WeightFixture& fixture,
                     bench::L2FlushBuffer& flush, CudaExecutionView cuda,
                     std::vector<Result>& results) {
    const std::int32_t min_tokens = *std::min_element(options.tokens.begin(), options.tokens.end());
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    const std::size_t workspace_bytes = ops::linear_split_workspace_capacity_bytes(
        ops::SingleProjectionWeight{fixture.weight, policy},
        std::array{q_rows, kv_rows, q_rows, kv_rows}, min_tokens, max_tokens);
    ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                           std::max<std::size_t>(workspace_bytes, 1));
    MemoryArena workspace(workspace_storage.span());
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens, 101U);
    DeviceBuffer q(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer gate(ninfer::bench::device_context().device(),
                      static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.span(), DType::BF16, {hidden, tokens});
        Tensor tq(q.span(), DType::BF16, {q_rows, tokens});
        Tensor tg(gate.span(), DType::BF16, {q_rows, tokens});
        Tensor tk(k.span(), DType::BF16, {kv_rows, tokens});
        Tensor tv(v.span(), DType::BF16, {kv_rows, tokens});
        const auto exact_workspace = ops::linear_split_workspace_capacity_bytes(
            ops::SingleProjectionWeight{fixture.weight, policy},
            std::array{q_rows, kv_rows, q_rows, kv_rows}, tokens, tokens);
        const auto launch = [&](CudaExecutionView cuda) {
            if (implicit_a16_entry) {
                ops::linear_split(x, fixture.weight, std::array{tq, tk, tg, tv}, cuda);
            } else {
                ops::linear_split(x, ops::SingleProjectionWeight{fixture.weight, policy},
                                  std::array{tq, tk, tg, tv}, workspace, cuda);
            }
        };
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, format, policy_name(policy), profile_cache, flush, cuda,
                           options.warmup, options.graph);
            continue;
        }
        const std::uint64_t logical = fixture.model_weight_bytes() + tensor_bytes(hidden, tokens) +
                                      tensor_bytes(2 * q_rows + 2 * kv_rows, tokens);
        const double flops = 2.0 * parent_rows * hidden * static_cast<double>(tokens);
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm))
                continue;
            workspace.reset_peak();
            const auto measurement = measure_public(launch, cache, flush, cuda, options.warmup,
                                                    options.repeat, options.graph);
            if (workspace.used() != 0 || workspace.peak_used() > exact_workspace)
                throw std::runtime_error("public projection exceeds its workspace query");
            append_result(results, format, policy_name(policy), tokens, cache, exact_workspace,
                          logical, flops, measurement, workspace_bytes, workspace.peak_used());
        }
    }
}

void run_q8_qkv(const Options& options, const char* label, std::int32_t hidden,
                bench::L2FlushBuffer& flush, CudaExecutionView cuda, std::vector<Result>& results) {
    constexpr std::int32_t q_rows      = 4096;
    constexpr std::int32_t kv_rows     = 1024;
    constexpr std::int32_t parent_rows = 6144;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight weight =
        bench::make_row_split_weight(QType::Q8_G32_FP16, parent_rows, hidden, hidden, 505U);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens, 103U);
    DeviceBuffer q(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(ninfer::bench::device_context().device(),
                   static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.span(), DType::BF16, {hidden, tokens});
        Tensor tq(q.span(), DType::BF16, {q_rows, tokens});
        Tensor tk(k.span(), DType::BF16, {kv_rows, tokens});
        Tensor tv(v.span(), DType::BF16, {kv_rows, tokens});
        const auto launch = [&](CudaExecutionView cuda) {
            ops::linear_split(x, weight.weight, std::array{tq, tk, tv}, cuda);
        };
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, label, "a16", profile_cache, flush, cuda, options.warmup,
                           options.graph);
            continue;
        }
        const std::uint64_t logical = weight.model_weight_bytes() + tensor_bytes(hidden, tokens) +
                                      tensor_bytes(q_rows + 2 * kv_rows, tokens);
        const double flops = 2.0 * parent_rows * hidden * static_cast<double>(tokens);
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm))
                continue;
            append_result(results, label, "a16", tokens, cache, 0, logical, flops,
                          measure_public(launch, cache, flush, cuda, options.warmup, options.repeat,
                                         options.graph));
        }
    }
}

template <class Fixture>
void run_segmented(const Options& options, const char* label, int hidden, int parent_rows,
                   std::span<const int> rows, Fixture& weight, bench::L2FlushBuffer& flush,
                   CudaExecutionView cuda, std::vector<Result>& results,
                   ops::LinearPolicy policy = ops::LinearPolicy::A16Only,
                   std::uint64_t logical_weight_bytes = 0) {
    const int max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    const int min_tokens = *std::min_element(options.tokens.begin(), options.tokens.end());
    const auto capacity  = ops::linear_split_workspace_capacity_bytes(
        ops::SingleProjectionWeight{weight.weight, policy}, rows, min_tokens,
        max_tokens);
    auto scratch = bench::make_buffer(std::max<std::size_t>(capacity, 1));
    MemoryArena arena(scratch.span());
    auto input = bench::make_bf16(std::size_t(hidden) * max_tokens, 103U);
    std::vector<DeviceBuffer> storage;
    for (int row : rows) storage.push_back(bench::make_buffer(std::size_t(row) * max_tokens * 2));
    for (int tokens : options.tokens) {
        Tensor x(input.span(), DType::BF16, {hidden, tokens});
        std::vector<Tensor> outputs;
        for (std::size_t i = 0; i < rows.size(); ++i)
            outputs.emplace_back(storage[i].span(), DType::BF16,
                                 std::initializer_list<std::int32_t>{rows[i], tokens});
        const auto exact = ops::linear_split_workspace_capacity_bytes(
            ops::SingleProjectionWeight{weight.weight, policy}, rows, tokens,
            tokens);
        const auto launch = [&](CudaExecutionView c) {
            ops::linear_split(
                x, ops::SingleProjectionWeight{weight.weight, policy}, outputs,
                arena, c);
        };
        for (auto cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm))
                continue;
            if (options.profile) {
                profile_public(launch, label, policy_name(policy), cache, flush, cuda, options.warmup,
                               options.graph);
                continue;
            }
            arena.reset_peak();
            auto timing = measure_public(launch, cache, flush, cuda, options.warmup, options.repeat,
                                         options.graph);
            if (arena.used() != 0 || arena.peak_used() > exact)
                throw std::runtime_error("segmented Linear workspace exceeded capacity");
            append_result(results, label, policy_name(policy), tokens, cache, exact,
                          (logical_weight_bytes ? logical_weight_bytes : weight.model_weight_bytes()) +
                              tensor_bytes(hidden + parent_rows, tokens),
                          2.0 * parent_rows * hidden * tokens, timing, capacity, arena.peak_used());
        }
    }
}

void run_fp8(const Options& options, bench::L2FlushBuffer& flush, CudaExecutionView cuda,
             std::vector<Result>& results) {
    auto weight = bench::make_fp8_weight(14336, 5120);
    run_four_output(options, "fp8", options.fp8_policy, false, 5120, 6144, 1024, 14336, weight,
                    flush, cuda, results);
}

void run_fp8_segmented(const Options& options, bench::L2FlushBuffer& flush, CudaExecutionView cuda,
                       std::vector<Result>& results) {
    const int parent_rows = options.parent_rows ? options.parent_rows : options.n;
    if (parent_rows < options.n) throw std::invalid_argument("prefix exceeds FP8 parent");
    auto weight = bench::make_fp8_weight(parent_rows, options.k);
    const WeightParent parent{weight_geometry(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale,
        std::array<std::uint64_t, 2>{std::uint64_t(parent_rows), std::uint64_t(options.k)}),
        weight.weight.device, static_cast<const std::byte*>(weight.weight.payload)};
    weight.weight = native_weight(WeightView{{std::uint64_t(options.n), std::uint64_t(options.k)},
        {{&parent, 0, std::uint64_t(options.n) * options.k}}});
    std::string label = "fp8-segmented_n" + std::to_string(options.n) + "_k" + std::to_string(options.k);
    if (parent_rows != options.n) label += "_parent" + std::to_string(parent_rows);
    run_segmented(options, label.c_str(), options.k, options.n, options.rows, weight, flush, cuda,
                  results, options.fp8_policy, std::uint64_t(options.n) * (options.k + 2));
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) return;
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("failed to open CSV output");
    output << "entry,format,policy,cache,T,workspace_bytes,logical_bytes,useful_flops,"
              "median_us,min_us,p95_us,execution,graph_nodes,workspace_capacity_bytes,workspace_"
              "peak_bytes\n";
    for (const Result& result : results) {
        output << "linear_split," << result.format << ',' << result.policy << ','
               << cache_name(result.cache) << ',' << result.tokens << ',' << result.workspace_bytes
               << ',' << result.logical_bytes << ',' << result.useful_flops << ','
               << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << ',' << (result.graph_nodes ? "graph" : "eager") << ',';
        output << result.graph_nodes << ',' << result.workspace_capacity_bytes << ','
               << result.workspace_peak_bytes << '\n';
    }
}

bool selected(Format configured, Format candidate) {
    return configured == Format::All || configured == candidate;
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

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        std::printf("# gpu=%s sm=%d%d cuda_runtime=%d\n", props.name, props.major, props.minor,
                    CUDART_VERSION);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        bench::L2FlushBuffer flush(kFlushBytes);
        std::vector<Result> results;

        if (selected(options.format, Format::Q4Q5)) { run_q4q5(options, flush, cuda, results); }
        if (selected(options.format, Format::Q8Qgkv)) {
            auto weight = bench::make_row_split_weight(QType::Q8_G32_FP16, 9216, 2048, 2048, 507U);
            run_four_output(options, "q8-qgkv", ops::LinearPolicy::A16Only, true, 2048, 4096, 512,
                            9216, weight, flush, cuda, results);
        }
        if (selected(options.format, Format::Q8Qkv)) {
            run_q8_qkv(options, "q8-qkv", 2048, flush, cuda, results);
        }
        if (selected(options.format, Format::Q8DFlash2Qkv)) {
            run_q8_qkv(options, "q8-dflash2-qkv", 5120, flush, cuda, results);
        }
        if (options.format == Format::Bf16Segmented || options.format == Format::Q8Segmented) {
            std::string label = options.format == Format::Bf16Segmented ? "bf16" : "q8";
            label += "_n" + std::to_string(options.n) + "_k" + std::to_string(options.k);
            for (int row : options.rows) label += "_" + std::to_string(row);
            if (options.format == Format::Bf16Segmented) {
                auto weight = bench::make_direct_bf16_weight(options.n, options.k, 505U);
                run_segmented(options, label.c_str(), options.k, options.n, options.rows, weight,
                              flush, cuda, results);
            } else {
                auto weight = bench::make_row_split_weight(QType::Q8_G32_FP16, options.n, options.k,
                                                           options.k, 505U);
                run_segmented(options, label.c_str(), options.k, options.n, options.rows, weight,
                              flush, cuda, results);
            }
        }

        if (selected(options.format, Format::Bf16)) {
            auto weight = bench::make_direct_bf16_weight(14336, 5120);
            run_four_output(options, "bf16", ops::LinearPolicy::A16Only, false, 5120, 6144, 1024,
                            14336, weight, flush, cuda, results);
        }
        if (selected(options.format, Format::Nvfp4)) {
            auto weight = bench::make_nvfp4_weight(14336, 5120);
            run_four_output(options, "nvfp4", options.nvfp4_policy, false, 5120, 6144, 1024, 14336,
                            weight, flush, cuda, results);
        }
        if (selected(options.format, Format::Fp8)) { run_fp8(options, flush, cuda, results); }
        if (options.format == Format::Fp8Segmented)
            run_fp8_segmented(options, flush, cuda, results);
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_linear_split_multi_output_bench: %s\n", error.what());
        return 1;
    }
}
