// Public-Op benchmark for every registered Attention input-projection contract.
// Production dispatch is owned exclusively by attn_input_proj().

#include "ninfer/ops/attn_input_proj.h"

#include "core/device.h"
#include "direct_bf16_weight.cuh"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
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


enum class Format : std::uint8_t { Q4Q5, W8Qgkv, W8Qkv, W8DFlash2Qkv, Bf16, Nvfp4, Fp8, All };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    Format format                  = Format::All;
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
    const char* format;
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
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_attn_input_proj_bench "
                 "[--format q4q5|w8-qgkv|w8-qkv|w8-dflash2-qkv|bf16|nvfp4|fp8|all] "
                 "[--nvfp4-policy a16|a4] [--fp8-policy a16|a8] "
                 "[--tokens T,...] [--cache cold|warm|both] [--execution eager|graph] "
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
            else if (value == "w8-qgkv")
                options.format = Format::W8Qgkv;
            else if (value == "w8-qkv")
                options.format = Format::W8Qkv;
            else if (value == "w8-dflash2-qkv")
                options.format = Format::W8DFlash2Qkv;
            else if (value == "bf16")
                options.format = Format::Bf16;
            else if (value == "nvfp4")
                options.format = Format::Nvfp4;
            else if (value == "fp8")
                options.format = Format::Fp8;
            else if (value == "all")
                options.format = Format::All;
            else
                usage("--format expects q4q5, w8-qgkv, w8-qkv, w8-dflash2-qkv, bf16, nvfp4, "
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
Measurement measure_public(Launch&& launch, CacheState cache, DeviceBuffer& flush,
                           cudaStream_t stream, int warmup, int repeat, bool graph) {
    if (graph) {
        bench::TimedGraph captured;
        captured.capture(stream, launch);
        return {cache == CacheState::Cold
                    ? bench::measure_cold_graph(captured, flush, stream, warmup, repeat)
                    : bench::measure_graph(captured, stream, warmup, repeat),
                captured.nodes()};
    }
    return {cache == CacheState::Cold
                ? bench::measure_cold_launch(std::forward<Launch>(launch), flush, stream, warmup,
                                             repeat)
                : bench::measure_launch(std::forward<Launch>(launch), stream, warmup, repeat),
            0};
}

template <class Launch>
void profile_public(Launch&& launch, const char* format, const char* policy, CacheState cache,
                    DeviceBuffer& flush, cudaStream_t stream, int warmup, bool graph) {
    bench::TimedGraph captured;
    if (graph) captured.capture(stream, launch);
    const auto invoke = [&] {
        if (graph)
            captured.launch(stream);
        else
            launch(stream);
    };
    for (int index = 0; index < warmup; ++index) { invoke(); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=attn_input_proj format=%s policy=%s dispatch=public execution=%s "
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
    std::printf("entry=attn_input_proj format=%-8s policy=%-3s cache=%-4s execution=%-5s "
                "graph_nodes=%zu T=%4d "
                "workspace=%9zu capacity=%9zu peak=%9zu median=%9.3f us min=%9.3f us p95=%9.3f us "
                "logical=%8.1f GB/s math=%8.2f TFLOP/s\n",
                result.format, result.policy, cache_name(result.cache),
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

__device__ unsigned mix_bits(unsigned v) {
    v ^= v >> 16;
    v *= 0x7feb352dU;
    v ^= v >> 15;
    v *= 0x846ca68bU;
    return v ^ (v >> 16);
}

__global__ void fill_input(__nv_bfloat16* x, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = __float2bfloat16_rn(
            (float(mix_bits(static_cast<unsigned>(i) + 913U) >> 8) * (2.f / 16777216.f) - 1.f) *
            .01f);
}

DeviceBuffer varied_input(std::size_t n) {
    DeviceBuffer x(n * 2);
    fill_input<<<(n + 255) / 256, 256>>>(static_cast<__nv_bfloat16*>(x.p), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return x;
}

template <bool Five>
__device__ unsigned group_code(unsigned group, int lane) {
    constexpr int span = Five ? 31 : 15, limit = Five ? 15 : 7;
    return static_cast<unsigned>(int(mix_bits(group * 64 + lane + 175U) % span) - limit) &
           (Five ? 31 : 15);
}

template <bool Five>
__global__ void fill_groupwise(std::uint8_t* low, std::uint8_t* high, std::uint16_t* scales,
                               unsigned groups) {
    const unsigned group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    for (int i = 0; i < 32; ++i)
        low[group * 32 + i] = (group_code<Five>(group, 2 * i) & 15) |
                              ((group_code<Five>(group, 2 * i + 1) & 15) << 4);
    if constexpr (Five)
        for (int i = 0; i < 8; ++i) {
            unsigned byte = 0;
            for (int bit = 0; bit < 8; ++bit)
                byte |= (group_code<Five>(group, i * 8 + bit) >> 4) << bit;
            high[group * 8 + i] = byte;
        }
    scales[group] = 0x3000U + (mix_bits(group + 1345U) & 1023U);
}

void vary_groupwise(bench::PackedQuantizedWeight& weight) {
    auto* bytes       = static_cast<std::uint8_t*>(weight.storage.p);
    auto* scales      = reinterpret_cast<std::uint16_t*>(bytes + weight.scale_offset);
    const auto groups = static_cast<unsigned>(weight.scale_bytes / 2);
    if (weight.weight.qtype == QType::Q4G64_F16S)
        fill_groupwise<false><<<(groups + 255) / 256, 256>>>(bytes, nullptr, scales, groups);
    else
        fill_groupwise<true>
            <<<(groups + 255) / 256, 256>>>(bytes, bytes + weight.high_offset, scales, groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void fill_fp8(std::uint8_t* codes, std::uint16_t* scales, std::size_t n, int rows) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned code = mix_bits(static_cast<unsigned>(i) + 7413U) & 255;
        if ((code & 127) == 127) --code; // E4M3FN has no infinities; exclude its two NaN words.
        codes[i] = code;
    }
    if (i < rows) scales[i] = 0x3b80U + (mix_bits(static_cast<unsigned>(i) + 873U) & 255U);
}

void run_q4q5(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
              std::vector<Result>& results) {
    constexpr std::int32_t hidden      = 5120;
    constexpr std::int32_t q_rows      = 6144;
    constexpr std::int32_t kv_rows     = 1024;
    constexpr std::int32_t parent_rows = q_rows + kv_rows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight qk = bench::make_row_split_weight(
        QType::Q4G64_F16S, parent_rows, hidden, hidden, {0x31, 0x00, 0x3c00});
    bench::PackedQuantizedWeight gv = bench::make_row_split_weight(
        QType::Q5G64_F16S, parent_rows, hidden, hidden, {0x31, 0xa5, 0x3c00});
    vary_groupwise(qk);
    vary_groupwise(gv);
    DeviceBuffer input = varied_input(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer q(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.p, DType::BF16, {hidden, tokens});
        Tensor tq(q.p, DType::BF16, {q_rows, tokens});
        Tensor tg(gate.p, DType::BF16, {q_rows, tokens});
        Tensor tk(k.p, DType::BF16, {kv_rows, tokens});
        Tensor tv(v.p, DType::BF16, {kv_rows, tokens});
        const auto launch = [&](cudaStream_t launch_stream) {
            ops::attn_input_proj(x, qk.weight, gv.weight, tq, tg, tk, tv, launch_stream);
        };
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, "q4q5", "a16", profile_cache, flush, stream, options.warmup,
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
                          measure_public(launch, cache, flush, stream, options.warmup,
                                         options.repeat, options.graph));
        }
    }
}

template <class WeightFixture>
void run_four_output(const Options& options, const char* format, QType qtype,
                     ops::LinearPolicy policy, bool implicit_a16_entry, std::int32_t hidden,
                     std::int32_t q_rows, std::int32_t kv_rows, std::int32_t parent_rows,
                     WeightFixture& fixture, DeviceBuffer& flush, cudaStream_t stream,
                     std::vector<Result>& results) {
    const std::int32_t min_tokens = *std::min_element(options.tokens.begin(), options.tokens.end());
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    const std::size_t workspace_bytes = ops::attn_input_proj_workspace_capacity_bytes(
        qtype, parent_rows, hidden, policy, min_tokens, max_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
    DeviceBuffer input = qtype == QType::FP8_E4M3FN_ROW_BF16S
                             ? varied_input(static_cast<std::size_t>(hidden) * max_tokens)
                             : bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer q(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.p, DType::BF16, {hidden, tokens});
        Tensor tq(q.p, DType::BF16, {q_rows, tokens});
        Tensor tg(gate.p, DType::BF16, {q_rows, tokens});
        Tensor tk(k.p, DType::BF16, {kv_rows, tokens});
        Tensor tv(v.p, DType::BF16, {kv_rows, tokens});
        const auto exact_workspace = ops::attn_input_proj_workspace_capacity_bytes(
            qtype, parent_rows, hidden, policy, tokens, tokens);
        const auto launch = [&](cudaStream_t launch_stream) {
            if (implicit_a16_entry) {
                ops::attn_input_proj(x, fixture.weight, tq, tg, tk, tv, launch_stream);
            } else {
                ops::attn_input_proj(x, fixture.weight, tq, tg, tk, tv, policy, workspace,
                                     launch_stream);
            }
        };
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, format, policy_name(policy), profile_cache, flush, stream,
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
            const auto measurement = measure_public(launch, cache, flush, stream, options.warmup,
                                                    options.repeat, options.graph);
            if (workspace.used() != 0 || workspace.peak_used() > exact_workspace)
                throw std::runtime_error("public projection exceeds its workspace query");
            append_result(results, format, policy_name(policy), tokens, cache, exact_workspace,
                          logical, flops, measurement, workspace_bytes, workspace.peak_used());
        }
    }
}

void run_w8_qkv(const Options& options, const char* label, std::int32_t hidden, DeviceBuffer& flush,
                cudaStream_t stream, std::vector<Result>& results) {
    constexpr std::int32_t q_rows      = 4096;
    constexpr std::int32_t kv_rows     = 1024;
    constexpr std::int32_t parent_rows = 6144;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight weight = bench::make_row_split_weight(
        QType::W8G32_F16S, parent_rows, hidden, hidden, {0x31, 0x00, 0x3c00});
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer q(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.p, DType::BF16, {hidden, tokens});
        Tensor tq(q.p, DType::BF16, {q_rows, tokens});
        Tensor tk(k.p, DType::BF16, {kv_rows, tokens});
        Tensor tv(v.p, DType::BF16, {kv_rows, tokens});
        const auto launch = [&](cudaStream_t launch_stream) {
            ops::attn_input_proj(x, weight.weight, tq, tk, tv, launch_stream);
        };
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, label, "a16", profile_cache, flush, stream, options.warmup,
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
                          measure_public(launch, cache, flush, stream, options.warmup,
                                         options.repeat, options.graph));
        }
    }
}

void run_fp8(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
             std::vector<Result>& results) {
    auto weight = bench::make_fp8_weight(14336, 5120);
    auto* data  = static_cast<std::uint8_t*>(weight.storage.p);
    fill_fp8<<<(weight.low_bytes + 255) / 256, 256>>>(
        data, reinterpret_cast<std::uint16_t*>(data + weight.scale_offset), weight.low_bytes,
        14336);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    run_four_output(options, "fp8", QType::FP8_E4M3FN_ROW_BF16S, options.fp8_policy, false, 5120,
                    6144, 1024, 14336, weight, flush, stream, results);
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
        output << "attn_input_proj," << result.format << ',' << result.policy << ','
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
        int device            = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        std::printf("# gpu=%s sm=%d%d cuda_runtime=%d\n", props.name, props.major, props.minor,
                    CUDART_VERSION);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        std::vector<Result> results;

        if (selected(options.format, Format::Q4Q5)) { run_q4q5(options, flush, stream, results); }
        if (selected(options.format, Format::W8Qgkv)) {
            auto weight = bench::make_row_split_weight(QType::W8G32_F16S, 9216, 2048, 2048,
                                                       {0x31, 0x00, 0x3c00});
            run_four_output(options, "w8-qgkv", QType::W8G32_F16S, ops::LinearPolicy::A16Only, true,
                            2048, 4096, 512, 9216, weight, flush, stream, results);
        }
        if (selected(options.format, Format::W8Qkv)) {
            run_w8_qkv(options, "w8-qkv", 2048, flush, stream, results);
        }
        if (selected(options.format, Format::W8DFlash2Qkv)) {
            run_w8_qkv(options, "w8-dflash2-qkv", 5120, flush, stream, results);
        }
        if (selected(options.format, Format::Bf16)) {
            auto weight = bench::make_direct_bf16_weight(14336, 5120);
            run_four_output(options, "bf16", QType::BF16_CTRL, ops::LinearPolicy::A16Only, false,
                            5120, 6144, 1024, 14336, weight, flush, stream, results);
        }
        if (selected(options.format, Format::Nvfp4)) {
            auto weight = bench::make_nvfp4_weight(14336, 5120);
            run_four_output(options, "nvfp4", QType::NVFP4, options.nvfp4_policy, false, 5120, 6144,
                            1024, 14336, weight, flush, stream, results);
        }
        if (selected(options.format, Format::Fp8)) { run_fp8(options, flush, stream, results); }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_attn_input_proj_bench: %s\n", error.what());
        return 1;
    }
}
