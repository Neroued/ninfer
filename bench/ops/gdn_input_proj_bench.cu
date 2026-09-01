// Public-Op benchmark for every registered GDN input-projection contract.
// Production dispatch is owned exclusively by gdn_input_proj().

#include "ninfer/ops/gdn_input_proj.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

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
constexpr double kRtx5090DramGBs  = 1792.0;

enum class Format : std::uint8_t { Q4Q5, W8, Nvfp4, Fp8, All };
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
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_gdn_input_proj_bench "
                 "[--format q4q5|w8|nvfp4|fp8|all] [--nvfp4-policy a16|a4] "
                 "[--fp8-policy a16|a8] "
                 "[--tokens T,...] [--cache cold|warm|both] [--warmup N] [--repeat N] "
                 "[--profile] [--csv-out PATH]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno             = 0;
    char* end         = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
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
            else if (value == "w8")
                options.format = Format::W8;
            else if (value == "nvfp4")
                options.format = Format::Nvfp4;
            else if (value == "fp8")
                options.format = Format::Fp8;
            else if (value == "all")
                options.format = Format::All;
            else
                usage("--format expects q4q5, w8, nvfp4, fp8, or all");
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
bench::ColdTiming measure_public(Launch&& launch, CacheState cache, DeviceBuffer& flush,
                                 cudaStream_t stream, int warmup, int repeat) {
    return cache == CacheState::Cold
               ? bench::measure_cold_launch(std::forward<Launch>(launch), flush, stream, warmup,
                                            repeat)
               : bench::measure_launch(std::forward<Launch>(launch), stream, warmup, repeat);
}

template <class Launch>
void profile_public(Launch&& launch, const char* format, const char* policy, CacheState cache,
                    DeviceBuffer& flush, cudaStream_t stream, int warmup) {
    for (int index = 0; index < warmup; ++index) { launch(stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=gdn_input_proj format=%s policy=%s dispatch=public cache=%s\n",
                format, policy, cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::uint64_t tensor_bytes(std::int32_t rows, std::int32_t tokens) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(tokens) * 2ULL;
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = static_cast<double>(result.logical_bytes) / seconds / 1.0e9;
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=gdn_input_proj format=%-6s policy=%-3s cache=%-4s T=%4d "
                "workspace=%9zu median=%9.3f us min=%9.3f us p95=%9.3f us "
                "logical=%8.1f GB/s (%5.1f%% of %.0f) math=%8.2f TFLOP/s\n",
                result.format, result.policy, cache_name(result.cache), result.tokens,
                result.workspace_bytes, result.timing.median_us, result.timing.min_us,
                result.timing.p95_us, gbps, gbps / kRtx5090DramGBs * 100.0, kRtx5090DramGBs,
                tflops);
}

void append_result(std::vector<Result>& results, const char* format, const char* policy,
                   std::int32_t tokens, CacheState cache, std::size_t workspace_bytes,
                   std::uint64_t logical_bytes, double useful_flops, bench::ColdTiming timing) {
    Result result{format,          policy,        tokens,       cache,
                  workspace_bytes, logical_bytes, useful_flops, timing};
    report(result);
    results.push_back(result);
}

template <class WorkspaceCapacity, class Launch>
void measure_points(const Options& options, const char* format, const char* policy,
                    std::int32_t hidden, std::int32_t output_rows, std::uint64_t weight_bytes,
                    WorkspaceCapacity&& workspace_capacity, Launch&& make_launch,
                    DeviceBuffer& flush, cudaStream_t stream, std::vector<Result>& results) {
    const CacheState profile_cache =
        options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    for (const std::int32_t tokens : options.tokens) {
        auto launch                       = make_launch(tokens);
        const std::size_t workspace_bytes = workspace_capacity(tokens);
        if (options.profile) {
            profile_public(launch, format, policy, profile_cache, flush, stream, options.warmup);
            continue;
        }
        const std::uint64_t logical =
            weight_bytes + tensor_bytes(hidden, tokens) + tensor_bytes(output_rows, tokens);
        const double flops = 2.0 * static_cast<double>(output_rows) * hidden * tokens;
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm)) {
                continue;
            }
            append_result(
                results, format, policy, tokens, cache, workspace_bytes, logical, flops,
                measure_public(launch, cache, flush, stream, options.warmup, options.repeat));
        }
    }
}

void run_q4q5(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
              std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kOutputRows = kQkRows + kValueRows + kZRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight qk = bench::make_row_split_weight(
        QType::Q4G64_F16S, kQkRows, kHidden, kHidden, {0x31, 0x00, 0x3c00});
    bench::PackedQuantizedWeight value_z = bench::make_row_split_weight(
        QType::Q5G64_F16S, kValueRows + kZRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
    DeviceBuffer qkv(static_cast<std::size_t>(kQkRows + kValueRows) * max_tokens * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    const std::int32_t min_tokens = *std::min_element(options.tokens.begin(), options.tokens.end());
    const std::size_t workspace_bytes =
        ops::q4_q5_gdn_input_proj_workspace_capacity_bytes(min_tokens, max_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
    const auto make_launch = [&](std::int32_t tokens) {
        return [&, tokens](cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor tqkv(qkv.p, DType::BF16, {kQkRows + kValueRows, tokens});
            Tensor tz(z.p, DType::BF16, {kZRows, tokens});
            ops::gdn_input_proj(x, qk.weight, value_z.weight, tqkv, tz, workspace,
                                launch_stream);
        };
    };
    measure_points(
        options, "q4q5", "a16", kHidden, kOutputRows,
        qk.model_weight_bytes() + value_z.model_weight_bytes(),
        [](std::int32_t) { return std::size_t{0}; }, make_launch, flush, stream, results);
}

void run_w8(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
            std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 2048;
    constexpr std::int32_t kQkvRows    = 8192;
    constexpr std::int32_t kZRows      = 4096;
    constexpr std::int32_t kOutputRows = kQkvRows + kZRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_row_split_weight(
        QType::W8G32_F16S, kOutputRows, kHidden, kHidden, {0x31, 0x00, 0x3c00});
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
    DeviceBuffer qkv(static_cast<std::size_t>(kQkvRows) * max_tokens * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    const auto make_launch = [&](std::int32_t tokens) {
        return [&, tokens](cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor tqkv(qkv.p, DType::BF16, {kQkvRows, tokens});
            Tensor tz(z.p, DType::BF16, {kZRows, tokens});
            ops::gdn_input_proj(x, parent.weight, tqkv, tz, launch_stream);
        };
    };
    const auto workspace_capacity = [](std::int32_t tokens) {
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::W8G32_F16S, kOutputRows, kHidden, ops::LinearPolicy::A16Only, tokens, tokens);
    };
    measure_points(options, "w8", "a16", kHidden, kOutputRows, parent.model_weight_bytes(),
                   workspace_capacity, make_launch, flush, stream, results);
}

void run_nvfp4(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
               std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkvRows    = 10240;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kOutputRows = kQkvRows + kZRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_nvfp4_weight(kOutputRows, kHidden);
    const std::size_t maximum_workspace = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kOutputRows, kHidden, options.nvfp4_policy, max_tokens, max_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(maximum_workspace, 256));
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
    DeviceBuffer qkv(static_cast<std::size_t>(kQkvRows) * max_tokens * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    const auto make_launch = [&](std::int32_t tokens) {
        return [&, tokens](cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor tqkv(qkv.p, DType::BF16, {kQkvRows, tokens});
            Tensor tz(z.p, DType::BF16, {kZRows, tokens});
            ops::gdn_input_proj(x, parent.weight, tqkv, tz, options.nvfp4_policy, workspace,
                                launch_stream);
        };
    };
    const auto workspace_capacity = [&](std::int32_t tokens) {
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, kOutputRows, kHidden,
                                                            options.nvfp4_policy, tokens, tokens);
    };
    measure_points(options, "nvfp4", policy_name(options.nvfp4_policy), kHidden, kOutputRows,
                   parent.model_weight_bytes(), workspace_capacity, make_launch, flush, stream,
                   results);
}

void run_fp8(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
             std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkvRows    = 10240;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kOutputRows = kQkvRows + kZRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_fp8_weight(kOutputRows, kHidden);
    const std::size_t maximum_workspace = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kOutputRows, kHidden, options.fp8_policy, 1, max_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(maximum_workspace, 256));
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
    DeviceBuffer qkv(static_cast<std::size_t>(kQkvRows) * max_tokens * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    const auto make_launch = [&](std::int32_t tokens) {
        return [&, tokens](cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor tqkv(qkv.p, DType::BF16, {kQkvRows, tokens});
            Tensor tz(z.p, DType::BF16, {kZRows, tokens});
            ops::gdn_input_proj(x, parent.weight, tqkv, tz, options.fp8_policy, workspace,
                                launch_stream);
        };
    };
    const auto workspace_capacity = [&](std::int32_t tokens) {
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, kOutputRows, kHidden, options.fp8_policy, tokens, tokens);
    };
    measure_points(options, "fp8", policy_name(options.fp8_policy), kHidden, kOutputRows,
                   parent.model_weight_bytes(), workspace_capacity, make_launch, flush, stream,
                   results);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,format,policy,cache,T,workspace_bytes,logical_bytes,useful_flops,"
              "median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << "gdn_input_proj," << result.format << ',' << result.policy << ','
               << cache_name(result.cache) << ',' << result.tokens << ',' << result.workspace_bytes
               << ',' << result.logical_bytes << ',' << result.useful_flops << ','
               << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << '\n';
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
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        std::vector<Result> results;

        if (selected(options.format, Format::Q4Q5)) { run_q4q5(options, flush, stream, results); }
        if (selected(options.format, Format::W8)) { run_w8(options, flush, stream, results); }
        if (selected(options.format, Format::Nvfp4)) { run_nvfp4(options, flush, stream, results); }
        if (selected(options.format, Format::Fp8)) { run_fp8(options, flush, stream, results); }

        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gdn_input_proj_bench: %s\n", error.what());
        return 1;
    }
}
