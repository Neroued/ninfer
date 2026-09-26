// Cold-cache benchmark and route crossover tuner for BF16 LinearAdd [5120,6144].

#include "core/weight.h"
#include "ninfer/ops/linear_add.h"

#include "core/device.h"
#include "direct_bf16_weight.cuh"
#include "ninfer_bench_common.h"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

constexpr std::int32_t kRows               = 5120;
constexpr std::int32_t kHidden             = 6144;
constexpr double kRtx5090DramGBs           = 1792.0;
constexpr double kRtx5090SustainedReadGBs  = 1674.5;
constexpr double kRtx5090DenseBf16Tflops   = 209.5;
constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20;
constexpr int kDefaultWarmup               = 3;
constexpr int kDefaultRepeat               = 20;

enum class Route : std::uint8_t {
    Production,
    Decode,
    SmallT,
    Mma,
    All,
};

struct Options {
    std::vector<std::int32_t> tokens{
        1,  2,  3,  4,  5,   6,   7,   8,   9,   10,   11,   12,   13,   14,   15, 16,
        17, 18, 19, 20, 21,  22,  23,  24,  25,  26,   27,   28,   29,   30,   31, 32,
        40, 48, 64, 96, 127, 128, 129, 256, 512, 1023, 1024, 1025, 1536, 2048,
    };
    Route route               = Route::Production;
    int warmup                = kDefaultWarmup;
    int repeat                = kDefaultRepeat;
    std::uint64_t flush_bytes = kDefaultFlushBytes;
    bool profile              = false;
    std::string csv_out;
};

struct Result {
    std::string route;
    std::int32_t tokens = 0;
    bench::ColdTiming timing{};
    std::uint64_t logical_bytes = 0;
    double useful_flops         = 0.0;
    double effective_gbs        = 0.0;
    double read_pct             = 0.0;
    double useful_tflops        = 0.0;
    double tc_pct               = 0.0;
    double memory_floor_us      = 0.0;
    double compute_floor_us     = 0.0;
    double roofline_pct         = 0.0;
};

std::vector<std::int32_t> parse_list(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("--t-sweep requires positive int32 values");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    return result;
}

std::vector<std::int32_t> parse_range(std::string_view raw) {
    const std::size_t first  = raw.find(':');
    const std::size_t second = first == std::string_view::npos ? first : raw.find(':', first + 1);
    if (first == std::string_view::npos || second == std::string_view::npos ||
        raw.find(':', second + 1) != std::string_view::npos) {
        throw std::invalid_argument("--sweep must be BEGIN:END:STEP");
    }
    const long begin = std::stol(std::string(raw.substr(0, first)));
    const long end   = std::stol(std::string(raw.substr(first + 1, second - first - 1)));
    const long step  = std::stol(std::string(raw.substr(second + 1)));
    if (begin <= 0 || end < begin || step <= 0 || end > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("--sweep requires 0 < BEGIN <= END and STEP > 0");
    }
    std::vector<std::int32_t> result;
    for (long value = begin; value <= end; value += step) {
        result.push_back(static_cast<std::int32_t>(value));
        if (value > end - step) { break; }
    }
    return result;
}

Route parse_route(std::string_view raw) {
    if (raw == "production") { return Route::Production; }
    if (raw == "decode") { return Route::Decode; }
    if (raw == "small-t") { return Route::SmallT; }
    if (raw == "mma") { return Route::Mma; }
    if (raw == "all") { return Route::All; }
    throw std::invalid_argument("--route must be production|decode|small-t|mma|all");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--t-sweep") {
            options.tokens = parse_list(next("--t-sweep value"));
        } else if (argument == "--sweep") {
            options.tokens = parse_range(next("--sweep value"));
        } else if (argument == "--route") {
            options.route = parse_route(next("--route value"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (argument == "--flush-bytes") {
            options.flush_bytes = std::stoull(std::string(next("--flush-bytes value")));
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s [--t-sweep 1,4,... | --sweep BEGIN:END:STEP] "
                        "[--route production|decode|small-t|mma|all] [--warmup N] [--repeat N] "
                        "[--flush-bytes N] [--csv-out PATH] [--profile]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.tokens.empty() || options.warmup < 0 || options.repeat <= 0 ||
        options.flush_bytes == 0) {
        throw std::invalid_argument("invalid empty sweep, timing count, or flush size");
    }
    std::sort(options.tokens.begin(), options.tokens.end());
    options.tokens.erase(std::unique(options.tokens.begin(), options.tokens.end()),
                         options.tokens.end());
    if (options.profile && (options.tokens.size() != 1 || options.route == Route::All)) {
        throw std::invalid_argument("--profile requires one T and one concrete route");
    }
    return options;
}

bool route_supports(Route route, std::int32_t tokens) {
    switch (route) {
    case Route::Production:
    case Route::Mma:
        return tokens > 0;
    case Route::Decode:
        return tokens == 1;
    case Route::SmallT:
        return tokens >= ops::detail::kBf16LinearAddSmallTMinTokens &&
               tokens <= ops::detail::kBf16LinearAddSmallTMaxTokens;
    case Route::All:
        break;
    }
    return false;
}

std::string route_name(Route route, std::int32_t tokens) {
    switch (route) {
    case Route::Production:
        return ops::detail::bf16_linear_add_schedule_name(
            ops::detail::bf16_linear_add_select(kRows, kHidden, tokens));
    case Route::Decode:
        return "candidate.decode";
    case Route::SmallT:
        return "candidate.small_t";
    case Route::Mma:
        return "candidate.mma";
    case Route::All:
        break;
    }
    throw std::invalid_argument("route does not name one launch");
}

void launch_route(Route route, const Tensor& x, const Weight& weight, Tensor& residual,
                  WorkspaceArena& workspace, cudaStream_t stream) {
    switch (route) {
    case Route::Production:
        ops::linear_add(x, weight, residual, workspace, stream);
        return;
    case Route::Decode:
        ops::detail::bf16_linear_add_decode_launch(x, weight, residual, stream);
        return;
    case Route::SmallT:
        ops::detail::bf16_linear_add_small_t_launch(x, weight, residual, stream);
        return;
    case Route::Mma:
        ops::detail::bf16_linear_add_mma_launch(x, weight, residual, stream);
        return;
    case Route::All:
        break;
    }
    throw std::invalid_argument("route does not resolve to one launch");
}

Result make_result(std::string route, std::int32_t tokens, const bench::ColdTiming& timing,
                   std::uint64_t weight_bytes) {
    const std::uint64_t x_bytes        = 2ULL * static_cast<std::uint64_t>(kHidden) * tokens;
    const std::uint64_t residual_bytes = 2ULL * static_cast<std::uint64_t>(kRows) * tokens;
    const std::uint64_t logical_bytes  = weight_bytes + x_bytes + 2ULL * residual_bytes;
    const double useful_flops =
        2.0 * static_cast<double>(kRows) * static_cast<double>(kHidden) * tokens;
    const double seconds = timing.median_us * 1.0e-6;
    const double memory_floor_us =
        static_cast<double>(logical_bytes) / (kRtx5090DramGBs * 1.0e9) * 1.0e6;
    const double compute_floor_us  = useful_flops / (kRtx5090DenseBf16Tflops * 1.0e12) * 1.0e6;
    const double roofline_floor_us = std::max(memory_floor_us, compute_floor_us);

    return {
        std::move(route),
        tokens,
        timing,
        logical_bytes,
        useful_flops,
        static_cast<double>(logical_bytes) / seconds / 1.0e9,
        (static_cast<double>(logical_bytes) / seconds / 1.0e9) / kRtx5090SustainedReadGBs * 100.0,
        useful_flops / seconds / 1.0e12,
        (useful_flops / seconds / 1.0e12) / kRtx5090DenseBf16Tflops * 100.0,
        memory_floor_us,
        compute_floor_us,
        roofline_floor_us / timing.median_us * 100.0,
    };
}

void print_result(const Result& result) {
    const char* bound = result.memory_floor_us >= result.compute_floor_us ? "memory" : "compute";
    std::printf("T=%-4d %-36s median=%8.3f us  %7.1f GB/s READ=%6.2f%%  "
                "%7.2f TFLOP/s TC=%6.2f%%  %-7s roofline=%6.2f%%\n",
                result.tokens, result.route.c_str(), result.timing.median_us, result.effective_gbs,
                result.read_pct, result.useful_tflops, result.tc_pct, bound, result.roofline_pct);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    out << "route,T,median_us,min_us,p95_us,logical_bytes,useful_flops,effective_gbs,"
           "sustained_read_pct,useful_tflops,bf16_tc_pct,memory_floor_us,compute_floor_us,"
           "roofline_pct,warmup,repeat,flush_bytes\n";
    for (const Result& result : results) {
        out << result.route << ',' << result.tokens << ',' << result.timing.median_us << ','
            << result.timing.min_us << ',' << result.timing.p95_us << ',' << result.logical_bytes
            << ',' << result.useful_flops << ',' << result.effective_gbs << ',' << result.read_pct
            << ',' << result.useful_tflops << ',' << result.tc_pct << ',' << result.memory_floor_us
            << ',' << result.compute_floor_us << ',' << result.roofline_pct << ',' << options.warmup
            << ',' << options.repeat << ',' << options.flush_bytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options         = parse_options(argc, argv);
        const std::int32_t max_tokens = options.tokens.back();
        cudaStream_t stream           = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        DeviceBuffer flush(options.flush_bytes);
        DeviceBuffer input    = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
        DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(kRows) * max_tokens);
        bench::DirectBf16Weight weight = bench::make_direct_bf16_weight(kRows, kHidden, 0x61U);
        WorkspaceArena workspace(1);

        if (options.profile) {
            const std::int32_t tokens = options.tokens.front();
            if (!route_supports(options.route, tokens)) {
                throw std::invalid_argument("selected route does not support T");
            }
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor out(residual.p, DType::BF16, {kRows, tokens});
            for (int index = 0; index < options.warmup; ++index) {
                bench::flush_l2(flush, stream);
                launch_route(options.route, x, weight.weight, out, workspace, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE route=%s T=%d\n", route_name(options.route, tokens).c_str(),
                        tokens);
            CUDA_CHECK(cudaProfilerStart());
            launch_route(options.route, x, weight.weight, out, workspace, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        constexpr Route kConcreteRoutes[]{
            Route::Production,
            Route::Decode,
            Route::SmallT,
            Route::Mma,
        };
        for (const std::int32_t tokens : options.tokens) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor out(residual.p, DType::BF16, {kRows, tokens});
            const auto measure = [&](Route route) {
                if (!route_supports(route, tokens)) {
                    if (options.route == Route::All) { return; }
                    throw std::invalid_argument("selected route does not support T");
                }
                CUDA_CHECK(cudaMemsetAsync(
                    out.data, 0, 2ULL * static_cast<std::uint64_t>(kRows) * tokens, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto launch = [&](cudaStream_t launch_stream) {
                    launch_route(route, x, weight.weight, out, workspace, launch_stream);
                };
                const auto prepare = [&](cudaStream_t launch_stream) {
                    CUDA_CHECK(cudaMemsetAsync(out.data, 0,
                                               2ULL * static_cast<std::uint64_t>(kRows) * tokens,
                                               launch_stream));
                };
                Result result =
                    make_result(route_name(route, tokens), tokens,
                                bench::measure_cold_launch_prepared(prepare, launch, flush, stream,
                                                                    options.warmup, options.repeat),
                                weight.model_weight_bytes());
                print_result(result);
                results.push_back(std::move(result));
            };
            if (options.route == Route::All) {
                for (const Route route : kConcreteRoutes) { measure(route); }
            } else {
                measure(options.route);
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_bf16_linear_add_bench: %s\n", error.what());
        return 1;
    }
}
