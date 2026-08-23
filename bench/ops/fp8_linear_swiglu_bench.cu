// Cold-cache public Op benchmark for the registered row-scaled FP8 LinearSwiGLU profile.

#include "ninfer/ops/linear_swiglu.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kGateUpRows      = 34816;
constexpr std::int32_t kOutputRows      = 17408;
constexpr std::int32_t kHidden          = 5120;
constexpr std::size_t kFlushBytes       = 256ULL << 20;
constexpr double kFp8Fp32AccumulatePeak = 419.0;

struct Options {
    ops::LinearPolicy policy = ops::LinearPolicy::AllowA8;
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 16, 32, 48, 1024};
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
};

std::vector<std::int32_t> parse_tokens(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("--t-sweep values must be positive int32");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--policy") {
            const std::string_view value = next("--policy value");
            if (value == "a16") {
                options.policy = ops::LinearPolicy::A16Only;
            } else if (value == "a8") {
                options.policy = ops::LinearPolicy::AllowA8;
            } else {
                throw std::invalid_argument("--policy must be a16 or a8");
            }
        } else if (argument == "--t-sweep") {
            options.tokens = parse_tokens(next("--t-sweep value"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s [--policy a16|a8] [--t-sweep 1,2,...] [--warmup N] "
                        "[--repeat N] [--profile]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.profile && options.tokens.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    return options;
}

const char* policy_name(ops::LinearPolicy policy) {
    return policy == ops::LinearPolicy::AllowA8 ? "A8" : "A16";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto [min_it, max_it] =
            std::minmax_element(options.tokens.begin(), options.tokens.end());
        const std::int32_t min_t = *min_it;
        const std::int32_t max_t = *max_it;

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t);
        DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_t * sizeof(std::uint16_t));
        bench::PackedQuantizedWeight packed  = bench::make_fp8_weight(kGateUpRows, kHidden);
        const std::size_t workspace_capacity = ops::linear_swiglu_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, kGateUpRows, kHidden, options.policy, min_t, max_t);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));

        const auto launch = [&](std::int32_t tokens, cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
            ops::linear_swiglu(x, packed.weight, out, options.policy, workspace, launch_stream);
        };

        if (options.profile) {
            const std::int32_t tokens = options.tokens.front();
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                bench::flush_l2(flush, stream);
                launch(tokens, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_swiglu weight_type=FP8 policy=%s T=%d\n",
                        policy_name(options.policy), tokens);
            std::fflush(stdout);
            CUDA_CHECK(cudaProfilerStart());
            launch(tokens, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::printf("# fp8_fp32_accumulate_peak_tflops=%.1f cache=cold\n", kFp8Fp32AccumulatePeak);
        std::printf("%-4s %6s %11s %11s %11s %10s %10s %8s\n", "pol", "T", "median_us", "min_us",
                    "p95_us", "eff_GB/s", "TFLOP/s", "TC_%");
        for (const std::int32_t tokens : options.tokens) {
            const auto timing = bench::measure_cold_launch(
                [&](cudaStream_t launch_stream) { launch(tokens, launch_stream); }, flush, stream,
                options.warmup, options.repeat);
            const double seconds = timing.median_us * 1.0e-6;
            const double flops   = 2.0 * static_cast<double>(kGateUpRows) * kHidden * tokens;
            const double bytes   = static_cast<double>(packed.model_weight_bytes()) +
                                 2.0 * static_cast<double>(kHidden + kOutputRows) * tokens;
            const double tflops = flops / seconds / 1.0e12;
            const bool tensor_route =
                options.policy == ops::LinearPolicy::AllowA8 && (tokens == 1 || tokens >= 3);
            const double tensor_percent = tensor_route ? 100.0 * tflops / kFp8Fp32AccumulatePeak
                                                       : std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(tensor_percent)) {
                std::printf("%-4s %6d %11.3f %11.3f %11.3f %10.1f %10.2f %8.2f\n",
                            policy_name(options.policy), tokens, timing.median_us, timing.min_us,
                            timing.p95_us, bytes / seconds / 1.0e9, tflops, tensor_percent);
            } else {
                std::printf("%-4s %6d %11.3f %11.3f %11.3f %10.1f %10.2f %8s\n",
                            policy_name(options.policy), tokens, timing.median_us, timing.min_us,
                            timing.p95_us, bytes / seconds / 1.0e9, tflops, "-");
            }
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_fp8_linear_swiglu_bench: %s\n", error.what());
        return 1;
    }
}
