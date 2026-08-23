// Cold-cache public Op benchmark for the registered row-scaled FP8 LinearAdd profiles.

#include "ninfer/ops/linear_add.h"

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
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kRows            = 5120;
constexpr std::size_t kFlushBytes       = 256ULL << 20;
constexpr double kFp8Fp32AccumulatePeak = 419.0;

struct Options {
    std::int32_t k           = 0;
    ops::LinearPolicy policy = ops::LinearPolicy::AllowA8;
    std::vector<std::int32_t> t_sweep{1, 2, 4, 8, 16, 20, 21, 22, 24, 25, 32, 48, 1024};
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
    std::string csv_out;
};

struct Result {
    std::int32_t tokens;
    bench::ColdTiming timing;
    double effective_gbs;
    double useful_tflops;
    double tensor_percent;
};

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
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
        if (argument == "--k") {
            options.k = std::stoi(std::string(next("--k")));
        } else if (argument == "--policy") {
            const std::string_view value = next("--policy");
            if (value == "a16") {
                options.policy = ops::LinearPolicy::A16Only;
            } else if (value == "a8") {
                options.policy = ops::LinearPolicy::AllowA8;
            } else {
                throw std::invalid_argument("--policy must be a16 or a8");
            }
        } else if (argument == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat")));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out");
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s --k 6144|17408 [--policy a16|a8] [--t-sweep 1,2,...] "
                        "[--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.k != 6144 && options.k != 17408) {
        throw std::invalid_argument("FP8 LinearAdd supports N=5120 and K=6144|17408");
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.profile && options.t_sweep.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    return options;
}

const char* policy_name(ops::LinearPolicy policy) {
    return policy == ops::LinearPolicy::AllowA8 ? "A8" : "A16";
}

bool uses_tensor_cores(const Options& options, std::int32_t tokens) {
    if (options.policy != ops::LinearPolicy::AllowA8) { return false; }
    return options.k == 6144 ? tokens >= 22 : tokens >= 25;
}

void write_csv(const Options& options, const std::vector<Result>& results,
               std::uint64_t weight_bytes) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open CSV: " + options.csv_out); }
    out << "op,weight_type,policy,N,K,T,weight_bytes,median_us,min_us,p95_us,effective_gbs,"
           "useful_tflops,tensor_peak_percent,warmup,repeat,flush_bytes\n";
    for (const Result& result : results) {
        out << "linear_add,FP8_E4M3FN_ROW_BF16S," << policy_name(options.policy) << ',' << kRows
            << ',' << options.k << ',' << result.tokens << ',' << weight_bytes << ','
            << result.timing.median_us << ',' << result.timing.min_us << ',' << result.timing.p95_us
            << ',' << result.effective_gbs << ',' << result.useful_tflops << ',';
        if (std::isfinite(result.tensor_percent)) { out << result.tensor_percent; }
        out << ',' << options.warmup << ',' << options.repeat << ',' << kFlushBytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto [min_it, max_it] =
            std::minmax_element(options.t_sweep.begin(), options.t_sweep.end());
        const std::int32_t min_t = *min_it;
        const std::int32_t max_t = *max_it;

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input    = bench::make_bf16(static_cast<std::size_t>(options.k) * max_t);
        DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(kRows) * max_t);
        bench::PackedQuantizedWeight packed  = bench::make_fp8_weight(kRows, options.k);
        const std::size_t workspace_capacity = ops::linear_add_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, kRows, options.k, options.policy, min_t, max_t);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));

        const auto launch = [&](std::int32_t tokens, cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {options.k, tokens});
            Tensor out(residual.p, DType::BF16, {kRows, tokens});
            ops::linear_add(x, packed.weight, out, options.policy, workspace, launch_stream);
        };

        if (options.profile) {
            const std::int32_t tokens = options.t_sweep.front();
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                bench::flush_l2(flush, stream);
                launch(tokens, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_add weight_type=FP8 policy=%s N=%d K=%d T=%d\n",
                        policy_name(options.policy), kRows, options.k, tokens);
            std::fflush(stdout);
            CUDA_CHECK(cudaProfilerStart());
            launch(tokens, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        results.reserve(options.t_sweep.size());
        std::printf("# fp8_fp32_accumulate_peak_tflops=%.1f cache=cold\n", kFp8Fp32AccumulatePeak);
        std::printf("%-4s %8s %8s %6s %11s %11s %11s %10s %10s %8s\n", "pol", "N", "K", "T",
                    "median_us", "min_us", "p95_us", "eff_GB/s", "TFLOP/s", "TC_%");
        for (const std::int32_t tokens : options.t_sweep) {
            const bench::ColdTiming timing = bench::measure_cold_launch(
                [&](cudaStream_t launch_stream) { launch(tokens, launch_stream); }, flush, stream,
                options.warmup, options.repeat);
            const double seconds = timing.median_us * 1.0e-6;
            const double flops   = 2.0 * static_cast<double>(kRows) * options.k * tokens;
            const double bytes   = static_cast<double>(packed.model_weight_bytes()) +
                                 2.0 * static_cast<double>(options.k) * tokens +
                                 4.0 * static_cast<double>(kRows) * tokens;
            const double tflops         = flops / seconds / 1.0e12;
            const double tensor_percent = uses_tensor_cores(options, tokens)
                                              ? 100.0 * tflops / kFp8Fp32AccumulatePeak
                                              : std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(tensor_percent)) {
                std::printf("%-4s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %10.2f %8.2f\n",
                            policy_name(options.policy), kRows, options.k, tokens, timing.median_us,
                            timing.min_us, timing.p95_us, bytes / seconds / 1.0e9, tflops,
                            tensor_percent);
            } else {
                std::printf("%-4s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %10.2f %8s\n",
                            policy_name(options.policy), kRows, options.k, tokens, timing.median_us,
                            timing.min_us, timing.p95_us, bytes / seconds / 1.0e9, tflops, "-");
            }
            results.push_back({tokens, timing, bytes / seconds / 1.0e9, tflops, tensor_percent});
        }
        write_csv(options, results, packed.model_weight_bytes());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_fp8_linear_add_bench: %s\n", error.what());
        return 1;
    }
}
