#include "common/roofline.h"
#include "core/weight.h"
#include "core/buffer.h"
#include "ninfer/ops/linear_add.h"

#include "core/device.h"
#include "op_measurement.cuh"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
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

constexpr std::int32_t kRows      = 5120;
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Options {
    std::int32_t n           = 0;
    std::int32_t k           = 0;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::vector<std::int32_t> t_sweep{1, 4, 8, 16, 17, 128, 512, 1024};
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
    double nvfp4_percent;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--n") {
            options.n = bench::measurement::integer(next("--n"));
        } else if (argument == "--k") {
            options.k = bench::measurement::integer(next("--k"));
        } else if (argument == "--policy") {
            const std::string_view value = next("--policy");
            if (value == "a16") {
                options.policy = ops::LinearPolicy::A16Only;
            } else if (value == "a4") {
                options.policy = ops::LinearPolicy::AllowA4;
            } else {
                throw std::invalid_argument("--policy must be a16 or a4");
            }
        } else if (argument == "--t-sweep") {
            options.t_sweep = bench::measurement::integers(next("--t-sweep"));
        } else if (argument == "--warmup") {
            options.warmup = bench::measurement::integer(next("--warmup"), 0);
        } else if (argument == "--repeat") {
            options.repeat = bench::measurement::integer(next("--repeat"));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out");
        } else if (argument == "--help" || argument == "-h") {
            std::printf(
                "Usage: %s --n 5120 --k 6144|17408 [--policy a16|a4] "
                "[--t-sweep 1,4,...] [--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n",
                argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.n != kRows || (options.k != 6144 && options.k != 17408)) {
        throw std::invalid_argument("NVFP4 LinearAdd supports N=5120 and K=6144|17408");
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
    return policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16";
}

void write_csv(const Options& options, const std::vector<Result>& results,
               std::uint64_t weight_bytes) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open CSV: " + options.csv_out); }
    out << "op,weight_type,policy,N,K,T,weight_bytes,median_us,min_us,p95_us,effective_gbs,"
           "useful_tflops,nvfp4_peak_percent,warmup,repeat,flush_bytes\n";
    for (const Result& result : results) {
        out << "linear_add,NVFP4," << policy_name(options.policy) << ',' << options.n << ','
            << options.k << ',' << result.tokens << ',' << weight_bytes << ','
            << result.timing.median_us << ',' << result.timing.min_us << ',' << result.timing.p95_us
            << ',' << result.effective_gbs << ',' << result.useful_tflops << ','
            << bench::format_metric(result.nvfp4_percent, 6, "") << ',' << options.warmup << ','
            << options.repeat << ',' << kFlushBytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());
        const std::int32_t min_t =
            *std::min_element(options.t_sweep.begin(), options.t_sweep.end());
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        bench::L2FlushBuffer flush(kFlushBytes);
        DeviceBuffer input    = bench::make_bf16(static_cast<std::size_t>(options.k) * max_t, 101U);
        DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(options.n) * max_t, 103U);
        bench::SavedBuffer residual_initial(residual);
        const auto restore = [&](CudaExecutionView cuda) { residual_initial.restore(cuda); };
        bench::PackedQuantizedWeight packed  = bench::make_nvfp4_weight(options.n, options.k);
        const std::size_t workspace_capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, options.n, options.k, options.policy, min_t, max_t);
        ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                               std::max<std::size_t>(workspace_capacity, 256));
        MemoryArena workspace(workspace_storage.span());

        const auto make_launch = [&](std::int32_t tokens) {
            return [&, tokens](CudaExecutionView cuda) {
                Tensor x(input.span(), DType::BF16, {options.k, tokens});
                Tensor out(residual.span(), DType::BF16, {options.n, tokens});
                ops::linear_add(x, packed.weight, out, options.policy, workspace, cuda);
            };
        };

        if (options.profile) {
            const std::int32_t tokens = options.t_sweep.front();
            const auto launch         = make_launch(tokens);
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                restore(cuda);
                bench::flush_l2(flush, cuda);
                launch(cuda);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            restore(cuda);
            bench::flush_l2(flush, cuda);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_add weight_type=NVFP4 policy=%s N=%d K=%d T=%d\n",
                        policy_name(options.policy), options.n, options.k, tokens);
            std::fflush(stdout);
            CUDA_CHECK(cudaProfilerStart());
            launch(cuda);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        results.reserve(options.t_sweep.size());
        std::printf("# reference_target=%s dense_nvfp4_fp32_accumulate_peak_tflops=%s cache=cold\n",
                    bench::kRoofline.target,
                    bench::format_metric(bench::kRoofline.nvfp4_f32_tflops, 1).c_str());
        std::printf("%-11s %3s %8s %8s %6s %11s %11s %11s %10s %10s %8s\n", "op", "pol", "N", "K",
                    "T", "median_us", "min_us", "p95_us", "eff_GB/s", "TFLOP/s", "NVFP4_%");
        for (const std::int32_t tokens : options.t_sweep) {
            const auto launch              = make_launch(tokens);
            const bench::ColdTiming timing = bench::measure_cold_launch_prepared(
                restore, launch, flush, cuda, options.warmup, options.repeat);
            const double seconds      = timing.median_us * 1.0e-6;
            const double useful_flops = 2.0 * static_cast<double>(options.n) * options.k * tokens;
            const double model_bytes  = static_cast<double>(packed.model_weight_bytes()) +
                                       2.0 * static_cast<double>(options.k) * tokens +
                                       4.0 * static_cast<double>(options.n) * tokens;
            const double useful_tflops = useful_flops / seconds / 1.0e12;
            const double effective_gbs = model_bytes / seconds / 1.0e9;
            const bool a4              = options.policy == ops::LinearPolicy::AllowA4 &&
                            tokens >= (options.k == 6144 ? 17 : 8);
            const double nvfp4_percent =
                a4 ? 100.0 * useful_tflops / bench::kRoofline.nvfp4_f32_tflops
                   : std::numeric_limits<double>::quiet_NaN();
            std::printf("%-11s %3s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %10.2f %8s\n",
                        "linear_add", policy_name(options.policy), options.n, options.k, tokens,
                        timing.median_us, timing.min_us, timing.p95_us, effective_gbs,
                        useful_tflops, bench::format_metric(nvfp4_percent).c_str());
            results.push_back({tokens, timing, effective_gbs, useful_tflops, nvfp4_percent});
        }
        write_csv(options, results, packed.model_weight_bytes());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_nvfp4_linear_add_bench: %s\n", error.what());
        return 1;
    }
}
