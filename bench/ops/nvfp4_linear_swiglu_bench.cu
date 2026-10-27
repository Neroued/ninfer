#include "common/roofline.h"
#include "core/weight.h"
#include "core/platform.h"
#include "core/buffer.h"
#include "ninfer/ops/linear_swiglu.h"

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

constexpr std::int32_t kHidden    = 5120;
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Options {
    int gate_up_rows         = 34816;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::vector<std::int32_t> t_sweep{1, 4, 8, 16};
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
        if (argument == "--rows") {
            options.gate_up_rows = bench::measurement::integer(next("--rows"));
        } else if (argument == "--policy") {
            const std::string_view value = next("--policy");
            if (value == "a16") {
                options.policy  = ops::LinearPolicy::A16Only;
                options.t_sweep = {1, 4, 8, 16};
            } else if (value == "a4") {
                options.policy  = ops::LinearPolicy::AllowA4;
                options.t_sweep = {5, 48, 49, 56, 64, 65, 96, 97, 128, 256, 1024};
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
            std::printf("Usage: %s --policy a16|a4 [--rows N] [--t-sweep 1,4,...] [--warmup N] "
                        "[--repeat N] "
                        "[--profile] [--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.gate_up_rows != 34816
#if NINFER_TARGET_RTX5090
        && options.gate_up_rows != 28672
#endif
    )
        throw std::invalid_argument("unsupported gate/up rows");
    if (options.profile && options.t_sweep.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    if (options.policy == ops::LinearPolicy::A16Only &&
        *std::max_element(options.t_sweep.begin(), options.t_sweep.end()) > 16) {
        throw std::invalid_argument("A16 policy is registered only through T=16");
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
        out << "linear_swiglu,NVFP4," << policy_name(options.policy) << ',' << options.gate_up_rows
            << ',' << kHidden << ',' << result.tokens << ',' << weight_bytes << ','
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
        const int kGateUpRows = options.gate_up_rows, kOutputRows = kGateUpRows / 2;
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());
        const std::int32_t min_t =
            *std::min_element(options.t_sweep.begin(), options.t_sweep.end());
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        bench::L2FlushBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t, 101U);
        DeviceBuffer output(ninfer::bench::device_context().device(),
                            static_cast<std::size_t>(kOutputRows) * max_t * sizeof(std::uint16_t));
        bench::PackedQuantizedWeight packed  = bench::make_nvfp4_weight(kGateUpRows, kHidden);
        const std::size_t workspace_capacity = ops::linear_swiglu_workspace_capacity_bytes(
            QType::NVFP4, kGateUpRows, kHidden, options.policy, min_t, max_t);
        ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                               std::max<std::size_t>(workspace_capacity, 256));
        MemoryArena workspace(workspace_storage.span());

        const auto make_launch = [&](std::int32_t tokens) {
            return [&, tokens](CudaExecutionView cuda) {
                Tensor x(input.span(), DType::BF16, {kHidden, tokens});
                Tensor out(output.span(), DType::BF16, {kOutputRows, tokens});
                ops::linear_swiglu(x, packed.weight, out, options.policy, workspace, cuda);
            };
        };

        if (options.profile) {
            const std::int32_t tokens = options.t_sweep.front();
            const auto launch         = make_launch(tokens);
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                bench::flush_l2(flush, cuda);
                launch(cuda);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench::flush_l2(flush, cuda);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_swiglu weight_type=NVFP4 policy=%s T=%d\n",
                        policy_name(options.policy), tokens);
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
        std::printf("%-14s %3s %8s %8s %6s %11s %11s %11s %10s %10s %8s\n", "op", "pol", "N", "K",
                    "T", "median_us", "min_us", "p95_us", "eff_GB/s", "TFLOP/s", "NVFP4_%");
        for (const std::int32_t tokens : options.t_sweep) {
            const auto launch = make_launch(tokens);
            const bench::ColdTiming timing =
                bench::measure_cold_launch(launch, flush, cuda, options.warmup, options.repeat);
            const double seconds      = timing.median_us * 1.0e-6;
            const double useful_flops = 2.0 * static_cast<double>(kGateUpRows) * kHidden * tokens;
            const double model_bytes  = static_cast<double>(packed.model_weight_bytes()) +
                                       2.0 * static_cast<double>(kHidden + kOutputRows) * tokens;
            const double useful_tflops = useful_flops / seconds / 1.0e12;
            const double effective_gbs = model_bytes / seconds / 1.0e9;
            const bool a4              = options.policy == ops::LinearPolicy::AllowA4 &&
                            tokens >= (options.gate_up_rows == 28672 ? 8 : 5);
            const double nvfp4_percent =
                a4 ? 100.0 * useful_tflops / bench::kRoofline.nvfp4_f32_tflops
                   : std::numeric_limits<double>::quiet_NaN();
            std::printf("%-14s %3s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %10.2f %8s\n",
                        "linear_swiglu", policy_name(options.policy), kGateUpRows, kHidden, tokens,
                        timing.median_us, timing.min_us, timing.p95_us, effective_gbs,
                        useful_tflops, bench::format_metric(nvfp4_percent).c_str());
            results.push_back({tokens, timing, effective_gbs, useful_tflops, nvfp4_percent});
        }
        write_csv(options, results, packed.model_weight_bytes());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_nvfp4_linear_swiglu_bench: %s\n", error.what());
        return 1;
    }
}
