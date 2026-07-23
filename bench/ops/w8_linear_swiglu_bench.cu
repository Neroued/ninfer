// Cold-cache benchmark and candidate tuner for the Qwen3.6-35B-A3B W8 LinearSwiGLU Op.

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/silu_mul.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

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

constexpr std::int32_t kGateUpRows = 12288;
constexpr std::int32_t kOutputRows = 6144;
constexpr std::int32_t kHidden     = 2048;
constexpr std::size_t kFlushBytes  = 256ULL << 20;

struct Options {
    std::vector<std::int32_t> t_sweep{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  24,  31,  32,
        33, 48, 63, 64, 65, 80, 95, 96, 97, 127, 128, 129, 192, 256, 384, 512, 768, 896, 1024};
    int warmup = 5;
    int repeat = 30;
    std::string csv_out;
    bool profile         = false;
    bool production_only = false;
};

struct Stats {
    double median_us;
    double min_us;
    double p95_us;
};

struct Result {
    std::string path;
    std::int32_t t;
    Stats timing;
};

struct DevicePackedWeight {
    explicit DevicePackedWeight(std::size_t bytes) : storage(bytes) {}

    bench::DBuf storage;
    Weight weight{};
};

__global__ void fill_w8_codes(std::uint8_t* codes, std::uint64_t count) {
    const std::uint64_t begin = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t step  = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = begin; index < count; index += step) {
        codes[index] = static_cast<std::uint8_t>((index * 17ULL + 31ULL) & 0xffULL);
    }
}

__global__ void fill_w8_scales(std::uint8_t* scales, std::uint64_t groups) {
    const std::uint64_t begin = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t step  = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t group = begin; group < groups; group += step) {
        scales[group * 2]     = 0x00u;
        scales[group * 2 + 1] = 0x3cu;
    }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

DevicePackedWeight make_w8_weight() {
    constexpr std::int32_t group = 32;
    const std::uint64_t code_bytes =
        static_cast<std::uint64_t>(kGateUpRows) * static_cast<std::uint64_t>(kHidden);
    const std::uint64_t groups       = code_bytes / group;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = groups * 2;
    DevicePackedWeight packed(static_cast<std::size_t>(scale_offset + scale_bytes));

    constexpr int block  = 256;
    const int code_grid  = static_cast<int>(std::min<std::uint64_t>(
        65535, std::max<std::uint64_t>(1, (code_bytes + block - 1) / block)));
    const int scale_grid = static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (groups + block - 1) / block)));
    fill_w8_codes<<<code_grid, block>>>(static_cast<std::uint8_t*>(packed.storage.p), code_bytes);
    fill_w8_scales<<<scale_grid, block>>>(
        static_cast<std::uint8_t*>(packed.storage.p) + scale_offset, groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = packed.weight;
    weight.payload          = packed.storage.p;
    weight.payload_bytes    = packed.storage.bytes;
    weight.high_plane_bytes = 0;
    weight.qtype            = QType::W8G32_F16S;
    weight.layout           = QuantLayout::RowSplit;
    weight.group_size       = group;
    weight.qdata            = packed.storage.p;
    weight.qhigh            = nullptr;
    weight.scales           = static_cast<std::uint8_t*>(packed.storage.p) + scale_offset;
    weight.n                = kGateUpRows;
    weight.k                = kHidden;
    weight.group            = group;
    weight.scale_dtype      = DType::FP16;
    weight.ndim             = 2;
    weight.shape[0]         = kGateUpRows;
    weight.shape[1]         = kHidden;
    weight.padded_shape[0]  = kGateUpRows;
    weight.padded_shape[1]  = kHidden;
    return packed;
}

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
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[i];
        };
        if (arg == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep value"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--profile") {
            options.profile = true;
        } else if (arg == "--production-only") {
            options.production_only = true;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--t-sweep 1,2,...] [--warmup N] [--repeat N] "
                        "[--csv-out PATH] [--profile] [--production-only]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.profile && options.t_sweep.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    return options;
}

template <class Launch>
Stats measure_cold(Launch&& launch, bench::DBuf& flush, cudaStream_t stream, int warmup,
                   int repeat) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int index = 0; index < warmup; ++index) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2], samples.front(),
            samples[std::min(samples.size() - 1, static_cast<std::size_t>(0.95 * samples.size()))]};
}

void append(std::vector<Result>& results, const char* path, std::int32_t t, Stats timing,
            std::size_t weight_bytes) {
    const double useful_flops = 2.0 * kGateUpRows * kHidden * static_cast<double>(t);
    const double useful_bytes =
        static_cast<double>(weight_bytes) + 2.0 * static_cast<double>((kHidden + kOutputRows) * t);
    const double tflops = useful_flops / timing.median_us / 1.0e6;
    const double gbs    = useful_bytes / timing.median_us / 1.0e3;
    std::printf("T=%-4d %-24s median=%8.3f us  useful=%7.2f TFLOP/s %7.1f GB/s\n", t, path,
                timing.median_us, tflops, gbs);
    results.push_back({path, t, timing});
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    out << "path,T,median_us,min_us,p95_us,warmup,repeat\n";
    for (const Result& result : results) {
        out << result.path << ',' << result.t << ',' << result.timing.median_us << ','
            << result.timing.min_us << ',' << result.timing.p95_us << ',' << options.warmup << ','
            << options.repeat << '\n';
    }
}

ops::detail::W8KernelVariant variant_for(std::int32_t t, std::int32_t tile_cols) {
    return (t % tile_cols) == 0 ? ops::detail::W8KernelVariant::Full
                                : ops::detail::W8KernelVariant::Predicated;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::DBuf flush(kFlushBytes);
        bench::DBuf input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t);
        bench::DBuf output(static_cast<std::size_t>(kOutputRows) * max_t * 2);
        bench::DBuf gate_up(static_cast<std::size_t>(kGateUpRows) * max_t * 2);
        DevicePackedWeight packed = make_w8_weight();
        WorkspaceArena workspace(1);
        std::vector<Result> results;

        for (const std::int32_t t : options.t_sweep) {
            Tensor x(input.p, DType::BF16, {kHidden, t});
            Tensor out(output.p, DType::BF16, {kOutputRows, t});
            Tensor parent(gate_up.p, DType::BF16, {kGateUpRows, t});
            const auto plan = ops::detail::w8_linear_swiglu_resolve_plan(
                {kGateUpRows, kOutputRows, kHidden, kHidden, t});

            if (options.profile) {
                ops::linear_swiglu(x, packed.weight, out, workspace, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::printf("profile T=%d route=%s\n", t,
                            ops::detail::w8_linear_swiglu_schedule_name(plan.schedule));
                continue;
            }

            const auto run = [&](const char* path, auto&& launch) {
                append(results, path, t,
                       measure_cold(launch, flush, stream, options.warmup, options.repeat),
                       packed.storage.bytes);
            };
            run(ops::detail::w8_linear_swiglu_schedule_name(plan.schedule),
                [&](cudaStream_t candidate_stream) {
                    ops::linear_swiglu(x, packed.weight, out, workspace, candidate_stream);
                });
            if (options.production_only) { continue; }

            run("composed.linear+silu", [&](cudaStream_t candidate_stream) {
                ops::linear(x, packed.weight, parent, workspace, candidate_stream);
                ops::silu_mul(parent.slice(0, 0, kOutputRows),
                              parent.slice(0, kOutputRows, kOutputRows), out, candidate_stream);
            });
            if (t == 1) {
                run("decode_pair_r4", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_swiglu_decode_pair_r4_launch(x, packed.weight, out,
                                                                        candidate_stream);
                });
                run("decode_pair_r8", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_swiglu_decode_pair_launch(x, packed.weight, out,
                                                                     candidate_stream);
                });
                run("decode_pair_r16", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_swiglu_decode_pair_r16_launch(x, packed.weight, out,
                                                                         candidate_stream);
                });
            }
            if (t >= 2 && t <= 32) {
                run("splitk_mma_exact_t", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_swiglu_splitk_exact_t_launch(
                        ops::detail::W8KernelVariant::None, x, packed.weight, out,
                        candidate_stream);
                });
            }
            run("mma_r32_c32", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r32_c32_launch(
                    variant_for(t, 32), x, packed.weight, out, candidate_stream);
            });
            run("mma_r32_c48", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r32_c48_launch(
                    variant_for(t, 48), x, packed.weight, out, candidate_stream);
            });
            run("mma_r32_c64", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r32_c64_launch(
                    variant_for(t, 64), x, packed.weight, out, candidate_stream);
            });
            run("mma_r32_c80", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r32_c80_launch(
                    variant_for(t, 80), x, packed.weight, out, candidate_stream);
            });
            run("mma_r32_c96", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r32_c96_launch(
                    variant_for(t, 96), x, packed.weight, out, candidate_stream);
            });
            run("mma_r32_c128", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r32_c128_launch(
                    variant_for(t, 128), x, packed.weight, out, candidate_stream);
            });
            run("mma_r64_c32", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r64_c32_launch(
                    variant_for(t, 32), x, packed.weight, out, candidate_stream);
            });
            run("mma_r64_c48", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r64_c48_launch(
                    variant_for(t, 48), x, packed.weight, out, candidate_stream);
            });
            run("mma_r64_c64", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r64_c64_launch(
                    variant_for(t, 64), x, packed.weight, out, candidate_stream);
            });
            run("mma_r64_c80", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r64_c80_launch(
                    variant_for(t, 80), x, packed.weight, out, candidate_stream);
            });
            run("mma_r64_c96", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r64_c96_launch(
                    variant_for(t, 96), x, packed.weight, out, candidate_stream);
            });
            run("mma_r64_c128", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r64_c128_launch(
                    variant_for(t, 128), x, packed.weight, out, candidate_stream);
            });
            run("mma_r128_c64", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r128_c64_launch(
                    variant_for(t, 64), x, packed.weight, out, candidate_stream);
            });
            run("mma_r128_c80", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_swiglu_mma_r128_c80_launch(
                    variant_for(t, 80), x, packed.weight, out, candidate_stream);
            });
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_w8_linear_swiglu_bench: %s\n", error.what());
        return 1;
    }
}
