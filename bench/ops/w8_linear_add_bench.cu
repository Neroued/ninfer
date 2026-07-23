// Cold-cache benchmark and candidate tuner for the Qwen3.6-35B-A3B W8 LinearAdd Op.

#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/residual_add.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"
#include "ops/linear_add/w8/w8_linear_add_kernels.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"

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

constexpr std::int32_t kRows      = 2048;
constexpr std::int32_t kHidden    = 6144;
constexpr std::size_t kFlushBytes = 256ULL << 20;
constexpr std::int32_t kGroupSize = 32;

struct Options {
    std::vector<std::int32_t> t_sweep{
        1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,
        24,  31,  32,  33,  48,  63,  64,  65,  80,  95,  96,  97,  112, 127, 128, 129, 160,
        192, 224, 256, 320, 384, 448, 512, 576, 640, 641, 704, 768, 832, 896, 960, 1024};
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
    const std::uint64_t code_bytes =
        static_cast<std::uint64_t>(kRows) * static_cast<std::uint64_t>(kHidden);
    const std::uint64_t groups       = code_bytes / kGroupSize;
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
    weight.group_size       = kGroupSize;
    weight.qdata            = packed.storage.p;
    weight.qhigh            = nullptr;
    weight.scales           = static_cast<std::uint8_t*>(packed.storage.p) + scale_offset;
    weight.n                = kRows;
    weight.k                = kHidden;
    weight.group            = kGroupSize;
    weight.scale_dtype      = DType::FP16;
    weight.ndim             = 2;
    weight.shape[0]         = kRows;
    weight.shape[1]         = kHidden;
    weight.padded_shape[0]  = kRows;
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
    const double useful_flops = 2.0 * kRows * kHidden * static_cast<double>(t);
    const double useful_bytes =
        static_cast<double>(weight_bytes) + 2.0 * static_cast<double>((kHidden + 2 * kRows) * t);
    const double tflops = useful_flops / timing.median_us / 1.0e6;
    const double gbs    = useful_bytes / timing.median_us / 1.0e3;
    std::printf("T=%-4d %-32s median=%8.3f us  useful=%7.2f TFLOP/s %7.1f GB/s\n", t, path,
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

ops::detail::W8KernelVariant variant_for(std::int32_t t, std::int32_t tile_cols,
                                         std::int32_t tile_rows = 32) {
    return (t % tile_cols) == 0 && (kRows % tile_rows) == 0
               ? ops::detail::W8KernelVariant::Full
               : ops::detail::W8KernelVariant::Predicated;
}

void launch_composed_control(const Tensor& x, const Weight& weight, Tensor& projected,
                             Tensor& residual, cudaStream_t stream) {
    using S = ops::detail::W8ScheduleId;
    S schedule;
    int tile_cols;
    if (x.ne[1] <= 32) {
        schedule  = S::SimtR8C4;
        tile_cols = 4;
    } else if (x.ne[1] <= 640) {
        schedule  = S::MmaR32C128;
        tile_cols = 128;
    } else {
        schedule  = S::MmaR64C128;
        tile_cols = 128;
    }
    ops::detail::w8_rowsplit_launch_candidate(schedule, variant_for(x.ne[1], tile_cols), x, weight,
                                              projected, stream);
    ops::residual_add(projected, residual, stream);
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
        bench::DBuf input    = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t);
        bench::DBuf residual = bench::make_bf16(static_cast<std::size_t>(kRows) * max_t);
        bench::DBuf projected(static_cast<std::size_t>(kRows) * max_t * 2);
        DevicePackedWeight packed = make_w8_weight();
        WorkspaceArena workspace(1);
        std::vector<Result> results;

        for (const std::int32_t t : options.t_sweep) {
            Tensor x(input.p, DType::BF16, {kHidden, t});
            Tensor out(residual.p, DType::BF16, {kRows, t});
            Tensor parent(projected.p, DType::BF16, {kRows, t});
            const auto plan = ops::detail::w8_linear_add_resolve_plan({kRows, kHidden, kHidden, t});

            if (options.profile) {
                ops::linear_add(x, packed.weight, out, workspace, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::printf("profile T=%d route=%s\n", t,
                            ops::detail::w8_linear_add_schedule_name(plan.schedule));
                continue;
            }

            const auto run = [&](const char* path, auto&& launch) {
                append(results, path, t,
                       measure_cold(launch, flush, stream, options.warmup, options.repeat),
                       packed.storage.bytes);
            };
            run(ops::detail::w8_linear_add_schedule_name(plan.schedule),
                [&](cudaStream_t candidate_stream) {
                    ops::linear_add(x, packed.weight, out, workspace, candidate_stream);
                });
            run("composed.projection+residual_add", [&](cudaStream_t candidate_stream) {
                launch_composed_control(x, packed.weight, parent, out, candidate_stream);
            });
            if (options.production_only) { continue; }

            if (t == 1) {
                run("decode_r4", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_add_decode_r4_launch(x, packed.weight, out,
                                                                candidate_stream);
                });
                run("decode_r8", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_add_decode_r8_launch(x, packed.weight, out,
                                                                candidate_stream);
                });
                run("decode_r16", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_add_decode_r16_launch(x, packed.weight, out,
                                                                 candidate_stream);
                });
            }
            run("simt_r8_c4", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_add_simt_r8_c4_launch(variant_for(t, 4), x, packed.weight,
                                                             out, candidate_stream);
            });
            run("simt_r8_c8", [&](cudaStream_t candidate_stream) {
                ops::detail::w8_linear_add_simt_r8_c8_launch(variant_for(t, 8), x, packed.weight,
                                                             out, candidate_stream);
            });
            if (t >= 2 && t <= 32) {
                run("splitk_mma_exact_t", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_add_splitk_mma_launch(ops::detail::W8KernelVariant::None,
                                                                 x, packed.weight, out,
                                                                 candidate_stream);
                });
            }
            if (t >= 33 && t <= 65) {
                run("splitk_mma_32_plus_tail", [&](cudaStream_t candidate_stream) {
                    ops::detail::w8_linear_add_splitk_mma_composite_launch(x, packed.weight, out,
                                                                           candidate_stream);
                });
            }
#define RUN_MMA(NAME, TILE_ROWS, TILE_COLS)                                                        \
    run(#NAME, [&](cudaStream_t candidate_stream) {                                                \
        ops::detail::w8_linear_add_##NAME##_launch(variant_for(t, TILE_COLS, TILE_ROWS), x,        \
                                                   packed.weight, out, candidate_stream);          \
    })
            RUN_MMA(mma_r32_c32, 32, 32);
            RUN_MMA(mma_r32_c48, 32, 48);
            RUN_MMA(mma_r32_c64, 32, 64);
            RUN_MMA(mma_r32_c80, 32, 80);
            RUN_MMA(mma_r32_c96, 32, 96);
            RUN_MMA(mma_r32_c112, 32, 112);
            RUN_MMA(mma_r32_c128, 32, 128);
            RUN_MMA(mma_r48_c64, 48, 64);
            RUN_MMA(mma_r48_c80, 48, 80);
            RUN_MMA(mma_r48_c96, 48, 96);
            RUN_MMA(mma_r48_c112, 48, 112);
            RUN_MMA(mma_r48_c128, 48, 128);
            RUN_MMA(mma_r64_c32, 64, 32);
            RUN_MMA(mma_r64_c48, 64, 48);
            RUN_MMA(mma_r64_c64, 64, 64);
            RUN_MMA(mma_r64_c80, 64, 80);
            RUN_MMA(mma_r64_c96, 64, 96);
            RUN_MMA(mma_r64_c112, 64, 112);
            RUN_MMA(mma_r64_c128, 64, 128);
            RUN_MMA(mma_r128_c64, 128, 64);
            RUN_MMA(mma_r128_c80, 128, 80);
#undef RUN_MMA
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_w8_linear_add_bench: %s\n", error.what());
        return 1;
    }
}
