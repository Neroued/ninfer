// Complete-Op benchmark for Qwen3.6-35B W8 attention/GDN multi-output projections.

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/scatter.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"

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
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden     = 2048;
constexpr std::int32_t kAttnQRows  = 4096;
constexpr std::int32_t kAttnKvRows = 512;
constexpr std::int32_t kAttnRows   = 9216;
constexpr std::int32_t kGdnQkvRows = 8192;
constexpr std::int32_t kGdnZRows   = 4096;
constexpr std::int32_t kGdnRows    = 12288;
constexpr std::size_t kFlushBytes  = 256ULL << 20;

enum class OpSelection { All, Attention, Gdn, GdnSnapshot };

struct Options {
    OpSelection op = OpSelection::All;
    std::vector<std::int32_t> t_sweep{1, 2, 4, 8, 13, 14, 16, 17, 32, 64, 128, 129, 256, 512, 1024};
    int warmup = 5;
    int repeat = 30;
    std::string csv_out;
};

struct Stats {
    double median_us;
    double min_us;
    double p95_us;
};

struct Result {
    std::string op;
    std::string path;
    std::int32_t t;
    Stats timing;
};

struct DevicePackedWeight {
    explicit DevicePackedWeight(std::size_t bytes) : storage(bytes) {}

    bench::DBuf storage;
    Weight weight{};
};

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

DevicePackedWeight make_w8_weight(std::int32_t rows) {
    constexpr std::int32_t group = 32;
    const std::uint64_t groups =
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(kHidden / group);
    const std::uint64_t code_bytes   = groups * group;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = groups * 2;
    DevicePackedWeight packed(static_cast<std::size_t>(scale_offset + scale_bytes));
    CUDA_CHECK(cudaMemset(packed.storage.p, 0x03, code_bytes));
    constexpr int block = 256;
    const int grid      = static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (groups + block - 1) / block)));
    fill_w8_scales<<<grid, block>>>(static_cast<std::uint8_t*>(packed.storage.p) + scale_offset,
                                    groups);
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
    weight.n                = rows;
    weight.k                = kHidden;
    weight.group            = group;
    weight.scale_dtype      = DType::FP16;
    weight.ndim             = 2;
    weight.shape[0]         = rows;
    weight.shape[1]         = kHidden;
    weight.padded_shape[0]  = rows;
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
        if (arg == "--op") {
            const std::string_view value = next("--op value");
            if (value == "all")
                options.op = OpSelection::All;
            else if (value == "attention")
                options.op = OpSelection::Attention;
            else if (value == "gdn")
                options.op = OpSelection::Gdn;
            else if (value == "gdn-snapshot")
                options.op = OpSelection::GdnSnapshot;
            else
                throw std::invalid_argument("--op must be all, attention, gdn, or gdn-snapshot");
        } else if (arg == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep value"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--op all|attention|gdn|gdn-snapshot] "
                        "[--t-sweep 1,2,...] "
                        "[--warmup N] [--repeat N] [--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
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
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
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

void append(std::vector<Result>& results, const char* op, const char* path, std::int32_t t,
            Stats timing, std::int32_t rows) {
    const double tflops = 2.0 * rows * kHidden * static_cast<double>(t) / timing.median_us / 1.0e6;
    std::printf("%-9s T=%-4d %-26s median=%8.3f us  useful=%7.2f TFLOP/s\n", op, t, path,
                timing.median_us, tflops);
    results.push_back({op, path, t, timing});
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    out << "op,path,T,median_us,min_us,p95_us,warmup,repeat\n";
    for (const Result& result : results) {
        out << result.op << ',' << result.path << ',' << result.t << ',' << result.timing.median_us
            << ',' << result.timing.min_us << ',' << result.timing.p95_us << ',' << options.warmup
            << ',' << options.repeat << '\n';
    }
}

ops::detail::W8KernelVariant simt_variant(std::int32_t t) {
    return (t % 4) == 0 ? ops::detail::W8KernelVariant::Full
                        : ops::detail::W8KernelVariant::Predicated;
}

ops::detail::W8KernelVariant mma_variant(std::int32_t t) {
    return (t % 128) == 0 ? ops::detail::W8KernelVariant::Full
                          : ops::detail::W8KernelVariant::Predicated;
}

bench::DBuf make_i32(std::int32_t value) {
    bench::DBuf device(sizeof(value));
    CUDA_CHECK(cudaMemcpy(device.p, &value, sizeof(value), cudaMemcpyHostToDevice));
    return device;
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
        WorkspaceArena workspace(1);
        std::vector<Result> results;

        if (options.op == OpSelection::All || options.op == OpSelection::Attention) {
            DevicePackedWeight packed = make_w8_weight(kAttnRows);
            bench::DBuf parent(static_cast<std::size_t>(kAttnRows) * max_t * 2);
            bench::DBuf q(static_cast<std::size_t>(kAttnQRows) * max_t * 2);
            bench::DBuf gate(static_cast<std::size_t>(kAttnQRows) * max_t * 2);
            bench::DBuf k(static_cast<std::size_t>(kAttnKvRows) * max_t * 2);
            bench::DBuf v(static_cast<std::size_t>(kAttnKvRows) * max_t * 2);
            for (const std::int32_t t : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {kHidden, t});
                Tensor out_parent(parent.p, DType::BF16, {kAttnRows, t});
                Tensor tq(q.p, DType::BF16, {kAttnQRows, t});
                Tensor tg(gate.p, DType::BF16, {kAttnQRows, t});
                Tensor tk(k.p, DType::BF16, {kAttnKvRows, t});
                Tensor tv(v.p, DType::BF16, {kAttnKvRows, t});
                const auto run = [&](const char* path, auto&& launch) {
                    append(results, "attention", path, t,
                           measure_cold(launch, flush, stream, options.warmup, options.repeat),
                           kAttnRows);
                };
                run("production", [&](cudaStream_t s) {
                    ops::attn_input_proj(x, packed.weight, tq, tg, tk, tv, workspace, s);
                });
                if (t == 1) {
                    run("decode_direct", [&](cudaStream_t s) {
                        ops::detail::w8_attn_input_decode_launch(x, packed.weight, tq, tg, tk, tv,
                                                                 s);
                    });
                }
                run("simt_r8_c4", [&](cudaStream_t s) {
                    ops::detail::w8_attn_input_simt_r8_c4_launch(simt_variant(t), x, packed.weight,
                                                                 tq, tg, tk, tv, s);
                });
                if (t >= 2 && t <= 16) {
                    run("splitk_mma_exact_t", [&](cudaStream_t s) {
                        ops::detail::w8_attn_input_splitk_mma_launch(
                            ops::detail::W8KernelVariant::None, x, packed.weight, tq, tg, tk, tv,
                            s);
                    });
                }
                run("mma_r32_c128", [&](cudaStream_t s) {
                    ops::detail::w8_attn_input_mma_r32_c128_launch(mma_variant(t), x, packed.weight,
                                                                   tq, tg, tk, tv, s);
                });
                run("mma_r64_c128", [&](cudaStream_t s) {
                    ops::detail::w8_attn_input_mma_r64_c128_launch(mma_variant(t), x, packed.weight,
                                                                   tq, tg, tk, tv, s);
                });
                run("parent_linear", [&](cudaStream_t s) {
                    ops::linear(x, packed.weight, out_parent, workspace, s);
                });
                run("parent_plus_extract", [&](cudaStream_t s) {
                    ops::linear(x, packed.weight, out_parent, workspace, s);
                    ops::extract_bf16_columns(out_parent, 0, tq, s);
                    ops::extract_bf16_columns(out_parent, 4096, tk, s);
                    ops::extract_bf16_columns(out_parent, 4608, tg, s);
                    ops::extract_bf16_columns(out_parent, 8704, tv, s);
                });
            }
        }

        if (options.op == OpSelection::All || options.op == OpSelection::Gdn) {
            DevicePackedWeight packed = make_w8_weight(kGdnRows);
            bench::DBuf parent(static_cast<std::size_t>(kGdnRows) * max_t * 2);
            bench::DBuf qkv(static_cast<std::size_t>(kGdnQkvRows) * max_t * 2);
            bench::DBuf z(static_cast<std::size_t>(kGdnZRows) * max_t * 2);
            for (const std::int32_t t : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {kHidden, t});
                Tensor out_parent(parent.p, DType::BF16, {kGdnRows, t});
                Tensor tqkv(qkv.p, DType::BF16, {kGdnQkvRows, t});
                Tensor tz(z.p, DType::BF16, {kGdnZRows, t});
                const auto run = [&](const char* path, auto&& launch) {
                    append(results, "gdn", path, t,
                           measure_cold(launch, flush, stream, options.warmup, options.repeat),
                           kGdnRows);
                };
                run("production", [&](cudaStream_t s) {
                    ops::gdn_input_proj(x, packed.weight, tqkv, tz, workspace, s);
                });
                if (t == 1) {
                    run("decode_direct", [&](cudaStream_t s) {
                        ops::detail::w8_gdn_input_decode_launch(x, packed.weight, tqkv, tz, s);
                    });
                }
                if (t >= 2 && t <= 96) {
                    run("small_t_splitk_mma", [&](cudaStream_t s) {
                        ops::detail::w8_gdn_input_splitk_mma_launch(
                            ops::detail::W8KernelVariant::None, x, packed.weight, tqkv, tz, s);
                    });
                }
                run("mma_r64_c128", [&](cudaStream_t s) {
                    ops::detail::w8_gdn_input_mma_r64_c128_launch(mma_variant(t), x, packed.weight,
                                                                  tqkv, tz, s);
                });
                run("parent_linear", [&](cudaStream_t s) {
                    ops::linear(x, packed.weight, out_parent, workspace, s);
                });
                run("parent_plus_extract", [&](cudaStream_t s) {
                    ops::linear(x, packed.weight, out_parent, workspace, s);
                    ops::extract_bf16_columns(out_parent, 0, tqkv, s);
                    ops::extract_bf16_columns(out_parent, 8192, tz, s);
                });
            }
        }

        if (options.op == OpSelection::All || options.op == OpSelection::GdnSnapshot) {
            constexpr std::int32_t kQueryRows = 2048;
            constexpr std::int32_t kKeyRows   = 2048;
            constexpr std::int32_t kValueRows = 4096;
            constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
            const std::int32_t slots          = max_t + 1;
            DevicePackedWeight packed         = make_w8_weight(kGdnRows);
            bench::DBuf conv_weight = bench::make_bf16(static_cast<std::size_t>(kChannels) * 4);
            bench::DBuf conv_states =
                bench::make_bf16(static_cast<std::size_t>(kChannels) * 3 * slots);
            bench::DBuf initial_slot = make_i32(max_t);
            bench::DBuf q(static_cast<std::size_t>(kQueryRows) * max_t * 2);
            bench::DBuf k(static_cast<std::size_t>(kKeyRows) * max_t * 2);
            bench::DBuf v(static_cast<std::size_t>(kValueRows) * max_t * 2);
            bench::DBuf z(static_cast<std::size_t>(kValueRows) * max_t * 2);
            bench::DBuf qkv(static_cast<std::size_t>(kChannels) * max_t * 2);
            bench::DBuf convolved(static_cast<std::size_t>(kChannels) * max_t * 2);
            WorkspaceArena snapshot_workspace(
                std::max<std::size_t>(1, ops::gdn_input_proj_conv_snapshot_workspace_bytes(
                                             kQueryRows, kKeyRows, kValueRows, max_t)));

            for (const std::int32_t t : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {kHidden, t});
                Tensor tw(conv_weight.p, DType::BF16, {kChannels, 4});
                Tensor states(conv_states.p, DType::BF16, {kChannels, 3, slots});
                Tensor slot(initial_slot.p, DType::I32, {1});
                Tensor tq(q.p, DType::BF16, {kQueryRows, t});
                Tensor tk(k.p, DType::BF16, {kKeyRows, t});
                Tensor tv(v.p, DType::BF16, {kValueRows, t});
                Tensor tz(z.p, DType::BF16, {kValueRows, t});
                Tensor tqkv(qkv.p, DType::BF16, {kChannels, t});
                Tensor tconvolved(convolved.p, DType::BF16, {kChannels, t});
                const auto run = [&](const char* path, auto&& launch) {
                    append(results, "gdn_snap", path, t,
                           measure_cold(launch, flush, stream, options.warmup, options.repeat),
                           kGdnRows);
                };

                const auto snapshot_plan = ops::detail::w8_gdn_input_snapshot_resolve_plan(
                    {kHidden, kChannels, kValueRows, kGdnRows, kHidden, t});
                run(ops::detail::w8_gdn_input_snapshot_schedule_name(snapshot_plan.schedule),
                    [&](cudaStream_t s) {
                        ops::gdn_input_proj_conv_snapshot(x, packed.weight, tw, states, slot, tq,
                                                          tk, tv, tz, snapshot_workspace, s);
                    });
                run("composed_control", [&](cudaStream_t s) {
                    ops::gdn_input_proj(x, packed.weight, tqkv, tz, workspace, s);
                    ops::causal_conv1d_silu_snapshot(tqkv, tw, states, slot, tconvolved, s);
                    ops::extract_bf16_columns(tconvolved, 0, tq, s);
                    ops::extract_bf16_columns(tconvolved, kQueryRows, tk, s);
                    ops::extract_bf16_columns(tconvolved, kQueryRows + kKeyRows, tv, s);
                });
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_w8_input_proj_bench: %s\n", error.what());
        return 1;
    }
}
