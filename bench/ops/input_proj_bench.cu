// Fixed-shape production/control benchmark for the 27B Attention and GDN input projections.
// Controls reproduce only the superseded Small-T compositions and are intentionally benchmark-
// local: production dispatch has no fallback to them.

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/scatter.h"

#include "core/device.h"
#include "ninfer_bench_common.h"

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

constexpr std::int32_t kHidden         = 5120;
constexpr std::int32_t kQueryRows      = 6144;
constexpr std::int32_t kKvRows         = 1024;
constexpr std::int32_t kParentRows     = kQueryRows + kKvRows;
constexpr std::int32_t kGdnQkRows      = 4096;
constexpr std::int32_t kGdnValueRows   = 6144;
constexpr std::int32_t kGdnRows        = kGdnQkRows + kGdnValueRows;
constexpr std::int32_t kGdnKeyRows     = 2048;
constexpr std::int32_t kGdnSlots       = 7;
constexpr std::int32_t kMaxBenchTokens = 16384;
constexpr std::size_t kFlushBytes      = 256ULL << 20;

enum class OpSelection { All, Attention, Gdn };

struct Options {
    OpSelection op = OpSelection::All;
    std::vector<std::int32_t> t_sweep{1,  2,  3,  4,  5,  6,  7,  8,   9,   10,
                                      11, 12, 13, 14, 15, 16, 17, 128, 129, 1024};
    int warmup = 5;
    int repeat = 50;
    std::string csv_out;
};

struct Stats {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

struct Result {
    std::string op;
    std::string path;
    std::int32_t t = 0;
    Stats timing;
};

struct DevicePackedWeight {
    explicit DevicePackedWeight(std::size_t bytes) : storage(bytes) {}

    bench::DBuf storage;
    Weight weight{};
};

__global__ void fill_scales(std::uint8_t* scales, std::uint64_t groups) {
    const std::uint64_t start  = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t group = start; group < groups; group += stride) {
        scales[group * 2]     = 0x00u;
        scales[group * 2 + 1] = 0x3cu;
    }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

DevicePackedWeight make_weight(QType qtype, std::int32_t rows) {
    const std::int32_t groups_per_row = kHidden / 64;
    const std::uint64_t groups =
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(groups_per_row);
    const std::uint64_t low_bytes    = groups * 32;
    const std::uint64_t high_bytes   = qtype == QType::Q5G64_F16S ? groups * 8 : 0;
    const std::uint64_t high_offset  = align_up(low_bytes, 256);
    const std::uint64_t scale_offset = high_offset + align_up(high_bytes, 256);
    const std::uint64_t scale_bytes  = groups * 2;
    DevicePackedWeight packed(static_cast<std::size_t>(scale_offset + scale_bytes));

    CUDA_CHECK(cudaMemset(packed.storage.p, 0x53, packed.storage.bytes));
    constexpr int block = 256;
    const int grid      = static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (groups + block - 1) / block)));
    fill_scales<<<grid, block>>>(static_cast<std::uint8_t*>(packed.storage.p) + scale_offset,
                                 groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = packed.weight;
    weight.payload          = packed.storage.p;
    weight.payload_bytes    = packed.storage.bytes;
    weight.high_plane_bytes = high_bytes;
    weight.qtype            = qtype;
    weight.layout           = QuantLayout::RowSplit;
    weight.group_size       = 64;
    weight.qdata            = packed.storage.p;
    weight.qhigh =
        high_bytes == 0 ? nullptr : static_cast<std::uint8_t*>(packed.storage.p) + high_offset;
    weight.scales          = static_cast<std::uint8_t*>(packed.storage.p) + scale_offset;
    weight.n               = rows;
    weight.k               = kHidden;
    weight.group           = 64;
    weight.scale_dtype     = DType::FP16;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = kHidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = kHidden;
    return packed;
}

Weight row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    if (row_begin < 0 || rows <= 0 || row_begin + rows > parent.n) {
        throw std::invalid_argument("benchmark row view is out of range");
    }
    const std::uint64_t groups_per_row = parent.padded_shape[1] / parent.group;
    const std::uint64_t low_row_bytes  = groups_per_row * 32;
    const std::uint64_t high_row_bytes = parent.qtype == QType::Q5G64_F16S ? groups_per_row * 8 : 0;
    const std::uint64_t scale_row_bytes = groups_per_row * 2;
    Weight view                         = parent;
    view.qdata                          = static_cast<const std::uint8_t*>(parent.qdata) +
                 static_cast<std::uint64_t>(row_begin) * low_row_bytes;
    view.qhigh  = high_row_bytes == 0 ? nullptr
                                      : static_cast<const std::uint8_t*>(parent.qhigh) +
                                           static_cast<std::uint64_t>(row_begin) * high_row_bytes;
    view.scales = static_cast<const std::uint8_t*>(parent.scales) +
                  static_cast<std::uint64_t>(row_begin) * scale_row_bytes;
    view.n               = rows;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    return view;
}

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        if (token.empty()) { throw std::invalid_argument("empty --t-sweep element"); }
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
        const auto next = [&](const char* name) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + name); }
            return argv[i];
        };
        if (arg == "--op") {
            const std::string_view value = next("--op value");
            if (value == "all") {
                options.op = OpSelection::All;
            } else if (value == "attention") {
                options.op = OpSelection::Attention;
            } else if (value == "gdn") {
                options.op = OpSelection::Gdn;
            } else {
                throw std::invalid_argument("--op must be all, attention, or gdn");
            }
        } else if (arg == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep value"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--op all|attention|gdn] [--t-sweep 1,2,...] "
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

void append_result(std::vector<Result>& results, std::string op, std::string path, std::int32_t t,
                   Stats timing) {
    std::printf("%-9s T=%-3d %-31s median=%8.3f us min=%8.3f us p95=%8.3f us\n", op.c_str(), t,
                path.c_str(), timing.median_us, timing.min_us, timing.p95_us);
    results.push_back({std::move(op), std::move(path), t, timing});
}

void write_csv(const std::string& path, const std::vector<Result>& results,
               const Options& options) {
    if (path.empty()) { return; }
    const std::filesystem::path output(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream stream(output);
    if (!stream) { throw std::runtime_error("failed to open CSV: " + path); }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    stream << "op,path,T,median_us,min_us,p95_us,warmup,repeat,build_type,gpu,cuda_runtime\n";
    for (const Result& result : results) {
        stream << result.op << ',' << result.path << ',' << result.t << ','
               << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << ',' << options.warmup << ',' << options.repeat << ','
#ifdef NDEBUG
               << "Release"
#else
               << "Debug"
#endif
               << ',' << properties.name << ',' << runtime << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());
        if (max_t > kMaxBenchTokens) {
            throw std::invalid_argument("input-projection benchmark allocation limit is T<=16384");
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::DBuf flush(kFlushBytes);
        bench::DBuf input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t);
        const std::size_t workspace_bytes =
            options.op == OpSelection::Attention
                ? 1
                : std::max<std::size_t>(1, ops::gdn_input_proj_conv_snapshot_workspace_bytes(
                                               kGdnKeyRows, kGdnKeyRows, kGdnValueRows, max_t));
        WorkspaceArena workspace(workspace_bytes);
        std::vector<Result> results;

        if (options.op != OpSelection::Gdn) {
            DevicePackedWeight query_key  = make_weight(QType::Q4G64_F16S, kParentRows);
            DevicePackedWeight gate_value = make_weight(QType::Q5G64_F16S, kParentRows);
            const Weight query            = row_view(query_key.weight, 0, kQueryRows);
            const Weight key              = row_view(query_key.weight, kQueryRows, kKvRows);
            const Weight gate             = row_view(gate_value.weight, 0, kQueryRows);
            const Weight value            = row_view(gate_value.weight, kQueryRows, kKvRows);
            bench::DBuf q(static_cast<std::size_t>(kQueryRows) * max_t * 2);
            bench::DBuf g(static_cast<std::size_t>(kQueryRows) * max_t * 2);
            bench::DBuf k(static_cast<std::size_t>(kKvRows) * max_t * 2);
            bench::DBuf v(static_cast<std::size_t>(kKvRows) * max_t * 2);

            for (const std::int32_t t : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {kHidden, t});
                Tensor tq(q.p, DType::BF16, {kQueryRows, t});
                Tensor tg(g.p, DType::BF16, {kQueryRows, t});
                Tensor tk(k.p, DType::BF16, {kKvRows, t});
                Tensor tv(v.p, DType::BF16, {kKvRows, t});
                const auto production = [&](cudaStream_t launch_stream) {
                    ops::attn_input_proj(x, query_key.weight, gate_value.weight, tq, tg, tk, tv,
                                         workspace, launch_stream);
                };
                append_result(
                    results, "attention", "production_parent_split", t,
                    measure_cold(production, flush, stream, options.warmup, options.repeat));
                if (t <= 16) {
                    const auto control = [&](cudaStream_t launch_stream) {
                        ops::linear(x, query, tq, workspace, launch_stream);
                        ops::linear(x, gate, tg, workspace, launch_stream);
                        ops::linear(x, key, tk, workspace, launch_stream);
                        ops::linear(x, value, tv, workspace, launch_stream);
                    };
                    append_result(
                        results, "attention", "control_four_projection", t,
                        measure_cold(control, flush, stream, options.warmup, options.repeat));
                }
            }
        }

        if (options.op != OpSelection::Attention) {
            DevicePackedWeight qk_weight    = make_weight(QType::Q4G64_F16S, kGdnQkRows);
            DevicePackedWeight value_weight = make_weight(QType::Q5G64_F16S, kGdnValueRows);
            bench::DBuf qkv(static_cast<std::size_t>(kGdnRows) * max_t * 2);
            bench::DBuf qkv_conv(static_cast<std::size_t>(kGdnRows) * max_t * 2);
            bench::DBuf qk_tmp(static_cast<std::size_t>(kGdnQkRows) * max_t * 2);
            bench::DBuf value_tmp(static_cast<std::size_t>(kGdnValueRows) * max_t * 2);
            bench::DBuf query(static_cast<std::size_t>(kGdnKeyRows) * max_t * 2);
            bench::DBuf key(static_cast<std::size_t>(kGdnKeyRows) * max_t * 2);
            bench::DBuf value_out(static_cast<std::size_t>(kGdnValueRows) * max_t * 2);
            bench::DBuf conv_weight = bench::make_bf16(static_cast<std::size_t>(kGdnRows) * 4);
            bench::DBuf conv_states =
                bench::make_zeros(static_cast<std::size_t>(kGdnRows) * 3 * kGdnSlots * 2);
            bench::DBuf initial_slot(sizeof(std::int32_t));
            constexpr std::int32_t kInitialSlot = 6;
            CUDA_CHECK(cudaMemcpy(initial_slot.p, &kInitialSlot, sizeof(kInitialSlot),
                                  cudaMemcpyHostToDevice));

            for (const std::int32_t t : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {kHidden, t});
                Tensor out(qkv.p, DType::BF16, {kGdnRows, t});
                Tensor convolved(qkv_conv.p, DType::BF16, {kGdnRows, t});
                Tensor qk(qk_tmp.p, DType::BF16, {kGdnQkRows, t});
                Tensor value(value_tmp.p, DType::BF16, {kGdnValueRows, t});
                Tensor tq(query.p, DType::BF16, {kGdnKeyRows, t});
                Tensor tk(key.p, DType::BF16, {kGdnKeyRows, t});
                Tensor tv(value_out.p, DType::BF16, {kGdnValueRows, t});
                Tensor conv_w(conv_weight.p, DType::BF16, {kGdnRows, 4});
                Tensor states(conv_states.p, DType::BF16, {kGdnRows, 3, kGdnSlots});
                Tensor initial(initial_slot.p, DType::I32, {1});
                const auto production = [&](cudaStream_t launch_stream) {
                    ops::gdn_input_proj(x, qk_weight.weight, value_weight.weight, out, workspace,
                                        launch_stream);
                };
                append_result(
                    results, "gdn", "production_direct", t,
                    measure_cold(production, flush, stream, options.warmup, options.repeat));
                if (t <= 6) {
                    const auto fused_snapshot = [&](cudaStream_t launch_stream) {
                        ops::gdn_input_proj_conv_snapshot(x, qk_weight.weight, value_weight.weight,
                                                          conv_w, states, initial, tq, tk, tv,
                                                          workspace, launch_stream);
                    };
                    append_result(results, "gdn", "fused_projection_conv_snapshot", t,
                                  measure_cold(fused_snapshot, flush, stream, options.warmup,
                                               options.repeat));
                    const auto composed_snapshot = [&](cudaStream_t launch_stream) {
                        ops::gdn_input_proj(x, qk_weight.weight, value_weight.weight, out,
                                            workspace, launch_stream);
                        ops::causal_conv1d_silu_snapshot(out, conv_w, states, initial, convolved,
                                                         launch_stream);
                        ops::extract_bf16_columns(convolved, 0, tq, launch_stream);
                        ops::extract_bf16_columns(convolved, kGdnKeyRows, tk, launch_stream);
                        ops::extract_bf16_columns(convolved, 2 * kGdnKeyRows, tv, launch_stream);
                    };
                    append_result(results, "gdn", "composed_projection_conv_snapshot", t,
                                  measure_cold(composed_snapshot, flush, stream, options.warmup,
                                               options.repeat));
                }
                if (t <= 16) {
                    const auto projections = [&](cudaStream_t launch_stream) {
                        ops::linear(x, qk_weight.weight, qk, workspace, launch_stream);
                        ops::linear(x, value_weight.weight, value, workspace, launch_stream);
                    };
                    append_result(
                        results, "gdn", "control_projection_only", t,
                        measure_cold(projections, flush, stream, options.warmup, options.repeat));
                    const auto materialize_copy = [&](cudaStream_t launch_stream) {
                        projections(launch_stream);
                        CUDA_CHECK(cudaMemcpy2DAsync(out.data, out.nb[1], qk.data, qk.nb[1],
                                                     static_cast<std::size_t>(kGdnQkRows) * 2, t,
                                                     cudaMemcpyDeviceToDevice, launch_stream));
                        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::uint8_t*>(out.data) +
                                                         static_cast<std::size_t>(kGdnQkRows) * 2,
                                                     out.nb[1], value.data, value.nb[1],
                                                     static_cast<std::size_t>(kGdnValueRows) * 2, t,
                                                     cudaMemcpyDeviceToDevice, launch_stream));
                    };
                    append_result(results, "gdn", "control_materialize_copy", t,
                                  measure_cold(materialize_copy, flush, stream, options.warmup,
                                               options.repeat));
                }
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        write_csv(options.csv_out, results, options);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_input_proj_bench: %s\n", error.what());
        return 1;
    }
}
