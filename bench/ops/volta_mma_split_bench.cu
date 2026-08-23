// Split-K tuning bench for the Volta fused-dequant tensor-core GEMMs (sm_70 only).
//
// The public op benches go through the registered plans and so can only ever show the split
// count the table already picked. This one calls launch_q{4,5}_volta_mma directly with an
// explicit `splits_override`, at an arbitrary (n, k) rather than one registered profile's, so
// the split table can be re-derived per exact production shape.
//
// It reports two numbers per point:
//
//   total   the whole launcher -- memset of the fp32 workspace, the main kernel, the
//           narrowing pass. This is the figure the table is chosen on.
//   fixed   memset + narrowing alone, measured from an equivalent pair on the same buffers.
//           Independent of the split count, so it bounds what removing the global split-K
//           accumulator (rather than retuning it) could win.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Options {
    std::string qtype = "q4";
    std::int32_t n    = 0;
    std::int32_t k    = 0;
    std::vector<std::int32_t> tokens{12, 16, 24, 32};
    std::vector<int> splits{1, 2, 4, 8, 16};
    int warmup = 5;
    int repeat = 30;
};

std::vector<std::int32_t> parse_ints(std::string_view raw, const char* label) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument(std::string(label) + " values must be positive int32");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument(std::string(label) + " must not be empty"); }
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
        if (argument == "--qtype") {
            options.qtype = std::string(next("--qtype value"));
        } else if (argument == "--n") {
            options.n = std::stoi(std::string(next("--n value")));
        } else if (argument == "--k") {
            options.k = std::stoi(std::string(next("--k value")));
        } else if (argument == "--t-sweep") {
            options.tokens = parse_ints(next("--t-sweep value"), "--t-sweep");
        } else if (argument == "--splits") {
            const auto raw = parse_ints(next("--splits value"), "--splits");
            options.splits.assign(raw.begin(), raw.end());
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s --qtype q4|q5 --n N --k K [--t-sweep 12,16,...] "
                        "[--splits 1,2,4,...] [--warmup N] [--repeat N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.qtype != "q4" && options.qtype != "q5" && options.qtype != "q4qpn") {
        throw std::invalid_argument("--qtype must be q4, q5 or q4qpn");
    }
    if (options.n <= 0 || options.k <= 0) {
        throw std::invalid_argument("--n and --k are required and must be positive");
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    return options;
}

#ifdef NINFER_VOLTA_BUILD
// Stands in for the launcher's narrowing pass so the split-independent cost can be priced on
// its own. Same shape of work: read the fp32 workspace, write BF16 through the column stride.
__global__ void narrow_probe_kernel(const float* __restrict__ partial,
                                    __nv_bfloat16* __restrict__ out, int n, int t) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(n) * t) { return; }
    out[i] = __float2bfloat16(partial[i]);
}
#endif

} // namespace

int main(int argc, char** argv) {
#ifndef NINFER_VOLTA_BUILD
    (void)argc;
    (void)argv;
    std::fprintf(stderr, "ninfer_volta_mma_split_bench: sm_70 build only\n");
    return 1;
#else
    try {
        const Options options = parse_options(argc, argv);
        const bool is_qpn     = options.qtype == "q4qpn";
        const bool is_q4      = options.qtype == "q4" || is_qpn;
        const QType qtype     = is_q4 ? QType::Q4G64_F16S : QType::Q5G64_F16S;
        const std::int32_t max_t =
            *std::max_element(options.tokens.begin(), options.tokens.end());

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(options.k) * max_t);
        DeviceBuffer output(static_cast<std::size_t>(options.n) * max_t * sizeof(__nv_bfloat16));
        bench::PackedQuantizedWeight packed =
            bench::make_row_split_weight(qtype, options.n, options.k, options.k);

        const std::size_t workspace_bytes =
            is_q4 ? ops::detail::q4_volta_mma_workspace_bytes(options.n, options.k, max_t)
                  : ops::detail::q5_volta_mma_workspace_bytes(options.n, options.k, max_t);
        // The forced-split sweep can pick a count the table would not, so size the arena for the
        // accumulator unconditionally rather than trusting the (single-split-aware) query.
        const std::size_t forced_bytes =
            static_cast<std::size_t>(options.n) * max_t * sizeof(float);
        WorkspaceArena workspace(std::max<std::size_t>(forced_bytes, 256));
        DeviceBuffer probe(std::max<std::size_t>(forced_bytes, 256));
        (void)workspace_bytes;

        std::printf("# %s n=%d k=%d  weight=%.1f MiB\n", options.qtype.c_str(), options.n,
                    options.k, static_cast<double>(packed.model_weight_bytes()) / (1 << 20));

        for (const std::int32_t tokens : options.tokens) {
            const double weight_bytes = static_cast<double>(packed.model_weight_bytes());
            double best_us            = 0.0;
            int best_splits           = 0;

            // The quadpair-split-N kernel has no split-K dimension at all -- its CTA's warps
            // split K and reduce in shared memory -- so it is measured once per T, not per split.
            if (is_qpn) {
                if (!ops::detail::q4_volta_qpn_supported(options.n, options.k, tokens)) {
                    std::printf("T=%-3d qpn unsupported\n\n", tokens);
                    continue;
                }
                const auto launch = [&](cudaStream_t launch_stream) {
                    Tensor x(input.p, DType::BF16, {options.k, tokens});
                    Tensor out(output.p, DType::BF16, {options.n, tokens});
                    ops::detail::launch_q4_volta_qpn(x, packed.weight, out, launch_stream);
                };
                const auto timing = bench::measure_cold_launch(launch, flush, stream,
                                                               options.warmup, options.repeat);
                std::printf("T=%-3d qpn         total=%8.2f us  min=%8.2f  %7.1f GB/s\n\n",
                            tokens, timing.median_us, timing.min_us,
                            weight_bytes / (timing.median_us * 1.0e-6) / 1.0e9);
                continue;
            }

            for (const int splits : options.splits) {
                // Each split needs at least one kKStep=32 run of K to itself; skip rather than
                // launch a configuration the kernel's own guard rejects.
                if ((options.k / splits) < 32) { continue; }
                const auto launch = [&](cudaStream_t launch_stream) {
                    Tensor x(input.p, DType::BF16, {options.k, tokens});
                    Tensor out(output.p, DType::BF16, {options.n, tokens});
                    if (is_q4) {
                        ops::detail::launch_q4_volta_mma(x, packed.weight, out, workspace,
                                                         launch_stream, /*weight_row_offset=*/0,
                                                         splits);
                    } else {
                        ops::detail::launch_q5_volta_mma(x, packed.weight, out,
                                                         /*add_residual=*/false,
                                                         /*weight_row_offset=*/0, workspace,
                                                         launch_stream, splits);
                    }
                };
                const auto timing = bench::measure_cold_launch(launch, flush, stream,
                                                               options.warmup, options.repeat);
                if (best_splits == 0 || timing.median_us < best_us) {
                    best_us     = timing.median_us;
                    best_splits = splits;
                }
                std::printf("T=%-3d splits=%-3d total=%8.2f us  min=%8.2f  %7.1f GB/s\n", tokens,
                            splits, timing.median_us, timing.min_us,
                            weight_bytes / (timing.median_us * 1.0e-6) / 1.0e9);
            }

            const std::size_t fixed_bytes =
                static_cast<std::size_t>(options.n) * tokens * sizeof(float);
            const auto fixed = bench::measure_cold_launch(
                [&](cudaStream_t launch_stream) {
                    CUDA_CHECK(cudaMemsetAsync(probe.p, 0, fixed_bytes, launch_stream));
                    const std::int64_t count = static_cast<std::int64_t>(options.n) * tokens;
                    narrow_probe_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                                          launch_stream>>>(
                        static_cast<const float*>(probe.p),
                        static_cast<__nv_bfloat16*>(output.p), options.n, tokens);
                    CUDA_CHECK(cudaGetLastError());
                },
                flush, stream, options.warmup, options.repeat);

            std::printf("T=%-3d best splits=%-3d %8.2f us   fixed(memset+narrow)=%7.2f us "
                        "(%4.1f%% of best)\n\n",
                        tokens, best_splits, best_us, fixed.median_us,
                        100.0 * fixed.median_us / best_us);
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_volta_mma_split_bench: %s\n", error.what());
        return 1;
    }
#endif
}
