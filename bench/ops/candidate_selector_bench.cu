// Cold-cache CUDA Graph benchmark for candidate_selector_path.

#include "ninfer/ops/candidate_selector.h"
#include "core/buffer.h"

#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kCandidates       = 16;
constexpr std::int32_t kMaxSteps         = 15;
constexpr std::int32_t kRank             = 256;
constexpr std::int32_t kCodebookRows     = 248320;
constexpr std::int32_t kTokenDomain      = 248077;
constexpr std::size_t kDefaultFlushBytes = 256ULL << 20;

enum class Mode : std::uint8_t {
    Greedy,
    Stochastic,
    Mixed,
};

struct Options {
    std::int32_t batch_size = 0;
    int steps               = 0;
    int warmup              = 8;
    int repeat              = 50;
    std::size_t flush_bytes = kDefaultFlushBytes;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const auto next = [&](const char* label) -> const char* {
            if (index + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[++index];
        };
        if (!std::strcmp(argv[index], "--steps")) {
            options.steps = measurement::integer(next("steps"), 0);
        } else if (!std::strcmp(argv[index], "--batch")) {
            options.batch_size = measurement::integer(next("batch"), 0);
        } else if (!std::strcmp(argv[index], "--warmup")) {
            options.warmup = measurement::integer(next("warmup"), 0);
        } else if (!std::strcmp(argv[index], "--repeat")) {
            options.repeat = measurement::integer(next("repeat"));
        } else if (!std::strcmp(argv[index], "--flush-mib")) {
            const auto mib = measurement::integer<std::size_t>(
                next("flush MiB"), 1, std::numeric_limits<std::size_t>::max() >> 20);
            options.flush_bytes = mib << 20;
        } else if (!std::strcmp(argv[index], "--help") || !std::strcmp(argv[index], "-h")) {
            std::printf("usage: %s [--steps 1..15] [--batch 1..8] [--warmup N] [--repeat N] "
                        "[--flush-mib N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argv[index]));
        }
    }
    if (options.steps < 0 || options.steps > 15) throw std::invalid_argument("invalid steps");
    if (options.batch_size < 0 || options.batch_size > 8) {
        throw std::invalid_argument("batch must be in [1,8]");
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repeat positive");
    }
    return options;
}

const char* mode_name(Mode mode) {
    if (mode == Mode::Greedy) return "greedy";
    if (mode == Mode::Stochastic) return "stochastic";
    return "mixed";
}

struct Fixture {
    int kSteps;
    std::array<std::int32_t, kCandidates * kMaxSteps * 8> host_candidates{};
    std::array<float, kCandidates * kMaxSteps * 8> host_unary{};
    std::array<std::int32_t, 8> host_anchors{};
    std::array<std::int32_t, 8> host_positions{};
    DeviceBuffer candidate_ids{ninfer::bench::device_context().device(),
                               host_candidates.size() * sizeof(std::int32_t)};
    DeviceBuffer unary_scores{ninfer::bench::device_context().device(),
                              host_unary.size() * sizeof(float)};
    DeviceBuffer projected_hidden =
        make_bf16(static_cast<std::size_t>(kRank) * kMaxSteps * 8, 101U);
    DeviceBuffer anchors{ninfer::bench::device_context().device(),
                         host_anchors.size() * sizeof(std::int32_t)};
    DeviceBuffer predecessor_codebook{ninfer::bench::device_context().device(),
                                      static_cast<std::size_t>(kRank) * kCodebookRows *
                                          sizeof(std::uint16_t)};
    DeviceBuffer successor_codebook{ninfer::bench::device_context().device(),
                                    static_cast<std::size_t>(kRank) * kCodebookRows *
                                        sizeof(std::uint16_t)};
    DeviceBuffer base_positions{ninfer::bench::device_context().device(),
                                host_positions.size() * sizeof(std::int32_t)};
    DeviceBuffer configs{ninfer::bench::device_context().device(), 8 * sizeof(ops::SamplingConfig)};
    DeviceBuffer drafts{ninfer::bench::device_context().device(),
                        static_cast<std::size_t>(kMaxSteps) * 8 * sizeof(std::int32_t)};
    DeviceBuffer proposal_q{ninfer::bench::device_context().device(),
                            static_cast<std::size_t>(kCandidates) * kMaxSteps * 8 * sizeof(float)};

    explicit Fixture(int steps) : kSteps(steps) {
        for (std::int32_t batch = 0; batch < 8; ++batch) {
            host_anchors[static_cast<std::size_t>(batch)]   = 220000 + batch * 131;
            host_positions[static_cast<std::size_t>(batch)] = 4096 + batch * 97;
            for (std::int32_t step = 0; step < kSteps; ++step) {
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    const std::size_t offset =
                        (static_cast<std::size_t>(batch) * kSteps + step) * kCandidates + candidate;
                    host_candidates[offset] = static_cast<std::int32_t>(
                        (offset * 7919ULL + 17ULL) % static_cast<std::size_t>(kTokenDomain));
                    host_unary[offset] = fixture::uniform(offset, 205U, -1.F, 1.F);
                }
            }
        }
        candidate_ids.copy_from_host(host_candidates.data(), candidate_ids.size());
        unary_scores.copy_from_host(host_unary.data(), unary_scores.size());
        anchors.copy_from_host(host_anchors.data(), anchors.size());
        base_positions.copy_from_host(host_positions.data(), base_positions.size());
        CUDA_CHECK(fixture::fill_values(static_cast<__nv_bfloat16*>(predecessor_codebook.data()),
                                        predecessor_codebook.size() / 2, 201U, -.05F, .05F));
        CUDA_CHECK(fixture::fill_values(static_cast<__nv_bfloat16*>(successor_codebook.data()),
                                        successor_codebook.size() / 2, 203U, -.05F, .05F));
    }

    void set_mode(Mode mode, std::int32_t batch_size) {
        std::vector<ops::SamplingConfig> host(static_cast<std::size_t>(batch_size));
        for (std::int32_t batch = 0; batch < batch_size; ++batch) {
            host[static_cast<std::size_t>(batch)].temperature =
                mode == Mode::Greedy || (mode == Mode::Mixed && (batch & 1) != 0) ? 0.0F : 0.8F;
            host[static_cast<std::size_t>(batch)].seed = 20260905ULL + batch;
        }
        configs.copy_from_host(host.data(), host.size() * sizeof(ops::SamplingConfig));
    }

    void tensors(std::int32_t batch_size, Tensor& ids, Tensor& unary, Tensor& hidden,
                 Tensor& anchor, Tensor& predecessor, Tensor& successor, Tensor& positions,
                 Tensor& draft, Tensor& q) {
        ids         = Tensor(candidate_ids.span(), DType::I32, {kCandidates, kSteps, batch_size});
        unary       = Tensor(unary_scores.span(), DType::FP32, {kCandidates, kSteps, batch_size});
        hidden      = Tensor(projected_hidden.span(), DType::BF16, {kRank, kSteps, batch_size});
        anchor      = Tensor(anchors.span(), DType::I32, {batch_size});
        predecessor = Tensor(predecessor_codebook.span(), DType::BF16, {kRank, kCodebookRows});
        successor   = Tensor(successor_codebook.span(), DType::BF16, {kRank, kCodebookRows});
        positions   = Tensor(base_positions.span(), DType::I32, {batch_size});
        draft       = Tensor(drafts.span(), DType::I32, {kSteps, batch_size});
        q           = Tensor(proposal_q.span(), DType::FP32, {kCandidates, kSteps, batch_size});
    }
};

void run(std::int32_t batch_size, Mode mode, const Options& options, Fixture& fixture,
         bench::L2FlushBuffer& flush, CudaExecutionView cuda) {
    fixture.set_mode(mode, batch_size);
    Tensor ids;
    Tensor unary;
    Tensor hidden;
    Tensor anchor;
    Tensor predecessor;
    Tensor successor;
    Tensor positions;
    Tensor draft;
    Tensor q;
    fixture.tensors(batch_size, ids, unary, hidden, anchor, predecessor, successor, positions,
                    draft, q);
    const auto* configs = static_cast<const ops::SamplingConfig*>(fixture.configs.data());

    const auto capacity = ops::candidate_selector_path_workspace_capacity_bytes(
        fixture.kSteps, fixture.kSteps, batch_size, batch_size);
    ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                           std::max<std::size_t>(capacity, 1));
    MemoryArena workspace(workspace_storage.span());
    TimedGraph graph;
    graph.capture(cuda, [&](CudaExecutionView cuda) {
        ops::candidate_selector_path(ids, unary, hidden, anchor, predecessor, successor, positions,
                                     configs, draft, q, workspace, cuda);
    });
    const auto timing = measure_cold_graph(graph, flush, cuda, options.warmup, options.repeat);
    std::printf("%d,%d,%s,%.3f,%.3f,%.3f,%zu,%zu\n", fixture.kSteps, batch_size, mode_name(mode),
                timing.median_us, timing.min_us, timing.p95_us, graph.nodes(),
                workspace.peak_used());
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        bench::L2FlushBuffer flush(options.flush_bytes);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::printf("# gpu=%s public=candidate_selector_path geometry=C16_K1..15_R256 "
                    "cache=cold flush_mib=%zu execution=graph\n",
                    properties.name, options.flush_bytes >> 20);
        std::printf("K,B,mode,median_us,min_us,p95_us,graph_nodes,workspace_bytes\n");
        for (int steps = 1; steps <= 15; ++steps) {
            if (options.steps && options.steps != steps) continue;
            Fixture fixture(steps);
            for (std::int32_t batch_size = 1; batch_size <= 8; ++batch_size) {
                if (options.batch_size != 0 && options.batch_size != batch_size) continue;
                for (const Mode mode : std::array{Mode::Greedy, Mode::Stochastic, Mode::Mixed}) {
                    if (mode == Mode::Mixed && batch_size == 1) continue;
                    run(batch_size, mode, options, fixture, flush, cuda);
                }
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_candidate_selector_bench: %s\n", error.what());
        return 1;
    }
}
