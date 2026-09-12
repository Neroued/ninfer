// Cold-cache CUDA Graph benchmark for candidate_selector_path.

#include "ninfer/ops/candidate_selector.h"

#include "ninfer_bench_common.h"
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

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
            options.steps = std::atoi(next("steps"));
        } else if (!std::strcmp(argv[index], "--batch")) {
            options.batch_size = std::atoi(next("batch"));
        } else if (!std::strcmp(argv[index], "--warmup")) {
            options.warmup = std::atoi(next("warmup"));
        } else if (!std::strcmp(argv[index], "--repeat")) {
            options.repeat = std::atoi(next("repeat"));
        } else if (!std::strcmp(argv[index], "--flush-mib")) {
            const long mib = std::strtol(next("flush MiB"), nullptr, 10);
            if (mib <= 0) { throw std::invalid_argument("flush MiB must be positive"); }
            options.flush_bytes = static_cast<std::size_t>(mib) << 20;
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
    DeviceBuffer candidate_ids{host_candidates.size() * sizeof(std::int32_t)};
    DeviceBuffer unary_scores{host_unary.size() * sizeof(float)};
    DeviceBuffer projected_hidden = make_bf16(static_cast<std::size_t>(kRank) * kMaxSteps * 8);
    DeviceBuffer anchors{host_anchors.size() * sizeof(std::int32_t)};
    DeviceBuffer predecessor_codebook{static_cast<std::size_t>(kRank) * kCodebookRows *
                                      sizeof(std::uint16_t)};
    DeviceBuffer successor_codebook{static_cast<std::size_t>(kRank) * kCodebookRows *
                                    sizeof(std::uint16_t)};
    DeviceBuffer base_positions{host_positions.size() * sizeof(std::int32_t)};
    DeviceBuffer configs{8 * sizeof(ops::SamplingConfig)};
    DeviceBuffer drafts{static_cast<std::size_t>(kMaxSteps) * 8 * sizeof(std::int32_t)};
    DeviceBuffer proposal_q{static_cast<std::size_t>(kCandidates) * kMaxSteps * 8 * sizeof(float)};

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
                    host_unary[offset] =
                        static_cast<float>((candidate * 13 + step * 7 + batch * 3) % 29) / 32.0F;
                }
            }
        }
        candidate_ids.copy_from_host(host_candidates.data(), candidate_ids.bytes);
        unary_scores.copy_from_host(host_unary.data(), unary_scores.bytes);
        anchors.copy_from_host(host_anchors.data(), anchors.bytes);
        base_positions.copy_from_host(host_positions.data(), base_positions.bytes);
        CUDA_CHECK(cudaMemset(predecessor_codebook.p, 0x3f, predecessor_codebook.bytes));
        CUDA_CHECK(cudaMemset(successor_codebook.p, 0x3f, successor_codebook.bytes));
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
        ids         = Tensor(candidate_ids.p, DType::I32, {kCandidates, kSteps, batch_size});
        unary       = Tensor(unary_scores.p, DType::FP32, {kCandidates, kSteps, batch_size});
        hidden      = Tensor(projected_hidden.p, DType::BF16, {kRank, kSteps, batch_size});
        anchor      = Tensor(anchors.p, DType::I32, {batch_size});
        predecessor = Tensor(predecessor_codebook.p, DType::BF16, {kRank, kCodebookRows});
        successor   = Tensor(successor_codebook.p, DType::BF16, {kRank, kCodebookRows});
        positions   = Tensor(base_positions.p, DType::I32, {batch_size});
        draft       = Tensor(drafts.p, DType::I32, {kSteps, batch_size});
        q           = Tensor(proposal_q.p, DType::FP32, {kCandidates, kSteps, batch_size});
    }
};

void run(std::int32_t batch_size, Mode mode, const Options& options, Fixture& fixture,
         DeviceBuffer& flush, cudaStream_t stream) {
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
    const auto* configs = static_cast<const ops::SamplingConfig*>(fixture.configs.p);

    const auto route    = ops::detail::candidate_selector_path_route(fixture.kSteps, batch_size);
    const auto capacity = ops::candidate_selector_path_workspace_capacity_bytes(
        fixture.kSteps, fixture.kSteps, batch_size, batch_size);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 1));
    TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t s) {
        ops::candidate_selector_path(ids, unary, hidden, anchor, predecessor, successor, positions,
                                     configs, draft, q, workspace, s);
    });
    const auto timing = measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
    std::printf("%d,%d,%s,%s,%.3f,%.3f,%.3f,%zu,%zu\n", fixture.kSteps, batch_size, mode_name(mode),
                ops::detail::selector_route_name(route), timing.median_us, timing.min_us,
                timing.p95_us, graph.nodes(), workspace.peak_used());
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
        DeviceBuffer flush(options.flush_bytes);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::printf("# gpu=%s public=candidate_selector_path geometry=C16_K1..15_R256 "
                    "cache=cold flush_mib=%zu execution=graph\n",
                    properties.name, options.flush_bytes >> 20);
        std::printf("K,B,mode,route,median_us,min_us,p95_us,graph_nodes,workspace_bytes\n");
        for (int steps = 1; steps <= 15; ++steps) {
            if (options.steps && options.steps != steps) continue;
            Fixture fixture(steps);
            for (std::int32_t batch_size = 1; batch_size <= 8; ++batch_size) {
                if (options.batch_size != 0 && options.batch_size != batch_size) continue;
                for (const Mode mode : std::array{Mode::Greedy, Mode::Stochastic, Mode::Mixed}) {
                    if (mode == Mode::Mixed && batch_size == 1) continue;
                    run(batch_size, mode, options, fixture, flush, stream);
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
