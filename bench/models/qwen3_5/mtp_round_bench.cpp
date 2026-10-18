#include "round_fixture.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q35 = ninfer::models::qwen3_5;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_27b.ninfer";
    int device                     = 0;
    int warmup                     = 2;
    int repetitions                = 10;
    std::uint32_t draft_tokens     = 5;
    ninfer::ProposalHead proposal  = ninfer::ProposalHead::Optimized;
    bool use_cuda_graph            = true;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninfer>] [--device <id>] [--warmup <n>] [--reps <n>]"
                 " [--draft-tokens <1..5>] [--proposal-head full|optimized]"
                 " [--no-cuda-graph]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> const char* {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs value");
            }
            return argv[index];
        };
        if (argument == "--artifact") {
            options.artifact = value("--artifact");
        } else if (argument == "--device") {
            options.device = std::stoi(value("--device"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value("--warmup"));
        } else if (argument == "--reps") {
            options.repetitions = std::stoi(value("--reps"));
        } else if (argument == "--draft-tokens") {
            options.draft_tokens = static_cast<std::uint32_t>(std::stoul(value("--draft-tokens")));
        } else if (argument == "--proposal-head") {
            const std::string_view head(value("--proposal-head"));
            if (head == "full") {
                options.proposal = ninfer::ProposalHead::Full;
            } else if (head == "optimized") {
                options.proposal = ninfer::ProposalHead::Optimized;
            } else {
                throw std::invalid_argument("--proposal-head must be full or optimized");
            }
        } else if (argument == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argc > 0 ? argv[0] : "ninfer_qwen3_5_mtp_round_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (options.warmup < 0) { throw std::invalid_argument("--warmup must be nonnegative"); }
    if (options.repetitions <= 0) { throw std::invalid_argument("--reps must be positive"); }
    if (options.draft_tokens == 0 || options.draft_tokens > 5) {
        throw std::invalid_argument("--draft-tokens must be in [1,5]");
    }
    return options;
}

using RoundMeasurement = ninfer::bench::qwen3_5::RoundMeasurement;

int run(const Options& options) {
    if (!std::filesystem::exists(options.artifact)) {
        std::cout << "SKIP: artifact not present: " << options.artifact.string() << '\n';
        return 0;
    }

    const std::vector<ninfer::TokenId> seed{248045, 846, 198, 5834, 248046, 198};
    const std::uint32_t measured_rounds =
        static_cast<std::uint32_t>(options.warmup + options.repetitions);
    ninfer::EngineOptions engine;
    engine.artifact_path       = options.artifact;
    engine.device              = options.device;
    engine.max_context         = static_cast<std::uint32_t>(seed.size() + 64ULL +
                                                            static_cast<std::uint64_t>(measured_rounds) *
                                                                (options.draft_tokens + 1ULL) +
                                                            2ULL * options.draft_tokens);
    engine.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(engine.max_context);
    engine.prefill_chunk       = 128;
    engine.kv_cache            = ninfer::KvCacheStorage::BFloat16;
    engine.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    engine.speculative.draft_tokens  = options.draft_tokens;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = options.use_cuda_graph;

    engine           = ninfer::runtime::normalize_engine_options(std::move(engine));
    auto constructed = ninfer::runtime::construct_model(engine);
    auto& instance =
        *std::get<std::unique_ptr<ninfer::bench::qwen3_5::Instance>>(constructed.instance);
    ninfer::bench::qwen3_5::RoundFixture fixture(instance);
    const auto active_sequence = fixture.bind(seed, ninfer::runtime::LaneId{0},
                                              1 + measured_rounds * (options.draft_tokens + 1));
    const std::array<q35::SequenceHandle, 1> active_sequences{active_sequence};

    constexpr std::uint64_t rounds_before = 0;
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        (void)fixture.measure(active_sequences, options.draft_tokens);
    }

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        measurements.push_back(fixture.measure(active_sequences, options.draft_tokens));
    }
    const auto aborted                   = fixture.release(active_sequence);
    const ninfer::SpeculativeStats stats = aborted.speculative;
    if (stats.rounds - rounds_before != measured_rounds || stats.fallback_steps != 0) {
        throw std::runtime_error("benchmark did not stay on the native MTP proposal/verify path");
    }

    std::vector<double> milliseconds;
    double host_call_ms = 0;
    milliseconds.reserve(measurements.size());
    std::uint64_t licensed_tokens = 0;
    for (const RoundMeasurement& measurement : measurements) {
        milliseconds.push_back(measurement.wall_ms);
        host_call_ms += measurement.host_call_ms;
        licensed_tokens += measurement.licensed_tokens;
    }
    const double mean_ms =
        std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) / measurements.size();
    const auto [minimum, maximum] = std::minmax_element(milliseconds.begin(), milliseconds.end());
    const double mean_licensed =
        static_cast<double>(licensed_tokens) / static_cast<double>(measurements.size());

    std::cout << "format,ninfer_qwen3_5_mtp_round_bench_v3\n";
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << ninfer::bench::qwen3_5::device_name(options.device) << '\n';
    std::cout << "draft_tokens," << options.draft_tokens << '\n';
    std::cout << "proposal_head,"
              << (options.proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full")
              << '\n';
    std::cout << "cuda_graph," << (options.use_cuda_graph ? "true" : "false") << '\n';
    std::cout << "warmup," << options.warmup << '\n';
    std::cout << "repetitions," << options.repetitions << '\n';
    std::cout << "mtp_round_wall_mean_ms," << mean_ms << '\n';
    std::cout << "mtp_round_wall_min_ms," << *minimum << '\n';
    std::cout << "mtp_round_wall_max_ms," << *maximum << '\n';
    std::cout << "mtp_round_host_call_mean_ms," << host_call_ms / measurements.size() << '\n';
    std::cout << "mean_licensed_tokens," << mean_licensed << '\n';
    std::cout << "accepted_draft_tokens," << stats.accepted_tokens << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_5_mtp_round_bench: " << error.what() << '\n';
        return 1;
    }
}
