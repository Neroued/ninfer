#include "round_fixture.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q35 = ninfer::models::qwen3_5;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_35b_a3b.ninfer";
    int device                     = 0;
    int warmup                     = 2;
    int repetitions                = 10;
    std::uint32_t context_tokens   = 128;
    std::uint32_t draft_tokens     = 15;
    std::uint32_t batch_size       = 1;
    ninfer::ProposalHead proposal  = ninfer::ProposalHead::Optimized;
    bool use_cuda_graph            = true;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninfer>] [--device <id>] [--context <tokens>]"
                 " [--warmup <n>] [--reps <n>] [--draft-tokens <1..15>]"
                 " [--batch <1..8>]"
                 " [--proposal-head full|optimized] [--no-cuda-graph]\n";
}

std::uint32_t parse_u32(const char* text, const char* label) {
    std::size_t consumed      = 0;
    const unsigned long value = std::stoul(text, &consumed);
    if (text[consumed] != '\0' || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(label) + " is not a uint32");
    }
    return static_cast<std::uint32_t>(value);
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
        } else if (argument == "--context") {
            options.context_tokens = parse_u32(value("--context"), "context");
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value("--warmup"));
        } else if (argument == "--reps") {
            options.repetitions = std::stoi(value("--reps"));
        } else if (argument == "--draft-tokens") {
            options.draft_tokens = parse_u32(value("--draft-tokens"), "draft-tokens");
        } else if (argument == "--batch") {
            options.batch_size = parse_u32(value("--batch"), "batch");
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
            print_usage(argc > 0 ? argv[0] : "ninfer_qwen3_5_dflash_round_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (options.warmup < 1) {
        throw std::invalid_argument("--warmup must be positive so measured rounds are steady");
    }
    if (options.repetitions <= 0) { throw std::invalid_argument("--reps must be positive"); }
    if (options.context_tokens < 16) {
        throw std::invalid_argument("--context must be at least 16 tokens");
    }
    if (options.draft_tokens == 0 || options.draft_tokens > 15) {
        throw std::invalid_argument("--draft-tokens must be in [1,15]");
    }
    if (options.batch_size == 0 || options.batch_size > ninfer::kMaximumConcurrency) {
        throw std::invalid_argument("--batch must be in [1,8]");
    }
    return options;
}

std::vector<ninfer::TokenId> prompt_tokens(std::uint32_t count) {
    std::vector<ninfer::TokenId> prompt{
        248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
        1710,   248046, 198, 248045, 74455, 198,   248068, 198,
    };
    prompt.insert(prompt.begin() + 9, count - prompt.size(), 374);
    return prompt;
}

using RoundMeasurement = ninfer::bench::qwen3_5::RoundMeasurement;

template <class T>
double mean(const std::vector<T>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

int run(const Options& options) {
    if (!std::filesystem::exists(options.artifact)) {
        std::cout << "SKIP: artifact not present: " << options.artifact.string() << '\n';
        return 0;
    }

    const std::uint32_t measured_rounds =
        static_cast<std::uint32_t>(options.warmup + options.repetitions);
    const std::uint64_t block = options.draft_tokens + 1ULL;
    const std::uint64_t per_request_capacity =
        options.context_tokens + (measured_rounds + 1ULL) * block;
    if (per_request_capacity > 262144) {
        throw std::invalid_argument("context and measured rounds exceed native capacity");
    }

    ninfer::EngineOptions engine;
    engine.artifact_path             = options.artifact;
    engine.device                    = options.device;
    engine.max_context               = static_cast<std::uint32_t>(per_request_capacity);
    engine.prefill_chunk             = 128;
    engine.kv_cache                  = ninfer::KvCacheStorage::BFloat16;
    engine.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    engine.speculative.draft_tokens  = options.draft_tokens;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = options.use_cuda_graph;
    engine.max_concurrency           = options.batch_size;

    engine           = ninfer::runtime::normalize_engine_options(std::move(engine));
    auto constructed = ninfer::runtime::construct_model(engine);
    auto& instance =
        *std::get<std::unique_ptr<ninfer::bench::qwen3_5::Instance>>(constructed.instance);
    ninfer::bench::qwen3_5::RoundFixture fixture(instance);
    std::array<q35::SequenceHandle, ninfer::kMaximumConcurrency> active_sequences{};
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        active_sequences[lane] =
            fixture.bind(prompt_tokens(options.context_tokens), ninfer::runtime::LaneId{lane},
                         1 + measured_rounds * (options.draft_tokens + 1));
    }

    const auto active_span =
        std::span<const q35::SequenceHandle>(active_sequences.data(), options.batch_size);
    RoundMeasurement warmup_state;
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        warmup_state = fixture.measure(active_span, options.draft_tokens);
    }

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        measurements.push_back(fixture.measure(active_span, options.draft_tokens));
    }
    const auto& before = warmup_state.stats;
    const auto& after  = measurements.back().stats;
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        if (after[lane].rounds - before[lane].rounds !=
                static_cast<std::uint64_t>(options.repetitions) ||
            after[lane].fallback_steps != before[lane].fallback_steps) {
            throw std::runtime_error("benchmark left the complete DFlash round path");
        }
    }

    std::vector<double> host_call_ms;
    std::vector<double> wall_ms;
    std::uint64_t licensed_tokens = 0;
    for (const RoundMeasurement& measurement : measurements) {
        host_call_ms.push_back(measurement.host_call_ms);
        wall_ms.push_back(measurement.wall_ms);
        licensed_tokens += measurement.licensed_tokens;
    }
    const auto [wall_min, wall_max] = std::minmax_element(wall_ms.begin(), wall_ms.end());
    std::uint64_t accepted          = 0;
    std::uint64_t drafted           = 0;
    std::array<std::uint64_t, 15> accepted_per_position{};
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        accepted += after[lane].accepted_tokens - before[lane].accepted_tokens;
        drafted += after[lane].drafted_tokens - before[lane].drafted_tokens;
        for (std::size_t position = 0; position < options.draft_tokens; ++position) {
            accepted_per_position[position] += after[lane].accepted_per_position[position] -
                                               before[lane].accepted_per_position[position];
        }
    }
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        (void)fixture.release(active_sequences[lane]);
    }
    const double mean_host_call_ms = mean(host_call_ms);
    const double mean_wall_ms      = mean(wall_ms);
    const double mean_licensed_per_batch =
        static_cast<double>(licensed_tokens) / static_cast<double>(options.repetitions);
    const double mean_licensed_per_request =
        mean_licensed_per_batch / static_cast<double>(options.batch_size);

    std::cout << "format,ninfer_qwen3_5_dflash_round_bench_v4\n";
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << ninfer::bench::qwen3_5::device_name(options.device) << '\n';
    std::cout << "context_tokens," << options.context_tokens << '\n';
    std::cout << "draft_tokens," << options.draft_tokens << '\n';
    std::cout << "batch_size," << options.batch_size << '\n';
    std::cout << "proposal_head,"
              << (options.proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full")
              << '\n';
    std::cout << "cuda_graph," << (options.use_cuda_graph ? "true" : "false") << '\n';
    std::cout << "warmup," << options.warmup << '\n';
    std::cout << "repetitions," << options.repetitions << '\n';
    std::cout << "steady_round_wall_min_ms," << *wall_min << '\n';
    std::cout << "steady_round_wall_max_ms," << *wall_max << '\n';
    std::cout << "steady_round_wall_mean_ms," << mean_wall_ms << '\n';
    std::cout << "steady_round_host_call_mean_ms," << mean_host_call_ms << '\n';
    std::cout << "mean_licensed_tokens_per_batch," << mean_licensed_per_batch << '\n';
    std::cout << "mean_licensed_tokens_per_request," << mean_licensed_per_request << '\n';
    std::cout << "drafted_tokens," << drafted << '\n';
    std::cout << "accepted_draft_tokens," << accepted << '\n';
    std::cout << "acceptance_rate,"
              << (drafted == 0 ? 0.0 : static_cast<double>(accepted) / drafted) << '\n';
    std::cout << "published_tokens_per_second," << 1000.0 * mean_licensed_per_batch / mean_wall_ms
              << '\n';
    std::cout << "accepted_per_position";
    for (std::size_t position = 0; position < options.draft_tokens; ++position) {
        std::cout << ',' << accepted_per_position[position];
    }
    std::cout << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_5_dflash_round_bench: " << error.what() << '\n';
        return 1;
    }
}
