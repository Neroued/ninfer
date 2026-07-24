#include <ninfer/targets/qwen3_6_35b_a3b/package.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/request_memory.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace target = ninfer::targets::qwen3_6_35b_a3b;
using Clock      = std::chrono::steady_clock;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_35b_a3b.ninfer";
    int device                     = 0;
    int warmup                     = 2;
    int repetitions                = 10;
    std::uint32_t context_tokens   = 128;
    std::uint32_t draft_tokens     = 15;
    ninfer::ProposalHead proposal  = ninfer::ProposalHead::Optimized;
    bool use_cuda_graph            = true;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninfer>] [--device <id>] [--context <tokens>]"
                 " [--warmup <n>] [--reps <n>] [--draft-tokens <1..15>]"
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
            print_usage(argc > 0 ? argv[0] : "ninfer_qwen3_6_35b_a3b_dflash_round_bench");
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

struct RoundMeasurement {
    float gpu_ms                  = 0.0F;
    double wall_ms                = 0.0;
    std::uint32_t licensed_tokens = 0;
};

RoundMeasurement measure_round(target::Package::Program& program, ninfer::DeviceContext& device,
                               std::uint32_t draft_tokens) {
    ninfer::CudaEventTimer timer(device);
    const auto wall_start = Clock::now();
    timer.start();
    auto round = program.decode_round(
        ninfer::runtime::RoundBudget{.generated_tokens_remaining = draft_tokens + 1});
    const float gpu_ms = timer.stop_ms();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();
    const std::uint32_t licensed = static_cast<std::uint32_t>(round.tokens.size());
    program.resolve_pending(licensed, false);
    return {.gpu_ms = gpu_ms, .wall_ms = wall_ms, .licensed_tokens = licensed};
}

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
    const std::uint64_t block    = options.draft_tokens + 1ULL;
    const std::uint64_t capacity = options.context_tokens + (measured_rounds + 1ULL) * block;
    if (capacity > 262144) {
        throw std::invalid_argument("context and measured rounds exceed native capacity");
    }

    ninfer::EngineOptions engine;
    engine.artifact_path             = options.artifact;
    engine.device                    = options.device;
    engine.max_context               = static_cast<std::uint32_t>(capacity);
    engine.prefill_chunk             = 128;
    engine.kv_cache                  = ninfer::KvCacheStorage::BFloat16;
    engine.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    engine.speculative.draft_tokens  = options.draft_tokens;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = options.use_cuda_graph;

    ninfer::DeviceContext device(options.device);
    ninfer::artifact::Reader reader(options.artifact);
    if (reader.model_id() != target::Package::model_id) {
        throw std::invalid_argument("artifact model_id does not match qwen3.6-35b-a3b");
    }
    ninfer::artifact::Binder binder(reader);
    auto load_plan = target::Package::plan_load(binder, engine);
    auto materialized =
        ninfer::artifact::materialize(reader, load_plan.materialization(), device, nullptr);
    auto model =
        target::Package::construct_loaded_model(std::move(load_plan), std::move(materialized));
    auto frontend = target::Package::make_frontend(*model);
    auto prompt   = frontend.prepare_tokens(prompt_tokens(options.context_tokens), false);

    auto sequence = target::Package::plan_sequence(device, engine);
    auto program  = target::Package::create_program(*model, std::move(sequence), device);
    ninfer::runtime::RequestMemory request_memory(device);
    ninfer::ExecutionOptions execution;
    execution.requested_output_tokens = 1 + measured_rounds * (options.draft_tokens + 1);
    execution.allow_prefix_reuse      = false;
    auto request_plan                 = program->plan_request(prompt, execution);
    request_memory.ensure(request_plan.summary().transient_bytes,
                          request_plan.summary().transient_alignment);
    auto first =
        program->begin(std::move(prompt), std::move(request_plan), request_memory.region());
    program->resolve_pending(static_cast<std::uint32_t>(first.round.tokens.size()), false);

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        (void)measure_round(*program, device, options.draft_tokens);
    }
    const ninfer::SpeculativeStats before = program->speculative_stats();

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        measurements.push_back(measure_round(*program, device, options.draft_tokens));
    }
    const ninfer::SpeculativeStats after = program->speculative_stats();
    if (after.rounds - before.rounds != static_cast<std::uint64_t>(options.repetitions) ||
        after.fallback_steps != before.fallback_steps) {
        throw std::runtime_error("benchmark left the complete DFlash round path");
    }

    std::vector<float> gpu_ms;
    std::vector<double> wall_ms;
    std::uint64_t licensed_tokens = 0;
    for (const RoundMeasurement& measurement : measurements) {
        gpu_ms.push_back(measurement.gpu_ms);
        wall_ms.push_back(measurement.wall_ms);
        licensed_tokens += measurement.licensed_tokens;
    }
    const auto [gpu_min, gpu_max] = std::minmax_element(gpu_ms.begin(), gpu_ms.end());
    const std::uint64_t accepted  = after.accepted_tokens - before.accepted_tokens;
    const std::uint64_t drafted   = after.drafted_tokens - before.drafted_tokens;
    const double mean_gpu_ms      = mean(gpu_ms);
    const double mean_wall_ms     = mean(wall_ms);
    const double mean_licensed =
        static_cast<double>(licensed_tokens) / static_cast<double>(options.repetitions);

    std::cout << "format,ninfer_qwen3_6_35b_a3b_dflash_round_bench_v1\n";
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << device.props.name << '\n';
    std::cout << "context_tokens," << options.context_tokens << '\n';
    std::cout << "draft_tokens," << options.draft_tokens << '\n';
    std::cout << "proposal_head,"
              << (options.proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full")
              << '\n';
    std::cout << "cuda_graph," << (options.use_cuda_graph ? "true" : "false") << '\n';
    std::cout << "warmup," << options.warmup << '\n';
    std::cout << "repetitions," << options.repetitions << '\n';
    std::cout << "steady_round_gpu_mean_ms," << mean_gpu_ms << '\n';
    std::cout << "steady_round_gpu_min_ms," << *gpu_min << '\n';
    std::cout << "steady_round_gpu_max_ms," << *gpu_max << '\n';
    std::cout << "steady_round_wall_mean_ms," << mean_wall_ms << '\n';
    std::cout << "mean_licensed_tokens," << mean_licensed << '\n';
    std::cout << "drafted_tokens," << drafted << '\n';
    std::cout << "accepted_draft_tokens," << accepted << '\n';
    std::cout << "acceptance_rate,"
              << (drafted == 0 ? 0.0 : static_cast<double>(accepted) / drafted) << '\n';
    std::cout << "published_tokens_per_second," << 1000.0 * mean_licensed / mean_wall_ms << '\n';
    std::cout << "accepted_per_position";
    for (std::size_t position = 0; position < after.accepted_per_position.size(); ++position) {
        std::cout << ','
                  << after.accepted_per_position[position] - before.accepted_per_position[position];
    }
    std::cout << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_6_35b_a3b_dflash_round_bench: " << error.what() << '\n';
        return 1;
    }
}
