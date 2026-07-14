#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/request_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
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

namespace target = ninfer::targets::qwen3_6_27b_rtx5090;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_27b_rtx5090.ninfer";
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
            print_usage(argc > 0 ? argv[0] : "ninfer_qwen3_6_27b_mtp_round_bench");
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

struct RoundMeasurement {
    float milliseconds            = 0.0F;
    std::uint32_t licensed_tokens = 0;
};

RoundMeasurement measure_round(target::Package::Program& program, ninfer::DeviceContext& device,
                               std::uint32_t draft_tokens) {
    ninfer::CudaEventTimer timer(device);
    timer.start();
    auto round = program.decode_round(
        ninfer::runtime::RoundBudget{.generated_tokens_remaining = draft_tokens + 1});
    const float milliseconds     = timer.stop_ms();
    const std::uint32_t licensed = static_cast<std::uint32_t>(round.tokens.size());
    program.resolve_pending(licensed, false);
    return RoundMeasurement{.milliseconds = milliseconds, .licensed_tokens = licensed};
}

int run(const Options& options) {
    if (!std::filesystem::exists(options.artifact)) {
        std::cout << "SKIP: artifact not present: " << options.artifact.string() << '\n';
        return 0;
    }

    ninfer::DeviceContext device(options.device);
    ninfer::artifact::Reader reader(options.artifact);
    if (reader.model_id() != target::Package::model_id) {
        throw std::invalid_argument("artifact model_id does not match qwen3.6-27b");
    }
    ninfer::artifact::Binder binder(reader);
    auto load_plan = target::Package::plan_load(binder);
    auto materialized =
        ninfer::artifact::materialize(reader, load_plan.materialization(), device, nullptr);
    auto model =
        target::Package::construct_loaded_model(std::move(load_plan), std::move(materialized));
    auto frontend = target::Package::make_frontend(*model);
    const std::vector<ninfer::TokenId> seed{248045, 846, 198, 5834, 248046, 198};
    auto prompt = frontend.prepare_tokens(seed, false);

    const std::uint32_t measured_rounds =
        static_cast<std::uint32_t>(options.warmup + options.repetitions);
    ninfer::EngineOptions engine;
    engine.artifact_path             = options.artifact;
    engine.device                    = options.device;
    engine.max_context               = static_cast<std::uint32_t>(seed.size() + 64ULL +
                                                                  static_cast<std::uint64_t>(measured_rounds) *
                                                                      (options.draft_tokens + 1ULL) +
                                                                  2ULL * options.draft_tokens);
    engine.prefill_chunk             = 128;
    engine.kv_cache                  = ninfer::KvCacheStorage::BFloat16;
    engine.speculative.draft_tokens  = options.draft_tokens;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = options.use_cuda_graph;

    auto sequence = target::Package::plan_sequence(*model, device, engine);
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

    const std::uint64_t rounds_before = program->speculative_stats().rounds;
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        (void)measure_round(*program, device, options.draft_tokens);
    }

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        measurements.push_back(measure_round(*program, device, options.draft_tokens));
    }
    const ninfer::SpeculativeStats stats = program->speculative_stats();
    if (stats.rounds - rounds_before != measured_rounds || stats.fallback_steps != 0) {
        throw std::runtime_error("benchmark did not stay on the native MTP proposal/verify path");
    }

    std::vector<float> milliseconds;
    milliseconds.reserve(measurements.size());
    std::uint64_t licensed_tokens = 0;
    for (const RoundMeasurement& measurement : measurements) {
        milliseconds.push_back(measurement.milliseconds);
        licensed_tokens += measurement.licensed_tokens;
    }
    const double mean_ms =
        std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) / measurements.size();
    const auto [minimum, maximum] = std::minmax_element(milliseconds.begin(), milliseconds.end());
    const double mean_licensed =
        static_cast<double>(licensed_tokens) / static_cast<double>(measurements.size());

    std::cout << "format,ninfer_qwen3_6_27b_mtp_round_bench_v1\n";
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << device.props.name << '\n';
    std::cout << "draft_tokens," << options.draft_tokens << '\n';
    std::cout << "proposal_head,"
              << (options.proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full")
              << '\n';
    std::cout << "cuda_graph," << (options.use_cuda_graph ? "true" : "false") << '\n';
    std::cout << "warmup," << options.warmup << '\n';
    std::cout << "repetitions," << options.repetitions << '\n';
    std::cout << "mtp_round_mean_ms," << mean_ms << '\n';
    std::cout << "mtp_round_min_ms," << *minimum << '\n';
    std::cout << "mtp_round_max_ms," << *maximum << '\n';
    std::cout << "mean_licensed_tokens," << mean_licensed << '\n';
    std::cout << "accepted_draft_tokens," << stats.accepted_tokens << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_6_27b_mtp_round_bench: " << error.what() << '\n';
        return 1;
    }
}
