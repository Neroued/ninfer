#include <ninfer/targets/qwen3_6_35b_a3b/package.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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

struct RoundMeasurement {
    float gpu_ms                  = 0.0F;
    double wall_ms                = 0.0;
    std::uint32_t licensed_tokens = 0;
    std::array<ninfer::SpeculativeStats, ninfer::kMaximumConcurrency> stats{};
};

RoundMeasurement measure_round(target::Package::Program& program, ninfer::DeviceContext& device,
                               std::span<const target::Package::SequenceHandle> sequences,
                               std::uint32_t batch_size, std::uint32_t draft_tokens) {
    std::array<ninfer::runtime::RoundBudget, ninfer::kMaximumConcurrency> budgets{};
    for (std::uint32_t row = 0; row < batch_size; ++row) {
        budgets[row] = {.generated_tokens_remaining = draft_tokens + 1};
    }
    const auto budget_span =
        std::span<const ninfer::runtime::RoundBudget>(budgets.data(), batch_size);
    ninfer::CudaEventTimer timer(device);
    const auto wall_start = Clock::now();
    timer.start();
    auto pending = program.decode(sequences, budget_span);
    if (pending.row_counts().size() != batch_size) {
        throw std::runtime_error("DFlash benchmark round returned invalid row counts");
    }
    std::array<ninfer::runtime::CommitDecision, ninfer::kMaximumConcurrency> decisions{};
    std::uint32_t licensed = 0;
    for (std::uint32_t row = 0; row < batch_size; ++row) {
        const std::int32_t count = pending.row_counts()[row];
        if (count <= 0 || count > static_cast<std::int32_t>(draft_tokens + 1U)) {
            throw std::runtime_error("DFlash benchmark round returned an invalid row extent");
        }
        decisions[row].accepted_tokens = static_cast<std::uint32_t>(count);
        licensed += decisions[row].accepted_tokens;
    }
    const auto committed = program.commit(
        std::move(pending),
        std::span<const ninfer::runtime::CommitDecision>(decisions.data(), batch_size));
    const float gpu_ms = timer.stop_ms();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();
    RoundMeasurement result{.gpu_ms = gpu_ms, .wall_ms = wall_ms, .licensed_tokens = licensed};
    for (std::uint32_t row = 0; row < batch_size; ++row) {
        result.stats[row] = committed.rows[row].speculative;
    }
    return result;
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
    const std::uint64_t block = options.draft_tokens + 1ULL;
    const std::uint64_t per_request_capacity =
        options.context_tokens + (measured_rounds + 1ULL) * block;
    const std::uint64_t aligned_request_capacity = (per_request_capacity + 63ULL) & ~63ULL;
    const std::uint64_t capacity                 = options.batch_size == 1
                                                       ? per_request_capacity
                                                       : aligned_request_capacity * options.batch_size;
    if (per_request_capacity > 262144 || capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("context and measured rounds exceed native capacity");
    }

    ninfer::EngineOptions engine;
    engine.artifact_path = options.artifact;
    engine.device        = options.device;
    engine.max_context   = static_cast<std::uint32_t>(per_request_capacity);
    engine.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(capacity));
    engine.prefill_chunk             = 128;
    engine.kv_cache                  = ninfer::KvCacheStorage::BFloat16;
    engine.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    engine.speculative.draft_tokens  = options.draft_tokens;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = options.use_cuda_graph;
    engine.max_concurrency           = options.batch_size;
    engine.context_cache.enabled                = false;
    engine.context_cache.device_state_slots     = 0;
    engine.context_cache.host_state_slots       = 0;
    engine.context_cache.host_kv_capacity_bytes = 0;
    engine.context_cache.max_private_continuations = options.batch_size;
    engine.context_cache.max_shared_prefixes               = 0;
    engine.context_cache.max_long_anchors_per_continuation = 0;

    ninfer::DeviceContext device(options.device);
    ninfer::artifact::Reader reader(options.artifact);
    const auto weights_profile = target::Package::resolve_weights(reader.identity());
    ninfer::artifact::Binder binder(reader);
    auto load_plan        = target::Package::plan_load(binder, engine, weights_profile);
    auto planner          = target::Package::make_sequence_planner(device, engine, weights_profile);
    const auto resolution = ninfer::runtime::resolve_kv_capacity(
        engine.kv_capacity, planner.capacity_curve(), std::numeric_limits<std::size_t>::max());
    auto sequence                      = std::move(planner).finalize(resolution.main_page_groups);
    const std::uint64_t weight_bytes   = load_plan.materialization().device_capacity_bytes;
    const std::uint64_t sequence_bytes = sequence.device_reservation_bytes();
    if (sequence_bytes > std::numeric_limits<std::uint64_t>::max() - weight_bytes) {
        throw std::overflow_error("benchmark device-memory requirement overflows uint64");
    }
    const std::uint64_t required_bytes = weight_bytes + sequence_bytes;
    std::size_t free_bytes             = 0;
    std::size_t total_bytes            = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (required_bytes > free_bytes) {
        throw std::invalid_argument(
            "benchmark configuration requires " + std::to_string(required_bytes) +
            " bytes of device memory, but only " + std::to_string(free_bytes) + " bytes are free");
    }
    auto materialized =
        ninfer::artifact::materialize(reader, load_plan.materialization(), device, nullptr);
    auto model =
        target::Package::construct_loaded_model(std::move(load_plan), std::move(materialized));
    auto frontend = target::Package::make_frontend(*model, engine);
    auto program  = target::Package::create_program(*model, std::move(sequence), device);
    ninfer::runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens = 1 + measured_rounds * (options.draft_tokens + 1);
    execution.allow_prefix_reuse      = false;
    std::array<target::Package::SequenceHandle, ninfer::kMaximumConcurrency> active_sequences{};
    const auto machine_cost = ninfer::runtime::generic_context_machine_cost_model();
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        auto prompt       = frontend.prepare_tokens(prompt_tokens(options.context_tokens), false);
        auto request_base = program->plan_request(prompt, execution);
        auto request_plan =
            program->inspect_admission(prompt, request_base, ninfer::runtime::LaneId{lane}, nullptr,
                                       nullptr, std::nullopt, false, machine_cost);
        if (!request_plan) { throw std::runtime_error("benchmark root admission was rejected"); }
        auto resource_plan = program->seal_identity(*request_plan, prompt);
        if (!resource_plan) {
            throw std::runtime_error("benchmark root resources were not sealed");
        }
        const auto reserved =
            program->start_resource_transaction(std::move(*resource_plan), std::move(prompt), {});
        if (reserved != ninfer::runtime::ContextTransactionReserveStatus::Reserved) {
            throw std::runtime_error("benchmark root materialization was not reserved");
        }
        std::optional<target::Package::MaterializationResult> published;
        for (;;) {
            auto transaction = program->progress_context_transaction({});
            if (std::holds_alternative<ninfer::runtime::ContextTransactionInProgress>(
                    transaction)) {
                continue;
            }
            if (!std::holds_alternative<target::Package::MaterializationResult>(transaction)) {
                program->finalize_context_transaction();
                throw std::runtime_error("benchmark root returned the wrong transaction result");
            }
            published.emplace(
                std::get<target::Package::MaterializationResult>(std::move(transaction)));
            break;
        }
        if (published->status != ninfer::runtime::ContextTransactionStatus::Published ||
            !published->published) {
            program->finalize_context_transaction();
            throw std::runtime_error("benchmark root materialization was not published");
        }
        auto started = std::move(*published->published);
        program->finalize_context_transaction();
        active_sequences[lane] = started.sequence;
        std::optional<target::Package::PrefillProgress> progress;
        progress.emplace(program->advance_prefill(active_sequences[lane]));
        while (!progress->complete) {
            progress.reset();
            progress.emplace(program->advance_prefill(active_sequences[lane]));
        }
        if (!progress->pending || progress->pending->tokens().size() != 1) {
            throw std::runtime_error("benchmark seed prefill did not license exactly one token");
        }
        const std::array<ninfer::runtime::CommitDecision, 1> begin_decision{
            ninfer::runtime::CommitDecision{.accepted_tokens = 1}};
        (void)program->commit(std::move(*progress->pending), begin_decision);
    }

    const auto active_span = std::span<const target::Package::SequenceHandle>(
        active_sequences.data(), options.batch_size);
    RoundMeasurement warmup_state;
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        warmup_state =
            measure_round(*program, device, active_span, options.batch_size, options.draft_tokens);
    }

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        measurements.push_back(
            measure_round(*program, device, active_span, options.batch_size, options.draft_tokens));
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

    std::vector<float> gpu_ms;
    std::vector<double> wall_ms;
    std::uint64_t licensed_tokens = 0;
    for (const RoundMeasurement& measurement : measurements) {
        gpu_ms.push_back(measurement.gpu_ms);
        wall_ms.push_back(measurement.wall_ms);
        licensed_tokens += measurement.licensed_tokens;
    }
    const auto [gpu_min, gpu_max] = std::minmax_element(gpu_ms.begin(), gpu_ms.end());
    std::uint64_t accepted        = 0;
    std::uint64_t drafted         = 0;
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
        const auto aborted = program->abort(active_sequences[lane]);
        if (aborted.status != ninfer::runtime::ConsumeStatus::Consumed) {
            throw std::runtime_error("benchmark could not release an active sequence");
        }
    }
    const double mean_gpu_ms  = mean(gpu_ms);
    const double mean_wall_ms = mean(wall_ms);
    const double mean_licensed_per_batch =
        static_cast<double>(licensed_tokens) / static_cast<double>(options.repetitions);
    const double mean_licensed_per_request =
        mean_licensed_per_batch / static_cast<double>(options.batch_size);

    std::cout << "format,ninfer_qwen3_6_35b_a3b_dflash_round_bench_v2\n";
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << device.props.name << '\n';
    std::cout << "context_tokens," << options.context_tokens << '\n';
    std::cout << "draft_tokens," << options.draft_tokens << '\n';
    std::cout << "batch_size," << options.batch_size << '\n';
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
        std::cerr << "ninfer_qwen3_6_35b_a3b_dflash_round_bench: " << error.what() << '\n';
        return 1;
    }
}
