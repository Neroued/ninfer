#include <ninfer/targets/qwen3_6_27b/package.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/phase_trace.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/request_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace target = ninfer::targets::qwen3_6_27b;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_27b.ninfer";
    int device                     = 0;
    int warmup                     = 2;
    int repetitions                = 10;
    std::uint32_t draft_tokens     = 5;
    ninfer::ProposalHead proposal  = ninfer::ProposalHead::Optimized;
    bool use_cuda_graph            = true;
    bool phase_timing              = false;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninfer>] [--device <id>] [--warmup <n>] [--reps <n>]"
                 " [--draft-tokens <1..5>] [--proposal-head full|optimized]"
                 " [--no-cuda-graph] [--phase-timing]\n"
              << "  --phase-timing  attribute each round to target verification, the per-draft\n"
                 "                  proposal steps, and acceptance/commit. Implies --no-cuda-graph,\n"
                 "                  because a replayed graph re-records no host-visible events.\n";
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
        } else if (argument == "--phase-timing") {
            options.phase_timing = true;
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
    // A captured graph replays without re-recording events, so the trace would see nothing.
    if (options.phase_timing) { options.use_cuda_graph = false; }
    return options;
}

struct RoundMeasurement {
    float milliseconds            = 0.0F;
    std::uint32_t licensed_tokens = 0;
    std::vector<ninfer::PhaseTrace::Interval> phases;
};

RoundMeasurement measure_round(target::Package::Program& program, ninfer::DeviceContext& device,
                               std::uint32_t draft_tokens) {
    constexpr std::array<std::uint32_t, 1> lanes{0};
    // The budget has to cover this round's verify width *and* leave room for the next round's
    // proposal, because mtp_prepare_next_round sizes the next extent as (remaining - 1) and a
    // round entering with extent 0 is a fallback, not a speculative round. At exactly
    // draft_tokens + 1 that is guaranteed to happen -- which is why this bench only ever passed
    // its native-path assertion at the default --draft-tokens 5, where high acceptance made it
    // intermittent rather than certain. Serving passes the request's whole remaining output here,
    // so steady state is what a generous budget reproduces.
    const std::array<ninfer::runtime::RoundBudget, 1> budgets{
        ninfer::runtime::RoundBudget{.generated_tokens_remaining = 4 * (draft_tokens + 1)}};
    ninfer::CudaEventTimer timer(device);
    timer.start();
    const auto round = program.decode_batch(lanes, budgets);
    const std::uint32_t licensed =
        round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts.front());
    const std::array<std::uint32_t, 1> accepted{licensed};
    constexpr std::array<std::uint8_t, 1> terminal{0};
    constexpr std::array<std::uint8_t, 1> cancelled{0};
    program.resolve_pending_batch(lanes, accepted, terminal, cancelled);
    // resolve_pending_batch closes the round, so this interval is the commit the schedule body
    // itself cannot mark.
    ninfer::phase_trace_mark("resolve_pending");
    const float milliseconds = timer.stop_ms();
    std::vector<ninfer::PhaseTrace::Interval> phases;
    if (ninfer::PhaseTrace::instance().enabled()) {
        phases = ninfer::PhaseTrace::instance().collect();
    }
    return RoundMeasurement{.milliseconds   = milliseconds,
                            .licensed_tokens = licensed,
                            .phases          = std::move(phases)};
}

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
    // Two rounds of headroom. Sized exactly, the last measured round can find its output budget
    // spent and fall back to an ordinary decode step, which fails the native-path assertion below
    // -- the bench only ever passed at its default --draft-tokens 5 for that reason.
    const std::uint64_t budget_rounds = static_cast<std::uint64_t>(measured_rounds) + 2ULL;
    engine.max_context         = static_cast<std::uint32_t>(seed.size() + 64ULL +
                                                            budget_rounds *
                                                                (options.draft_tokens + 1ULL) +
                                                            2ULL * options.draft_tokens);
    engine.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(engine.max_context);
    engine.prefill_chunk       = 128;
    engine.kv_cache            = ninfer::KvCacheStorage::BFloat16;
    engine.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    engine.speculative.draft_tokens  = options.draft_tokens;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = options.use_cuda_graph;

    ninfer::DeviceContext device(options.device);
    ninfer::artifact::Reader reader(options.artifact);
    const auto weights_profile = target::Package::resolve_weights(reader.identity());
    ninfer::artifact::Binder binder(reader);
    auto load_plan = target::Package::plan_load(binder, engine, weights_profile);
    auto materialized =
        ninfer::artifact::materialize(reader, load_plan.materialization(), device, nullptr);
    auto model =
        target::Package::construct_loaded_model(std::move(load_plan), std::move(materialized));
    auto frontend = target::Package::make_frontend(*model, engine);
    auto prompt   = frontend.prepare_tokens(seed, false);

    auto planner          = target::Package::make_sequence_planner(device, engine, weights_profile);
    const auto resolution = ninfer::runtime::resolve_kv_capacity(
        engine.kv_capacity, planner.capacity_curve(), std::numeric_limits<std::size_t>::max());
    auto sequence                      = std::move(planner).finalize(resolution.main_page_groups);
    const std::size_t request_capacity = sequence.request_transient_capacity_bytes();
    auto program = target::Package::create_program(*model, std::move(sequence), device);
    ninfer::runtime::RequestMemory request_memory(device, request_capacity);
    ninfer::runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens =
        static_cast<std::uint32_t>(1ULL + budget_rounds * (options.draft_tokens + 1ULL));
    execution.allow_prefix_reuse      = false;
    auto request_base                 = program->plan_request_base(prompt, execution);
    auto request_plan                 = program->plan_request_for_lane(0, prompt, request_base);
    request_memory.activate(request_plan.summary().transient_bytes,
                            request_plan.summary().transient_alignment);
    const auto first = program->start_prefill_lane(0, std::move(prompt), std::move(request_plan),
                                                   request_memory.region());
    request_memory.deactivate();
    if (!first.complete || first.round.tokens.size() != 1) {
        throw std::runtime_error("benchmark seed prefill did not complete in one scheduling unit");
    }
    program->resolve_prefill_lane(0, false);

    // Enabled after warm-up would miss nothing, but enabling here keeps the warm-up rounds on the
    // exact same code path as the measured ones. Capacity covers every mark the k-deep body can
    // emit: open + prepare_verify + verify/accept/commit + prepare_next + align + propose[0]
    // + (k-1) ar steps + egress + resolve, with slack.
    if (options.phase_timing) {
        ninfer::PhaseTrace::instance().enable(device.stream,
                                              16 + static_cast<std::size_t>(options.draft_tokens));
    }
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        (void)measure_round(*program, device, options.draft_tokens);
    }

    // Snapshot after warm-up, not before. A round that enters with no drafts carried in counts as
    // a fallback rather than a speculative round, and the rounds right after prefill legitimately
    // do that while the proposal pipeline fills; asserting across warm-up made this bench fail at
    // every --draft-tokens except its default 5.
    const std::uint64_t rounds_before    = program->speculative_stats_lane(0).rounds;
    const std::uint64_t fallbacks_before = program->speculative_stats_lane(0).fallback_steps;

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        measurements.push_back(measure_round(*program, device, options.draft_tokens));
    }
    const ninfer::SpeculativeStats stats = program->speculative_stats_lane(0);
    const std::uint64_t measured_fallbacks = stats.fallback_steps - fallbacks_before;
    if (stats.rounds - rounds_before != static_cast<std::uint64_t>(options.repetitions) ||
        measured_fallbacks != 0) {
        throw std::runtime_error(
            "benchmark did not stay on the native MTP proposal/verify path: rounds=" +
            std::to_string(stats.rounds - rounds_before) +
            " expected=" + std::to_string(options.repetitions) +
            " fallback_steps=" + std::to_string(measured_fallbacks));
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

    if (options.phase_timing) {
        if (ninfer::PhaseTrace::instance().saw_capture()) {
            throw std::runtime_error(
                "phase timing saw a capturing stream; the schedule must run without CUDA graphs");
        }
        // Sum repeats of a label within a round first -- the k-1 autoregressive draft steps all
        // report under one name -- then average those per-round totals across repetitions.
        std::map<std::string, double> totals;
        std::map<std::string, std::uint64_t> occurrences;
        std::vector<std::string> order;
        for (const RoundMeasurement& measurement : measurements) {
            std::map<std::string, double> round_totals;
            for (const ninfer::PhaseTrace::Interval& phase : measurement.phases) {
                if (totals.find(phase.label) == totals.end() &&
                    round_totals.find(phase.label) == round_totals.end()) {
                    order.push_back(phase.label);
                }
                round_totals[phase.label] += phase.milliseconds;
                occurrences[phase.label] += 1;
            }
            for (const auto& [label, value] : round_totals) { totals[label] += value; }
        }
        const double rounds = static_cast<double>(measurements.size());
        double attributed   = 0.0;
        for (const std::string& label : order) { attributed += totals[label] / rounds; }
        std::cout << "phase_timing,true\n";
        for (const std::string& label : order) {
            const double per_round = totals[label] / rounds;
            std::cout << "phase_ms," << label << ',' << per_round << ','
                      << (100.0 * per_round / mean_ms) << '%' << ','
                      << (static_cast<double>(occurrences[label]) / rounds) << "x\n";
        }
        std::cout << "phase_attributed_ms," << attributed << '\n';
        std::cout << "phase_unattributed_ms," << (mean_ms - attributed) << '\n';
    }
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
