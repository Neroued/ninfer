#include "ninfer/engine.h"
#include "product/config/options.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;

void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}

void lifecycle(Engine& engine) {
    PromptInput chat;
    chat.options.enable_thinking = false;
    chat.messages.push_back(
        {.role  = ChatRole::User,
         .parts = {{.kind = MessagePartKind::Text, .text = "Choose a route."}}});
    RequestOptions constrained;
    constrained.constraint = OutputConstraint::choice({"route_search", "route_answer"});
    constrained.execution.requested_output_tokens = 32;
    constrained.execution.allow_prefix_reuse      = false;
    RequestOptions long_request;
    long_request.execution.requested_output_tokens = 96;
    long_request.execution.allow_prefix_reuse      = false;
    long_request.stop.include_model_defaults       = false;
    auto prepared_chat                             = engine.prepare(std::move(chat));
    std::atomic<bool> published                    = false;
    auto long_decode = engine.submit(engine.prepare_tokens(std::vector<TokenId>(279, 198), false),
                                     long_request, OutputConsumerMode::Aggregate,
                                     {.first_token = [&](const auto&) { published = true; }});
    auto grammar     = engine.submit(std::move(prepared_chat), constrained);
    const auto cancelled =
        long_decode.wait(nullptr, CancellationView([&] { return published.load(); }));
    const auto result = grammar.wait();
    require(published && cancelled.finish_reason == FinishReason::Cancelled &&
                cancelled.stats.computed_prefill_tokens == 279 &&
                cancelled.generated_token_ids.size() <
                    long_request.execution.requested_output_tokens,
            "long prefill and cancellation did not retire the request");
    require(result.finish_reason == FinishReason::StopToken &&
                std::ranges::find(constrained.constraint->choices, result.content) !=
                    constrained.constraint->choices.end() &&
                engine.is_available(),
            "concurrent cancellation interrupted the constrained source");
    engine.reset_measurement_peaks();
    const auto snapshot = engine.snapshot();
    require(snapshot.in_flight.empty() && !snapshot.runtime.running_requests,
            "measurement cut retained a service or a request");
    auto stopping = engine.submit(engine.prepare_tokens({198, 41, 198}, false), long_request,
                                  OutputConsumerMode::Aggregate,
                                  {.first_token = [&](const auto&) { engine.request_stop(); }});
    try {
        (void)stopping.wait();
        throw std::runtime_error("stop did not terminate an active source");
    } catch (const RequestError& e) {
        require(e.kind() == RequestErrorKind::Unavailable && !engine.is_available(),
                "stop changed the public terminal reason");
    }
    std::cout << "lifecycle constrained=" << result.content
              << " cancelled_after=" << cancelled.generated_token_ids.size() << " stop=unavailable"
              << std::endl;
}

void generation(EngineOptions options, bool measure) {
    Engine engine(options);
    if (options.speculative.backend != SpeculativeBackend::None &&
        options.speculative.length_policy == SpeculativeLengthPolicy::Adaptive) {
        const auto& summary = engine.startup_report().speculative;
        require(summary && summary->calibrated && summary->calibration_contexts,
                "adaptive startup did not finish a calibration context");
        std::cout << "calibration anchors=" << summary->calibration_anchors
                  << " contexts=" << summary->calibration_contexts << std::endl;
    }
    RequestOptions request;
    constexpr auto output_tokens              = 9U;
    request.execution.requested_output_tokens = output_tokens;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const auto submit                         = [&](std::string_view prompt) {
        return engine.submit(engine.prepare_tokens(engine.tokenize_text(prompt), false), request);
    };
    constexpr auto a    = "One plus two equals";
    constexpr auto b    = "The capital of France is";
    const auto single_a = submit(a).wait();
    const auto single_b = submit(b).wait();
    require(single_a.generated_token_ids.size() == output_tokens &&
                single_b.generated_token_ids.size() == output_tokens,
            "single source did not deliver the requested tokens");
    auto pending_a    = submit(a);
    auto pending_b    = submit(b);
    const auto pair_a = pending_a.wait();
    const auto pair_b = pending_b.wait();
    require(pair_a.replica_id && pair_b.replica_id && pair_a.replica_id != pair_b.replica_id,
            "concurrent generations did not run on distinct replicas");
    require(pair_a.generated_token_ids.size() == output_tokens &&
                pair_b.generated_token_ids.size() == output_tokens,
            "concurrent generation did not deliver the requested tokens");
    engine.reset_measurement_peaks();
    const auto snapshot = engine.snapshot();
    const auto& service =
        snapshot.executions.at(static_cast<std::size_t>(ExecutionKind::ExpertService));
    require(service.units && !service.rows && !service.input_tokens && !service.scheduled_positions,
            "generation did not retire physical zero-source service");
    if (options.speculative.backend != SpeculativeBackend::None)
        require(single_a.stats.speculative.rounds && pair_a.stats.speculative.proposed_tokens &&
                    pair_b.stats.speculative.proposed_tokens,
                "speculative sources did not execute proposals");
    if (options.use_cuda_graph) {
        for (const auto& rank : snapshot.ranks)
            require(rank.compute_paths[static_cast<std::size_t>(ExecutionPath::Graph)] &&
                        rank.communication_graph.calls[4] && rank.communication_graph.calls[5],
                    "rank did not replay the expert send and receive");
    }
    std::cout << "generation paired owners=" << *pair_a.replica_id << ',' << *pair_b.replica_id
              << " service=" << service.units
              << " proposals=" << snapshot.runtime.speculative_proposed_tokens << std::endl;
    if (measure) {
        const auto before = engine.begin_measurement();
        std::vector<double> ttft, tpot;
        for (unsigned i = 0; i < 8; ++i) {
            auto first = submit(a), second = submit(b);
            for (auto* handle : {&first, &second}) {
                const auto result = handle->wait();
                require(result.generated_token_ids.size() == output_tokens,
                        "measurement request ended early");
                ttft.push_back(1000 * result.stats.timings.first_token_seconds.value());
                tpot.push_back(1000 * result.stats.timings.generation_wall_seconds /
                               (output_tokens - 1));
            }
        }
        const auto after = engine.end_measurement();
        std::ranges::sort(ttft);
        std::ranges::sort(tpot);
        std::cout << "paired requests=" << ttft.size() << " ttft_ms_p50=" << ttft[7]
                  << " ttft_ms_p95=" << ttft[15] << " tpot_ms_p50=" << tpot[7]
                  << " tpot_ms_p95=" << tpot[15] << std::endl;
        for (std::size_t i = 0; i < after.ranks.size(); ++i) {
            const auto& rank = after.ranks[i];
            const auto send  = [](const auto& r) {
                return r.communication_eager.payload_bytes[4] +
                       r.communication_graph.payload_bytes[4];
            };
            std::cout << "rank=" << rank.rank
                      << " expert_tx_bytes=" << send(rank) - send(before.ranks[i])
                      << " workspace_peak_bytes=" << rank.memory.workspace.measurement_peak_bytes
                      << " context_bytes="
                      << engine.startup_report().ranks.at(i).context_capacity_bytes << std::endl;
        }
    }
    lifecycle(engine);
}

void scoring(EngineOptions options) {
    options.purpose         = EnginePurpose::CausalScoring;
    options.max_concurrency = 1;
    Engine engine(options);
    // Three native Session chunks, with distinct predecessors at each boundary.
    std::vector<TokenId> tokens;
    for (int i = 0; i < 261; ++i) tokens.push_back(50 + (i * 17 + i / 3) % 89);
    const auto full   = engine.score_tokens(tokens, 1);
    const auto suffix = engine.score_tokens(tokens, 126);
    require(full.size() == tokens.size() - 1 && suffix.size() == tokens.size() - 126,
            "scoring lost targets across chunks");
    const auto valid_logprob = [](float score) { return std::isfinite(score) && score <= 0; };
    require(std::ranges::all_of(full, valid_logprob) && std::ranges::all_of(suffix, valid_logprob),
            "invalid causal log probability");
    engine.reset_measurement_peaks();
    const auto snapshot = engine.snapshot();
    const auto service  = static_cast<std::size_t>(ExecutionKind::ExpertService);
    require(snapshot.replicas.size() == 2 && snapshot.executions[service].units >= 6 &&
                !snapshot.executions[service].rows,
            "idle replica failed to serve every scoring chunk");
    for (const auto& replica : snapshot.replicas)
        require(replica.executions[static_cast<std::size_t>(ExecutionKind::Score)].units == 3 &&
                    replica.executions[service].units == 3,
                "score did not swap source replicas");
    std::cout << "scoring targets=" << full.size() << ',' << suffix.size()
              << " service=" << snapshot.executions[service].units << std::endl;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 5 || (argc == 5 && std::string_view(argv[4]) != "--measure"))
            throw std::invalid_argument(
                "usage: ninfer_expert_engine_test MODEL DEPLOYMENT [MODE [--measure]]");
        const std::string_view mode = argc >= 4 ? argv[3] : "eager";
        if (mode != "eager" && mode != "mtp-eager" && mode != "graph" && mode != "mtp-graph" &&
            mode != "adaptive-graph" && mode != "adaptive-eager")
            throw std::invalid_argument("unknown expert Engine test mode");
        ninfer::EngineOptions o;
        o.artifact_path = argv[1];
        o.deployment    = ninfer::product::read_deployment_file(argv[2]);
        const auto& p   = o.deployment->parallel;
        require(p.dp == 2 && p.tp == 1 && p.adp == 1 && p.ep == 2 && p.pp == 1,
                "this Engine test requires dp2/tp1/adp1/ep2/pp1");
        o.ple.storage                       = ninfer::PleStorage::Disk;
        o.prefill_chunk                     = 128;
        o.max_context                       = 512;
        o.max_concurrency                   = 2;
        o.context_cache.host_capacity_bytes = 0;
        o.context_capacity = ninfer::ContextCapacityPolicy::explicit_capacity(2ULL << 30U);
        o.use_cuda_graph   = mode.ends_with("graph");
        if (mode.starts_with("mtp") || mode.starts_with("adaptive"))
            o.speculative = {.backend       = ninfer::SpeculativeBackend::Mtp,
                             .draft_tokens  = 3,
                             .length_policy = mode.starts_with("adaptive")
                                                  ? SpeculativeLengthPolicy::Adaptive
                                                  : SpeculativeLengthPolicy::Fixed};
        generation(o, argc == 5);
        if (mode == "eager") scoring(o);
        std::cout << "PASS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
