#include "ninfer/engine.h"
#include "product/config/options.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;

void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}

void generation(EngineOptions options) {
    Engine engine(options);
    RequestOptions constrained;
    constrained.constraint = OutputConstraint::choice({"route_search", "route_answer"});
    constrained.execution.requested_output_tokens = 32;
    constrained.execution.allow_prefix_reuse      = false;
    auto prompt = engine.prepare_tokens(engine.tokenize_text("Choose a route."), false);
    RequestOptions ordinary;
    ordinary.execution.requested_output_tokens   = 64;
    ordinary.execution.allow_prefix_reuse        = false;
    ordinary.stop.include_model_defaults         = false;
    ordinary.execution.sampling.presence_penalty = 0.2F;
    std::atomic<bool> first = false, cancel = false;
    auto long_running = engine.submit(engine.prepare_tokens({41, 198, 50}, false), ordinary,
                                      OutputConsumerMode::Aggregate,
                                      {.first_token = [&](const auto&) { first = true; }});
    // Start the chunked source while the first source is decoding on a disjoint residency.
    auto cancelled = std::async(std::launch::async, [&] {
        return long_running.wait(nullptr, CancellationView([&] { return cancel.load(); }));
    });
    while (!first &&
           cancelled.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready) {}
    require(first, "pipeline failed to publish its first token");
    auto chunked_request                              = ordinary;
    chunked_request.execution.requested_output_tokens = 4;
    auto chunked = engine.submit(engine.prepare_tokens(std::vector<TokenId>(279, 198), false),
                                 chunked_request);
    auto grammar = engine.submit(std::move(prompt), constrained);
    std::size_t peak_frames = 0;
    bool mixed              = false;
    auto others =
        std::async(std::launch::async, [&] { return std::pair(chunked.wait(), grammar.wait()); });
    do {
        const auto snapshot = engine.snapshot();
        std::size_t active  = 0;
        bool prefill = false, decode = false;
        for (const auto& flight : snapshot.in_flight) {
            if (flight.work.kind == ExecutionKind::Prefill ||
                flight.work.kind == ExecutionKind::Decode) {
                ++active;
                prefill |= flight.work.kind == ExecutionKind::Prefill;
                decode |= flight.work.kind == ExecutionKind::Decode;
            }
        }
        peak_frames = std::max(peak_frames, active);
        mixed |= prefill && decode;
        if (mixed) cancel = true;
    } while (others.wait_for(std::chrono::milliseconds(2)) != std::future_status::ready);
    cancel                     = true;
    auto [long_result, choice] = others.get();
    auto stopped               = cancelled.get();
    require(peak_frames >= 2 && mixed, "no overlapping prefill and decode tickets observed");
    require(stopped.finish_reason == FinishReason::Cancelled &&
                !stopped.generated_token_ids.empty(),
            "cancellation did not retire the active decode");
    require(long_result.stats.computed_prefill_tokens == 279 &&
                long_result.generated_token_ids.size() == 4,
            "chunked prefill did not reach its own output boundary");
    require(choice.finish_reason == FinishReason::StopToken && choice.stats.constraint &&
                choice.stats.constraint->mask_upload_bytes && choice.stats.constraint->complete &&
                std::ranges::find(constrained.constraint->choices, choice.content) !=
                    constrained.constraint->choices.end(),
            "pipeline grammar or mask accounting did not reach the accepted language");
    engine.reset_measurement_peaks();
    const auto after = engine.snapshot();
    require(after.in_flight.empty() && !after.runtime.running_requests, "pipeline did not drain");
    for (const auto& rank : after.ranks) {
        require(!rank.host.supply_ns, "W=1 pipeline waited for a separate grammar command");
        const auto graph = rank.compute_paths[static_cast<std::size_t>(ExecutionPath::Graph)];
        require(!options.use_cuda_graph || graph, "pipeline stage did not replay a decode graph");
        const bool producer = rank.rank < options.deployment->parallel.tp;
        const auto tx       = rank.communication_eager.payload_bytes[producer ? 4 : 5];
        require(tx && !rank.communication_graph.calls[4] && !rank.communication_graph.calls[5],
                "pipeline edge did not transfer outside the graph");
        std::cout << "rank=" << rank.rank << " pp_payload=" << tx << " graph_replays=" << graph
                  << '\n';
    }
    std::cout << "overlapping_frames=" << peak_frames
              << " prefill=" << long_result.stats.computed_prefill_tokens
              << " grammar=" << choice.content
              << " mask_bytes=" << choice.stats.constraint->mask_upload_bytes
              << " cancelled_after=" << stopped.generated_token_ids.size() << std::endl;
    auto stopping = engine.submit(engine.prepare_tokens({41, 198, 50}, false), ordinary,
                                  OutputConsumerMode::Aggregate,
                                  {.first_token = [&](const auto&) { engine.request_stop(); }});
    try {
        (void)stopping.wait();
        throw std::runtime_error("pipeline stop left a generation running");
    } catch (const RequestError& e) {
        require(e.kind() == RequestErrorKind::Unavailable && !engine.is_available(),
                "pipeline stop lost the public terminal reason");
    }
}

void scoring(EngineOptions options) {
    options.purpose         = EnginePurpose::CausalScoring;
    options.max_concurrency = 1;
    Engine engine(options);
    std::vector<TokenId> tokens;
    for (int i = 0; i < 261; ++i) tokens.push_back(50 + (i * 17 + i / 3) % 89);
    const auto full   = engine.score_tokens(tokens, 1);
    const auto suffix = engine.score_tokens(tokens, 126);
    const auto finite = [](float p) { return std::isfinite(p) && p <= 0; };
    require(full.size() == 260 && suffix.size() == 135 && std::ranges::all_of(full, finite) &&
                std::ranges::all_of(suffix, finite),
            "pipeline scoring lost causal targets");
    engine.reset_measurement_peaks();
    require(engine.snapshot().in_flight.empty(), "pipeline scoring did not drain");
    std::cout << "score_targets=" << full.size() << ',' << suffix.size() << std::endl;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument(
                "usage: pipeline_engine MODEL DEPLOYMENT eager|graph|score");
        const std::string_view mode = argv[3];
        EngineOptions options;
        options.artifact_path = argv[1];
        options.deployment    = ninfer::product::read_deployment_file(argv[2]);
        require(options.deployment->parallel.pp == 2 && options.deployment->parallel.dp == 1,
                "fixture requires two stages in one replica");
        options.prefill_chunk                     = 128;
        options.max_context                       = 512;
        options.max_concurrency                   = 3;
        options.context_cache.host_capacity_bytes = 0;
        options.context_capacity = ContextCapacityPolicy::explicit_capacity(2ULL << 30);
        options.use_cuda_graph   = mode == "graph";
        if (mode == "score")
            scoring(options);
        else if (mode == "eager" || mode == "graph")
            generation(options);
        else
            throw std::invalid_argument("unknown pipeline test mode");
        std::cout << "PASS" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
