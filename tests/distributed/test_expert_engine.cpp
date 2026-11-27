#include "ninfer/engine.h"
#include "product/config/options.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;

void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}

void generation(EngineOptions options) {
    Engine engine(options);
    RequestOptions request;
    request.execution.requested_output_tokens = 3;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const auto submit                         = [&](std::string_view prompt) {
        return engine.submit(engine.prepare_tokens(engine.tokenize_text(prompt), false), request);
    };
    constexpr auto a    = "One plus two equals";
    constexpr auto b    = "The capital of France is";
    const auto single_a = submit(a).wait();
    const auto single_b = submit(b).wait();
    require(single_a.generated_token_ids.size() == 3 && single_b.generated_token_ids.size() == 3,
            "single source failed to generate three tokens");
    auto pending_a    = submit(a);
    auto pending_b    = submit(b);
    const auto pair_a = pending_a.wait();
    const auto pair_b = pending_b.wait();
    require(pair_a.replica_id && pair_b.replica_id && pair_a.replica_id != pair_b.replica_id,
            "concurrent generations did not run on distinct replicas");
    require(pair_a.generated_token_ids.size() == 3 && pair_b.generated_token_ids.size() == 3,
            "concurrent generation did not deliver the requested tokens");
    engine.reset_measurement_peaks();
    const auto snapshot = engine.snapshot();
    const auto& service =
        snapshot.executions.at(static_cast<std::size_t>(ExecutionKind::ExpertService));
    require(service.units && !service.rows && !service.input_tokens && !service.scheduled_positions,
            "generation did not retire physical zero-source service");
    std::cout << "generation paired owners=" << *pair_a.replica_id << ',' << *pair_b.replica_id
              << " service=" << service.units << std::endl;
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
        if (argc != 3)
            throw std::invalid_argument("usage: ninfer_expert_engine_test MODEL DEPLOYMENT");
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
        o.use_cuda_graph   = false;
        o.speculative      = {};
        generation(o);
        scoring(o);
        std::cout << "PASS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
