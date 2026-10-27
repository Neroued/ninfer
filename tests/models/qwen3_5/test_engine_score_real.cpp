#include "ninfer/engine.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk == 0 ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }
    const auto& budget = engine.startup_report().ranks.at(0);
    if (budget.minimum_context_capacity_bytes == 0 ||
        budget.context_capacity_bytes < budget.minimum_context_capacity_bytes ||
        engine.snapshot().ranks.at(0).memory.allocations.context_device_bytes !=
            budget.context_capacity_bytes) {
        std::cerr << "causal scoring did not materialize its required Context working set\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513);
    const std::vector<float> repeated = engine.score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    for (const auto* window : {&all, &suffix, &repeated}) {
        for (const float value : *window) {
            if (!std::isfinite(value) || value > 0.0F) {
                std::cerr << "causal scoring returned an invalid log probability\n";
                return 1;
            }
        }
    }
    const auto snapshot = engine.snapshot();
    if (snapshot.scoring.completed_jobs != 3 || snapshot.scoring.scored_tokens != 3584 ||
        snapshot.scoring.active_jobs != 0 || snapshot.runtime.generated_tokens != 0 ||
        snapshot.executions[static_cast<std::size_t>(ninfer::ExecutionKind::Score)].units < 3 ||
        snapshot.executions[static_cast<std::size_t>(ninfer::ExecutionKind::Score)].input_tokens !=
            4608) {
        std::cerr
            << "scoring observations mixed target scoring with generation or missed retirement\n";
        return 1;
    }
    std::cout << "OK causal_score_real\n";
    return 0;
}
