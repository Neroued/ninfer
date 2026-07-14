#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.prefill_chunk             = 1024;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

int exercise_prefix(ninfer::Engine& engine) {
    ninfer::RequestOptions first_options;
    first_options.execution.requested_output_tokens = 5;
    first_options.execution.sampling.temperature    = 0.0F;
    first_options.stop.include_model_defaults       = false;

    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), first_options);
    if (first.generated_token_ids.size() != 5) {
        std::cerr << "first request did not generate five tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 5;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), reuse_options);

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse) {
        std::cerr << "append reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << expected_reuse << '\n';
        return 1;
    }

    ninfer::RequestOptions reset_options       = reuse_options;
    reset_options.execution.allow_prefix_reuse = false;
    const ninfer::GenerationResult reset =
        engine.generate(engine.prepare_tokens(continuation), reset_options);
    if (reused.generated_token_ids != reset.generated_token_ids) {
        std::cerr << "reused continuation differs from a full reset continuation\n";
        return 1;
    }
    if (reused.speculative.rounds != reset.speculative.rounds ||
        reused.speculative.drafted_tokens != reset.speculative.drafted_tokens ||
        reused.speculative.accepted_tokens != reset.speculative.accepted_tokens ||
        reused.speculative.accepted_per_position != reset.speculative.accepted_per_position) {
        std::cerr << "reused MTP frontier differs from a full reset frontier\n";
        return 1;
    }

    return 0;
}

int verify_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_6_27b_rtx5090" || load.tensor_count != 1166 ||
        load.resource_count != 6 || load.host_to_device_bytes == 0 ||
        load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "second Engine construction has an incomplete load summary\n";
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.weights.capacity_bytes == 0 ||
        memory.weights.used_bytes != memory.weights.capacity_bytes) {
        std::cerr << "second Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 0;
    }

    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = exercise_prefix(engine); result != 0) { return result; }
    }

    // The second construction happens only after the first Engine, Program, loaded resources,
    // weight arena, and DeviceContext have all been destroyed in this process.
    {
        const ninfer::Engine engine(engine_options(artifact));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
    }
    std::cout << "ok\n";
    return 0;
}
