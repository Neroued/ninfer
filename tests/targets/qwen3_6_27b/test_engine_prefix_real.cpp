#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
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

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput chinese_chat(bool enable_thinking) {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "你好，简单介绍一下你自己。", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = enable_thinking;
    return input;
}

int exercise_registered_frontend(const ninfer::Engine& engine) {
    if (engine.count_tokens(chinese_chat(true)) != 16) {
        std::cerr << "registered tokenizer/chat template changed the thinking prompt golden\n";
        return 1;
    }
    if (engine.count_tokens(chinese_chat(false)) != 18) {
        std::cerr << "registered tokenizer/chat template changed the no-thinking prompt golden\n";
        return 1;
    }
    return 0;
}

int exercise_partial_mtp_terminal(ninfer::Engine& engine,
                                  const std::vector<ninfer::TokenId>& prompt,
                                  const ninfer::GenerationResult& baseline) {
    if (baseline.generated_token_ids.size() < 2 || baseline.speculative.rounds != 1 ||
        baseline.speculative.accepted_tokens == 0) {
        std::cerr << "baseline did not expose a multi-token first MTP round\n";
        return 1;
    }
    const ninfer::TokenId stop = baseline.generated_token_ids[1];
    if (stop == baseline.generated_token_ids[0]) {
        std::cerr << "baseline repeats its first token before the MTP stop boundary\n";
        return 1;
    }

    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 5;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    options.stop.token_ids.push_back(stop);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare_tokens(prompt), options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.size() != 2 ||
        stopped.generated_token_ids[0] != baseline.generated_token_ids[0] ||
        stopped.generated_token_ids[1] != stop) {
        std::cerr << "custom stop did not terminate inside the first MTP round\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), stopped.generated_token_ids.begin(),
                        stopped.generated_token_ids.end());
    continuation.push_back(198);
    ninfer::RequestOptions probe;
    probe.execution.requested_output_tokens = 1;
    probe.execution.sampling.temperature    = 0.0F;
    probe.execution.allow_prefix_reuse      = true;
    probe.stop.include_model_defaults       = false;
    const ninfer::GenerationResult after =
        engine.generate(engine.prepare_tokens(std::move(continuation)), probe);
    if (after.reused_prompt_tokens != 0) {
        std::cerr << "strict-prefix MTP terminal left provisional Program state reusable\n";
        return 1;
    }
    return 0;
}

int exercise_zero_suffix_gdn(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    ninfer::RequestOptions baseline_options;
    baseline_options.execution.requested_output_tokens = 8;
    baseline_options.execution.sampling.temperature    = 0.0F;
    baseline_options.execution.allow_prefix_reuse      = false;
    baseline_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(prompt), baseline_options);
    // With K=3 and no fallback, eight outputs can only finish on a four-token MTP return. The
    // committed target state therefore lives in snapshot slot 3 rather than slot 0.
    if (baseline.generated_token_ids.size() != 8 || baseline.speculative.rounds == 0 ||
        baseline.speculative.draft_window != 3 || baseline.speculative.fallback_steps != 0 ||
        1 + baseline.speculative.rounds + baseline.speculative.accepted_tokens !=
            baseline.generated_token_ids.size() ||
        baseline.speculative.accepted_per_position.size() != baseline.speculative.draft_window ||
        baseline.speculative.accepted_per_position.back() == 0) {
        std::cerr << "zero-suffix fixture did not end on a fully accepted MTP round: outputs="
                  << baseline.generated_token_ids.size()
                  << " rounds=" << baseline.speculative.rounds
                  << " draft_window=" << baseline.speculative.draft_window
                  << " accepted=" << baseline.speculative.accepted_tokens
                  << " fallbacks=" << baseline.speculative.fallback_steps << '\n';
        return 1;
    }

    std::vector<ninfer::TokenId> exact_frontier = prompt;
    exact_frontier.insert(exact_frontier.end(), baseline.generated_token_ids.begin(),
                          baseline.generated_token_ids.end() - 1);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 2;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(exact_frontier), reuse_options);
    if (reused.reused_prompt_tokens != exact_frontier.size()) {
        std::cerr << "zero-suffix reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << exact_frontier.size() << '\n';
        return 1;
    }
    if (reused.generated_token_ids.size() != 2 ||
        reused.generated_token_ids.front() != baseline.generated_token_ids.back() ||
        reused.speculative.fallback_steps != 1) {
        std::cerr << "zero-suffix reuse did not resume and take one ordinary target step\n";
        return 1;
    }

    ninfer::RequestOptions reset_options       = reuse_options;
    reset_options.execution.allow_prefix_reuse = false;
    const ninfer::GenerationResult reset =
        engine.generate(engine.prepare_tokens(exact_frontier), reset_options);
    if (reused.generated_token_ids != reset.generated_token_ids) {
        std::cerr << "zero-suffix reuse lost the committed MTP GDN snapshot\n";
        return 1;
    }
    return 0;
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

    if (const int result = exercise_zero_suffix_gdn(engine, prompt); result != 0) { return result; }

    if (const int result = exercise_partial_mtp_terminal(engine, prompt, first); result != 0) {
        return result;
    }

    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;

    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(std::move(input)), options);
    if (!result.prompt.has_media || result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "real Vision request did not complete through the public Engine\n";
        return 1;
    }
    return 0;
}

int verify_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_6_27b_rtx5090" || load.tensor_count != 1166 ||
        load.resource_count != 6 || load.host_to_device_bytes == 0 ||
        load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "Engine construction has an incomplete load summary\n";
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.weights.capacity_bytes == 0 ||
        memory.weights.used_bytes != memory.weights.capacity_bytes) {
        std::cerr << "Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }

    ninfer::Engine engine(engine_options(artifact));
    if (const int result = verify_loaded_product(engine); result != 0) { return result; }
    if (const int result = exercise_registered_frontend(engine); result != 0) { return result; }
    if (const int result = exercise_prefix(engine); result != 0) { return result; }
    if (const int result = exercise_vision(engine); result != 0) { return result; }
    std::cout << "ok\n";
    return 0;
}
