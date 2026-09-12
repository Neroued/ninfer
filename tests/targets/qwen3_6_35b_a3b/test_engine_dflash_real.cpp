#include "ninfer/engine.h"
#include "../qwen3_6/speculative_page_boundary.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions base_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = 128;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk  = 128;
    options.kv_cache       = ninfer::KvCacheStorage::BFloat16;
    options.use_cuda_graph = false;
    options.enable_vision  = false;
    return options;
}

ninfer::EngineOptions dflash_engine_options(const char* artifact, ninfer::ProposalHead proposal,
                                            std::uint32_t max_context) {
    ninfer::EngineOptions options     = base_engine_options(artifact);
    options.max_context               = max_context;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = proposal;
    options.use_cuda_graph            = true;
    options.context_cache.device_state_slots = 2;
    return options;
}

ninfer::EngineOptions dflash_vision_engine_options(const char* artifact) {
    ninfer::EngineOptions options =
        dflash_engine_options(artifact, ninfer::ProposalHead::Optimized, 4096);
    options.prefill_chunk   = 1024;
    options.max_concurrency = 2;
    options.enable_vision   = true;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
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

ninfer::PromptInput media_conversation(ninfer::MediaKind kind, std::string session = {}) {
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = kind;
    media.media.bytes       = gradient_ppm();
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = kind == ninfer::MediaKind::Image ? "inline.ppm" : "single-frame.ppm";

    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(std::move(media));
    user.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                             .text  = "Describe the visible pattern briefly.",
                                             .media = {}});

    ninfer::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    if (!session.empty()) {
        input.context_cache.session_key = std::move(session);
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    }
    return input;
}

ninfer::PromptInput media_followup(const ninfer::GenerationResult& first) {
    ninfer::PromptInput input =
        media_conversation(ninfer::MediaKind::Image, "dflash-vision-prefix-real");
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = first.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
    input.messages.push_back(std::move(assistant));

    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Give one more detail.", .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

ninfer::PromptInput initial_conversation() {
    ninfer::PromptInput input;
    input.options.enable_thinking   = false;
    input.context_cache.session_key = "dflash-boundary-real";
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;

    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Name one prime number.", .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

ninfer::PromptInput followup_conversation(const ninfer::GenerationResult& first,
                                          std::string followup) {
    ninfer::PromptInput input = initial_conversation();
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = first.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
    input.messages.push_back(std::move(assistant));

    ninfer::ChatMessage next;
    next.role = ninfer::ChatRole::User;
    next.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(followup), .media = {}});
    input.messages.push_back(std::move(next));
    return input;
}

ninfer::PromptInput altered_history_after_boundary() {
    ninfer::PromptInput input = initial_conversation();
    ninfer::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "This history was changed.", .media = {}});
    input.messages.push_back(std::move(assistant));

    ninfer::ChatMessage next;
    next.role = ninfer::ChatRole::User;
    next.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
    input.messages.push_back(std::move(next));
    return input;
}

int verify_dflash_load(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_6_35b_a3b" || load.weights_id != "groupwise-int" ||
        load.host_to_device_bytes == 0 || load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "DFlash Engine materialized an invalid artifact payload: target="
                  << load.target << " weights=" << load.weights_id << '\n';
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.max_context != 4352 || memory.kv_cache != ninfer::KvCacheStorage::BFloat16 ||
        memory.kv_payload_bytes == 0 || memory.weights.capacity_bytes == 0 ||
        memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || memory.vision_workspace.has_value() ||
        memory.workspace_logical_peak_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "DFlash Engine has an invalid frozen memory layout\n";
        return 1;
    }
    return 0;
}

int exercise_partial_terminal(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                              const std::vector<ninfer::TokenId>& reference) {
    for (std::size_t stop_index = 1; stop_index < reference.size(); ++stop_index) {
        const ninfer::TokenId stop = reference[stop_index];
        if (std::find(reference.begin(),
                      reference.begin() + static_cast<std::ptrdiff_t>(stop_index),
                      stop) != reference.begin() + static_cast<std::ptrdiff_t>(stop_index)) {
            continue;
        }
        ninfer::RequestOptions options = greedy_options(24, true);
        options.stop.token_ids.push_back(stop);
        const ninfer::GenerationResult stopped =
            engine.generate(engine.prepare_tokens(prompt), options);
        if (stopped.finish_reason != ninfer::FinishReason::StopToken) { continue; }
        if (stopped.generated_token_ids.empty() || stopped.generated_token_ids.back() != stop) {
            std::cerr << "partial DFlash terminal did not end at the configured stop token\n";
            return 1;
        }

        const std::uint64_t fully_licensed = 1 + stopped.speculative.rounds +
                                             stopped.speculative.accepted_tokens +
                                             stopped.speculative.fallback_steps;
        if (stopped.generated_token_ids.size() >= fully_licensed) { continue; }

        std::vector<ninfer::TokenId> continuation = prompt;
        continuation.insert(continuation.end(), stopped.generated_token_ids.begin(),
                            stopped.generated_token_ids.end());
        continuation.push_back(198);
        const ninfer::GenerationResult reused = engine.generate(
            engine.prepare_tokens(std::move(continuation)), greedy_options(2, true));
        const std::uint32_t expected_reuse =
            static_cast<std::uint32_t>(prompt.size() + stopped.generated_token_ids.size() - 1);
        if (reused.reused_prompt_tokens != expected_reuse ||
            reused.generated_token_ids.size() != 2) {
            std::cerr << "partial DFlash terminal did not publish its exact context frontier: "
                      << "reused=" << reused.reused_prompt_tokens << " expected=" << expected_reuse
                      << '\n';
            return 1;
        }
        return 0;
    }
    std::cerr << "fixed DFlash fixture exposed no terminal stop inside a licensed batch\n";
    return 1;
}

int exercise_boundary_restore(ninfer::Engine& engine) {
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(initial_conversation()), greedy_options(2, true));
    if (first.generated_token_ids.size() != 2) {
        std::cerr << "DFlash boundary fixture did not establish resident state\n";
        return 1;
    }

    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(followup_conversation(first, "Answer with one digit.")),
                        greedy_options(4, true));
    if (restored.reused_prompt_tokens == 0 || restored.generated_token_ids.size() != 4) {
        std::cerr << "DFlash assistant-boundary restore did not reuse its cache snapshot: reused="
                  << restored.reused_prompt_tokens
                  << " path=" << static_cast<int>(restored.prefix_reuse_path)
                  << " source_prompt=" << first.prompt.prompt_tokens
                  << " outputs=" << restored.generated_token_ids.size() << '\n';
        return 1;
    }
    if (restored.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        restored.speculative.rounds == 0) {
        std::cerr << "DFlash assistant-boundary restore did not return to speculative decode\n";
        return 1;
    }
    return 0;
}

int exercise_long_boundary_restore(ninfer::Engine& engine) {
    constexpr std::uint32_t generated_tokens = 4100;
    const ninfer::GenerationResult long_run  = engine.generate(
        engine.prepare(initial_conversation()), greedy_options(generated_tokens, true));
    if (long_run.generated_token_ids.size() != generated_tokens) {
        std::cerr << "DFlash long-restore fixture did not cross the cyclic-cache window\n";
        return 1;
    }
    const std::uint32_t resident_frontier = long_run.prompt.prompt_tokens + generated_tokens - 1;
    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(altered_history_after_boundary()), greedy_options(2, true));
    if (restored.reused_prompt_tokens == 0 ||
        resident_frontier - restored.reused_prompt_tokens <= 4096 ||
        restored.generated_token_ids.size() != 2 ||
        restored.speculative.backend != ninfer::SpeculativeBackend::DFlash) {
        std::cerr << "DFlash long-distance restore did not use the saved cyclic cache: resident="
                  << resident_frontier << " restored=" << restored.reused_prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_vision_dflash(const char* artifact, const std::vector<ninfer::TokenId>& text_prompt) {
    constexpr std::uint32_t outputs = 12;
    ninfer::Engine engine(dflash_vision_engine_options(artifact));
    const ninfer::MemorySummary before = engine.memory_summary();
    if (!before.vision_workspace || before.workspace.capacity_bytes == 0 ||
        before.vision_workspace->general_capacity_bytes == 0 ||
        before.vision_workspace->handoff_capacity_bytes == 0) {
        std::cerr << "combined DFlash and Vision Engine has no frozen Vision workspace\n";
        return 1;
    }

    const ninfer::GenerationResult image =
        engine.generate(engine.prepare(media_conversation(ninfer::MediaKind::Image)),
                        greedy_options(outputs, false));
    if (!image.prompt.has_media || image.generated_token_ids.size() != outputs ||
        image.finish_reason != ninfer::FinishReason::OutputLimit ||
        image.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        image.speculative.rounds == 0 || !(image.timings.vision_seconds > 0.0)) {
        std::cerr << "DFlash image generation did not complete through speculative decode: outputs="
                  << image.generated_token_ids.size() << " rounds=" << image.speculative.rounds
                  << " vision=" << image.timings.vision_seconds
                  << " drafted=" << image.speculative.drafted_tokens
                  << " accepted=" << image.speculative.accepted_tokens
                  << " fallbacks=" << image.speculative.fallback_steps << '\n';
        return 1;
    }

    const ninfer::GenerationResult video =
        engine.generate(engine.prepare(media_conversation(ninfer::MediaKind::Video)),
                        greedy_options(outputs, false));
    if (!video.prompt.has_media || video.generated_token_ids.size() != outputs ||
        video.finish_reason != ninfer::FinishReason::OutputLimit ||
        video.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        video.speculative.rounds == 0 || !(video.timings.vision_seconds > 0.0)) {
        std::cerr << "DFlash video generation did not complete through speculative decode: outputs="
                  << video.generated_token_ids.size() << " rounds=" << video.speculative.rounds
                  << " vision=" << video.timings.vision_seconds << '\n';
        return 1;
    }

    auto text_request =
        engine.submit(engine.prepare_tokens(text_prompt), greedy_options(outputs, false));
    auto image_request = engine.submit(engine.prepare(media_conversation(ninfer::MediaKind::Image)),
                                       greedy_options(outputs, false));
    const ninfer::GenerationResult mixed_text  = text_request.wait();
    const ninfer::GenerationResult mixed_image = image_request.wait();
    if (mixed_text.generated_token_ids.size() != outputs ||
        mixed_image.generated_token_ids.size() != outputs ||
        mixed_text.finish_reason != ninfer::FinishReason::OutputLimit ||
        mixed_image.finish_reason != ninfer::FinishReason::OutputLimit ||
        mixed_text.prompt.has_media || !mixed_image.prompt.has_media ||
        mixed_text.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        mixed_image.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        mixed_text.speculative.rounds == 0 || mixed_image.speculative.rounds == 0 ||
        mixed_text.timings.vision_seconds != 0.0 || !(mixed_image.timings.vision_seconds > 0.0)) {
        std::cerr << "mixed text/Vision DFlash CUDA Graph batch did not complete independently\n";
        return 1;
    }

    const ninfer::GenerationResult first = engine.generate(
        engine.prepare(media_conversation(ninfer::MediaKind::Image, "dflash-vision-prefix-real")),
        greedy_options(4, true));
    if (!first.prompt.has_media || first.generated_token_ids.size() != 4 ||
        first.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        first.speculative.rounds == 0 || !(first.timings.vision_seconds > 0.0)) {
        std::cerr << "retained DFlash Vision prefix did not complete speculative decode\n";
        return 1;
    }
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(media_followup(first)), greedy_options(4, true));
    const ninfer::GenerationResult fresh =
        engine.generate(engine.prepare(media_followup(first)), greedy_options(4, false));
    if (reused.reused_prompt_tokens == 0 || reused.timings.vision_seconds != 0.0 ||
        reused.generated_token_ids.size() != 4 || fresh.generated_token_ids.size() != 4 ||
        reused.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        fresh.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        reused.speculative.rounds == 0 || fresh.speculative.rounds == 0 ||
        !(fresh.timings.vision_seconds > 0.0)) {
        std::cerr << "DFlash multimodal continuation did not preserve reusable context: reused="
                  << reused.reused_prompt_tokens << " vision=" << reused.timings.vision_seconds
                  << " rounds=" << reused.speculative.rounds << '\n';
        return 1;
    }

    const ninfer::MemorySummary after = engine.memory_summary();
    if (!after.vision_workspace || after.vision_workspace->handoff_active_bytes != 0 ||
        after.vision_workspace->handoff_peak_bytes == 0 ||
        after.workspace_logical_peak_bytes == 0 ||
        after.workspace_logical_peak_bytes > after.workspace.capacity_bytes) {
        std::cerr
            << "combined DFlash and Vision execution exceeded or bypassed its workspace plan\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_35B_A3B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_35B_A3B_WEIGHTS is not set\n";
        return 77;
    }

    const std::vector<ninfer::TokenId> prompt{
        248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
        1710,   248046, 198, 248045, 74455, 198,   248068, 198,
    };
    if (const int result = exercise_vision_dflash(artifact, prompt); result != 0) { return result; }

    {
        ninfer::EngineOptions options =
            dflash_engine_options(artifact, ninfer::ProposalHead::Full, 128);
        options.max_concurrency = 2;
        ninfer::Engine full(std::move(options));
        auto first  = full.submit(full.prepare_tokens(prompt), greedy_options(17, false));
        auto second = full.submit(full.prepare_tokens(prompt), greedy_options(9, false));
        const ninfer::GenerationResult first_result  = first.wait();
        const ninfer::GenerationResult second_result = second.wait();
        const auto valid = [&](const ninfer::GenerationResult& result, std::size_t count) {
            return result.generated_token_ids.size() == count &&
                   result.finish_reason == ninfer::FinishReason::OutputLimit &&
                   result.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                   result.speculative.rounds != 0;
        };
        if (!valid(first_result, 17) || !valid(second_result, 9)) {
            std::cerr << "concurrent full-head DFlash Graph requests did not complete\n";
            return 1;
        }
    }

    ninfer::Engine engine(dflash_engine_options(artifact, ninfer::ProposalHead::Optimized, 4352));
    if (const int result = verify_dflash_load(engine); result != 0) { return result; }
    ninfer::test::speculative_page_boundary(engine);
    engine.reset_memory_peaks();
    const ninfer::GenerationResult dflash =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(24, false));
    if (dflash.generated_token_ids.size() != 24 ||
        dflash.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "optimized-head DFlash Graph request did not complete\n";
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.workspace_logical_peak_bytes == 0 ||
        memory.workspace_logical_peak_bytes > memory.workspace.capacity_bytes) {
        std::cerr << "DFlash request did not report a valid planned workspace phase\n";
        return 1;
    }
    if (dflash.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        dflash.speculative.rounds == 0) {
        std::cerr << "DFlash fixture did not execute speculative decode\n";
        return 1;
    }
    if (const int result = exercise_boundary_restore(engine); result != 0) { return result; }
    if (const int result = exercise_partial_terminal(engine, prompt, dflash.generated_token_ids);
        result != 0) {
        return result;
    }
    if (const int result = exercise_long_boundary_restore(engine); result != 0) { return result; }

    std::cout << "ok\n";
    return 0;
}
