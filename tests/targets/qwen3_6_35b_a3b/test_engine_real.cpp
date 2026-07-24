#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.prefill_chunk             = 1024;
    options.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = true;
    options.use_cuda_graph            = true;
    return options;
}

ninfer::EngineOptions maximum_engine_options(const char* artifact) {
    ninfer::EngineOptions options     = engine_options(artifact);
    options.max_context               = 262144;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 5;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
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

int verify_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_6_35b_a3b" || load.tensor_count != 883 || load.resource_count != 6 ||
        load.host_to_device_bytes != 22'360'191'904ULL ||
        load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "35B Engine construction has an incomplete load summary: target="
                  << load.target << " tensors=" << load.tensor_count
                  << " resources=" << load.resource_count << " h2d=" << load.host_to_device_bytes
                  << '\n';
        return 1;
    }

    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.weights.capacity_bytes != 22'360'207'360ULL ||
        memory.weights.used_bytes != memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.workspace.capacity_bytes == 0) {
        std::cerr << "35B Engine construction has an invalid memory summary\n";
        return 1;
    }
    return 0;
}

int exercise_text_mtp_and_prefix(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(5, false));
    if (first.generated_token_ids.size() != 5 || first.speculative.draft_window != 3 ||
        first.speculative.rounds != 1 || first.speculative.drafted_tokens != 3) {
        std::cerr << "35B fixed prompt did not execute one complete K=3 MTP round: outputs="
                  << first.generated_token_ids.size() << " rounds=" << first.speculative.rounds
                  << " drafted=" << first.speculative.drafted_tokens << '\n';
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse || reused.generated_token_ids.size() != 2 ||
        reused.speculative.fallback_steps != 1) {
        std::cerr << "35B append/prefix reuse or ordinary fallback is incorrect: reused="
                  << reused.reused_prompt_tokens << " expected=" << expected_reuse
                  << " outputs=" << reused.generated_token_ids.size()
                  << " fallbacks=" << reused.speculative.fallback_steps << '\n';
        return 1;
    }

    if (first.generated_token_ids[0] == first.generated_token_ids[1]) {
        std::cerr << "35B partial-terminal fixture repeats its first token\n";
        return 1;
    }
    ninfer::RequestOptions stop_options = greedy_options(5, false);
    stop_options.stop.token_ids.push_back(first.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare_tokens(prompt), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.size() != 2) {
        std::cerr << "35B custom stop did not terminate inside an MTP round\n";
        return 1;
    }

    std::vector<ninfer::TokenId> stopped_continuation = prompt;
    stopped_continuation.insert(stopped_continuation.end(), stopped.generated_token_ids.begin(),
                                stopped.generated_token_ids.end());
    stopped_continuation.push_back(198);
    const ninfer::GenerationResult stopped_reuse = engine.generate(
        engine.prepare_tokens(std::move(stopped_continuation)), greedy_options(1, true));
    const std::uint32_t expected_stopped_reuse =
        static_cast<std::uint32_t>(prompt.size() + stopped.generated_token_ids.size() - 1);
    if (stopped_reuse.reused_prompt_tokens != expected_stopped_reuse) {
        std::cerr << "35B partial MTP terminal reused " << stopped_reuse.reused_prompt_tokens
                  << ", expected " << expected_stopped_reuse << '\n';
        return 1;
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

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(std::move(input)), greedy_options(1, false));
    if (!result.prompt.has_media || result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "35B Vision request did not complete through the public Engine\n";
        return 1;
    }
    return 0;
}

int exercise_maximum_configuration(const char* artifact) {
    ninfer::Engine engine(maximum_engine_options(artifact));
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.max_context != 262144 || memory.kv_cache != ninfer::KvCacheStorage::Int8Group64 ||
        memory.kv_payload_bytes != 3'045'064'704ULL ||
        memory.sequence.used_bytes != memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes < 1024ULL * 1024ULL * 1024ULL) {
        std::cerr << "35B maximum configuration does not match the planned 256K layout: context="
                  << memory.max_context << " kv_payload=" << memory.kv_payload_bytes
                  << " sequence=" << memory.sequence.capacity_bytes
                  << " workspace=" << memory.workspace.capacity_bytes << '\n';
        return 1;
    }

    std::vector<ninfer::TokenId> oversized(262145, 198);
    bool rejected = false;
    try {
        (void)engine.generate(engine.prepare_tokens(std::move(oversized)),
                              greedy_options(1, false));
    } catch (const std::invalid_argument&) { rejected = true; } catch (const std::out_of_range&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "35B maximum configuration accepted an over-capacity request\n";
        return 1;
    }

    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult probe =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(1, false));
    if (probe.generated_token_ids.size() != 1) {
        std::cerr << "35B Program mutated while rejecting an over-capacity request\n";
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

    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
        if (const int result = exercise_text_mtp_and_prefix(engine); result != 0) { return result; }
        if (const int result = exercise_vision(engine); result != 0) { return result; }
    }
    if (const int result = exercise_maximum_configuration(artifact); result != 0) { return result; }
    std::cout << "ok\n";
    return 0;
}
