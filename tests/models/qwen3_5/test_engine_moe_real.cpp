#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 4096;
    options.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.kv_cache                          = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend               = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens          = 3;
    options.speculative.proposal_head         = ninfer::ProposalHead::Optimized;
    options.enable_vision                     = true;
    options.use_cuda_graph                    = std::getenv("NINFER_TEST_EAGER") == nullptr;
    options.context_cache.host_capacity_bytes = 0;
    return options;
}

ninfer::EngineOptions maximum_engine_options(const char* artifact) {
    ninfer::EngineOptions options     = engine_options(artifact);
    options.max_context               = 262144;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(262144);
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 5;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
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
    const ninfer::LoadSummary load = engine.startup_report().load;
    if (load.architecture != "Qwen3_5MoeForCausalLM" || load.model_name.empty() ||
        load.host_to_device_bytes == 0 || load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "35B Engine construction has an invalid load summary: target="
                  << load.architecture << " weights=" << load.prefill_signature << '\n';
        return 1;
    }

    const auto memory  = engine.snapshot().ranks.at(0).memory;
    const auto* vision = memory.vision ? &*memory.vision : nullptr;
    if (memory.allocations.device_weights_bytes == 0 ||

        memory.allocations.persistent_device_bytes == 0 ||

        memory.allocations.workspace_device_bytes == 0 || vision == nullptr ||
        vision->aggregate_prompt_tokens != 4096 || vision->max_item_tokens != 4096 ||
        vision->general_capacity_bytes == 0 || vision->encode_required_bytes == 0 ||
        vision->handoff_offset_bytes > memory.allocations.workspace_device_bytes ||
        vision->handoff_capacity_bytes == 0 ||
        vision->handoff_capacity_bytes >
            memory.allocations.workspace_device_bytes - vision->handoff_offset_bytes ||
        vision->handoff_active_bytes != 0 || vision->handoff_peak_bytes != 0 ||
        memory.workspace.measurement_peak_bytes != 0 ||
        (engine.startup_report().ranks.at(0).graphs.has_value()) !=
            engine.options().use_cuda_graph) {
        std::cerr << "35B Engine construction has an invalid memory summary: weights="
                  << memory.allocations.device_weights_bytes
                  << " sequence=" << memory.allocations.persistent_device_bytes
                  << " workspace=" << memory.allocations.workspace_device_bytes
                  << " logical_peak=" << memory.workspace.measurement_peak_bytes
                  << " graph=" << engine.startup_report().ranks.at(0).graphs.has_value()
                  << " expected_graph=" << engine.options().use_cuda_graph << '\n';
        if (vision)
            std::cerr << "vision: prompt=" << vision->aggregate_prompt_tokens
                      << " item=" << vision->max_item_tokens
                      << " general=" << vision->general_capacity_bytes
                      << " encode_peak=" << vision->encode_required_bytes
                      << " handoff=" << vision->handoff_offset_bytes << '+'
                      << vision->handoff_capacity_bytes
                      << " active=" << vision->handoff_active_bytes
                      << " peak=" << vision->handoff_peak_bytes << '\n';
        else
            std::cerr << "vision workspace is absent\n";
        return 1;
    }
    return 0;
}

int exercise_text_mtp_and_prefix(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(5, true));
    if (first.generated_token_ids.size() != 5 ||
        first.stats.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
        first.stats.speculative.rounds == 0) {
        std::cerr << "35B fixed prompt did not complete through MTP: outputs="
                  << first.generated_token_ids.size() << '\n';
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
    if (reused.stats.reused_prompt_tokens != expected_reuse ||
        reused.generated_token_ids.size() != 2) {
        std::cerr << "35B append/prefix reuse is incorrect: reused="
                  << reused.stats.reused_prompt_tokens << " expected=" << expected_reuse
                  << " outputs=" << reused.generated_token_ids.size() << '\n';
        return 1;
    }

    const auto stop_reference =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(5, false));
    if (stop_reference.generated_token_ids.size() != 5 ||
        stop_reference.generated_token_ids[0] == stop_reference.generated_token_ids[1]) {
        std::cerr << "35B partial-terminal fixture repeats its first token\n";
        return 1;
    }
    ninfer::RequestOptions stop_options = greedy_options(5, false);
    stop_options.stop.token_ids.push_back(stop_reference.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare_tokens(prompt), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.empty() || stopped.stats.speculative.rounds == 0 ||
        stopped.generated_token_ids.back() != stop_options.stop.token_ids.front()) {
        std::cerr << "35B custom stop did not terminate inside an MTP round\n";
        return 1;
    }

    std::vector<ninfer::TokenId> stopped_continuation = prompt;
    stopped_continuation.insert(stopped_continuation.end(), stopped.generated_token_ids.begin(),
                                stopped.generated_token_ids.end());
    stopped_continuation.push_back(198);
    const ninfer::GenerationResult after_stop = engine.generate(
        engine.prepare_tokens(std::move(stopped_continuation)), greedy_options(1, false));
    if (after_stop.generated_token_ids.size() != 1 ||
        after_stop.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "35B request after partial MTP terminal did not finish its output budget\n";
        return 1;
    }

    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    engine.reset_measurement_peaks();
    const auto before = engine.snapshot().ranks.at(0).memory;
    if (!before.vision || before.vision->handoff_active_bytes != 0 ||
        before.vision->handoff_peak_bytes != 0) {
        std::cerr << "35B Vision handoff peak did not reset before Vision\n";
        return 1;
    }
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
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
    const auto after = engine.snapshot().ranks.at(0).memory;
    if (!after.vision ||
        after.vision->handoff_offset_bytes != before.vision->handoff_offset_bytes ||
        after.vision->handoff_capacity_bytes != before.vision->handoff_capacity_bytes ||
        after.vision->handoff_active_bytes != 0 || after.vision->handoff_peak_bytes == 0 ||
        after.vision->handoff_peak_bytes > after.vision->handoff_capacity_bytes ||
        after.workspace.measurement_peak_bytes == 0 ||
        after.workspace.measurement_peak_bytes > after.allocations.workspace_device_bytes) {
        std::cerr << "35B Vision request did not use the startup-frozen workspace handoff\n";
        return 1;
    }
    return 0;
}

int exercise_maximum_configuration(const char* artifact) {
    ninfer::Engine engine(maximum_engine_options(artifact));
    const auto memory  = engine.snapshot().ranks.at(0).memory;
    const auto* vision = memory.vision ? &*memory.vision : nullptr;
    if (engine.description().max_context != 262144 ||
        engine.description().kv_cache != ninfer::KvCacheStorage::Int8Group64 ||
        memory.kv_payload_bytes == 0 || memory.allocations.persistent_device_bytes == 0 ||

        memory.allocations.workspace_device_bytes == 0 || vision == nullptr ||
        vision->handoff_offset_bytes > memory.allocations.workspace_device_bytes ||
        vision->handoff_capacity_bytes == 0 ||
        vision->handoff_capacity_bytes >
            memory.allocations.workspace_device_bytes - vision->handoff_offset_bytes ||
        vision->handoff_active_bytes != 0 ||
        !engine.startup_report().ranks.at(0).graphs.has_value()) {
        std::cerr << "35B maximum configuration does not match the planned 256K layout: context="
                  << engine.description().max_context << " kv_payload=" << memory.kv_payload_bytes
                  << " sequence=" << memory.allocations.persistent_device_bytes
                  << " workspace=" << memory.allocations.workspace_device_bytes << '\n';
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
        std::cerr << "35B Engine was unusable after rejecting an over-capacity request\n";
        return 1;
    }
    const auto after = engine.snapshot().ranks.at(0).memory;
    if (after.workspace.measurement_peak_bytes == 0 ||
        after.workspace.measurement_peak_bytes > after.allocations.workspace_device_bytes) {
        std::cerr << "35B maximum configuration did not report its executed workspace phase\n";
        return 1;
    }
    return 0;
}

int exercise_expert_priority(const char* artifact) {
    auto options            = engine_options(artifact);
    options.speculative     = {};
    options.enable_vision   = false;
    options.max_concurrency = 8;
    options.prefill_chunk   = 2048;
    options.moe.expert_priority.emplace(40);
    auto& priority = *options.moe.expert_priority;
    for (std::size_t layer = 0; layer < priority.size(); ++layer) {
        for (std::uint32_t e = 0; e < 256; ++e) {
            priority[layer].push_back((255 - e + layer) % 256);
        }
    }
    ninfer::Engine engine(options);
    const auto before = engine.snapshot().ranks.at(0).memory;
    if (engine.options().prefill_chunk != 2048 || engine.description().prefill_chunk != 2048) {
        throw std::runtime_error("explicit prefill chunk changed during offload startup");
    }
    if (!before.experts || before.experts->target_experts_per_layer.size() != priority.size()) {
        throw std::runtime_error("expert priority cache was not materialized");
    }
    std::vector<ninfer::GenerationHandle> calls;
    for (unsigned i = 0; i < 8; ++i) {
        calls.push_back(
            engine.submit(engine.prepare_tokens(std::vector<ninfer::TokenId>(8 + i, 198)),
                          greedy_options(4, false)));
    }
    for (auto& call : calls) {
        if (call.wait().generated_token_ids.size() != 4) {
            throw std::runtime_error("priority cache lost a compact batch row");
        }
    }
    const auto warmed = engine.snapshot().ranks.at(0).memory.experts;
    if (warmed->ready_bytes != before.experts->ready_bytes ||
        warmed->uploaded_bytes != before.experts->uploaded_bytes) {
        throw std::runtime_error("stable small-T execution changed its expert replicas");
    }
    const auto result = engine.generate(
        engine.prepare_tokens(std::vector<ninfer::TokenId>(128, 198)), greedy_options(2, false));
    if (result.generated_token_ids.size() != 2 ||
        engine.snapshot().ranks.at(0).memory.experts->uploaded_bytes <=
            before.experts->uploaded_bytes) {
        throw std::runtime_error("priority cache did not stream its missing experts");
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool offload = argc == 2 && std::string_view(argv[1]) == "--offload";
        if (argc != 1 && !offload)
            throw std::invalid_argument("expected --offload or no arguments");
        const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
        if (artifact == nullptr || *artifact == '\0') {
            std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
            return 77;
        }

        {
            ninfer::Engine engine(engine_options(artifact));
            if (const int result = verify_loaded_product(engine); result != 0) { return result; }
            const auto before         = engine.snapshot().ranks.at(0).memory;
            const auto expected_chunk = offload ? 4096U : 1024U;
            if (engine.options().prefill_chunk != expected_chunk ||
                engine.description().prefill_chunk != expected_chunk) {
                throw std::runtime_error(
                    "Engine did not resolve the expected default prefill chunk");
            }
            if (before.experts.has_value() != offload) {
                throw std::runtime_error("MoE test did not select the requested residency");
            }
            if (const int result = exercise_text_mtp_and_prefix(engine); result != 0) {
                return result;
            }
            if (const int result = exercise_vision(engine); result != 0) { return result; }
        }
        if (!offload) {
            if (const int result = exercise_maximum_configuration(artifact); result != 0) {
                return result;
            }
        } else {
            if (const int result = exercise_expert_priority(artifact); result != 0) {
                return result;
            }
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
