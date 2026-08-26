#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 4096;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 1024;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.enable_vision                    = true;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 4;
    return options;
}

ninfer::EngineOptions host_restore_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.context_cache.max_cache_markers_per_request     = 0;
    return options;
}

ninfer::EngineOptions shared_replacement_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 4;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 1;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.context_cache.max_cache_markers_per_request     = 1;
    return options;
}

ninfer::EngineOptions last_alias_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 3;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.context_cache.max_cache_markers_per_request     = 0;
    return options;
}

ninfer::EngineOptions concurrent_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 512;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.max_concurrency                  = 8;
    options.max_pending_requests             = 8;
    options.context_cache.device_state_slots = 16;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 8;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.context_cache.max_cache_markers_per_request     = 0;
    return options;
}

std::vector<std::uint8_t> gradient_ppm(int width = 64, int height = 64) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput chinese_chat(bool enable_thinking) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
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

int exercise_full_prefill_chunk(ninfer::Engine& engine) {
    constexpr std::size_t kChunkTokens = 1024;
    std::vector<ninfer::TokenId> prompt(kChunkTokens, 198);
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(std::move(prompt)), options);
    if (result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "full-chunk prefill did not complete through the planned workspace\n";
        return 1;
    }
    return 0;
}

int exercise_abandoned_handle_capacity(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;

    {
        auto abandoned = engine.submit(engine.prepare_tokens(prompt), request);
        if (!abandoned) {
            std::cerr << "abandonment fixture did not create a generation handle\n";
            return 1;
        }
    }
    const auto crossed = engine.generate(engine.prepare_tokens(prompt), request);
    if (crossed.generated_token_ids.size() != 1) {
        std::cerr << "request after an abandoned handle did not complete\n";
        return 1;
    }

    auto first      = engine.submit(engine.prepare_tokens(prompt), request);
    auto second     = engine.submit(engine.prepare_tokens(prompt), request);
    bool overloaded = false;
    try {
        auto third = engine.submit(engine.prepare_tokens(prompt), request);
        (void)third;
    } catch (const ninfer::RequestError& error) {
        overloaded = error.kind() == ninfer::RequestErrorKind::Overloaded;
    }
    if (!overloaded) {
        std::cerr << "outstanding capacity was released twice or not enforced\n";
        return 1;
    }
    if (first.wait().generated_token_ids.size() != 1 ||
        second.wait().generated_token_ids.size() != 1) {
        std::cerr << "requests retained after the overload check did not complete\n";
        return 1;
    }
    return 0;
}

int exercise_zero_suffix_reuse(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    ninfer::RequestOptions baseline_options;
    baseline_options.execution.requested_output_tokens = 8;
    baseline_options.execution.sampling.temperature    = 0.0F;
    baseline_options.execution.allow_prefix_reuse      = true;
    baseline_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(prompt), baseline_options);
    if (baseline.generated_token_ids.size() != 8) {
        std::cerr << "zero-suffix baseline did not generate eight tokens\n";
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
        reused.generated_token_ids[0] != baseline.generated_token_ids.back()) {
        std::cerr << "zero-suffix reuse did not resume from the retained target frontier\n";
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

    if (const int result = exercise_zero_suffix_reuse(engine, prompt); result != 0) {
        return result;
    }

    return 0;
}

int exercise_host_restore(const char* artifact) {
    ninfer::Engine engine(host_restore_engine_options(artifact));
    auto options = [](std::uint32_t outputs, bool reuse) {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = outputs;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = reuse;
        request.stop.include_model_defaults       = false;
        return request;
    };

    const auto retained_input = [] {
        std::string text;
        text.reserve(6U * 300U);
        for (std::uint32_t index = 0; index < 300; ++index) { text += "alpha "; }
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "host-restore-real";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };

    const ninfer::GenerationResult retained =
        engine.generate(engine.prepare(retained_input()), options(5, true));
    if (retained.prompt.prompt_tokens <= 256 || retained.generated_token_ids.size() != 5) {
        std::cerr << "Host-restore source request did not complete\n";
        return 1;
    }

    ninfer::PromptInput continuation = retained_input();
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = retained.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = retained.content, .media = {}});
    continuation.messages.push_back(std::move(assistant));
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
    continuation.messages.push_back(std::move(followup));

    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare(continuation), options(2, false));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    if (baseline.generated_token_ids.size() != 2 ||
        after_pressure.state_d2h_count <= before_pressure.state_d2h_count ||
        after_pressure.main_kv_d2h_pages <= before_pressure.main_kv_d2h_pages ||
        after_pressure.backend_kv_d2h_pages <= before_pressure.backend_kv_d2h_pages) {
        std::cerr << "Host pressure did not demote the complete MTP checkpoint: state="
                  << after_pressure.state_d2h_count << " main=" << after_pressure.main_kv_d2h_pages
                  << " backend=" << after_pressure.backend_kv_d2h_pages
                  << " degraded=" << after_pressure.private_checkpoint_degradations
                  << " evicted=" << after_pressure.private_checkpoint_evictions << '\n';
        return 1;
    }

    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(std::move(continuation)), options(2, true));
    const ninfer::RuntimeStats after_restore = engine.runtime_stats();
    if (restored.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure ||
        restored.reused_prompt_tokens == 0 ||
        after_restore.state_h2d_count <= after_pressure.state_h2d_count ||
        after_restore.main_kv_h2d_pages <= after_pressure.main_kv_h2d_pages ||
        after_restore.backend_kv_h2d_pages <= after_pressure.backend_kv_h2d_pages) {
        std::cerr << "Host checkpoint was not restored exactly: path="
                  << static_cast<int>(restored.prefix_reuse_path)
                  << " reused=" << restored.reused_prompt_tokens
                  << " state=" << after_restore.state_h2d_count
                  << " main=" << after_restore.main_kv_h2d_pages
                  << " backend=" << after_restore.backend_kv_h2d_pages
                  << " degraded=" << after_restore.private_checkpoint_degradations
                  << " evicted=" << after_restore.private_checkpoint_evictions << '\n';
        return 1;
    }

    if (restored.generated_token_ids != baseline.generated_token_ids) {
        std::cerr << "Host checkpoint restore changed greedy output: restored=";
        for (const ninfer::TokenId token : restored.generated_token_ids) {
            std::cerr << token << ',';
        }
        std::cerr << " baseline=";
        for (const ninfer::TokenId token : baseline.generated_token_ids) {
            std::cerr << token << ',';
        }
        std::cerr << " restored_spec=" << restored.speculative.rounds << '/'
                  << restored.speculative.drafted_tokens << '/'
                  << restored.speculative.accepted_tokens << '/'
                  << restored.speculative.fallback_steps
                  << " baseline_spec=" << baseline.speculative.rounds << '/'
                  << baseline.speculative.drafted_tokens << '/'
                  << baseline.speculative.accepted_tokens << '/'
                  << baseline.speculative.fallback_steps
                  << " reused=" << restored.reused_prompt_tokens
                  << " transfers=" << after_restore.state_d2h_count << '/'
                  << after_restore.state_h2d_count << '/' << after_restore.main_kv_d2h_pages << '/'
                  << after_restore.main_kv_h2d_pages << '\n';
        return 1;
    }
    return 0;
}

int exercise_shared_replacement_and_full_capacity_reuse(const char* artifact) {
    ninfer::EngineOptions engine_options = shared_replacement_engine_options(artifact);
    engine_options.context_cache.max_private_continuations = 1;
    ninfer::Engine engine(std::move(engine_options));
    ninfer::RequestOptions capture_request;
    capture_request.execution.requested_output_tokens = 1;
    capture_request.execution.sampling.temperature    = 0.0F;
    capture_request.execution.allow_prefix_reuse      = true;
    capture_request.stop.include_model_defaults       = false;

    const auto tool_prompt = [](std::string tool_json, std::string question) {
        ninfer::PromptInput input;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        input.options.tool_jsons.push_back(std::move(tool_json));
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        return input;
    };

    const std::string alpha_tool =
        R"({"type":"function","function":{"name":"alpha","description":"stable lookup schema","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})";
    const std::string bravo_tool =
        R"({"type":"function","function":{"name":"bravo","description":"stable lookup schema","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})";

    const ninfer::GenerationResult first = engine.generate(
        engine.prepare(tool_prompt(alpha_tool, "Use alpha once.")), capture_request);
    const ninfer::GenerationResult replacement = engine.generate(
        engine.prepare(tool_prompt(bravo_tool, "Use bravo once.")), capture_request);
    const ninfer::RuntimeStats after_replacement = engine.runtime_stats();
    if (replacement.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not produce its deterministic stop token\n";
        return 1;
    }
    // Remove the exact private endpoint without publishing another shared marker. The following
    // identical Bravo prompt must therefore materialize from the retained shared prefix.
    ninfer::PromptInput filler;
    ninfer::ChatMessage filler_user;
    filler_user.role = ninfer::ChatRole::User;
    filler_user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Unrelated private endpoint.", .media = {}});
    filler.messages.push_back(std::move(filler_user));
    filler.options.enable_thinking = false;
    filler.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
    const ninfer::GenerationResult filled =
        engine.generate(engine.prepare(std::move(filler)), capture_request);
    if (filled.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not displace the exact private endpoint\n";
        return 1;
    }
    ninfer::RequestOptions full_capacity_request = capture_request;
    full_capacity_request.execution.requested_output_tokens =
        std::numeric_limits<std::uint32_t>::max();
    full_capacity_request.stop.token_ids          = {replacement.generated_token_ids.front()};
    full_capacity_request.stop.publish_stop_token = true;
    const ninfer::GenerationResult reused         = engine.generate(
        engine.prepare(tool_prompt(bravo_tool, "Use bravo once.")), full_capacity_request);
    const ninfer::RuntimeStats after_reuse = engine.runtime_stats();

    if (first.generated_token_ids.size() != 1 || reused.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not complete all requests\n";
        return 1;
    }
    if (reused.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        reused.reused_prompt_tokens == 0 || reused.reused_prompt_tokens % 64U == 0) {
        std::cerr << "single-slot shared replacement was not reusable at a non-aligned frontier: "
                  << "path=" << static_cast<int>(reused.prefix_reuse_path)
                  << " reused=" << reused.reused_prompt_tokens << '\n';
        return 1;
    }
    if (after_replacement.active_captures_completed < 2 ||
        after_reuse.active_captures_completed < after_replacement.active_captures_completed) {
        std::cerr << "shared replacement capture was skipped under full State/KV capacity: "
                  << after_replacement.active_captures_completed << '/'
                  << after_reuse.active_captures_completed << '\n';
        return 1;
    }
    return 0;
}

int exercise_last_private_alias_eviction(const char* artifact) {
    ninfer::Engine engine(last_alias_engine_options(artifact));
    std::string prompt_text;
    prompt_text.reserve(6U * 300U + 8U);
    for (std::uint32_t index = 0; index < 300; ++index) { prompt_text += "alpha "; }

    const auto input = [&](std::string session, ninfer::CacheRetentionHint retention) {
        ninfer::PromptInput prompt;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = prompt_text, .media = {}});
        prompt.messages.push_back(std::move(user));
        prompt.options.enable_thinking   = false;
        prompt.context_cache.session_key = std::move(session);
        prompt.context_cache.retention   = retention;
        return prompt;
    };
    if (engine.count_tokens(input("probe", ninfer::CacheRetentionHint::Disposable)) % 64U == 0) {
        prompt_text += "beta";
    }

    ninfer::RequestOptions one_token;
    one_token.execution.requested_output_tokens = 1;
    one_token.execution.sampling.temperature    = 0.0F;
    one_token.execution.allow_prefix_reuse      = true;
    one_token.stop.include_model_defaults       = false;

    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(input("last-alias-source", ninfer::CacheRetentionHint::LiveSession)),
        one_token);
    const ninfer::GenerationResult branch = engine.generate(
        engine.prepare(input("last-alias-branch", ninfer::CacheRetentionHint::Disposable)),
        one_token);
    if (source.generated_token_ids.size() != 1 || branch.generated_token_ids.size() != 1 ||
        branch.reused_prompt_tokens == 0 ||
        (branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint)) {
        std::cerr << "last-alias fixture did not establish two private prefix aliases: path="
                  << static_cast<int>(branch.prefix_reuse_path)
                  << " reused=" << branch.reused_prompt_tokens << '\n';
        return 1;
    }

    ninfer::RequestOptions full_capacity            = one_token;
    full_capacity.execution.requested_output_tokens = std::numeric_limits<std::uint32_t>::max();
    full_capacity.stop.token_ids                    = {source.generated_token_ids.front()};
    full_capacity.stop.publish_stop_token           = true;
    const ninfer::RuntimeStats before               = engine.runtime_stats();
    const ninfer::GenerationResult consumed         = engine.generate(
        engine.prepare(input("last-alias-source", ninfer::CacheRetentionHint::LiveSession)),
        full_capacity);
    const ninfer::RuntimeStats after = engine.runtime_stats();
    if (consumed.generated_token_ids.size() != 1 || consumed.reused_prompt_tokens == 0 ||
        (consumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         consumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint) ||
        after.private_checkpoint_evictions <= before.private_checkpoint_evictions ||
        after.device_main_kv_occupied_pages == 0 || after.device_main_kv_occupied_pages > 8 ||
        after.device_backend_kv_occupied_pages == 0 || after.device_backend_kv_occupied_pages > 8) {
        std::cerr << "last private prefix alias did not transfer into the active entitlement: path="
                  << static_cast<int>(consumed.prefix_reuse_path)
                  << " reused=" << consumed.reused_prompt_tokens
                  << " evictions=" << before.private_checkpoint_evictions << '/'
                  << after.private_checkpoint_evictions
                  << " main=" << after.device_main_kv_occupied_pages
                  << " backend=" << after.device_backend_kv_occupied_pages << '\n';
        return 1;
    }
    return 0;
}

int exercise_rewrite_checkpoints(ninfer::Engine& engine) {
    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    auto assistant_call = [&](std::string reasoning, std::string id, std::string key) {
        ninfer::ChatMessage message = text_message(ninfer::ChatRole::Assistant, "");
        message.reasoning_content   = std::move(reasoning);
        message.tool_calls.push_back(ninfer::ToolCall{
            .id = std::move(id), .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        return message;
    };
    auto input_with_history = [&](int completed_responses, bool preserve_thinking) {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (completed_responses >= 1) {
            input.messages.push_back(
                assistant_call("The first lookup should be alpha.", "call_alpha", "alpha"));
            ninfer::ChatMessage tool =
                text_message(ninfer::ChatRole::Tool, "{\"value\":17,\"next\":\"beta\"}");
            tool.tool_call_id = "call_alpha";
            input.messages.push_back(std::move(tool));
        }
        if (completed_responses >= 2) {
            input.messages.push_back(
                assistant_call("The alpha result requests beta.", "call_beta", "beta"));
            ninfer::ChatMessage tool = text_message(ninfer::ChatRole::Tool, "{\"value\":25}");
            tool.tool_call_id        = "call_beta";
            input.messages.push_back(std::move(tool));
        }
        input.options.preserve_thinking = preserve_thinking;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return input;
    };
    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 4;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input_with_history(0, true)), options(true));
    if (first.generated_token_ids.size() != 4 ||
        first.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "response-checkpoint source request did not complete from a cold lane\n";
        return 1;
    }

    const ninfer::GenerationResult exact_replay =
        engine.generate(engine.prepare(input_with_history(0, false)), options(true));
    if (exact_replay.generated_token_ids.size() != 4 ||
        exact_replay.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        exact_replay.reused_prompt_tokens == 0) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "pre-generation response checkpoint was not restored on an exact replay: "
                  << "path=" << static_cast<int>(exact_replay.prefix_reuse_path)
                  << " reused=" << exact_replay.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    const ninfer::GenerationResult exact_baseline =
        engine.generate(engine.prepare(input_with_history(0, false)), options(false));
    if (exact_replay.generated_token_ids != exact_baseline.generated_token_ids) {
        std::cerr << "Device rewrite restore changed greedy output: restored=";
        for (const ninfer::TokenId token : exact_replay.generated_token_ids) {
            std::cerr << token << ',';
        }
        std::cerr << " baseline=";
        for (const ninfer::TokenId token : exact_baseline.generated_token_ids) {
            std::cerr << token << ',';
        }
        std::cerr << " source=";
        for (const ninfer::TokenId token : first.generated_token_ids) { std::cerr << token << ','; }
        std::cerr << " restored_spec=" << exact_replay.speculative.rounds << '/'
                  << exact_replay.speculative.drafted_tokens << '/'
                  << exact_replay.speculative.accepted_tokens << '/'
                  << exact_replay.speculative.fallback_steps
                  << " baseline_spec=" << exact_baseline.speculative.rounds << '/'
                  << exact_baseline.speculative.drafted_tokens << '/'
                  << exact_baseline.speculative.accepted_tokens << '/'
                  << exact_baseline.speculative.fallback_steps
                  << " source_prompt=" << first.prompt.prompt_tokens
                  << " restored_prompt=" << exact_replay.prompt.prompt_tokens
                  << " baseline_prompt=" << exact_baseline.prompt.prompt_tokens
                  << " reused=" << exact_replay.reused_prompt_tokens << '\n';
        return 1;
    }

    const ninfer::GenerationResult first_replay =
        engine.generate(engine.prepare(input_with_history(1, true)), options(true));
    if (first_replay.generated_token_ids.size() != 4 ||
        first_replay.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        first_replay.reused_prompt_tokens == 0) {
        std::cerr << "normalized first response did not restore its response checkpoint: path="
                  << static_cast<int>(first_replay.prefix_reuse_path)
                  << " reused=" << first_replay.reused_prompt_tokens << '\n';
        return 1;
    }

    const ninfer::GenerationResult second_replay =
        engine.generate(engine.prepare(input_with_history(2, true)), options(true));
    if (second_replay.generated_token_ids.size() != 4 ||
        second_replay.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        second_replay.reused_prompt_tokens <= first_replay.reused_prompt_tokens) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "rolling response checkpoint did not advance across the tool loop: first="
                  << first_replay.reused_prompt_tokens
                  << " second=" << second_replay.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }

    const ninfer::GenerationResult mode_change =
        engine.generate(engine.prepare(input_with_history(2, false)), options(true));
    if (mode_change.generated_token_ids.size() != 4 ||
        mode_change.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        mode_change.reused_prompt_tokens == 0) {
        std::cerr << "preserve-thinking policy change discarded a compatible response checkpoint: "
                  << "path=" << static_cast<int>(mode_change.prefix_reuse_path)
                  << " reused=" << mode_change.reused_prompt_tokens << '\n';
        return 1;
    }

    ninfer::PromptInput branch_input = input_with_history(0, true);
    branch_input.messages.push_back(
        text_message(ninfer::ChatRole::User, "Summarize the conversation before answering."));
    const ninfer::GenerationResult branch =
        engine.generate(engine.prepare(std::move(branch_input)), options(true));
    ninfer::PromptInput branch_baseline_input = input_with_history(0, true);
    branch_baseline_input.messages.push_back(
        text_message(ninfer::ChatRole::User, "Summarize the conversation before answering."));
    const ninfer::GenerationResult branch_baseline =
        engine.generate(engine.prepare(std::move(branch_baseline_input)), options(false));
    if (branch.generated_token_ids.size() != 4 || branch.reused_prompt_tokens == 0 ||
        branch.reused_prompt_tokens >= branch.prompt.prompt_tokens ||
        (branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure) ||
        branch.generated_token_ids != branch_baseline.generated_token_ids) {
        std::cerr << "replacement user suffix did not reuse the stable conversation prefix: path="
                  << static_cast<int>(branch.prefix_reuse_path)
                  << " reused=" << branch.reused_prompt_tokens
                  << " prompt=" << branch.prompt.prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    const auto image_bytes = gradient_ppm();
    auto image_part        = [](const std::vector<std::uint8_t>& bytes, std::string name) {
        ninfer::MessagePart image;
        image.kind              = ninfer::MessagePartKind::Media;
        image.media.kind        = ninfer::MediaKind::Image;
        image.media.bytes       = bytes;
        image.media.media_type  = "image/x-portable-pixmap";
        image.media.source_name = std::move(name);
        return image;
    };
    auto assistant_message = [](const ninfer::GenerationResult& result) {
        ninfer::ChatMessage message;
        message.role              = ninfer::ChatRole::Assistant;
        message.reasoning_content = result.reasoning;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = result.content, .media = {}});
        return message;
    };
    auto first_input = [&](const std::vector<std::uint8_t>& bytes) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(image_part(bytes, "inline.ppm"));
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "vision-prefix-real";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };
    auto followup_input = [&](const std::vector<std::uint8_t>& bytes,
                              const ninfer::GenerationResult& first) {
        ninfer::PromptInput input = first_input(bytes);
        input.messages.push_back(assistant_message(first));
        ninfer::ChatMessage followup;
        followup.role = ninfer::ChatRole::User;
        followup.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "Give one more detail.", .media = {}});
        input.messages.push_back(std::move(followup));
        return input;
    };
    auto appended_media_input =
        [&](const std::vector<std::uint8_t>& old_bytes, const ninfer::GenerationResult& first,
            const ninfer::GenerationResult& second, const std::vector<std::uint8_t>& new_bytes) {
            ninfer::PromptInput input = followup_input(old_bytes, first);
            input.messages.push_back(assistant_message(second));
            ninfer::ChatMessage followup;
            followup.role = ninfer::ChatRole::User;
            followup.parts.push_back(image_part(new_bytes, "second.ppm"));
            followup.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = "Compare the images.", .media = {}});
            input.messages.push_back(std::move(followup));
            return input;
        };

    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 2;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    // The 1024 merged Vision columns begin after the chat prefix, so the same item necessarily
    // crosses a 1024-token prefill boundary. Its host payload may be released after the first
    // encode, while later chunks must continue to reuse the resident Vision transient.
    ninfer::RequestOptions cross_chunk_options            = options(false);
    cross_chunk_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult cross_chunk =
        engine.generate(engine.prepare(first_input(gradient_ppm(1024, 1024))), cross_chunk_options);
    if (!cross_chunk.prompt.has_media || cross_chunk.generated_token_ids.size() != 1) {
        std::cerr << "cross-chunk Vision item did not complete after releasing its host payload\n";
        return 1;
    }

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_input(image_bytes)), options(true));
    if (!first.prompt.has_media || first.generated_token_ids.size() != 2 ||
        first.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "real Vision request did not complete through the public Engine\n";
        return 1;
    }

    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(followup_input(image_bytes, first)), options(true));
    if (reused.reused_prompt_tokens == 0 || reused.timings.vision_seconds != 0.0 ||
        reused.generated_token_ids.size() != 2) {
        std::cerr << "same-media continuation did not reuse the resident Vision prefix: reused="
                  << reused.reused_prompt_tokens << " vision=" << reused.timings.vision_seconds
                  << '\n';
        return 1;
    }

    std::vector<std::uint8_t> second_image = image_bytes;
    second_image.back() ^= 0x5aU;
    const ninfer::GenerationResult appended = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(true));
    if (appended.reused_prompt_tokens == 0 || !(appended.timings.vision_seconds > 0.0) ||
        appended.generated_token_ids.size() != 2) {
        std::cerr << "new-media suffix did not preserve the old multimodal prefix: reused="
                  << appended.reused_prompt_tokens << " vision=" << appended.timings.vision_seconds
                  << '\n';
        return 1;
    }

    const ninfer::GenerationResult baseline = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(false));
    if (baseline.generated_token_ids != appended.generated_token_ids) {
        std::cerr << "multimodal prefix reuse changed greedy output\n";
        return 1;
    }

    std::vector<std::uint8_t> changed_prefix = image_bytes;
    changed_prefix[changed_prefix.size() - 2] ^= 0x33U;
    const ninfer::GenerationResult miss = engine.generate(
        engine.prepare(appended_media_input(changed_prefix, first, reused, second_image)),
        options(true));
    if (miss.reused_prompt_tokens != 0) {
        std::cerr << "changed media content incorrectly reused placeholder-token KV\n";
        return 1;
    }

    ninfer::RequestOptions mtp_options            = options(false);
    mtp_options.execution.requested_output_tokens = 5;
    const ninfer::GenerationResult mtp_baseline =
        engine.generate(engine.prepare(first_input(image_bytes)), mtp_options);
    if (mtp_baseline.generated_token_ids.size() != 5 ||
        mtp_baseline.generated_token_ids[0] == mtp_baseline.generated_token_ids[1]) {
        std::cerr << "multimodal stop fixture did not produce distinct leading tokens\n";
        return 1;
    }
    ninfer::RequestOptions stop_options       = mtp_options;
    stop_options.execution.allow_prefix_reuse = true;
    stop_options.stop.token_ids.push_back(mtp_baseline.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare(first_input(image_bytes)), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.size() != 2 ||
        stopped.generated_token_ids[0] != mtp_baseline.generated_token_ids[0] ||
        stopped.generated_token_ids[1] != mtp_baseline.generated_token_ids[1]) {
        std::cerr << "multimodal custom stop did not terminate at the selected token\n";
        return 1;
    }
    const ninfer::GenerationResult stopped_reuse =
        engine.generate(engine.prepare(followup_input(image_bytes, stopped)), options(true));
    if (stopped_reuse.reused_prompt_tokens == 0 || stopped_reuse.timings.vision_seconds != 0.0) {
        std::cerr << "multimodal stop discarded its reusable boundary: reused="
                  << stopped_reuse.reused_prompt_tokens
                  << " vision=" << stopped_reuse.timings.vision_seconds << '\n';
        return 1;
    }

    // Exact registered rendering prefix before the first image-pad column:
    // <|im_start|>user\n<|vision_start|>. Reusing it places the MTP bridge directly on the first
    // Vision merger column rather than on an ordinary token embedding.
    const std::vector<ninfer::TokenId> visual_prefix{248045, 846, 198, 248053};
    ninfer::RequestOptions source_options            = options(true);
    source_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult bridge_source =
        engine.generate(engine.prepare_tokens(visual_prefix), source_options);
    ninfer::RequestOptions bridge_options            = options(true);
    bridge_options.execution.requested_output_tokens = 5;
    const ninfer::GenerationResult visual_bridge =
        engine.generate(engine.prepare(first_input(image_bytes)), bridge_options);
    if (bridge_source.generated_token_ids.size() != 1 ||
        visual_bridge.reused_prompt_tokens != visual_prefix.size() ||
        !(visual_bridge.timings.vision_seconds > 0.0) || visual_bridge.speculative.rounds == 0) {
        std::cerr << "visual MTP bridge did not append the prefix and enter speculative decode: "
                  << "source_outputs=" << bridge_source.generated_token_ids.size()
                  << " reused=" << visual_bridge.reused_prompt_tokens
                  << " vision=" << visual_bridge.timings.vision_seconds
                  << " rounds=" << visual_bridge.speculative.rounds
                  << " fallbacks=" << visual_bridge.speculative.fallback_steps << '\n';
        return 1;
    }
    ninfer::RequestOptions bridge_baseline_options       = bridge_options;
    bridge_baseline_options.execution.allow_prefix_reuse = false;
    const ninfer::GenerationResult visual_bridge_baseline =
        engine.generate(engine.prepare(first_input(image_bytes)), bridge_baseline_options);
    if (visual_bridge.generated_token_ids != visual_bridge_baseline.generated_token_ids) {
        std::cerr << "visual MTP bridge changed greedy output relative to full prefill\n";
        return 1;
    }
    return 0;
}

ninfer::PromptInput session_turn(std::string session, std::string question) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.context_cache.session_key = std::move(session);
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    return input;
}

ninfer::RequestOptions fixed_output(std::uint32_t tokens, bool reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

int exercise_concurrent_resource_settlement(const char* artifact,
                                            std::string_view expected_target) {
    ninfer::Engine engine(concurrent_engine_options(artifact));
    if (engine.load_summary().target != expected_target) {
        std::cerr << "concurrent fixture loaded target " << engine.load_summary().target
                  << ", expected " << expected_target << '\n';
        return 1;
    }

    constexpr std::string_view kSession = "publication-order-real";
    constexpr std::string_view kOlderQuestion =
        "Describe deterministic scheduling using exactly one concise paragraph.";
    constexpr std::string_view kNewerQuestion =
        "Describe prefix caching using exactly one concise paragraph.";
    auto older = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kOlderQuestion))),
        fixed_output(24));
    auto newer = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    const ninfer::GenerationResult newer_result = newer.wait();
    const ninfer::GenerationResult older_result = older.wait();
    if (newer_result.generated_token_ids.size() != 2 ||
        older_result.generated_token_ids.size() != 24) {
        std::cerr << "concurrent session requests did not reach staggered terminal boundaries\n";
        return 1;
    }

    for (std::uint32_t index = 0; index < 6; ++index) {
        const std::string suffix              = std::to_string(index);
        const ninfer::GenerationResult filler = engine.generate(
            engine.prepare(session_turn("publication-filler-" + suffix,
                                        "Give one deterministic token for filler " + suffix + '.')),
            fixed_output(1));
        if (filler.generated_token_ids.size() != 1) {
            std::cerr << "session-order catalog filler did not complete\n";
            return 1;
        }
    }
    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult pressure    = engine.generate(
        engine.prepare(session_turn("publication-pressure",
                                       "Give one deterministic token for the pressure request.")),
        fixed_output(1));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    if (pressure.generated_token_ids.size() != 1 ||
        after_pressure.private_checkpoint_evictions <=
            before_pressure.private_checkpoint_evictions) {
        std::cerr << "full session catalog did not execute its canonical eviction\n";
        return 1;
    }

    const ninfer::GenerationResult replay = engine.generate(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    if (replay.generated_token_ids.size() != 2 || replay.reused_prompt_tokens == 0 ||
        replay.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << "late older finish exposed the newer session binding to pressure: path="
                  << static_cast<int>(replay.prefix_reuse_path)
                  << " reused=" << replay.reused_prompt_tokens << '\n';
        return 1;
    }

    {
        std::vector<ninfer::TokenId> long_prompt(400, 198);
        auto cancelled =
            engine.submit(engine.prepare_tokens(std::move(long_prompt)), fixed_output(32, false));
        if (!cancelled) {
            std::cerr << "materialization cancellation fixture did not create a handle\n";
            return 1;
        }
    }
    const ninfer::GenerationResult after_cancel = engine.generate(
        engine.prepare_tokens({248045, 846, 198, 5834, 248046, 198}), fixed_output(1, false));
    if (after_cancel.generated_token_ids.size() != 1) {
        std::cerr << "request after materialization cancellation did not complete\n";
        return 1;
    }

    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(8);
    for (std::uint32_t row = 0; row < 8; ++row) {
        std::vector<ninfer::TokenId> prompt{
            248045, 846, 198, static_cast<ninfer::TokenId>(1000 + row), 248046, 198};
        handles.push_back(
            engine.submit(engine.prepare_tokens(std::move(prompt)), fixed_output(row + 1, false)));
    }
    for (std::uint32_t row = 0; row < handles.size(); ++row) {
        const ninfer::GenerationResult result = handles[row].wait();
        if (result.generated_token_ids.size() != row + 1 ||
            result.finish_reason != ninfer::FinishReason::OutputLimit) {
            std::cerr << "C=8 staggered row " << row << " did not terminate independently\n";
            return 1;
        }
    }
    const ninfer::RuntimeStats settled = engine.runtime_stats();
    if (settled.running_requests != 0 || settled.materializing_requests != 0 ||
        settled.prefilling_requests != 0 || settled.decode_ready_requests != 0 ||
        settled.capture_pending_requests != 0 || settled.terminal_pending_requests != 0) {
        std::cerr << "C=8 terminal settlement left live logical membership: running="
                  << settled.running_requests << " materializing=" << settled.materializing_requests
                  << " prefill=" << settled.prefilling_requests
                  << " decode=" << settled.decode_ready_requests
                  << " capture=" << settled.capture_pending_requests
                  << " terminal=" << settled.terminal_pending_requests << '\n';
        return 1;
    }
    return 0;
}

int verify_loaded_product(const ninfer::Engine& engine, std::string_view expected_target) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != expected_target ||
        (load.weights_id != "groupwise-int" && load.weights_id != "nvfp4") ||
        load.host_to_device_bytes == 0 || load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "Engine construction has an invalid load summary: target=" << load.target
                  << " weights=" << load.weights_id << '\n';
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    const auto* vision = memory.vision_workspace ? &*memory.vision_workspace : nullptr;
    if (memory.weights.capacity_bytes == 0 || memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || vision == nullptr ||
        vision->aggregate_prompt_tokens != 4096 || vision->max_item_tokens != 4096 ||
        vision->general_capacity_bytes == 0 || vision->encode_peak_bytes == 0 ||
        vision->handoff_offset_bytes > memory.workspace.capacity_bytes ||
        vision->handoff_capacity_bytes == 0 ||
        vision->handoff_capacity_bytes >
            memory.workspace.capacity_bytes - vision->handoff_offset_bytes ||
        vision->handoff_active_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

} // namespace

int exercise_artifact(const char* artifact, std::string_view expected_target) {
    {
        ninfer::EngineOptions options             = engine_options(artifact);
        options.context_cache.device_state_slots  = 1;
        options.context_cache.max_shared_prefixes = 0;
        ninfer::Engine engine(std::move(options));
        if (const int result = verify_loaded_product(engine, expected_target); result != 0) {
            return result;
        }
        if (const int result = exercise_registered_frontend(engine); result != 0) { return result; }
        if (const int result = exercise_full_prefill_chunk(engine); result != 0) { return result; }
        if (const int result = exercise_rewrite_checkpoints(engine); result != 0) { return result; }
        if (const int result = exercise_prefix(engine); result != 0) { return result; }
        if (const int result = exercise_abandoned_handle_capacity(engine); result != 0) {
            return result;
        }
    }
    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = exercise_vision(engine); result != 0) { return result; }
    }
    if (const int result = exercise_host_restore(artifact); result != 0) { return result; }
    {
        // Production C=1/H=1 topology: the shared prefix occupies the only cache Device slot,
        // while the pre-generation ResponseReplay must still rotate without a session identity.
        ninfer::Engine engine(shared_replacement_engine_options(artifact));
        if (const int result = exercise_rewrite_checkpoints(engine); result != 0) { return result; }
    }
    if (const int result = exercise_shared_replacement_and_full_capacity_reuse(artifact);
        result != 0) {
        return result;
    }
    if (const int result = exercise_last_private_alias_eviction(artifact); result != 0) {
        return result;
    }
    if (const int result = exercise_concurrent_resource_settlement(artifact, expected_target);
        result != 0) {
        return result;
    }
    return 0;
}

int main() {
    const char* groupwise        = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* nvfp4            = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    const char* qwen38_groupwise = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    const char* qwen38_nvfp4     = std::getenv("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS");
    if ((groupwise == nullptr || *groupwise == '\0') && (nvfp4 == nullptr || *nvfp4 == '\0') &&
        (qwen38_groupwise == nullptr || *qwen38_groupwise == '\0') &&
        (qwen38_nvfp4 == nullptr || *qwen38_nvfp4 == '\0')) {
        std::cout << "skip: neither NINFER_QWEN3_6_27B_WEIGHTS nor "
                     "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS nor a Qwen3.8 equivalent is set\n";
        return 77;
    }
    if (groupwise != nullptr && *groupwise != '\0') {
        if (const int result = exercise_artifact(groupwise, "qwen3_6_27b"); result != 0) {
            return result;
        }
    }
    if (nvfp4 != nullptr && *nvfp4 != '\0') {
        if (const int result = exercise_artifact(nvfp4, "qwen3_6_27b"); result != 0) {
            return result;
        }
    }
    if (qwen38_groupwise != nullptr && *qwen38_groupwise != '\0') {
        if (const int result =
                exercise_concurrent_resource_settlement(qwen38_groupwise, "qwen3_8_27b");
            result != 0) {
            return result;
        }
    }
    if (qwen38_nvfp4 != nullptr && *qwen38_nvfp4 != '\0') {
        if (const int result = exercise_concurrent_resource_settlement(qwen38_nvfp4, "qwen3_8_27b");
            result != 0) {
            return result;
        }
    }
    std::cout << "ok\n";
    return 0;
}
