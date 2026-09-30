#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

} // namespace

RequestBasePlan ProgramImpl::plan_request(const PreparedPromptData& prompt,
                                          const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= execution::dimension(parameters.model.resources().public_token_count)) {
            throw std::invalid_argument("prompt contains token outside the public token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.media_payloads.empty() ||
        prompt.media_payloads.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    for (std::size_t i = 0; i < prompt.media_payloads.size(); ++i) {
        if (!prompt.media_payloads[i] ||
            prompt.media_payloads[i]->patch_elements !=
                prompt.vision_items[i].patch_count * kPreparedVisionPatchFeatures) {
            throw std::invalid_argument("prepared prompt media item payload has an invalid shape");
        }
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->sampling                       = translate_sampling(options.sampling);
    base->allow_prefix_reuse             = options.allow_prefix_reuse && context_cache.enabled;
    base->summary.publish_continuation   = base->allow_prefix_reuse && prompt.identity.reusable;
    const std::uint32_t reserved_context_tokens =
        base->summary.prompt_tokens + (base->summary.effective_output_tokens == 0
                                           ? 0U
                                           : base->summary.effective_output_tokens - 1U);
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const std::uint32_t mtp_tokens    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(reserved_context_tokens) + draft_window - 1ULL));
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_tokens);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    }
    if (prompt.has_media()) {
        if (!workspace_plan.vision) {
            throw std::logic_error("Vision prompt has no startup workspace plan");
        }
        auto vision = std::make_shared<qwen3_5::VisionControlPlan>(
            qwen3_5::plan_vision_control(prompt, *parameters.model.config().vision));
        std::uint32_t previous_end = 0;
        for (std::size_t index = 0; index < vision->items.size(); ++index) {
            const qwen3_5::VisionItemControlPlan& item = vision->items[index];
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && item.token_begin != 0
                    ? item.token_begin - 1
                    : item.token_begin;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (item.merged_count > workspace_plan.vision->max_merged_tokens ||
                execution::VisionContext::workspace_bytes(
                    *parameters.model.config().vision, *parameters.vision,
                    prompt.vision_items[index].patch_count, item.merged_count,
                    *workspace_plan.vision) > workspace_plan.vision->encode_peak_bytes) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = item.token_end;
        }
        base->vision_control_plan = std::move(vision);
    }
    if (prompt.identity.rewrite_checkpoint &&
        (prompt.identity.rewrite_checkpoint->frontier == 0 ||
         prompt.identity.rewrite_checkpoint->frontier > base->summary.prompt_tokens)) {
        throw std::invalid_argument(
            "rewrite checkpoint frontier must lie at or inside the prompt frontier");
    }

    // Scheduler service units from the root: one per prefill chunk and Vision item, one per further
    // generated token. Admission recomputes them for the prefix it resumes from.
    const std::uint32_t prompt_tokens = base->summary.prompt_tokens;
    std::uint64_t units               = 1ULL + (prompt_tokens - 1ULL) / prefill_chunk;
    if (base->vision_control_plan) { units += base->vision_control_plan->items.size(); }
    if (base->summary.effective_output_tokens != 0) {
        units += base->summary.effective_output_tokens - 1ULL;
    }
    base->summary.service_work_quanta = units;
    return RequestBasePlan(std::move(base));
}

} // namespace ninfer::models::qwen3_5::detail
