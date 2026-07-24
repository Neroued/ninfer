#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

void validate_sampling(const SamplingParameters& sampling) {
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

ops::SamplingConfig translate_sampling(const SamplingParameters& source) {
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

std::size_t checked_size_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t align_up_256(std::size_t value, const char* label) {
    constexpr std::size_t mask = 255;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error(label);
    }
    return (value + mask) & ~mask;
}

} // namespace

RequestPlan ProgramImplCore::plan_request(const PreparedPromptData& prompt,
                                          const ExecutionOptions& options) const {
    if (lifecycle == Lifecycle::Active || lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.patches.empty()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto plan                             = std::make_unique<RequestPlanImpl>();
    plan->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    plan->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - plan->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    plan->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    plan->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    plan->summary.transient_alignment    = 1;
    plan->summary.transient_bytes        = 0;
    plan->sampling                       = translate_sampling(options.sampling);

    if (options.allow_prefix_reuse && prompt.identity.reusable &&
        lifecycle == Lifecycle::Resident) {
        const bool dflash_append_ready = speculative_backend != SpeculativeBackend::DFlash ||
                                         (!pending_context_valid && dflash_context_frontier == E);
        if (E != 0 && dflash_append_ready &&
            qwen3_6::detail::prefix_matches(prompt, ledger, prefix_identity, E)) {
            plan->reuse      = ReusePath::AppendAtFrontier;
            plan->reuse_base = E;
        } else if (boundary.valid && boundary.boundary != 0 &&
                   (speculative_backend != SpeculativeBackend::DFlash ||
                    (dflash_boundary_valid && dflash_boundary_frontier == boundary.boundary)) &&
                   boundary.boundary < prompt.token_ids.size() &&
                   qwen3_6::detail::prefix_matches(prompt, ledger, prefix_identity,
                                                   boundary.boundary)) {
            plan->reuse      = ReusePath::RestoreBoundary;
            plan->reuse_base = boundary.boundary;
        }
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    const bool mtp_capacity =
        speculative_backend == SpeculativeBackend::Mtp &&
        static_cast<std::uint64_t>(plan->summary.prompt_tokens) + 2ULL * draft_window <= capacity;
    if (mtp_capacity) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier && tail_hidden_valid &&
                   decoder->mtp_cache() != nullptr &&
                   (plan->reuse_base == 0 || mtp_kv_valid >= plan->reuse_base - 1)) {
            plan->prepare_mtp      = true;
            plan->needs_mtp_bridge = plan->reuse_base != 0;
        } else if (plan->reuse == ReusePath::RestoreBoundary && decoder->mtp_cache() != nullptr &&
                   boundary.hidden_valid && boundary.mtp_prefix_valid &&
                   mtp_kv_valid >= plan->reuse_base - 1) {
            plan->prepare_mtp      = true;
            plan->needs_mtp_bridge = true;
        }
    }

    plan->prepare_dflash =
        speculative_backend == SpeculativeBackend::DFlash &&
        static_cast<std::uint64_t>(plan->summary.prompt_tokens) + draft_window + 1ULL <= capacity;
    if (plan->needs_mtp_bridge && plan->reuse_base < plan->summary.prompt_tokens &&
        prompt.token_types[plan->reuse_base] != 0) {
        // A bridge consumes the first suffix token as the MTP shifted embedding. Reusing target
        // state remains valid when that token is visual, but the one-column bridge currently has
        // no Vision embedding input. Keep the target prefix and fall back to ordinary decode.
        plan->prepare_mtp      = false;
        plan->needs_mtp_bridge = false;
    }

    if (prompt.has_media()) {
        VisionPrefillPlan vision;
        vision.control = qwen3_6::build_vision_control(prompt);
        vision.uses.reserve(vision.control.items.size());
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (std::size_t index = 0; index < vision.control.items.size(); ++index) {
            const qwen3_6::VisionItemControl& item = vision.control.items[index];
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            if (!vision.uses.empty() && begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > plan->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            if (schedule::VisionContext::workspace_bytes(item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        if (!vision.uses.empty()) {
            const std::size_t output_elements =
                checked_size_mul(static_cast<std::size_t>(VisionConfig::output_hidden), max_merged,
                                 "vision item output elements overflow size_t");
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes =
                align_up_256(checked_size_mul(output_elements, dtype_size(DType::BF16),
                                              "vision item output bytes overflow size_t"),
                             "vision item output alignment overflows size_t");
            plan->vision = std::move(vision);
        }
    }

    if (prompt.identity.assistant_content_boundary) {
        const std::uint32_t candidate = *prompt.identity.assistant_content_boundary;
        if (candidate > plan->reuse_base && candidate <= plan->summary.prompt_tokens) {
            plan->snapshot_boundary = candidate;
        }
    }
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
