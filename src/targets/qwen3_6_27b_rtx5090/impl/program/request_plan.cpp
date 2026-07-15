#include "targets/qwen3_6_27b_rtx5090/impl/program/program.h"

#include "targets/qwen3_6_27b_rtx5090/impl/config.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/schedule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {
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

bool matches_prefix(std::span<const TokenId> incoming, const std::vector<TokenId>& resident,
                    std::size_t count) {
    if (incoming.size() < count || resident.size() < count) { return false; }
    return std::equal(incoming.begin(), incoming.begin() + static_cast<std::ptrdiff_t>(count),
                      resident.begin());
}

} // namespace

RequestPlan::RequestPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

RequestPlan::RequestPlan(RequestPlan&&) noexcept            = default;
RequestPlan& RequestPlan::operator=(RequestPlan&&) noexcept = default;
RequestPlan::~RequestPlan()                                 = default;

const runtime::RequestPlanSummary& RequestPlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

RequestPlan Program::Impl::plan_request(const PreparedPromptData& prompt,
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
    validate_sampling(options.sampling);

    auto plan                             = std::make_unique<RequestPlan::Impl>();
    plan->multimodal                      = prompt.has_media();
    plan->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    plan->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - plan->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    plan->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    plan->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    plan->summary.transient_alignment    = prompt.has_media() ? 256 : 1;
    plan->summary.transient_bytes =
        prompt.has_media() ? schedule::vision_workspace_bytes(prompt) : 0;
    plan->sampling = translate_sampling(options.sampling);

    const std::span<const TokenId> incoming(prompt.token_ids);
    if (options.allow_prefix_reuse && prompt.identity.reusable && !prompt.has_media() &&
        lifecycle == Lifecycle::Resident && !resident_multimodal) {
        if (E != 0 && matches_prefix(incoming, ledger, E)) {
            plan->reuse      = ReusePath::AppendAtFrontier;
            plan->reuse_base = E;
        } else if (boundary.valid && boundary.boundary != 0 &&
                   boundary.boundary < incoming.size() &&
                   matches_prefix(incoming, ledger, boundary.boundary)) {
            plan->reuse      = ReusePath::RestoreBoundary;
            plan->reuse_base = boundary.boundary;
        }
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    const bool mtp_capacity =
        mtp_k != 0 &&
        static_cast<std::uint64_t>(plan->summary.prompt_tokens) + 2ULL * mtp_k <= capacity;
    if (mtp_capacity) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier && tail_hidden_valid &&
                   mtp_kv != nullptr &&
                   (plan->reuse_base == 0 || mtp_kv_valid >= plan->reuse_base - 1)) {
            plan->prepare_mtp      = true;
            plan->needs_mtp_bridge = plan->reuse_base != 0;
        } else if (plan->reuse == ReusePath::RestoreBoundary && mtp_kv != nullptr &&
                   boundary.hidden_valid && boundary.mtp_prefix_valid &&
                   mtp_kv_valid >= plan->reuse_base - 1) {
            plan->prepare_mtp      = true;
            plan->needs_mtp_bridge = true;
        }
    }

    if (!prompt.has_media() && prompt.identity.assistant_content_boundary) {
        const std::uint32_t candidate = *prompt.identity.assistant_content_boundary;
        if (candidate > plan->reuse_base && candidate <= plan->summary.prompt_tokens) {
            plan->snapshot_boundary = candidate;
        }
    }
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
