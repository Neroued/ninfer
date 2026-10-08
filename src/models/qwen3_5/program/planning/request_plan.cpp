#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/execution_context.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>

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

} // namespace

bool RequestBasePlanImpl::accepts_capture(std::uint32_t frontier) const noexcept {
    if (frontier == 0 || frontier > summary.prompt_tokens) { return false; }
    if (vision_control_plan) {
        for (const auto& item : vision_control_plan->items) {
            if (item.token_begin < frontier && frontier < item.token_end) { return false; }
        }
    }
    return true;
}

CaptureGroup RequestBasePlanImpl::capture_group(std::uint32_t frontier) const {
    if (!capture_backing || !accepts_capture(frontier)) {
        throw std::logic_error("capture identity requires a legal prompt frontier");
    }
    return {.identity = capture_backing,
            .key      = {prefix_digests.at(frontier), frontier, prefix_identity_tag},
            .frontier = frontier};
}

RequestBasePlan ProgramImpl::plan_request(PreparedPromptData&& prompt,
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
    if (const auto& rewrite = prompt.identity.rewrite_checkpoint;
        rewrite &&
        (rewrite->recovery_frontier == 0 || rewrite->recovery_frontier > rewrite->frontier ||
         rewrite->frontier > prompt.token_ids.size())) {
        throw std::invalid_argument("input recovery is outside its typed rewrite prefix");
    }

    auto base                             = std::make_shared<RequestBasePlanImpl>();
    base->context_cache                   = prompt.context_cache;
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
    base->allow_prefix_reuse             = options.allow_prefix_reuse && prompt.identity.reusable;
    base->summary.publish_continuation =
        options.allow_prefix_reuse && prompt.identity.reusable && context_cache.enabled;
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


    base->prefix_digests.assign(prompt);
    base->prefix_identity_tag = static_cast<std::uint32_t>(speculative_backend) |
                                (static_cast<std::uint32_t>(proposal_head) << 8U) |
                                (static_cast<std::uint32_t>(kv_storage) << 16U);
    std::uint32_t previous = 0;
    for (const auto frontier : prompt.identity.rewrite_execution_frontiers) {
        if (frontier <= previous || frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite execution frontiers must be ordered prompt positions");
        }
        previous = frontier;
    }
    base->prompt = std::make_shared<const PreparedPromptData>(std::move(prompt));
    if (base->summary.publish_continuation) {
        const auto& prepared = *base->prompt;
        auto backing         = std::make_shared<PreparedCaptureBacking>();
        backing->digests     = base->prefix_digests;
        backing->ledger      = prepared.token_ids;
        backing->prefix_identity.assign(prepared);
        base->capture_backing = std::move(backing);
        const auto add        = [&](std::uint32_t frontier, runtime::CheckpointRole role) {
            if (!base->accepts_capture(frontier)) { return; }
            auto it =
                std::find_if(base->capture_groups.begin(), base->capture_groups.end(),
                                    [frontier](const auto& group) { return group.frontier == frontier; });
            if (it == base->capture_groups.end()) {
                base->capture_groups.push_back(base->capture_group(frontier));
                it = std::prev(base->capture_groups.end());
            }
            if (std::find(it->roles.begin(), it->roles.end(), role) == it->roles.end()) {
                it->roles.push_back(role);
            }
        };
        const auto& rewrite = prepared.identity.rewrite_checkpoint;
        add(rewrite ? rewrite->recovery_frontier : base->summary.prompt_tokens,
            runtime::CheckpointRole::InputReplay);
        for (const auto& opportunity : prepared.context_cache.opportunities) {
            add(opportunity.frontier, opportunity.kind == PromptCacheMarkerKind::PrivateLongAnchor
                                          ? runtime::CheckpointRole::LongAnchor
                                          : runtime::CheckpointRole::SharedPrefix);
        }
        std::sort(base->capture_groups.begin(), base->capture_groups.end(),
                  [](const auto& a, const auto& b) { return a.frontier < b.frontier; });
    }
    return RequestBasePlan(std::move(base));
}

std::uint32_t ProgramImpl::initial_mtp_extent(const RequestBasePlanImpl& base) const {
    const auto prompt = base.summary.prompt_tokens;
    return speculative_backend == SpeculativeBackend::Mtp
               ? std::min({draft_window,
                           base.summary.effective_output_tokens > 1
                               ? base.summary.effective_output_tokens - 2U
                               : 0U,
                           capacity > prompt ? capacity - prompt - 1U : 0U})
               : 0U;
}

UnitDemand ProgramImpl::prefill_unit(std::uint32_t prompt, std::uint32_t cursor,
                                     std::uint32_t mtp_extent) const {
    UnitDemand demand{.kind = ExecutionUnitKind::Prefill};
    demand.main_frontier    = std::min(prompt, cursor + prefill_chunk);
    demand.backend_frontier = backend_kv_cache() ? demand.main_frontier : 0;
    if (speculative_backend == SpeculativeBackend::Mtp && demand.main_frontier == prompt &&
        mtp_extent != 0) {
        demand.backend_frontier = std::min(capacity, prompt + mtp_extent - 1U);
    }
    return demand;
}

UnitDemand ProgramImpl::next_unit(const SequenceState& sequence, const RequestControl& request,
                                  ExecutionUnitKind kind, std::uint32_t tokens) const {
    UnitDemand demand{.kind = kind, .tokens = tokens};
    const auto frontier = sequence.execution_frontier;
    switch (kind) {
    case ExecutionUnitKind::Prefill: {
        if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
            throw std::logic_error("prefill demand requires a prefill cursor");
        }
        const auto& staged = *request.prefill;
        return prefill_unit(staged.prompt_tokens, staged.cursor, staged.initial_mtp_extent);
    }
    case ExecutionUnitKind::Replay:
        if (request.lifecycle != Lifecycle::Replaying) {
            throw std::logic_error("request is not replaying");
        }
        demand.main_frontier =
            std::min(request.replay_target, request.replay_cursor + prefill_chunk);
        demand.backend_frontier = backend_kv_cache() ? demand.main_frontier : 0;
        break;
    case ExecutionUnitKind::Decode: {
        if (request.lifecycle != Lifecycle::Active || tokens == 0 || frontier >= capacity) {
            throw std::logic_error("decode demand requires an active token and output budget");
        }
        const auto available    = std::min(tokens - 1U, capacity - frontier - 1U);
        const auto extent       = speculative_backend == SpeculativeBackend::None
                                      ? 0U
                                      : std::min({available, draft_window,
                                            speculative_backend == SpeculativeBackend::Mtp
                                                      ? sequence.mtp_draft_count
                                                      : draft_window});
        demand.main_frontier    = frontier + extent + 1U;
        demand.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                                      ? std::min(capacity, frontier + extent + draft_window)
                                  : speculative_backend == SpeculativeBackend::DFlash
                                      ? demand.main_frontier
                                      : 0U;
        break;
    }
    case ExecutionUnitKind::Control:
        if (request.lifecycle != Lifecycle::Active || tokens == 0 || tokens > capacity - frontier) {
            throw std::logic_error("control demand is outside the available context");
        }
        demand.main_frontier    = frontier + tokens;
        demand.backend_frontier = backend_kv_cache() ? demand.main_frontier : 0;
        break;
    case ExecutionUnitKind::Normalize:
        demand.main_frontier    = sequence.text_kv_valid;
        demand.backend_frontier = backend_kv_cache() ? frontier : 0;
        break;
    }
    return demand;
}

runtime::ResourceReservation ProgramImpl::reserve_units(std::span<const ExecutionUnit> units) {
    std::array<UnitDemand, kMaximumConcurrency> demands{};
    std::array<UnitDemand, kMaximumConcurrency> coverage{};
    std::array<std::uint32_t, kMaximumConcurrency> old_main{}, old_backend{};
    std::array<bool, kMaximumConcurrency> seen{};
    std::uint64_t main_needed = 0, backend_needed = 0;
    for (std::size_t index = 0; index < units.size(); ++index) {
        const auto& unit = units[index];
        if (!valid_sequence(unit.sequence)) {
            throw std::logic_error("execution unit has a stale lane");
        }
        const auto lane = ContractAccess::lane(unit.sequence).value;
        if (seen[lane]) { throw std::logic_error("duplicate lane in unit reservation"); }
        seen[lane]        = true;
        auto& request     = requests[lane];
        const auto& state = active_sequence(lane);
        demands[lane]     = next_unit(state, request, unit.kind, unit.tokens);
        coverage[lane]    = request.recovery ? request.recovery->coverage : demands[lane];
        if (demands[lane].main_frontier > coverage[lane].main_frontier ||
            demands[lane].backend_frontier > coverage[lane].backend_frontier) {
            throw std::logic_error("execution exceeds the retained recovery capacity");
        }
        if (request.permit) {
            if (request.permit->kind != unit.kind || request.permit->tokens != unit.tokens ||
                request.permit->main_frontier != demands[lane].main_frontier ||
                request.permit->backend_frontier != demands[lane].backend_frontier) {
                throw std::logic_error("an existing execution permit cannot be replaced");
            }
            continue;
        }
        const auto main = text_kv_addresses->growth_pages_for_tokens(state.kv->text,
                                                                     coverage[lane].main_frontier);
        old_main[lane]  = text_kv_addresses->reserved_growth_pages(state.kv->text);
        main_needed += main > old_main[lane] ? main - old_main[lane] : 0;
        if (state.kv->backend) {
            const auto backend = backend_kv_addresses->growth_pages_for_tokens(
                *state.kv->backend, coverage[lane].backend_frontier);
            old_backend[lane] = backend_kv_addresses->reserved_growth_pages(*state.kv->backend);
            backend_needed += backend > old_backend[lane] ? backend - old_backend[lane] : 0;
        }
    }
    const auto main_available = text_kv_pages->physical_pool().available_pages();
    const auto backend_available =
        backend_kv_pages ? backend_kv_pages->physical_pool().available_pages() : 0;
    if (main_needed > main_available || backend_needed > backend_available) {
        return {.reserved = false,
                .shortage = {.main_kv_pages = static_cast<std::uint32_t>(
                                 main_needed > main_available ? main_needed - main_available : 0),
                             .backend_kv_pages =
                                 static_cast<std::uint32_t>(backend_needed > backend_available
                                                                ? backend_needed - backend_available
                                                                : 0)}};
    }
    std::array<bool, kMaximumConcurrency> added{};
    try {
        for (const auto& unit : units) {
            const auto lane = ContractAccess::lane(unit.sequence).value;
            if (requests[lane].permit) { continue; }
            auto& state = active_sequence(lane);
            added[lane] = true;
            text_kv_addresses->reserve_growth(state.kv->text,
                                              text_kv_addresses->growth_pages_for_tokens(
                                                  state.kv->text, coverage[lane].main_frontier));
            if (state.kv->backend) {
                backend_kv_addresses->reserve_growth(
                    *state.kv->backend, backend_kv_addresses->growth_pages_for_tokens(
                                            *state.kv->backend, coverage[lane].backend_frontier));
            }
            requests[lane].permit = demands[lane];
        }
    } catch (...) {
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            if (!added[lane]) { continue; }
            auto& state = active_sequence(lane);
            text_kv_addresses->reserve_growth(state.kv->text, old_main[lane]);
            if (state.kv->backend) {
                backend_kv_addresses->reserve_growth(*state.kv->backend, old_backend[lane]);
            }
            requests[lane].permit.reset();
        }
        throw;
    }
    return {.reserved = true};
}

void ProgramImpl::release_units(std::span<const SequenceHandle> handles) noexcept {
    for (const auto handle : handles) {
        if (!valid_sequence(handle)) { continue; }
        settle_unit(ContractAccess::lane(handle).value);
    }
}

void ProgramImpl::require_unit(std::uint32_t lane, ExecutionUnitKind kind,
                               std::uint32_t tokens) const {
    if (!requests[lane].permit || requests[lane].permit->kind != kind ||
        requests[lane].permit->tokens != tokens) {
        throw std::logic_error("execution does not match its reserved unit");
    }
}

void ProgramImpl::settle_unit(std::uint32_t lane) noexcept {
    auto& state   = sequences[lane];
    auto& request = requests[lane];
    if (request.recovery && request.lifecycle != Lifecycle::Replaying &&
        request.lifecycle != Lifecycle::Pending &&
        (request.lifecycle == Lifecycle::Finishable ||
         state.execution_frontier > request.recovery->frontier ||
         state.ledger.size() > request.recovery->ledger_tokens ||
         (request.prefill && request.prefill->cursor > request.recovery->frontier))) {
        request.recovery.reset();
    }
    if (state.kv) {
        try {
            text_kv_addresses->settle_growth(
                state.kv->text, state.text_kv_valid,
                request.recovery ? request.recovery->coverage.main_frontier : 0);
            if (state.kv->backend) {
                backend_kv_addresses->settle_growth(
                    *state.kv->backend, backend_kv_valid(state),
                    request.recovery ? request.recovery->coverage.backend_frontier : 0);
            }
        } catch (...) { std::terminate(); }
    }
    request.permit.reset();
}
} // namespace ninfer::models::qwen3_5::detail
