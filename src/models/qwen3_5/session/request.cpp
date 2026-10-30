#include "models/qwen3_5/session/internal.h"
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

} // namespace

bool RequestBasePlanImpl::accepts_capture(std::uint32_t frontier) const noexcept {
    if (frontier == 0 || frontier > summary.prompt_tokens) { return false; }
    if (vision && !vision->can_checkpoint(frontier)) return false;
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

RequestBasePlan SessionImpl::plan_request(qwen::PreparedPromptData&& prompt,
                                          const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= static_cast<std::int32_t>(plan.profile->public_tokens)) {
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
            prompt.media_payloads[i]->features_per_patch != qwen::kPreparedVisionPatchFeatures ||
            prompt.media_payloads[i]->patch_elements !=
                prompt.vision_items[i].patch_count * qwen::kPreparedVisionPatchFeatures) {
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
    base->sampling                       = options.sampling;
    base->allow_prefix_reuse             = options.allow_prefix_reuse && prompt.identity.reusable;
    base->summary.publish_continuation =
        options.allow_prefix_reuse && prompt.identity.reusable && context_cache.enabled;
    if (prompt.has_media()) {
        if (!plan.vision_max_features) {
            throw std::logic_error("Vision prompt has no startup workspace plan");
        }
        base->vision = qwen::prepare_vision_input(prompt, *plan.profile->config.vision,
                                                  plan.vision_max_features);
    }

    base->prefix_digests.assign(prompt);
    base->prefix_identity_tag = static_cast<std::uint32_t>(speculative_backend) |
                                (static_cast<std::uint32_t>(plan.proposal_head) << 8U) |
                                (static_cast<std::uint32_t>(plan.kv_storage) << 16U);
    std::uint32_t previous = 0;
    for (const auto frontier : prompt.identity.rewrite_execution_frontiers) {
        if (frontier <= previous || frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite execution frontiers must be ordered prompt positions");
        }
        previous = frontier;
    }
    base->prompt = std::make_shared<const qwen::PreparedPromptData>(std::move(prompt));
    if (base->summary.publish_continuation) {
        const auto& prepared = *base->prompt;
        auto backing         = std::make_shared<PreparedCaptureBacking>();
        backing->digests     = base->prefix_digests;
        backing->ledger      = prepared.token_ids;
        backing->prefix_identity.assign(prepared);
        base->capture_backing = std::move(backing);
        const auto add        = [&](std::uint32_t frontier, runtime::CheckpointRole role,
                             SharedCandidateEvidence evidence = SharedCandidateEvidence::None) {
            if (!base->accepts_capture(frontier)) { return; }
            auto it =
                std::find_if(base->capture_groups.begin(), base->capture_groups.end(),
                                    [frontier](const auto& group) { return group.frontier == frontier; });
            if (it == base->capture_groups.end()) {
                base->capture_groups.push_back(base->capture_group(frontier));
                it = std::prev(base->capture_groups.end());
            }
            it->evidence |= evidence;
            if (std::find(it->roles.begin(), it->roles.end(), role) == it->roles.end()) {
                it->roles.push_back(role);
            }
        };
        const auto& rewrite = prepared.identity.rewrite_checkpoint;
        add(rewrite ? rewrite->recovery_frontier : base->summary.prompt_tokens,
            runtime::CheckpointRole::InputReplay);
        for (const auto& opportunity : prepared.context_cache.opportunities) {
            add(opportunity.frontier,
                opportunity.kind == PromptCacheMarkerKind::PrivateLongAnchor
                    ? runtime::CheckpointRole::LongAnchor
                    : runtime::CheckpointRole::SharedPrefix,
                opportunity.evidence);
        }
        std::sort(base->capture_groups.begin(), base->capture_groups.end(),
                  [](const auto& a, const auto& b) { return a.frontier < b.frontier; });
    }
    return RequestBasePlan(std::move(base));
}

UnitDemand SessionImpl::prefill_unit(std::uint32_t prompt, std::uint32_t cursor) const {
    UnitDemand out{.kind = ExecutionUnitKind::Prefill};
    out.main_frontier    = std::min(prompt, cursor + prefill_chunk);
    out.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                               ? (out.main_frontier ? out.main_frontier - 1 : 0)
                           : has_backend_pool() ? out.main_frontier
                                                : 0;
    return out;
}

UnitDemand SessionImpl::next_unit(const SequenceState& state, const RequestControl& request,
                                  ExecutionUnitKind kind, std::uint32_t tokens) const {
    UnitDemand out{.kind = kind, .tokens = tokens};
    const auto f = state.continuation.metadata.execution_frontier;
    switch (kind) {
    case ExecutionUnitKind::Prefill:
        if (request.lifecycle != Lifecycle::Prefilling || !request.prefill)
            throw std::logic_error("prefill demand requires a prefill cursor");
        out.main_frontier = window_end(state, request, request.prefill->cursor,
                                       request.prefill->prompt_tokens, vision::Window{});
        break;
    case ExecutionUnitKind::Replay:
        if (request.lifecycle != Lifecycle::Replaying)
            throw std::logic_error("request is not replaying");
        out.main_frontier = window_end(state, request, request.replay_cursor, request.replay_target,
                                       vision::Window{});
        break;
    case ExecutionUnitKind::Decode:
        if (request.lifecycle != Lifecycle::Active || !tokens || f >= capacity)
            throw std::logic_error("decode demand requires an active token and output budget");
        out.main_frontier = f + 1 +
                            (speculative_backend == SpeculativeBackend::None
                                 ? 0
                                 : std::min({tokens - 1, capacity - f - 1, draft_window}));
        break;
    case ExecutionUnitKind::Control:
        if (request.lifecycle != Lifecycle::Active || !tokens || tokens > capacity - f)
            throw std::logic_error("control demand is outside the available context");
        out.main_frontier = f + tokens;
        break;
    case ExecutionUnitKind::Normalize:
        out.main_frontier = f;
        break;
    }
    out.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                               ? (out.main_frontier ? out.main_frontier - 1 : 0)
                           : has_backend_pool() ? out.main_frontier
                                                : 0;
    return out;
}

ResourceReservation SessionImpl::reserve_units(std::span<const ExecutionUnit> units) {
    std::array<UnitDemand, kMaximumConcurrency> demands{};
    std::array<UnitDemand, kMaximumConcurrency> coverage{};
    std::array<std::uint32_t, kMaximumConcurrency> old_main{}, old_backend{};
    std::array<std::uint32_t, kMaximumConcurrency> add_main{}, add_backend{};
    std::array<bool, kMaximumConcurrency> seen{};
    for (std::size_t index = 0; index < units.size(); ++index) {
        const auto& unit = units[index];
        if (!valid_sequence(unit.sequence)) {
            throw std::logic_error("execution unit has a stale lane");
        }
        const auto lane = this->lane(unit.sequence);
        if (seen[lane]) { throw std::logic_error("duplicate lane in unit reservation"); }
        seen[lane]        = true;
        auto& request     = requests[lane];
        const auto& state = active_sequence(lane);
        demands[lane]     = next_unit(state, request, unit.kind, unit.tokens);
        if (unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) {
            auto& demand = demands[lane];
            demand.main_frontier =
                window_end(state, request, state.continuation.metadata.execution_frontier,
                           demand.main_frontier, vision_window);
            demand.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                                          ? (demand.main_frontier ? demand.main_frontier - 1 : 0)
                                      : has_backend_pool() ? demand.main_frontier
                                                           : 0;
        }
        if ((unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) &&
            demands[lane].main_frontier == state.continuation.metadata.execution_frontier &&
            state.continuation.metadata.execution_frontier < request.base->summary.prompt_tokens)
            return {.status = runtime::UnitReservationStatus::ExecutionUnavailable};
        if (request.permit) {
            const auto& permit = *request.permit;
            if (permit.kind != unit.kind || permit.tokens != unit.tokens)
                throw std::logic_error("an existing execution permit cannot change its operation");
            if (unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) {
                // A permit retains capacity through its frontier. A checkpoint or an execution
                // window may shorten the next prefix, but cannot enlarge that retained grant.
                demands[lane].main_frontier =
                    std::min(demands[lane].main_frontier, permit.main_frontier);
                demands[lane].backend_frontier =
                    std::min(demands[lane].backend_frontier, permit.backend_frontier);
            } else if (permit.main_frontier != demands[lane].main_frontier ||
                       permit.backend_frontier != demands[lane].backend_frontier)
                throw std::logic_error("an existing execution permit cannot change its frontier");
        }
        coverage[lane] = request.recovery ? request.recovery->coverage : demands[lane];
        if (demands[lane].main_frontier > coverage[lane].main_frontier ||
            demands[lane].backend_frontier > coverage[lane].backend_frontier)
            throw std::logic_error("execution exceeds the retained recovery capacity");
        if (request.permit) continue;
        const auto main = text_kv_histories->write_pages(state.kv->text.handle(),
                                                         state.continuation.metadata.text_kv_valid,
                                                         coverage[lane].main_frontier);
        old_main[lane]  = text_kv_histories->reserved_growth_pages(state.kv->text.handle());
        add_main[lane]  = main > old_main[lane] ? main - old_main[lane] : 0;
        if (state.kv->backend) {
            const auto backend = backend_kv_histories->write_pages(state.kv->backend->handle(),
                                                                   backend_kv_valid(state),
                                                                   coverage[lane].backend_frontier);
            old_backend[lane] =
                backend_kv_histories->reserved_growth_pages(state.kv->backend->handle());
            add_backend[lane] = backend > old_backend[lane] ? backend - old_backend[lane] : 0;
        }
    }
    ContextDemand bundle;
    std::array<context::ContextDemandId, kMaximumConcurrency> main_ids, backend_ids;
    for (const auto& unit : units) {
        const auto lane = this->lane(unit.sequence);
        if (requests[lane].permit) continue;
        const auto& state = active_sequence(lane);
        main_ids[lane] =
            bundle.add_growth(*text_kv_histories, state.kv->text.handle(), add_main[lane]);
        if (state.kv->backend)
            backend_ids[lane] = bundle.add_growth(*backend_kv_histories,
                                                  state.kv->backend->handle(), add_backend[lane]);
    }
    auto assessment = directory.assess(bundle);
    if (!assessment.ready()) return {.assessment = std::move(assessment)};
    auto reserved = directory.reserve(bundle);
    if (!reserved) return {.assessment = std::move(reserved.assessment)};
    std::array<bool, kMaximumConcurrency> added{};
    try {
        for (const auto& unit : units) {
            const auto lane = this->lane(unit.sequence);
            if (requests[lane].permit) { continue; }
            auto& state = active_sequence(lane);
            added[lane] = true;
            auto main   = reserved.reservation->take(main_ids[lane]);
            text_kv_histories->prepare_growth(state.kv->text.handle(), main);
            text_kv_histories->adopt_growth(state.kv->text.handle(), std::move(main));
            if (state.kv->backend) {
                auto backend = reserved.reservation->take(backend_ids[lane]);
                backend_kv_histories->prepare_growth(state.kv->backend->handle(), backend);
                backend_kv_histories->adopt_growth(state.kv->backend->handle(), std::move(backend));
            }
            requests[lane].permit = demands[lane];
        }
    } catch (...) {
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            if (!added[lane]) { continue; }
            auto& state = active_sequence(lane);
            text_kv_histories->shrink_growth(state.kv->text.handle(), old_main[lane]);
            if (state.kv->backend) {
                backend_kv_histories->shrink_growth(state.kv->backend->handle(), old_backend[lane]);
            }
            requests[lane].permit.reset();
        }
        throw;
    }
    return {.status = runtime::UnitReservationStatus::Granted};
}

void SessionImpl::release_units(std::span<const SequenceHandle> handles) noexcept {
    try {
        for (const auto handle : handles) {
            if (!valid_sequence(handle) || execution_pending(handle)) continue;
            const auto l  = lane(handle);
            auto& state   = sequences[l];
            auto& request = requests[l];
            if (state.kv) {
                text_kv_histories->shrink_growth(
                    state.kv->text.handle(),
                    request.recovery
                        ? text_kv_histories->write_pages(state.kv->text.handle(),
                                                         state.continuation.metadata.text_kv_valid,
                                                         request.recovery->coverage.main_frontier)
                        : 0);
                if (state.kv->backend)
                    backend_kv_histories->shrink_growth(
                        state.kv->backend->handle(),
                        request.recovery ? backend_kv_histories->write_pages(
                                               state.kv->backend->handle(), backend_kv_valid(state),
                                               request.recovery->coverage.backend_frontier)
                                         : 0);
            }
            request.permit.reset();
        }
    } catch (...) { std::terminate(); }
}

void SessionImpl::require_unit(std::uint32_t lane, ExecutionUnitKind kind,
                               std::uint32_t tokens) const {
    if (!requests[lane].permit || requests[lane].permit->kind != kind ||
        requests[lane].permit->tokens != tokens)
        throw std::logic_error("execution does not match its reserved unit");
}

bool SessionImpl::prepare_settlement(std::uint32_t lane, const ContinuationMetadata& continuation,
                                     Lifecycle lifecycle, std::size_t ledger_tokens,
                                     context::ContextTransaction& transaction) {
    auto& state   = sequences[lane];
    auto& request = requests[lane];
    const bool retain =
        request.recovery && (lifecycle == Lifecycle::Replaying || lifecycle == Lifecycle::Pending ||
                             (lifecycle != Lifecycle::Finishable &&
                              continuation.execution_frontier <= request.recovery->frontier &&
                              ledger_tokens <= request.recovery->ledger_tokens));
    if (state.kv) {
        text_kv_histories->prepare_commit(state.kv->text, continuation.text_kv_valid,
                                          retain ? request.recovery->coverage.main_frontier : 0,
                                          transaction);
        if (state.kv->backend)
            backend_kv_histories->prepare_commit(
                *state.kv->backend, continuation.backend_kv_valid,
                retain ? request.recovery->coverage.backend_frontier : 0, transaction);
    }
    return retain;
}

void SessionImpl::adopt_unit(std::uint32_t lane, const ContinuationMetadata& continuation,
                             bool state_written, bool retain_recovery) noexcept {
    auto& state   = sequences[lane];
    auto& request = requests[lane];
    if (state_written) complete_state_write(state);
    state.continuation.metadata = continuation;
    if (!retain_recovery) request.recovery.reset();
    request.permit.reset();
}

} // namespace ninfer::models::qwen3_5::detail
