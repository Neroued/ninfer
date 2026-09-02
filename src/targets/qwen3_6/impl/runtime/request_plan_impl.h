#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/runtime/rebuild_work.h"

#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
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

std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                  std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

std::uint64_t segmented_prefill_chunks(std::uint32_t begin, std::uint32_t end,
                                       std::uint32_t prefill_chunk,
                                       std::span<const CaptureGroup> captures,
                                       std::span<const std::uint32_t> rewrite_frontiers) noexcept {
    if (begin >= end || prefill_chunk == 0) { return 0; }
    std::uint64_t chunks        = 0;
    std::uint32_t segment_begin = begin;
    std::size_t capture_index   = 0;
    std::size_t rewrite_index   = 0;
    while (capture_index < captures.size() || rewrite_index < rewrite_frontiers.size()) {
        const std::uint32_t capture_frontier = capture_index < captures.size()
                                                   ? captures[capture_index].frontier
                                                   : std::numeric_limits<std::uint32_t>::max();
        const std::uint32_t rewrite_frontier = rewrite_index < rewrite_frontiers.size()
                                                   ? rewrite_frontiers[rewrite_index]
                                                   : std::numeric_limits<std::uint32_t>::max();
        const std::uint32_t frontier         = std::min(capture_frontier, rewrite_frontier);
        if (capture_frontier == frontier) { ++capture_index; }
        if (rewrite_frontier == frontier) { ++rewrite_index; }
        if (frontier <= segment_begin || frontier >= end) { continue; }
        const std::uint64_t segment = frontier - segment_begin;
        chunks += 1U + (segment - 1U) / prefill_chunk;
        segment_begin = frontier;
    }
    const std::uint64_t suffix = end - segment_begin;
    return chunks + 1U + (suffix - 1U) / prefill_chunk;
}

runtime::PrefillWork scheduled_prefill_work(std::uint32_t begin, std::uint32_t end,
                                            std::uint64_t vision_items,
                                            std::uint64_t vision_patches,
                                            std::uint32_t prefill_chunk,
                                            std::span<const CaptureGroup> captures,
                                            std::span<const std::uint32_t> rewrite_frontiers) {
    if (end < begin) { throw std::logic_error("prefill work interval is reversed"); }
    runtime::PrefillWork work =
        runtime::make_prefill_work(begin, end - begin, vision_items, vision_patches, prefill_chunk);
    work.chunks = segmented_prefill_chunks(begin, end, prefill_chunk, captures, rewrite_frontiers);
    return work;
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits,
                                     std::span<const CaptureGroup> captures,
                                     std::span<const std::uint32_t> rewrite_frontiers) noexcept {
    std::uint64_t prefill_units = 0;
    std::uint32_t segment_begin = reuse_base;
    std::size_t capture_index   = 0;
    std::size_t rewrite_index   = 0;
    while (capture_index < captures.size() || rewrite_index < rewrite_frontiers.size()) {
        const std::uint32_t capture_frontier = capture_index < captures.size()
                                                   ? captures[capture_index].frontier
                                                   : std::numeric_limits<std::uint32_t>::max();
        const std::uint32_t rewrite_frontier = rewrite_index < rewrite_frontiers.size()
                                                   ? rewrite_frontiers[rewrite_index]
                                                   : std::numeric_limits<std::uint32_t>::max();
        const std::uint32_t frontier         = std::min(capture_frontier, rewrite_frontier);
        if (capture_frontier == frontier) { ++capture_index; }
        if (rewrite_frontier == frontier) { ++rewrite_index; }
        if (frontier <= segment_begin || frontier >= summary.prompt_tokens) { continue; }
        const std::uint64_t segment = frontier - segment_begin;
        prefill_units += 1ULL + (segment - 1ULL) / prefill_chunk;
        segment_begin = frontier;
    }
    const std::uint64_t suffix = summary.prompt_tokens - segment_begin;
    prefill_units += suffix == 0 ? 1ULL : 1ULL + (suffix - 1ULL) / prefill_chunk;
    // A shared promotion at the selected reuse base is offered before the ordinary zero/suffix
    // prefill step. It executes no model work, but it is still one scheduler service unit.
    prefill_units += static_cast<std::uint64_t>(
        std::count_if(captures.begin(), captures.end(), [reuse_base](const CaptureGroup& capture) {
            return capture.frontier == reuse_base;
        }));
    prefill_units += prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

std::uint32_t capture_identity_tag(SpeculativeBackend backend, ProposalHead proposal,
                                   KvCacheStorage storage) noexcept {
    return static_cast<std::uint32_t>(backend) | (static_cast<std::uint32_t>(proposal) << 8U) |
           (static_cast<std::uint32_t>(storage) << 16U);
}

runtime::PrefillWork rebuild_work_at_frontier(const PreparedPromptData& prompt,
                                              std::uint32_t frontier, std::uint32_t prefill_chunk,
                                              std::span<const CaptureGroup> captures,
                                              std::span<const std::uint32_t> rewrite_frontiers) {
    std::uint64_t vision_items   = 0;
    std::uint64_t vision_patches = 0;
    for (const qwen3_6::VisionItem& item : prompt.vision_items) {
        bool fully_consumed = !item.token_spans.empty();
        for (const qwen3_6::TokenSpan& span : item.token_spans) {
            if (span.begin > std::numeric_limits<std::size_t>::max() - span.count ||
                span.begin + span.count > frontier) {
                fully_consumed = false;
                break;
            }
        }
        if (!fully_consumed) { continue; }
        if (vision_items == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Vision rebuild item count exceeds uint64");
        }
        ++vision_items;
        if (item.patch_count > std::numeric_limits<std::uint64_t>::max() - vision_patches) {
            throw std::overflow_error("Vision rebuild work exceeds uint64");
        }
        vision_patches += static_cast<std::uint64_t>(item.patch_count);
    }
    return scheduled_prefill_work(0, frontier, vision_items, vision_patches, prefill_chunk,
                                  captures, rewrite_frontiers);
}

detail::PhysicalDeviceResources
convertible_source_resources(detail::PhysicalDeviceResources active,
                             detail::PhysicalDeviceResources source) noexcept {
    return detail::PhysicalDeviceResources{
        .active_lanes     = 0,
        .state_slots      = std::min(active.state_slots, source.state_slots),
        .main_kv_pages    = std::min(active.main_kv_pages, source.main_kv_pages),
        .backend_kv_pages = std::min(active.backend_kv_pages, source.backend_kv_pages),
    };
}

detail::PhysicalDeviceResources
additional_resources(detail::PhysicalDeviceResources active,
                     detail::PhysicalDeviceResources converted) noexcept {
    return detail::PhysicalDeviceResources{
        .active_lanes     = active.active_lanes,
        .state_slots      = active.state_slots - converted.state_slots,
        .main_kv_pages    = active.main_kv_pages - converted.main_kv_pages,
        .backend_kv_pages = active.backend_kv_pages - converted.backend_kv_pages,
    };
}

detail::PhysicalResources positive_difference(detail::PhysicalResources value,
                                              detail::PhysicalResources removed) noexcept {
    const auto subtract = [](auto left, auto right) { return left > right ? left - right : 0; };
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes = static_cast<std::uint32_t>(
                    subtract(value.device.active_lanes, removed.device.active_lanes)),
                .state_slots = static_cast<std::uint32_t>(
                    subtract(value.device.state_slots, removed.device.state_slots)),
                .main_kv_pages = static_cast<std::uint32_t>(
                    subtract(value.device.main_kv_pages, removed.device.main_kv_pages)),
                .backend_kv_pages = static_cast<std::uint32_t>(
                    subtract(value.device.backend_kv_pages, removed.device.backend_kv_pages)),
            },
        .host =
            {
                .state_slots = static_cast<std::uint32_t>(
                    subtract(value.host.state_slots, removed.host.state_slots)),
                .kv_bytes =
                    static_cast<std::size_t>(subtract(value.host.kv_bytes, removed.host.kv_bytes)),
            },
    };
}

} // namespace

RequestBasePlan ProgramImplCore::plan_request(const PreparedPromptData& prompt,
                                              const runtime::ResolvedExecutionOptions& options) {
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
    base->allow_prefix_reuse             = options.allow_prefix_reuse;
    base->summary.publish_continuation =
        options.allow_prefix_reuse && prompt.identity.reusable && context_cache.enabled;
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
    detail::PhysicalDeviceResources root_active{
        .active_lanes     = 1,
        .state_slots      = 1U,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        if (!workspace_plan.vision) {
            throw std::logic_error("Vision prompt has no startup workspace plan");
        }
        auto vision =
            std::make_shared<qwen3_6::VisionControlPlan>(qwen3_6::plan_vision_control(prompt));
        std::uint32_t previous_end = 0;
        for (std::size_t index = 0; index < vision->items.size(); ++index) {
            const qwen3_6::VisionItemControlPlan& item = vision->items[index];
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && item.token_begin != 0
                    ? item.token_begin - 1
                    : item.token_begin;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (item.merged_count > workspace_plan.vision->max_merged_tokens ||
                schedule::VisionContext::workspace_bytes(prompt.vision_items[index].patch_count,
                                                         item.merged_count) >
                    workspace_plan.vision->encode_peak_bytes) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = item.token_end;
        }
        base->vision_control_plan = std::move(vision);
    }

    if (prompt.identity.rewrite_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.rewrite_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite checkpoint frontier must lie at or inside the prompt frontier");
        }
        base->rewrite_checkpoint = candidate;
    }
    std::uint32_t previous_rewrite_frontier = 0;
    for (const std::uint32_t frontier : prompt.identity.rewrite_execution_frontiers) {
        if (frontier == 0 || frontier > base->summary.prompt_tokens ||
            frontier <= previous_rewrite_frontier) {
            throw std::invalid_argument(
                "rewrite execution frontiers must be ordered unique prompt positions");
        }
        previous_rewrite_frontier = frontier;
    }
    if (base->summary.publish_continuation) {
        base->prefix_digests.assign(prompt);
        base->prefix_identity_tag =
            capture_identity_tag(speculative_backend, proposal_head, kv_storage);
    }
    if (options.allow_prefix_reuse && prompt.identity.reusable && context_cache.enabled) {
        const auto add_capture = [&](std::uint32_t frontier, std::uint32_t input_order,
                                     std::optional<RewriteCheckpointKind> rewrite, bool shared,
                                     bool long_anchor, SharedCandidateEvidence evidence) {
            if (frontier == 0 || frontier > base->summary.prompt_tokens) {
                throw std::invalid_argument("capture opportunity frontier is invalid");
            }
            auto& groups = shared ? base->shared_candidates : base->capture_groups;
            auto existing =
                std::find_if(groups.begin(), groups.end(),
                             [&](const CaptureGroup& group) { return group.frontier == frontier; });
            if (existing == groups.end()) {
                CaptureGroup group;
                group.frontier    = frontier;
                group.input_order = input_order;
                groups.push_back(std::move(group));
                existing = std::prev(groups.end());
            }
            existing->input_order = std::min(existing->input_order, input_order);
            if (rewrite) { existing->rewrite = rewrite; }
            existing->shared      = existing->shared || shared;
            existing->long_anchor = existing->long_anchor || long_anchor;
            if (shared) { existing->shared_evidence |= evidence; }
        };
        if (base->rewrite_checkpoint) {
            add_capture(base->rewrite_checkpoint->frontier, 0, base->rewrite_checkpoint->kind,
                        false, false, SharedCandidateEvidence::None);
        }
        for (const qwen3_6::PreparedCacheOpportunity& opportunity :
             base->context_cache.opportunities) {
            add_capture(opportunity.frontier, opportunity.input_order, std::nullopt,
                        opportunity.kind == PromptCacheMarkerKind::SharedStablePrefix,
                        opportunity.kind == PromptCacheMarkerKind::PrivateLongAnchor,
                        opportunity.evidence);
        }
        std::sort(base->capture_groups.begin(), base->capture_groups.end(),
                  [](const CaptureGroup& left, const CaptureGroup& right) {
                      return std::tie(left.frontier, left.input_order) <
                             std::tie(right.frontier, right.input_order);
                  });
        std::sort(base->shared_candidates.begin(), base->shared_candidates.end(),
                  [](const CaptureGroup& left, const CaptureGroup& right) {
                      return std::tie(left.frontier, left.input_order) <
                             std::tie(right.frontier, right.input_order);
                  });
        std::shared_ptr<const PreparedCaptureBacking> capture_backing;
        if (!base->capture_groups.empty() || !base->shared_candidates.empty()) {
            auto backing = std::make_shared<PreparedCaptureBacking>();
            const std::uint32_t capture_frontier =
                base->capture_groups.empty() ? 0U : base->capture_groups.back().frontier;
            const std::uint32_t shared_frontier =
                base->shared_candidates.empty() ? 0U : base->shared_candidates.back().frontier;
            const std::uint32_t backing_frontier = std::max(capture_frontier, shared_frontier);
            backing->ledger.assign(prompt.token_ids.begin(),
                                   prompt.token_ids.begin() +
                                       static_cast<std::ptrdiff_t>(backing_frontier));
            backing->prefix_identity.assign(prompt);
            backing->prefix_identity.truncate(backing_frontier);
            capture_backing = std::move(backing);
        }
        for (CaptureGroup& group : base->capture_groups) {
            auto identity          = std::make_shared<PreparedCaptureIdentity>();
            identity->backing      = capture_backing;
            identity->rebuild_work = rebuild_work_at_frontier(
                prompt, group.frontier, prefill_chunk, base->capture_groups,
                prompt.identity.rewrite_execution_frontiers);
            identity->shortlist_key = qwen3_6::PrefixShortlistKey{
                .digests      = base->prefix_digests.at(group.frontier),
                .frontier     = group.frontier,
                .identity_tag = base->prefix_identity_tag,
            };
            group.identity = std::move(identity);
        }
        for (CaptureGroup& group : base->shared_candidates) {
            auto identity          = std::make_shared<PreparedCaptureIdentity>();
            identity->backing      = capture_backing;
            identity->rebuild_work = rebuild_work_at_frontier(
                prompt, group.frontier, prefill_chunk, base->capture_groups,
                prompt.identity.rewrite_execution_frontiers);
            identity->shortlist_key = qwen3_6::PrefixShortlistKey{
                .digests      = base->prefix_digests.at(group.frontier),
                .frontier     = group.frontier,
                .identity_tag = base->prefix_identity_tag,
            };
            group.identity = std::move(identity);
        }
    }
    root_active.state_slots = 1U;
    const detail::PhysicalResources root_vector{.device = root_active};
    base->root_demand = detail::PhysicalDemand{
        .active_entitlement       = root_vector,
        .reservation_added        = root_vector,
        .physical_peak_additional = root_vector,
        .final_added              = root_vector,
    };
    const std::size_t cold_prefill_splits =
        base->vision_control_plan ? base->vision_control_plan->items.size() : 0ULL;
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits,
                               base->capture_groups, prompt.identity.rewrite_execution_frontiers);
    base->root_rebuild_work =
        rebuild_work_at_frontier(prompt, base->summary.prompt_tokens, prefill_chunk,
                                 base->capture_groups, prompt.identity.rewrite_execution_frontiers);
    for (const CaptureGroup& group : base->capture_groups) {
        runtime_support::include_rebuild_boundary(base->root_rebuild_tail_begin, group.frontier,
                                                  base->summary.prompt_tokens);
    }
    for (const std::uint32_t frontier : prompt.identity.rewrite_execution_frontiers) {
        runtime_support::include_rebuild_boundary(base->root_rebuild_tail_begin, frontier,
                                                  base->summary.prompt_tokens);
    }
    return RequestBasePlan(std::move(base));
}

std::optional<AdmissionCandidate> ProgramImplCore::inspect_lane(
    std::uint32_t lane, const PreparedPromptData& prompt, const RequestBasePlan& base_plan,
    const SequenceState* source, const SharedPrefixState* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    if (request.lifecycle != Lifecycle::Empty) {
        throw std::logic_error("admission inspection requires a free active lane");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<AdmissionCandidateImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;
    plan->root_rebuild_work           = base.root_rebuild_work;
    plan->root_rebuild_tail_begin     = base.root_rebuild_tail_begin;

    if ((source != nullptr && shared_source != nullptr) ||
        ((source == nullptr && shared_source == nullptr) != !checkpoint.has_value())) {
        throw std::invalid_argument("inspection source and checkpoint must be paired");
    }
    if (shared_source != nullptr) {
        const runtime::CheckpointRef selected = *checkpoint;
        plan->selected_checkpoint             = selected;
        if (selected.kind != runtime::CheckpointKind::SharedStablePrefix || selected.ordinal != 0 ||
            selected.frontier == 0 || selected.frontier != shared_source->frontier ||
            !shared_source->identity || !shared_source->kv ||
            !state_store->valid(shared_source->state)) {
            throw std::logic_error("catalog shared-prefix summary disagrees with Program state");
        }
        if (!base.allow_prefix_reuse || !prompt.identity.reusable) { return std::nullopt; }
        const auto* shared_identity = shared_source->identity->prefix_identity();
        if (shared_identity == nullptr ||
            !qwen3_6::detail::prefix_matches(prompt, shared_source->identity->ledger(),
                                             *shared_identity, selected.frontier)) {
            return std::nullopt;
        }
        plan->reuse       = ReusePath::SharedStablePrefix;
        plan->reuse_base  = selected.frontier;
        plan->source_mode = runtime::PrivateSourceMode::Retain;
    } else if (source != nullptr) {
        const runtime::CheckpointRef selected = *checkpoint;
        plan->selected_checkpoint             = selected;
        plan->source_mode                     = must_retain_private_source
                                                    ? runtime::PrivateSourceMode::Retain
                                                    : runtime::PrivateSourceMode::ConsumeToActive;
        if (!base.allow_prefix_reuse || !prompt.identity.reusable) { return std::nullopt; }
        if (selected.kind == runtime::CheckpointKind::SessionEndpoint) {
            if (selected.ordinal != 0) {
                throw std::logic_error("private endpoint checkpoint ordinal is invalid");
            }
            if (selected.frontier == 0 || selected.frontier != source->execution_frontier) {
                throw std::logic_error("catalog endpoint summary disagrees with Program state");
            }
            if (!qwen3_6::detail::prefix_matches(prompt, source->ledger, source->prefix_identity,
                                                 selected.frontier)) {
                return std::nullopt;
            }
            plan->reuse      = ReusePath::PrivateEndpoint;
            plan->reuse_base = selected.frontier;
        } else if (selected.kind == runtime::CheckpointKind::LongAnchor) {
            const auto anchor =
                std::find_if(source->long_anchors.begin(), source->long_anchors.end(),
                             [&](const LongAnchorCheckpoint& candidate) {
                                 return candidate.frontier == selected.frontier &&
                                        candidate.ordinal == selected.ordinal;
                             });
            if (anchor == source->long_anchors.end() || selected.frontier == 0) {
                throw std::logic_error("catalog long-anchor summary disagrees with Program state");
            }
            if (!qwen3_6::detail::prefix_matches(prompt, source->ledger, source->prefix_identity,
                                                 selected.frontier)) {
                return std::nullopt;
            }
            plan->reuse       = ReusePath::PrivateLongAnchor;
            plan->reuse_base  = selected.frontier;
            plan->source_mode = runtime::PrivateSourceMode::Retain;
        } else {
            if (selected.ordinal != 0) {
                throw std::logic_error("private rewrite checkpoint ordinal is invalid");
            }
            if (!source->rewrite_checkpoint.valid ||
                selected.kind != checkpoint_kind(source->rewrite_checkpoint.kind) ||
                selected.frontier == 0 ||
                selected.frontier != source->rewrite_checkpoint.frontier) {
                throw std::logic_error("catalog rewrite summary disagrees with Program state");
            }
            if (!qwen3_6::detail::prefix_matches(prompt, source->ledger, source->prefix_identity,
                                                 selected.frontier)) {
                return std::nullopt;
            }
            plan->reuse      = restore_path(source->rewrite_checkpoint.kind);
            plan->reuse_base = selected.frontier;
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const bool append_ready =
            plan->reuse == ReusePath::PrivateEndpoint && source != nullptr &&
            source->tail_hidden_valid && decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || source->mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready =
            (is_rewrite_checkpoint_restore(plan->reuse) ||
             plan->reuse == ReusePath::PrivateLongAnchor ||
             plan->reuse == ReusePath::SharedStablePrefix) &&
            decoder->mtp_cache() != nullptr && plan->reuse_base != 0 &&
            ((source != nullptr && source->mtp_kv_valid >= plan->reuse_base - 1) ||
             (shared_source != nullptr && shared_source->backend_frontier >= plan->reuse_base - 1));
        if (plan->reuse != ReusePath::Root && !append_ready && !checkpoint_ready) {
            throw std::logic_error("published MTP checkpoint is not materializable");
        }
    }

    if ((is_rewrite_checkpoint_restore(plan->reuse) ||
         plan->reuse == ReusePath::PrivateLongAnchor ||
         plan->reuse == ReusePath::SharedStablePrefix) &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash ||
         (source != nullptr && (!source->kv || !source->kv->backend ||
                                source->dflash_context_frontier < plan->reuse_base)) ||
         (shared_source != nullptr && (!shared_source->kv || !shared_source->kv->backend ||
                                       shared_source->backend_frontier < plan->reuse_base)))) {
        throw std::logic_error("published DFlash checkpoint is not materializable");
    }

    const std::optional<RewriteCheckpointSpec>& desired = base.rewrite_checkpoint;
    const bool can_retain_rewrite =
        desired && source != nullptr &&
        plan->source_mode == runtime::PrivateSourceMode::ConsumeToActive &&
        can_retain_rewrite_checkpoint(prompt, *desired, *source, plan->reuse, plan->reuse_base);
    if (!base.summary.publish_continuation || !desired) {
        plan->rewrite_disposition = RewriteCheckpointDisposition::DropOptional;
    } else if (can_retain_rewrite) {
        plan->rewrite_disposition = RewriteCheckpointDisposition::RetainExisting;
    } else if (desired->frontier > plan->reuse_base) {
        plan->rewrite_disposition = RewriteCheckpointDisposition::ReplaceAtCommittedFrontier;
    } else {
        plan->rewrite_disposition = RewriteCheckpointDisposition::DropOptional;
    }

    // A consumed endpoint keeps the sparse anchors which already belong to the lineage and any
    // typed rewrite that remains compatible with the requested fallback boundary. These are part
    // of the active lineage entitlement, not additional catalog ownership. If an optional
    // checkpoint aliases the endpoint image, that image must remain immutable and the whole source
    // is preserved through the normal Fork path.
    if (source != nullptr && plan->reuse == ReusePath::PrivateEndpoint &&
        plan->source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
        const StateImageHandle endpoint =
            selected_state(*source, plan->reuse, plan->selected_checkpoint);
        std::vector<StateImageHandle> optional_states;
        optional_states.reserve(1U + source->long_anchors.size());
        if (plan->rewrite_disposition == RewriteCheckpointDisposition::RetainExisting &&
            source->rewrite_state) {
            optional_states.push_back(*source->rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : source->long_anchors) {
            optional_states.push_back(anchor.state);
        }
        if (std::find(optional_states.begin(), optional_states.end(), endpoint) !=
            optional_states.end()) {
            plan->source_mode = runtime::PrivateSourceMode::Retain;
        } else {
            std::vector<StateImageHandle> unique;
            unique.reserve(optional_states.size());
            for (const StateImageHandle state : optional_states) {
                if (!state_store->valid(state) ||
                    std::find(unique.begin(), unique.end(), state) != unique.end()) {
                    continue;
                }
                unique.push_back(state);
                const StateReplicaResidency residency = state_store->residency(state);
                if (residency == StateReplicaResidency::DeviceOnly ||
                    residency == StateReplicaResidency::Both) {
                    ++plan->active_optional_resources.device.state_slots;
                }
                if (residency == StateReplicaResidency::HostOnly ||
                    residency == StateReplicaResidency::Both) {
                    ++plan->active_optional_resources.host.state_slots;
                }
            }
        }
    }
    if (source != nullptr && plan->source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
        plan->state_fork_required =
            selected_state_requires_fork(*source, plan->reuse, plan->rewrite_disposition,
                                         plan->selected_checkpoint, plan->reuse_base);
    }
    if (source != nullptr && is_rewrite_checkpoint_restore(plan->reuse) &&
        plan->source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
        std::vector<StateImageHandle> optional_states;
        optional_states.reserve(1U + source->long_anchors.size());
        if (plan->rewrite_disposition == RewriteCheckpointDisposition::RetainExisting &&
            source->rewrite_state) {
            optional_states.push_back(*source->rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : source->long_anchors) {
            if (anchor.frontier > plan->reuse_base) { continue; }
            optional_states.push_back(anchor.state);
        }
        std::vector<StateImageHandle> unique;
        unique.reserve(optional_states.size());
        for (const StateImageHandle state : optional_states) {
            if (!state_store->valid(state) ||
                std::find(unique.begin(), unique.end(), state) != unique.end()) {
                continue;
            }
            unique.push_back(state);
            if (!state_exclusive_to_sequence(*source, state)) { continue; }
            const StateReplicaResidency residency = state_store->residency(state);
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++plan->active_optional_resources.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++plan->active_optional_resources.host.state_slots;
            }
        }
    }

    plan->capture_groups.reserve(base.capture_groups.size());
    for (CaptureGroup group : base.capture_groups) {
        if (group.frontier <= plan->reuse_base) { continue; }
        if (group.rewrite &&
            (plan->rewrite_disposition !=
                 RewriteCheckpointDisposition::ReplaceAtCommittedFrontier ||
             !desired || group.frontier != desired->frontier || *group.rewrite != desired->kind)) {
            group.rewrite.reset();
        }
        if (!group.rewrite && !group.shared && !group.long_anchor) { continue; }
        plan->capture_groups.push_back(std::move(group));
    }
    plan->shared_candidates.reserve(base.shared_candidates.size());
    for (CaptureGroup group : base.shared_candidates) {
        if (group.frontier >= plan->reuse_base) {
            plan->shared_candidates.push_back(std::move(group));
        }
    }
    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    plan->summary.prefix_reuse_path      = plan->reuse;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::Root) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::PrivateEndpoint) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (is_rewrite_checkpoint_restore(plan->reuse) ||
                   plan->reuse == ReusePath::PrivateLongAnchor ||
                   plan->reuse == ReusePath::SharedStablePrefix) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        }
    }

    if (base.vision_control_plan) {
        VisionPrefillPlan vision;
        vision.control_plan = base.vision_control_plan;
        vision.uses.reserve(base.vision_control_plan->items.size());
        std::size_t max_merged = 0;
        for (std::size_t index = 0; index < base.vision_control_plan->items.size(); ++index) {
            const qwen3_6::VisionItemControlPlan& item = base.vision_control_plan->items[index];
            if (item.token_end <= plan->reuse_base) { continue; }
            const std::uint32_t begin = plan->prepare_mtp && item.token_begin != 0
                                            ? item.token_begin - 1
                                            : item.token_begin;
            vision.uses.push_back(VisionUseSpan{
                .begin               = begin,
                .end                 = item.token_end,
                .prepared_item_index = static_cast<std::uint32_t>(index),
            });
            max_merged = std::max(max_merged, item.merged_count);
        }
        if (!vision.uses.empty()) {
            vision.max_merged_count = max_merged;
            plan->vision            = std::move(vision);
        }
    }

    const std::size_t prefill_splits = plan->vision ? plan->vision->uses.size() : 0ULL;
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, plan->reuse_base, prefill_chunk, prefill_splits,
                               plan->capture_groups, prompt.identity.rewrite_execution_frontiers);
    std::uint64_t remaining_vision_items   = 0;
    std::uint64_t remaining_vision_patches = 0;
    if (plan->vision) {
        std::vector<bool> counted(prompt.vision_items.size(), false);
        for (const VisionUseSpan& use : plan->vision->uses) {
            if (use.prepared_item_index >= counted.size()) {
                throw std::logic_error("Vision cost input references a missing item");
            }
            if (counted[use.prepared_item_index]) { continue; }
            counted[use.prepared_item_index] = true;
            if (remaining_vision_items == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("Vision prefill item count exceeds uint64");
            }
            ++remaining_vision_items;
            const std::size_t patches = prompt.vision_items[use.prepared_item_index].patch_count;
            if (patches > std::numeric_limits<std::uint64_t>::max() - remaining_vision_patches) {
                throw std::overflow_error("Vision prefill cost input exceeds uint64");
            }
            remaining_vision_patches += static_cast<std::uint64_t>(patches);
        }
    }
    plan->remaining_prefill_work =
        scheduled_prefill_work(plan->reuse_base, plan->summary.prompt_tokens,
                               remaining_vision_items, remaining_vision_patches, prefill_chunk,
                               plan->capture_groups, prompt.identity.rewrite_execution_frontiers);
    plan->transfer_requirements.reserve(4);
    const auto add_state_transfer = [&](runtime::ContextTransferDirection direction,
                                        bool dflash_local_only = false) {
        const TransferWork work = dflash_local_only
                                      ? dflash_local_transfer_work(state_images->host_layout())
                                      : state_image_transfer_work(state_images->host_layout());
        plan->transfer_requirements.push_back(runtime::ContextTransferRequirement{
            .resource   = runtime::ContextResourceClass::State,
            .direction  = direction,
            .units      = 1,
            .page_count = 0,
            .work       = work,
        });
    };
    const auto add_kv_transfer = [&](runtime::ContextResourceClass resource,
                                     runtime::ContextTransferDirection direction,
                                     const LogicalKVPageStore& pages, std::uint32_t page_count,
                                     std::uint32_t contiguous_runs = 1) {
        if (page_count == 0) { return; }
        const HostKVPageLayout layout = plan_host_kv_page_layout(pages.physical_pool().geometry());
        const TransferWork work =
            direction == runtime::ContextTransferDirection::DeviceToDevice
                ? plan_device_kv_copy_work(layout, page_count)
                : plan_host_kv_transfer_work(layout, page_count, contiguous_runs);
        plan->transfer_requirements.push_back(runtime::ContextTransferRequirement{
            .resource   = resource,
            .direction  = direction,
            .units      = work.payload_bytes,
            .page_count = page_count,
            .work       = work,
        });
    };
    const auto missing_kv_restore = [&](const KVAddressSpaceStore& addresses,
                                        const LogicalKVPageStore& pages,
                                        KVAddressSpaceHandle address, std::uint32_t required) {
        std::pair<std::uint32_t, std::uint32_t> out;
        if (required > addresses.mapped_pages(address)) {
            throw std::logic_error("checkpoint KV requirement exceeds address membership");
        }
        std::optional<HostKVPageReplica> previous;
        for (std::uint32_t page = 0; page < required; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.device_resident(logical)) { continue; }
            if (!pages.host_resident(logical)) {
                throw std::logic_error("checkpoint KV page has no restorable replica");
            }
            const HostKVPageReplica replica = pages.host_replica(logical);
            if (!previous || previous->extent != replica.extent ||
                previous->page_offset + 1U != replica.page_offset) {
                ++out.second;
            }
            previous = replica;
            ++out.first;
        }
        return out;
    };
    const detail::PhysicalResources source_resources =
        source != nullptr ? resident_resources(*source) : detail::PhysicalResources{};
    plan->source_resources = source_resources;
    if (source != nullptr || shared_source != nullptr) {
        const StateImageHandle selected =
            source != nullptr ? selected_state(*source, plan->reuse, plan->selected_checkpoint)
                              : shared_source->state;
        const StateReplicaResidency selected_state_residency = state_store->residency(selected);
        plan->needs_transfer = selected_state_residency == StateReplicaResidency::HostOnly;
        if (plan->needs_transfer) {
            add_state_transfer(runtime::ContextTransferDirection::HostToDevice);
        }
        const bool splits_private_both =
            source != nullptr && selected_state_residency == StateReplicaResidency::Both;
        const bool forks_device_state =
            (plan->source_mode == runtime::PrivateSourceMode::Retain && !splits_private_both) ||
            (plan->source_mode == runtime::PrivateSourceMode::ConsumeToActive &&
             plan->state_fork_required);
        if (speculative_backend == SpeculativeBackend::DFlash && forks_device_state &&
            selected_state_residency != StateReplicaResidency::HostOnly) {
            // DFlash has lane-local recurrent state which is copied eagerly when a retained
            // checkpoint forks. The copy completes on the compute stream in the publication
            // boundary, so it contributes candidate cost without turning the plan into an
            // asynchronous materialization.
            add_state_transfer(runtime::ContextTransferDirection::DeviceToDevice, true);
        }
        const SequenceKVBundle* source_kv =
            source != nullptr ? (source->kv ? &*source->kv : nullptr)
                              : (shared_source->kv ? &*shared_source->kv : nullptr);
        if (source_kv == nullptr) { throw std::logic_error("checkpoint has no KV address space"); }
        const std::uint32_t backend_frontier =
            backend_frontier_at(speculative_backend, plan->reuse_base);
        if (plan->source_mode == runtime::PrivateSourceMode::Retain) {
            plan->text_prefix_fork_required    = true;
            plan->backend_prefix_fork_required = source_kv->backend && backend_frontier != 0;
        } else if (source != nullptr) {
            plan->text_prefix_fork_required =
                partial_tail_cow_required(*text_kv_addresses, source_kv->text, plan->reuse_base);
            plan->backend_prefix_fork_required =
                source_kv->backend && backend_frontier != 0 &&
                partial_tail_cow_required(*backend_kv_addresses, *source_kv->backend,
                                          backend_frontier);
        }
        const std::uint32_t main_required = pages_for_tokens(plan->reuse_base);
        const std::uint32_t main_device =
            device_kv_prefix_pages(*text_kv_addresses, source_kv->text, plan->reuse_base);
        if (main_device > main_required) {
            throw std::logic_error("Text KV resident prefix exceeds checkpoint requirement");
        }
        const auto [main_missing, main_contiguous_runs] =
            missing_kv_restore(*text_kv_addresses, *text_kv_pages, source_kv->text, main_required);
        if (main_missing != main_required - main_device) {
            throw std::logic_error("Text KV restore inventory is inconsistent");
        }
        plan->needs_transfer = plan->needs_transfer || main_missing != 0;
        if (main_missing != 0) {
            add_kv_transfer(runtime::ContextResourceClass::MainKV,
                            runtime::ContextTransferDirection::HostToDevice, *text_kv_pages,
                            main_missing, main_contiguous_runs);
        }
        if (source_kv->backend && backend_frontier != 0) {
            const std::uint32_t backend_required = pages_for_tokens(backend_frontier);
            const std::uint32_t backend_device   = device_kv_prefix_pages(
                *backend_kv_addresses, *source_kv->backend, backend_frontier);
            if (backend_device > backend_required) {
                throw std::logic_error("Backend KV resident prefix exceeds checkpoint requirement");
            }
            const auto [backend_missing, backend_contiguous_runs] = missing_kv_restore(
                *backend_kv_addresses, *backend_kv_pages, *source_kv->backend, backend_required);
            if (backend_missing != backend_required - backend_device) {
                throw std::logic_error("Backend KV restore inventory is inconsistent");
            }
            plan->needs_transfer = plan->needs_transfer || backend_missing != 0;
            if (backend_missing != 0) {
                add_kv_transfer(runtime::ContextResourceClass::BackendKV,
                                runtime::ContextTransferDirection::HostToDevice, *backend_kv_pages,
                                backend_missing, backend_contiguous_runs);
            }
        }
        if (plan->text_prefix_fork_required) {
            const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
            if (plan->reuse_base % page_size != 0) {
                add_kv_transfer(runtime::ContextResourceClass::MainKV,
                                runtime::ContextTransferDirection::DeviceToDevice, *text_kv_pages,
                                1);
                plan->needs_transfer = true;
            }
            if (plan->backend_prefix_fork_required && backend_frontier % page_size != 0) {
                add_kv_transfer(runtime::ContextResourceClass::BackendKV,
                                runtime::ContextTransferDirection::DeviceToDevice,
                                *backend_kv_pages, 1);
                plan->needs_transfer = true;
            }
        }
    }
    detail::PhysicalDeviceResources active = base.root_demand.active_entitlement.device;
    if (plan->active_optional_resources.device.state_slots >
        std::numeric_limits<std::uint32_t>::max() - active.state_slots) {
        throw std::overflow_error("active optional StateImage entitlement overflow");
    }
    active.state_slots += plan->active_optional_resources.device.state_slots;
    if ((source != nullptr || shared_source != nullptr) &&
        plan->source_mode == runtime::PrivateSourceMode::Retain) {
        const SequenceKVBundle* source_kv =
            source != nullptr ? (source->kv ? &*source->kv : nullptr)
                              : (shared_source->kv ? &*shared_source->kv : nullptr);
        if (source_kv == nullptr) {
            throw std::logic_error("retained source has no KV address space");
        }
        const std::uint32_t main_full_pages =
            plan->reuse_base / static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t backend_frontier =
            backend_frontier_at(speculative_backend, plan->reuse_base);
        const std::uint32_t backend_full_pages =
            backend_frontier / static_cast<std::uint32_t>(kPagedKVPageSize);
        if (main_full_pages > active.main_kv_pages ||
            backend_full_pages > active.backend_kv_pages) {
            throw std::logic_error("retained prefix exceeds its active KV entitlement");
        }
        detail::PhysicalResources active_resources{
            .device =
                {
                    .active_lanes     = active.active_lanes,
                    .state_slots      = active.state_slots,
                    .main_kv_pages    = active.main_kv_pages - main_full_pages,
                    .backend_kv_pages = active.backend_kv_pages - backend_full_pages,
                },
        };
        detail::PhysicalResources source_replica_additions;
        const std::uint32_t main_required = pages_for_tokens(plan->reuse_base);
        const std::uint32_t main_device =
            device_kv_prefix_pages(*text_kv_addresses, source_kv->text, plan->reuse_base);
        if (main_device > main_required) {
            throw std::logic_error("retained Text KV replica count exceeds requirement");
        }
        source_replica_additions.device.main_kv_pages = main_required - main_device;
        if (source_kv->backend) {
            const std::uint32_t backend_required =
                backend_frontier == 0 ? 0U : pages_for_tokens(backend_frontier);
            const std::uint32_t backend_device = device_kv_prefix_pages(
                *backend_kv_addresses, *source_kv->backend, backend_frontier);
            if (backend_device > backend_required) {
                throw std::logic_error("retained Backend KV replica count exceeds requirement");
            }
            source_replica_additions.device.backend_kv_pages = backend_required - backend_device;
        }
        detail::PhysicalResources retained_tail_added;
        detail::PhysicalResources retained_tail_removed;
        detail::PhysicalResources retained_tail_preparation_peak;
        const auto plan_retained_tail_release =
            [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                KVAddressSpaceHandle address, std::uint32_t frontier, std::uint32_t active_pages,
                std::uint32_t missing_source_pages, bool prefix_fork, bool& staged,
                runtime::ContextResourceClass resource) {
                const std::uint32_t required = pages_for_tokens(frontier);
                const std::uint64_t final_without_release =
                    static_cast<std::uint64_t>(required) + active_pages;
                const std::uint32_t capacity = pages.physical_pool().capacity_pages();
                if (final_without_release <= capacity) { return true; }
                if (!prefix_fork || frontier == 0 ||
                    frontier % static_cast<std::uint32_t>(kPagedKVPageSize) == 0 ||
                    final_without_release != static_cast<std::uint64_t>(capacity) + 1U ||
                    required > addresses.mapped_pages(address)) {
                    return false;
                }
                const LogicalKVPageHandle tail = addresses.logical_page(address, required - 1U);
                if (pages.address_references(tail) != 1 || pages.writer_references(tail) != 0 ||
                    addresses.has_active_reference(tail)) {
                    return false;
                }
                staged                 = true;
                const bool backend     = resource == runtime::ContextResourceClass::BackendKV;
                std::uint32_t& removed = backend ? retained_tail_removed.device.backend_kv_pages
                                                 : retained_tail_removed.device.main_kv_pages;
                std::uint32_t& preparation =
                    backend ? retained_tail_preparation_peak.device.backend_kv_pages
                            : retained_tail_preparation_peak.device.main_kv_pages;
                removed     = 1;
                preparation = missing_source_pages + 1U;
                if (!pages.host_resident(tail)) {
                    const std::size_t stride =
                        plan_host_kv_page_layout(pages.physical_pool().geometry()).page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() -
                                     retained_tail_added.host.kv_bytes) {
                        throw std::overflow_error("retained KV tail Host backup size overflow");
                    }
                    retained_tail_added.host.kv_bytes += stride;
                    add_kv_transfer(resource, runtime::ContextTransferDirection::DeviceToHost,
                                    pages, 1);
                    plan->needs_transfer = true;
                }
                return true;
            };
        if (!plan_retained_tail_release(
                *text_kv_addresses, *text_kv_pages, source_kv->text, plan->reuse_base,
                active_resources.device.main_kv_pages,
                source_replica_additions.device.main_kv_pages, plan->text_prefix_fork_required,
                plan->text_retained_tail_release, runtime::ContextResourceClass::MainKV)) {
            return std::nullopt;
        }
        if (source_kv->backend) {
            if (!plan_retained_tail_release(
                    *backend_kv_addresses, *backend_kv_pages, *source_kv->backend, backend_frontier,
                    active_resources.device.backend_kv_pages,
                    source_replica_additions.device.backend_kv_pages,
                    plan->backend_prefix_fork_required, plan->backend_retained_tail_release,
                    runtime::ContextResourceClass::BackendKV)) {
                return std::nullopt;
            }
        }
        const bool splits_private_state = [&] {
            if (source == nullptr) { return false; }
            const StateImageHandle selected =
                selected_state(*source, plan->reuse, plan->selected_checkpoint);
            return state_store->residency(selected) == StateReplicaResidency::Both;
        }();
        detail::PhysicalResources added = active_resources;
        if (source_replica_additions.device.main_kv_pages >
                std::numeric_limits<std::uint32_t>::max() - added.device.main_kv_pages ||
            source_replica_additions.device.backend_kv_pages >
                std::numeric_limits<std::uint32_t>::max() - added.device.backend_kv_pages) {
            throw std::overflow_error("retained materialization demand overflow");
        }
        added.device.main_kv_pages += source_replica_additions.device.main_kv_pages;
        added.device.backend_kv_pages += source_replica_additions.device.backend_kv_pages;
        added.host.kv_bytes = retained_tail_added.host.kv_bytes;
        detail::PhysicalResources credit;
        detail::PhysicalResources removed;
        detail::PhysicalResources physical_peak = added;
        if (splits_private_state) {
            credit.device.state_slots  = 1;
            removed.device.state_slots = 1;
            if (physical_peak.device.state_slots == 0) {
                throw std::logic_error("StateImage identity split has no active destination");
            }
            --physical_peak.device.state_slots;
        }
        credit.device.main_kv_pages     = retained_tail_removed.device.main_kv_pages;
        credit.device.backend_kv_pages  = retained_tail_removed.device.backend_kv_pages;
        removed.device.main_kv_pages    = retained_tail_removed.device.main_kv_pages;
        removed.device.backend_kv_pages = retained_tail_removed.device.backend_kv_pages;
        const auto stage_peak           = [](std::uint32_t added_pages, std::uint32_t removed_pages,
                                   std::uint32_t preparation_pages) {
            const std::uint32_t settled =
                added_pages > removed_pages ? added_pages - removed_pages : 0U;
            return std::max(settled, preparation_pages);
        };
        if (plan->text_retained_tail_release) {
            physical_peak.device.main_kv_pages =
                stage_peak(added.device.main_kv_pages, retained_tail_removed.device.main_kv_pages,
                           retained_tail_preparation_peak.device.main_kv_pages);
        }
        if (plan->backend_retained_tail_release) {
            physical_peak.device.backend_kv_pages = stage_peak(
                added.device.backend_kv_pages, retained_tail_removed.device.backend_kv_pages,
                retained_tail_preparation_peak.device.backend_kv_pages);
        }
        plan->demand = detail::PhysicalDemand{
            .active_entitlement       = active_resources,
            .reservation_added        = added,
            .reservation_credit       = credit,
            .physical_peak_additional = physical_peak,
            .final_removed            = removed,
            .final_added              = added,
        };
        return AdmissionCandidate(std::move(plan));
    }
    detail::PhysicalDeviceResources exclusive_active = active;
    detail::PhysicalResources shared_replica_additions;
    detail::PhysicalResources transient_source_restores;
    detail::PhysicalDeviceResources conversions;
    detail::PhysicalHostResources retained_host = plan->active_optional_resources.host;
    std::size_t transient_host_bytes            = 0;
    const auto private_tail_restore =
        [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
            KVAddressSpaceHandle address, std::uint32_t frontier, bool prefix_fork) {
            std::pair<std::uint32_t, std::size_t> out;
            if (!prefix_fork || frontier == 0 ||
                frontier % static_cast<std::uint32_t>(kPagedKVPageSize) == 0) {
                return out;
            }
            const std::uint32_t required = pages_for_tokens(frontier);
            if (required > addresses.mapped_pages(address)) {
                throw std::logic_error("checkpoint KV requirement exceeds address membership");
            }
            const LogicalKVPageHandle tail = addresses.logical_page(address, required - 1U);
            if (pages.address_references(tail) != 1 || pages.device_resident(tail)) { return out; }
            if (!pages.host_resident(tail)) {
                throw std::logic_error("private COW tail has no restorable replica");
            }
            out.first  = 1;
            out.second = plan_host_kv_page_layout(pages.physical_pool().geometry()).page_stride;
            return out;
        };
    if (source != nullptr) {
        const StateImageHandle selected_state_handle =
            selected_state(*source, plan->reuse, plan->selected_checkpoint);
        const StateReplicaResidency selected_residency =
            state_store->residency(selected_state_handle);
        conversions.state_slots = plan->active_optional_resources.device.state_slots;
        if (!plan->state_fork_required &&
            (selected_residency == StateReplicaResidency::DeviceOnly ||
             selected_residency == StateReplicaResidency::Both)) {
            if (conversions.state_slots == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("active StateImage conversion overflow");
            }
            ++conversions.state_slots;
        }

        if (!source->kv) { throw std::logic_error("private checkpoint has no KV address space"); }
        const std::uint32_t main_shared =
            shared_kv_prefix_pages(*text_kv_addresses, source->kv->text, plan->reuse_base);
        const std::uint32_t main_missing_shared = missing_shared_device_kv_prefix_pages(
            *text_kv_addresses, source->kv->text, plan->reuse_base);
        if (main_shared > exclusive_active.main_kv_pages) {
            throw std::logic_error("shared Main KV exceeds the active entitlement");
        }
        exclusive_active.main_kv_pages -= main_shared;
        shared_replica_additions.device.main_kv_pages = main_missing_shared;
        const std::uint32_t main_shared_device =
            shared_device_kv_prefix_pages(*text_kv_addresses, source->kv->text, plan->reuse_base);
        conversions.main_kv_pages =
            device_kv_prefix_pages(*text_kv_addresses, source->kv->text, plan->reuse_base) -
            main_shared_device;
        if (source->kv->backend) {
            const std::uint32_t backend_frontier =
                backend_frontier_at(speculative_backend, plan->reuse_base);
            const std::uint32_t backend_shared = shared_kv_prefix_pages(
                *backend_kv_addresses, *source->kv->backend, backend_frontier);
            const std::uint32_t backend_missing_shared = missing_shared_device_kv_prefix_pages(
                *backend_kv_addresses, *source->kv->backend, backend_frontier);
            if (backend_shared > exclusive_active.backend_kv_pages) {
                throw std::logic_error("shared Backend KV exceeds the active entitlement");
            }
            exclusive_active.backend_kv_pages -= backend_shared;
            shared_replica_additions.device.backend_kv_pages = backend_missing_shared;
            const std::uint32_t backend_shared_device        = shared_device_kv_prefix_pages(
                *backend_kv_addresses, *source->kv->backend, backend_frontier);
            conversions.backend_kv_pages =
                device_kv_prefix_pages(*backend_kv_addresses, *source->kv->backend,
                                       backend_frontier) -
                backend_shared_device;
        }
        retained_host.kv_bytes =
            host_kv_prefix_bytes(*text_kv_addresses, source->kv->text, plan->reuse_base);
        const auto [main_tail_restore, main_discarded_host_bytes] =
            private_tail_restore(*text_kv_addresses, *text_kv_pages, source->kv->text,
                                 plan->reuse_base, plan->text_prefix_fork_required);
        transient_source_restores.device.main_kv_pages = main_tail_restore;
        transient_host_bytes                           = main_discarded_host_bytes;
        if (source->kv->backend) {
            const std::uint32_t backend_frontier =
                backend_frontier_at(speculative_backend, plan->reuse_base);
            const std::size_t backend_bytes =
                host_kv_prefix_bytes(*backend_kv_addresses, *source->kv->backend, backend_frontier);
            if (backend_bytes > std::numeric_limits<std::size_t>::max() - retained_host.kv_bytes) {
                throw std::overflow_error("active Host KV entitlement overflow");
            }
            retained_host.kv_bytes += backend_bytes;
            const auto [backend_tail_restore, backend_discarded_host_bytes] =
                private_tail_restore(*backend_kv_addresses, *backend_kv_pages, *source->kv->backend,
                                     backend_frontier, plan->backend_prefix_fork_required);
            transient_source_restores.device.backend_kv_pages = backend_tail_restore;
            if (backend_discarded_host_bytes >
                std::numeric_limits<std::size_t>::max() - transient_host_bytes) {
                throw std::overflow_error("private COW Host KV bytes overflow");
            }
            transient_host_bytes += backend_discarded_host_bytes;
        }
    }
    detail::PhysicalHostResources preparation_retained_host = retained_host;
    if (transient_host_bytes >
        std::numeric_limits<std::size_t>::max() - preparation_retained_host.kv_bytes) {
        throw std::overflow_error("private COW preparation Host KV entitlement overflow");
    }
    preparation_retained_host.kv_bytes += transient_host_bytes;
    conversions = convertible_source_resources(exclusive_active, conversions);
    const detail::PhysicalDeviceResources exclusive_additional =
        additional_resources(exclusive_active, conversions);
    detail::PhysicalResources reservation_added{.device = exclusive_additional};
    if (shared_replica_additions.device.main_kv_pages >
            std::numeric_limits<std::uint32_t>::max() - reservation_added.device.main_kv_pages ||
        shared_replica_additions.device.backend_kv_pages >
            std::numeric_limits<std::uint32_t>::max() - reservation_added.device.backend_kv_pages) {
        throw std::overflow_error("shared KV restore demand overflow");
    }
    reservation_added.device.main_kv_pages += shared_replica_additions.device.main_kv_pages;
    reservation_added.device.backend_kv_pages += shared_replica_additions.device.backend_kv_pages;
    if (transient_source_restores.device.main_kv_pages >
            std::numeric_limits<std::uint32_t>::max() - reservation_added.device.main_kv_pages ||
        transient_source_restores.device.backend_kv_pages >
            std::numeric_limits<std::uint32_t>::max() - reservation_added.device.backend_kv_pages) {
        throw std::overflow_error("private COW source restore demand overflow");
    }
    reservation_added.device.main_kv_pages += transient_source_restores.device.main_kv_pages;
    reservation_added.device.backend_kv_pages += transient_source_restores.device.backend_kv_pages;
    const detail::PhysicalResources active_resources{.device = exclusive_active,
                                                     .host   = retained_host};
    detail::PhysicalResources final_added = active_resources;
    final_added.device.main_kv_pages += shared_replica_additions.device.main_kv_pages;
    final_added.device.backend_kv_pages += shared_replica_additions.device.backend_kv_pages;
    const detail::PhysicalResources reused_source{.device = conversions,
                                                  .host   = preparation_retained_host};
    const detail::PhysicalResources early_source_release =
        positive_difference(source_resources, reused_source);
    plan->demand = detail::PhysicalDemand{
        .active_entitlement       = active_resources,
        .reservation_added        = reservation_added,
        .reservation_credit       = {.device = conversions},
        .physical_peak_additional = positive_difference(reservation_added, early_source_release),
        .final_removed            = source_resources,
        .final_added              = final_added,
    };
    return AdmissionCandidate(std::move(plan));
}

void ProgramImplCore::select_shared_captures(AdmissionCandidate& candidate,
                                             const PreparedPromptData& prompt,
                                             std::span<const std::uint32_t> frontiers) {
    if (candidate.impl_ == nullptr || candidate.impl_->planning_revision != resource_revision()) {
        throw std::logic_error("shared capture selection observes a stale admission candidate");
    }
    if (!std::is_sorted(frontiers.begin(), frontiers.end()) ||
        std::adjacent_find(frontiers.begin(), frontiers.end()) != frontiers.end()) {
        throw std::invalid_argument("selected shared capture frontiers must be ordered unique");
    }
    AdmissionCandidateImpl& plan = *candidate.impl_;
    for (const std::uint32_t frontier : frontiers) {
        const auto selected =
            std::find_if(plan.shared_candidates.begin(), plan.shared_candidates.end(),
                         [&](const CaptureGroup& group) { return group.frontier == frontier; });
        if (selected == plan.shared_candidates.end() || frontier < plan.reuse_base ||
            !selected->shared || !selected->identity) {
            throw std::invalid_argument("selected shared capture frontier is unavailable");
        }
        auto existing =
            std::find_if(plan.capture_groups.begin(), plan.capture_groups.end(),
                         [&](const CaptureGroup& group) { return group.frontier == frontier; });
        if (existing == plan.capture_groups.end()) {
            plan.capture_groups.push_back(*selected);
        } else {
            existing->shared = true;
            existing->shared_evidence |= selected->shared_evidence;
            existing->input_order = std::min(existing->input_order, selected->input_order);
        }
    }
    plan.shared_candidates.clear();
    std::sort(plan.capture_groups.begin(), plan.capture_groups.end(),
              [](const CaptureGroup& left, const CaptureGroup& right) {
                  return std::tie(left.frontier, left.input_order) <
                         std::tie(right.frontier, right.input_order);
              });

    const std::size_t prefill_splits = plan.vision ? plan.vision->uses.size() : 0ULL;
    plan.summary.service_work_quanta =
        projected_service_work(plan.summary, plan.reuse_base, prefill_chunk, prefill_splits,
                               plan.capture_groups, prompt.identity.rewrite_execution_frontiers);
    std::uint64_t vision_items   = 0;
    std::uint64_t vision_patches = 0;
    if (plan.vision) {
        std::vector<bool> counted(prompt.vision_items.size(), false);
        for (const VisionUseSpan& use : plan.vision->uses) {
            if (use.prepared_item_index >= counted.size()) {
                throw std::logic_error("selected shared capture Vision use is invalid");
            }
            if (counted[use.prepared_item_index]) { continue; }
            counted[use.prepared_item_index] = true;
            ++vision_items;
            const std::size_t patches = prompt.vision_items[use.prepared_item_index].patch_count;
            if (patches > std::numeric_limits<std::uint64_t>::max() - vision_patches) {
                throw std::overflow_error("selected shared capture Vision work exceeds uint64");
            }
            vision_patches += patches;
        }
    }
    plan.remaining_prefill_work = scheduled_prefill_work(
        plan.reuse_base, plan.summary.prompt_tokens, vision_items, vision_patches, prefill_chunk,
        plan.capture_groups, prompt.identity.rewrite_execution_frontiers);
}

runtime::PrefillWork
ProgramImplCore::shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                                   const PreparedPromptData& prompt,
                                                   std::span<const std::uint32_t> frontiers) {
    if (candidate.impl_ == nullptr || candidate.impl_->planning_revision != resource_revision()) {
        throw std::logic_error("shared capture projection observes a stale admission candidate");
    }
    AdmissionCandidate projected(std::make_unique<AdmissionCandidateImpl>(*candidate.impl_));
    select_shared_captures(projected, prompt, frontiers);
    return projected.impl_->remaining_prefill_work;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
