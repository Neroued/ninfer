#pragma once

namespace ninfer::runtime {

template <class Instance>
void EngineCore<Instance>::reserve_active_capture(const std::shared_ptr<Request>& request,
                                                  EngineRequestState next) {
    request->model_state        = next;
    request->capture_pending    = true;
    request->post_capture_state = next;
    if (instance_.session->has_context_operation() ||
        instance_.session->execution_pending(*request->sequence)) {
        return;
    }
    if (scheduler_.progress_owner()) {
        instance_.session->skip_capture(*request->sequence);
        request->capture_pending = false;
        return;
    }
    if (!prepare_semantic_capture(request)) return;
    const auto started = instance_.session->start_capture(*request->sequence);
    if (started == ContextStart::Deferred) return;
    if (started == ContextStart::Stale) {
        request->capture_pending = false;
        return;
    }
    context_owner_ = request;
}

template <class Instance>
bool EngineCore<Instance>::prepare_semantic_capture(const std::shared_ptr<Request>& request) {
    auto& decision = capture_decisions_[request->lane->value];
    if (decision && decision->sequence != *request->sequence) { decision.reset(); }
    while (const auto capture = instance_.session->prepare_capture(*request->sequence)) {
        if (scheduler_.progress_owner()) {
            instance_.session->skip_capture(*request->sequence);
            decision.reset();
            continue;
        }
        if (instance_.session->capture_is_shared(*request->sequence) &&
            resources_.retain_shared_checkpoint(
                *instance_.session, *request->base_plan, capture->frontier,
                instance_.session->placement(*request->sequence), capture->evidence)) {
            instance_.session->skip_capture(*request->sequence, CaptureSkip::Retained);
            decision.reset();
            continue;
        }
        if (const auto source = resources_.capture_source(
                *instance_.session, *request->base_plan, capture->frontier,
                instance_.session->placement(*request->sequence))) {
            const auto shared = instance_.session->share_capture(*request->sequence, *source);
            if (!shared.empty()) {
                for (const auto checkpoint : shared)
                    resources_.publish(*instance_.session, request->continuation_owner, checkpoint);
                decision.reset();
                request_admission_check();
                continue;
            }
        }
        if (capture->reserved) {
            decision.reset();
            return true;
        }
        if (instance_.session->has_context_operation()) { return false; }
        const auto admission = resources_.capture_admission(
            *instance_.session, request->continuation_owner, *request->base_plan, capture->frontier,
            instance_.session->capture_is_input(*request->sequence)
                ? CheckpointRole::InputReplay : CheckpointRole::SharedPrefix);
        if (!decision || decision->frontier != capture->frontier) {
            decision.emplace(
                CaptureDecision{*request->sequence, capture->frontier,
                                resources_.begin_reclaim(*instance_.session, admission)});
        }
        if (!capture->host.empty()) {
            if (!instance_.session->can_place_host(capture->host)) {
                (void)instance_.session->release_redundant_host({}, request->sequence);
            }
            if (const auto victims =
                    resources_.host_victims(*instance_.session, capture->host, std::nullopt, {}, {},
                                            admission, &decision->reclaim)) {
                const auto loss  = resources_.release_cost(*instance_.session, *victims);
                const auto retry = instance_.session->prepare_capture(
                    *request->sequence, *victims, CaptureStorage::Host);
                if (retry && !retry->retired_points.empty()) {
                    if (retry->retired_points != *victims)
                        throw std::logic_error("capture consumed only part of its victim quote");
                    resources_.adopt_release(*instance_.session, retry->retired_points,
                                             decision->reclaim, loss);
                    request_admission_check();
                }
                if (retry && retry->reserved) {
                    decision.reset();
                    return true;
                }
            }
            const auto preserved = resources_.preserve_capture(
                *instance_.session, capture->assessment, decision->reclaim);
            if (preserved == ReclaimProgress::Transferring || preserved == ReclaimProgress::Deferred)
                return false;
            if (preserved == ReclaimProgress::Changed || preserved == ReclaimProgress::Retry)
                continue;
        }
        if (capture->assessment.requires_maintenance()) {
            const auto started = instance_.session->start_maintenance(capture->assessment);
            if (started == ContextStart::Started || started == ContextStart::Deferred) return false;
        }
        if (capture->assessment.waiting()) return false;
        if (const auto victims = resources_.capture_victims(
                *instance_.session, capture->assessment, decision->reclaim)) {
            const auto loss = resources_.release_cost(*instance_.session, *victims);
            const auto retry = instance_.session->prepare_capture(
                *request->sequence, *victims, CaptureStorage::Device);
            if (retry && !retry->retired_points.empty()) {
                if (retry->retired_points != *victims)
                    throw std::logic_error("capture consumed only part of its victim quote");
                resources_.adopt_release(*instance_.session, retry->retired_points,
                                         decision->reclaim, loss);
                request_admission_check();
            }
            if (retry && retry->reserved) {
                decision.reset();
                return true;
            }
        }
        instance_.session->skip_capture(*request->sequence);
        decision.reset();
    }
    decision.reset();
    return true;
}

template <class Instance>
void EngineCore<Instance>::ensure_base_plan(const std::shared_ptr<Request>& request) {
    if (!request->base_plan) {
        request->base_plan.emplace(instance_.session->plan_request(std::move(request->prompt),
                                                                   request->options.execution));
        const auto& summary = request->base_plan->summary();
        request->budget.emplace(summary.effective_output_tokens, summary.effective_limit_reason);
        request->generated.reserve(summary.effective_output_tokens);
    }
}

template <class Instance>
[[nodiscard]] std::optional<std::uint32_t> EngineCore<Instance>::free_lane() const noexcept {
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (!slots_[lane] && (!materializing_ || materializing_->destination.value != lane)) {
            return lane;
        }
    }
    return std::nullopt;
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::resident_empty() const noexcept {
    return !materializing_ &&
           std::all_of(slots_.begin(), slots_.end(), [](const auto& p) { return !p; });
}

template <class Instance>
void EngineCore<Instance>::resolve_context_progress(
    typename ModelContract::ContextProgress&& progress, HostPhaseMeasurement& boundary) {
    if (!progress.complete) {
        throw std::logic_error("Session emitted an incomplete context result");
    }
    const auto owner = materializing_ ? materializing_->request : context_owner_;
    record_context_work(progress, owner);
    if (progress.kind == decltype(progress.kind)::Maintenance) {
        request_admission_check();
        return;
    }
    if (owner && progress.kind == decltype(progress.kind)::Pause) {
        owner->speculative_stats = std::move(progress.request_speculative);
    }
    if (materializing_) {
        const auto control  = std::move(*materializing_);
        const auto& request = control.request;
        if (!progress.published) {
            materializing_.reset();
            complete_cancelled(request);
            request_admission_check();
            return;
        }
        auto carried = progress.private_points;
        if (control.resumed) {
            for (const auto point : resources_.points(request->continuation_owner)) {
                if (instance_.session->valid_checkpoint(point) &&
                    std::find(carried.begin(), carried.end(), point) == carried.end()) {
                    carried.push_back(point);
                }
            }
        }
        request->continuation_owner = resources_.adopt(
            *instance_.session, control.source, *request->base_plan, request->publication_order,
            control.resumed, carried, progress.retired_checkpoints);
        resources_.release_source(*instance_.session, request->id);
        request->lane      = control.destination;
        request->sequence  = progress.sequence;
        request->residency = progress.residency;
        if (!request->residency || request->residency->lane != control.destination) {
            throw std::logic_error("binding completed without the expected residency");
        }
        if (!request->sequence) {
            throw std::logic_error("binding completed without a native sequence");
        }
        request->placement = instance_.session->placement(*request->sequence);
        request->placement_tokens =
            std::max(request->placement_tokens, instance_.session->load_tokens(*request->sequence));
        request->model_state = progress.recovery.route == GenerationRecoveryRoute::Replay
                                   ? EngineRequestState::Replay
                               : control.resumed ? request->resume_phase
                                                 : EngineRequestState::Prefill;
        if (control.resumed) {
            request->recovery = progress.recovery;
            if (progress.recovery.route == GenerationRecoveryRoute::Snapshot) {
                ++cumulative_stats_.snapshot_restores;
                ++request->snapshot_restores;
            } else if (progress.recovery.route == GenerationRecoveryRoute::Direct) {
                ++cumulative_stats_.direct_restores;
                ++request->direct_restores;
            }
            request->suspended.reset();
            request->recovery_pending = instance_.session->recovery_pending(*request->sequence);
            if (request->paused_at) {
                request->paused_ns += elapsed_ns(*request->paused_at, Clock::now());
            }
            request->paused_at.reset();
            observe_scheduling(request, GenerationSchedulingTransition::Restored);
        } else {
            request->initial_binding_ns = elapsed_ns(*request->admitted_at, Clock::now());
            if (control.summary.reused_prompt_tokens) {
                ++cumulative_stats_.checkpoint_selections;
            } else {
                ++cumulative_stats_.root_selections;
            }
            cumulative_stats_.reused_prompt_tokens += control.summary.reused_prompt_tokens;
            cumulative_stats_.prompt_tokens += control.summary.prompt_tokens;
            cumulative_stats_.last_selected_frontier_tokens = control.summary.reused_prompt_tokens;
            // Generation start was published when the first binding obtained its resources.
        }
        finish_engine_phase(boundary, EngineHostPhase::Boundary);
        slots_[control.destination.value] = request;
        materializing_.reset();
        boundary = begin_host_phase();
    } else if (context_owner_) {
        const auto request = context_owner_;
        if (request->model_state == EngineRequestState::Pausing) {
            const auto lane = request->lane->value;
            if (!progress.published) {
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
                context_owner_.reset();
                complete_cancelled(request);
                boundary = begin_host_phase();
                return;
            }
            request->placement.reset();
            request->placement_tokens = 0;
            request->suspended        = std::move(progress.paused);
            request->model_state      = EngineRequestState::Paused;
            observe_scheduling(request, GenerationSchedulingTransition::Paused);
            paused_.push_back(request);
            std::sort(paused_.begin(), paused_.end(),
                      [](const auto& a, const auto& b) { return a->id < b->id; });
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            slots_[lane].reset();
            request->lane.reset();
            request->residency.reset();
            request->sequence.reset();
            request_admission_check();
            boundary = begin_host_phase();
        } else {
            if (progress.published) {
                ++cumulative_stats_.active_captures_completed;
            } else {
                ++cumulative_stats_.active_captures_aborted;
            }
            request->capture_pending = false;
            request->model_state     = request->post_capture_state;
            for (const auto point : progress.captured_checkpoints) {
                resources_.publish(*instance_.session, request->continuation_owner, point);
            }
        }
        context_owner_.reset();
    }
    request_admission_check();
    return;
}

template <class Instance>
ReclaimProgress EngineCore<Instance>::reclaim(const ContextAssessment& assessment, bool allow_pause,
                                              std::span<const Checkpoint> excluded,
                                              typename ResourceManagement::ReclaimCursor* cursor) {
    if (assessment.ready()) return ReclaimProgress::Retry;
    if (assessment.waiting()) return ReclaimProgress::Deferred;
    if (assessment.impossible()) return ReclaimProgress::Blocked;
    if (instance_.session->reclaim_capture_reservation(assessment)) {
        request_admission_check();
        return ReclaimProgress::Changed;
    }
    const auto progress =
        cursor ? resources_.reclaim(*instance_.session, assessment, excluded, *cursor)
               : resources_.reclaim(*instance_.session, assessment, excluded);
    if (progress != ReclaimProgress::Blocked) {
        if (progress == ReclaimProgress::Changed) {
            request_admission_check();
        }
        return progress;
    }
    if (cursor && cursor->rights.purpose != ReclaimPurpose::Execution)
        return ReclaimProgress::Blocked;
    std::vector<Checkpoint> paused_sources;
    for (auto it = paused_.rbegin(); it != paused_.rend(); ++it) {
        const auto& request = **it;
        if (!request.suspended || !request.suspended->has_snapshot()) { continue; }
        const auto handle = *request.suspended->snapshot_handle();
        if (std::find(excluded.begin(), excluded.end(), handle) == excluded.end()) {
            paused_sources.push_back(handle);
        }
    }
    const auto releases =
        instance_.session->plan_releases(paused_sources, excluded, assessment.demand);
    for (const auto& release : releases.releases) {
        if (instance_.session->useful_release_bytes(release.released, assessment) == 0 &&
            !instance_.session->releases_metadata(release.released, assessment)) {
            continue;
        }
        bool changed = false;
        for (const auto handle : release.sources) {
            const auto owner =
                std::find_if(paused_.begin(), paused_.end(), [&](const auto& request) {
                    return request->suspended && request->suspended->snapshot_handle() == handle;
                });
            if (owner == paused_.end() ||
                !instance_.session->revoke_snapshot(*(*owner)->suspended)) {
                continue;
            }
            (*owner)->recovery.reset();
            observe_scheduling(*owner, GenerationSchedulingTransition::SnapshotRevoked);
            changed = true;
        }
        if (changed) {
            request_admission_check();
            return ReclaimProgress::Changed;
        }
    }
    if (allow_pause && pause_reclaim_victim(assessment))
        return instance_.session->has_context_operation() ? ReclaimProgress::Transferring
                                                          : ReclaimProgress::Changed;
    return ReclaimProgress::Blocked;
}

template <class Instance>
bool EngineCore<Instance>::pause_reclaim_victim(const ContextAssessment& pressure) {
    std::vector<std::uint32_t> skipped;
    while (const auto victim = scheduler_.preemption_victim(slots_, max_concurrency_, skipped)) {
        if (pause_resident(*victim, PauseGoal::ReleaseCapacity, &pressure)) { return true; }
        skipped.push_back(*victim);
    }
    return false;
}

template <class Instance>
bool EngineCore<Instance>::pause_resident(std::uint32_t lane, PauseGoal goal,
                                          const ContextAssessment* pressure) {
    if (instance_.session->has_context_operation()) { return false; }
    auto request = slots_[lane];
    if (!request || !request->sequence || instance_.session->execution_pending(*request->sequence))
        return false;
    std::optional<PauseAction> selected;
    auto options = instance_.session->pause_options(*request->sequence);
    for (const auto& option : options) {
        if (goal == PauseGoal::RelocateOnly && option.action.kind != PauseKind::Relocate) {
            continue;
        }
        if (pressure &&
            !(goal == PauseGoal::RelocateOnly && option.action.kind == PauseKind::Relocate) &&
            !instance_.session->pause_contributes(*request->sequence, option, *pressure)) {
            continue;
        }
        if (option.action.kind == PauseKind::Relocate) {
            const auto target = instance_.session->assess(option.demand);
            if (!target.ready()) {
                auto cursor =
                    resources_.begin_reclaim(*instance_.session, std::nullopt,
                                             ReclaimRights{ReclaimPurpose::Execution, request->id});
                const auto progress =
                    resources_.reclaim(*instance_.session, target, option.sources, cursor);
                if (progress == ReclaimProgress::Transferring) { return true; }
                if (progress == ReclaimProgress::Changed) {
                    request_admission_check();
                    return true;
                }
                if (progress != ReclaimProgress::Retry ||
                    !instance_.session->assess(option.demand).ready()) {
                    continue;
                }
            }
        }
        std::vector<Checkpoint> victims;
        if (option.host_write) {
            if (!instance_.session->can_place_host(*option.host_write)) {
                (void)instance_.session->release_redundant_host({}, request->sequence);
            }
            std::vector<Checkpoint> later;
            for (auto it = paused_.rbegin(); it != paused_.rend(); ++it) {
                if ((*it)->id > request->id && (*it)->suspended &&
                    (*it)->suspended->has_snapshot()) {
                    later.push_back(*(*it)->suspended->snapshot_handle());
                }
            }
            auto candidate = resources_.host_victims(*instance_.session, *option.host_write,
                                                     std::nullopt, later);
            if (!candidate) { continue; }
            victims = std::move(*candidate);
        }
        if (!instance_.session->prepare_pause(*request->sequence, option.action, victims)) {
            continue;
        }
        for (const auto handle : victims) {
            if (resources_.erase(*instance_.session, handle)) { continue; }
            const auto owner = std::find_if(paused_.begin(), paused_.end(), [&](const auto& other) {
                return other->suspended && other->suspended->snapshot_handle() == handle;
            });
            if (owner == paused_.end() ||
                !instance_.session->revoke_snapshot(*(*owner)->suspended)) {
                throw std::logic_error("pause victim changed after preflight");
            }
            (*owner)->recovery.reset();
            observe_scheduling(*owner, GenerationSchedulingTransition::SnapshotRevoked);
        }
        selected = option.action;
        break;
    }
    if (!selected) { return false; }
    ExecutionCallScope call(*this);
    const auto pause_started = Clock::now();
    if (!instance_.session->start_pause(*request->sequence, *selected)) { return false; }
    call.finish();
    if (!request->is_replaying()) { request->resume_phase = request->model_state; }
    request->model_state = EngineRequestState::Pausing;
    request->paused_at   = pause_started;
    ++request->preemption_count;
    ++cumulative_stats_.preemptions;
    request->recovery.reset();
    observe_scheduling(request, GenerationSchedulingTransition::PauseStarted, pause_started);
    context_owner_ = request;
    capture_decisions_[lane].reset();
    return true;
}

template <class Instance>
void EngineCore<Instance>::finish_source_wait(const std::shared_ptr<Request>& request,
                                              Clock::time_point now) noexcept {
    if (!request->source_wait_started) { return; }
    request->admission.source_wait_seconds +=
        static_cast<double>(elapsed_ns(*request->source_wait_started, now)) * 1.0e-9;
    request->source_wait_started.reset();
}

template <class Instance>
void EngineCore<Instance>::retain_admission_source(
    const std::shared_ptr<Request>& request,
    const typename ResourceManagement::SourceChoice& choice, bool restoring) {
    resources_.retain_source(*instance_.session, request->id, choice);
    if (!restoring && choice.source.reused_tokens && !request->source_wait_started) {
        request->source_wait_started = Clock::now();
    } else if (!restoring && !choice.source.reused_tokens) {
        finish_source_wait(request, Clock::now());
    }
}

template <class Instance>
void EngineCore<Instance>::order_binding_sources(
    const std::shared_ptr<Request>& request, LaneId lane,
    std::vector<typename ResourceManagement::SourceChoice>& sources) {
    const auto placements = instance_.session->placements(*request->base_plan);
    if (placements.size() <= 1) { return; }
    std::vector<PlacementLoad> loads;
    std::vector<PlacementCandidate> targets;
    for (const auto placement : placements) {
        loads.push_back({placement});
        targets.push_back({placement});
    }
    const auto account = [&](const std::shared_ptr<Request>& active) {
        if (!active || !active->placement) { return; }
        auto found = std::find_if(loads.begin(), loads.end(), [&](const auto& load) {
            return load.placement == *active->placement;
        });
        if (found == loads.end()) {
            loads.push_back({*active->placement});
            found = std::prev(loads.end());
        }
        if (active->placement_tokens > UINT64_MAX - found->tokens) {
            throw std::overflow_error("placement token load overflow");
        }
        found->tokens += active->placement_tokens;
        ++found->requests;
    };
    for (const auto& active : slots_) { account(active); }
    if (materializing_) { account(materializing_->request); }
    const auto* resume = request->suspended ? &*request->suspended : nullptr;
    const auto kind    = request->resume_phase == EngineRequestState::Prefill ? UnitKind::Prefill
                         : request->resume_phase == EngineRequestState::ControlReady
                             ? UnitKind::Control
                             : UnitKind::Decode;
    const auto tokens =
        kind == UnitKind::Prefill ? 0U
        : kind == UnitKind::Control
            ? static_cast<std::uint32_t>(request->output.pending_control_tokens().size())
            : 1U;
    std::vector<typename ResourceManagement::SourceChoice> waiting;
    std::erase_if(sources, [&](const auto& choice) {
        auto assessment = instance_.session->assess_binding(*request->base_plan, lane,
                                                            choice.source, resume, kind, tokens);
        if (!assessment.source_valid) { return true; }
        if (!assessment.capacity_possible) {
            auto transferable = choice;
            const ReclaimRights rights{
                resume ? ReclaimPurpose::Execution : ReclaimPurpose::FreshAdmission, request->id};
            if (resources_.prepare_source(*instance_.session, *request->base_plan, transferable,
                                          request->id, rights, resume)) {
                assessment = instance_.session->assess_binding(
                    *request->base_plan, lane, transferable.source, resume, kind, tokens);
            }
        }
        if (!assessment.source_valid || !assessment.capacity_possible) { return true; }
        if (assessment.assessment.waiting()) {
            waiting.push_back(choice);
            return true;
        }
        auto target = std::find_if(targets.begin(), targets.end(), [&](const auto& target) {
            return target.placement == choice.source.placement;
        });
        if (target == targets.end()) {
            throw std::logic_error("binding candidate placement is absent from the model plan");
        }
        target->feasible = true;
        if (choice.source.local_device) {
            target->local_prefix_tokens =
                std::max<std::uint64_t>(target->local_prefix_tokens, choice.source.reused_tokens);
        }
        return false;
    });
    const auto weight  = std::max<std::uint64_t>(request->base_plan->summary().prompt_tokens,
                                                resume ? resume->frontier() : 0);
    const auto ordered = order_placements(loads, targets, weight, next_placement_tie_);
    std::vector<typename ResourceManagement::SourceChoice> result;
    result.reserve(sources.size() + waiting.size());
    for (const auto index : ordered) {
        for (auto& source : sources) {
            if (source.source.placement == targets[index].placement) {
                result.push_back(std::move(source));
            }
        }
    }
    result.insert(result.end(), std::make_move_iterator(waiting.begin()),
                  std::make_move_iterator(waiting.end()));
    sources = std::move(result);
}

template <class Instance>
bool EngineCore<Instance>::try_admit_one(bool restoring) {
    if (instance_.session->has_context_operation()) { return false; }
    const auto lane = free_lane();
    const auto owner     = scheduler_.progress_owner();
    const auto suspended = restoring ? std::find_if(paused_.begin(), paused_.end(),
                                                    [&](const auto& r) { return r->id == owner; })
                                     : paused_.end();
    if (restoring && suspended == paused_.end()) return false;
    if (!lane && restoring) {
        if (const auto victim = scheduler_.preemption_victim(slots_, max_concurrency_))
            return pause_resident(*victim);
        return false;
    }
    if (admission_decision_ && (admission_decision_->restoring != restoring ||
                                (restoring && *suspended != admission_decision_->request))) {
        admission_decision_.reset();
    }
    std::vector<std::shared_ptr<Request>> candidates;
    if (admission_decision_) {
        candidates.push_back(admission_decision_->request);
    } else if (restoring) {
        candidates.push_back(*suspended);
    } else {
        std::lock_guard lock(queue_mutex_);
        candidates = scheduler_.fresh_candidates(pending_, max_concurrency_);
    }
    for (const auto& request : candidates) {
        const ReclaimRights rights{
            restoring ? ReclaimPurpose::Execution : ReclaimPurpose::FreshAdmission, request->id};
        if (!admission_decision_) {
            if (!restoring) {
                scheduler_.admission_candidate_checked(request->id);
                if (request->admission_generation == admission_generation_) { continue; }
                request->admission_generation = admission_generation_;
            }
            try {
                ensure_base_plan(request);
            } catch (const RequestError&) {
                // Planning has not bound a sequence or submitted device work. A native
                // request rejection therefore ends only this pending request.
                if (restoring) throw;
                const auto error = std::current_exception();
                {
                    std::lock_guard lock(queue_mutex_);
                    std::erase(pending_, request);
                }
                complete_error(request, error);
                request_admission_check();
                return true;
            }
            const auto retained = resources_.retained_source(request->id);
            const auto revoked  = resources_.source_revocations(request->id);
            if (!restoring && revoked > request->admission.revoked_checkpoints) {
                request->admission.revoked_checkpoints = revoked;
                if (!retained) {
                    request->admission.fallback_reason = AdmissionFallbackReason::SourceRevoked;
                }
            }
            auto sources =
                resources_.candidates(*instance_.session, *request->base_plan,
                                      restoring ? request->suspended->frontier() : UINT32_MAX,
                                      restoring && request->continuation_owner
                                          ? std::optional(request->continuation_owner)
                                          : std::nullopt,
                                      request->id, request->publication_order,
                                      restoring ? &*request->suspended : nullptr);
            if (lane) { order_binding_sources(request, LaneId{*lane}, sources); }
            if (!restoring && !sources.empty()) {
                const auto preferred = sources.front().source.reused_tokens;
                if (!request->admission_observed) {
                    request->admission.preferred_reused_tokens = preferred;
                    request->admission_observed                = true;
                } else if (retained && preferred < retained->source.reused_tokens) {
                    request->admission.fallback_reason = AdmissionFallbackReason::CostChanged;
                }
            }
            admission_decision_.emplace(AdmissionDecision{
                request, restoring, std::move(sources), 0,
                resources_.begin_reclaim(*instance_.session, std::nullopt, rights)});
        }
        auto& decision = *admission_decision_;
        ContextAssessment assessment;
        bool deferred = false;
        std::optional<typename ResourceManagement::SourceChoice> waiting_choice;
        while (decision.current < decision.sources.size()) {
            auto& choice = decision.sources[decision.current];
            if (!resources_.prepare_source(*instance_.session, *request->base_plan, choice,
                                           request->id, std::nullopt,
                                           restoring ? &*request->suspended : nullptr)) {
                if (!restoring) {
                    request->admission.fallback_reason =
                        resources_.source_revocations(request->id) &&
                                !resources_.retained_source(request->id)
                            ? AdmissionFallbackReason::SourceRevoked
                            : AdmissionFallbackReason::SourceInvalid;
                }
                ++decision.current;
                continue;
            }
            if (!lane) {
                retain_admission_source(request, choice, restoring);
                deferred = true;
                break;
            }
            // A new selection replaces its old waiting reference before capacity is
            // checked. The replacement is retained before any old point is released.
            if (resources_.has_source_record(request->id)) {
                retain_admission_source(request, choice, restoring);
            }
            auto binding_started = Clock::now();
            const auto bind      = [&](const auto& source) {
                return instance_.session->start_binding(
                    *request->base_plan, LaneId{*lane}, source,
                    restoring ? &*request->suspended : nullptr,
                    request->resume_phase == EngineRequestState::Prefill ? UnitKind::Prefill
                         : request->resume_phase == EngineRequestState::ControlReady ? UnitKind::Control
                                                                                     : UnitKind::Decode,
                    request->resume_phase == EngineRequestState::Prefill ? 0U
                         : request->resume_phase == EngineRequestState::ControlReady
                             ? static_cast<std::uint32_t>(
                              request->output.pending_control_tokens().size())
                             : 1U);
            };
            auto reservation = bind(choice.source);
            if (!reservation && !reservation.source_valid) {
                if (!restoring) {
                    request->admission.fallback_reason =
                        resources_.source_revocations(request->id) &&
                                !resources_.retained_source(request->id)
                            ? AdmissionFallbackReason::SourceRevoked
                            : AdmissionFallbackReason::SourceInvalid;
                }
                ++decision.current;
                continue;
            }
            if (!reservation) {
                assessment = reservation.assessment;
                retain_admission_source(request, choice, restoring);
                const auto protected_source = choice.source.checkpoint;
                const auto excluded         = protected_source
                                                  ? std::span<const Checkpoint>(&*protected_source, 1)
                                                  : std::span<const Checkpoint>{};
                const auto reclaimed = reclaim(assessment, restoring, excluded, &decision.reclaim);
                if (reclaimed == ReclaimProgress::Transferring) return true;
                if (reclaimed == ReclaimProgress::Deferred) {
                    if (!waiting_choice) { waiting_choice = choice; }
                    deferred = true;
                    ++decision.current;
                    continue;
                }
                if (reclaimed == ReclaimProgress::Changed || reclaimed == ReclaimProgress::Retry)
                    continue;
                // Another waiting reference may be the only obstacle to a Move.
                // Grant its revocation only when the resulting complete bind succeeds.
                {
                    auto transferable = choice;
                    if (resources_.prepare_source(*instance_.session, *request->base_plan,
                                                  transferable, request->id, rights,
                                                  restoring ? &*request->suspended : nullptr) &&
                        (transferable.source.consume_source != choice.source.consume_source ||
                         transferable.source.retired_points != choice.source.retired_points)) {
                        const auto transfer_started = Clock::now();
                        auto attempt                = bind(transferable.source);
                        if (attempt.source_valid) {
                            reservation.capacity_possible |= attempt.capacity_possible;
                        }
                        if (attempt) {
                            binding_started = transfer_started;
                            reservation     = std::move(attempt);
                            choice          = std::move(transferable);
                        }
                    }
                }
                if (!reservation) {
                    // Resident progress can release/unlock capacity. A root chunk's
                    // smaller initial allocation is not a reason to discard this source.
                    if (reservation.capacity_possible &&
                        (!resident_empty() || (!restoring && !paused_.empty()))) {
                        if (!waiting_choice) { waiting_choice = choice; }
                        deferred = true;
                        ++decision.current;
                        continue;
                    }
                    if (!restoring) {
                        request->admission.fallback_reason =
                            reservation.capacity_possible
                                ? AdmissionFallbackReason::IsolatedCapacity
                                : AdmissionFallbackReason::CapacityLimit;
                    }
                    ++decision.current;
                    continue;
                }
            }
            request->placement = choice.source.placement;
            request->placement_tokens =
                std::max<std::uint64_t>(request->base_plan->summary().prompt_tokens,
                                        restoring ? request->suspended->frontier() : 0);
            next_placement_tie_.value = choice.source.placement.value + 1;
            resources_.binding_started(request->id, reservation.retired_points,
                                       reservation.consumed_source);
            const auto& source = choice.source;
            const BeginSummary begin{.prompt_tokens = request->base_plan->summary().prompt_tokens,
                                     .reused_prompt_tokens = source.reused_tokens,
                                     .prefix_reuse_path    = source.reused_tokens
                                                                 ? PrefixReusePath::Checkpoint
                                                                 : PrefixReusePath::Root};
            if (restoring) {
                std::erase(paused_, request);
                observe_scheduling(request, GenerationSchedulingTransition::RestoreStarted,
                                   binding_started);
            } else {
                std::lock_guard lock(queue_mutex_);
                scheduler_.admitted(pending_, *request, max_concurrency_);
                std::erase(pending_, request);
            }
            request->model_state = EngineRequestState::Materializing;
            materializing_.emplace(
                MaterializingRequest{request, LaneId{*lane}, begin, restoring, choice});
            admission_decision_.reset();
            if (!restoring) {
                finish_source_wait(request, binding_started);
                request->admission.revoked_checkpoints = resources_.source_revocations(request->id);
                request->admitted_at                   = binding_started;
                request->host_timing.queue_wait_ns =
                    elapsed_ns(request->submitted, *request->admitted_at);
                request->queue_wait_recorded = true;
                publish_generation_start(request, begin);
            }
            return true;
        }
        if (waiting_choice &&
            resources_.prepare_source(*instance_.session, *request->base_plan, *waiting_choice,
                                      request->id, std::nullopt,
                                      restoring ? &*request->suspended : nullptr)) {
            retain_admission_source(request, *waiting_choice, restoring);
        }
        admission_decision_.reset();
        if (deferred) {
            if (restoring) {
                return false;
            }
            continue;
        }
        if (admission_check_pending_.load(std::memory_order_acquire)) { return true; }
        if (resident_empty()) {
            throw std::logic_error("isolated request cannot reserve its first legal unit after "
                                   "context reclamation");
        }
        if (restoring) {
            return false;
        }
    }
    return false;
}

template <class Instance>
void EngineCore<Instance>::observe_scheduling(const std::shared_ptr<Request>& request,
                                              GenerationSchedulingTransition transition,
                                              Clock::time_point now) {
    if (!request->observation.scheduling) { return; }
    if (now == Clock::time_point{}) { now = Clock::now(); }
    const GenerationSchedulingObservation observation{
        .transition = transition,
        .route      = request->recovery ? request->recovery->route : GenerationRecoveryRoute::None,
        .phase      = observe_request(*request).phase,
        .engine_request_id       = request->id,
        .steady_ns               = elapsed_ns(Clock::time_point{}, now),
        .elapsed_ns              = elapsed_ns(request->submitted, now),
        .preemption_index        = request->preemption_count,
        .global_prefill_tokens   = cumulative_stats_.computed_prefill_tokens,
        .global_decode_tokens    = cumulative_stats_.committed_decode_tokens,
        .global_replayed_tokens  = cumulative_stats_.replayed_tokens,
        .request_prefill_tokens  = request->computed_prompt_tokens,
        .request_decode_tokens   = request->committed_decode_tokens,
        .request_replayed_tokens = request->replayed_tokens,
    };
    {
        std::lock_guard lock(request->mutex);
        request->events.emplace_back(
            std::make_unique<GenerationSchedulingObservation>(observation));
    }
    request->cv.notify_one();
}

template <class Instance>
void EngineCore<Instance>::update_recovery(const std::shared_ptr<Request>& request, bool released) {
    if (!request->recovery_pending) { return; }
    request->recovery_pending =
        !released && request->sequence && instance_.session->recovery_pending(*request->sequence);
    if (!request->recovery_pending) {
        if (!released) {
            observe_scheduling(request, GenerationSchedulingTransition::RecoveryComplete);
        }
        request_admission_check();
    }
}

template <class Instance>
typename EngineCore<Instance>::ExecutionUnit
EngineCore<Instance>::execution_unit(const Request& request) const {
    return {*request.sequence,
            request.is_control_ready()  ? UnitKind::Control
            : request.is_decode_ready() ? UnitKind::Decode
            : request.is_replaying()    ? UnitKind::Replay
                                        : UnitKind::Prefill,
            request.is_control_ready()
                ? static_cast<std::uint32_t>(request.output.pending_control_tokens().size())
            : request.is_decode_ready()
                ? (request.recovery_pending
                       ? 1U
                       : request.output.model_token_budget_remaining(request.budget->remaining()))
                : 0U};
}

template <class Instance>
std::optional<std::uint32_t> EngineCore<Instance>::next_prefill(
    const std::array<std::shared_ptr<Request>, kMaximumConcurrency>& candidates) const {
    const auto preferred = instance_.session->prefill_continuation();
    if (preferred)
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = candidates[lane];
            if (request && request->sequence == preferred && !request->capture_pending &&
                (request->is_prefilling() || request->is_replaying()))
                return lane;
        }
    return scheduler_.next_prefill(candidates, max_concurrency_);
}

template <class Instance>
typename EngineCore<Instance>::ResidentReservation
EngineCore<Instance>::reserve_resident_unit(std::uint32_t lane) {
    const auto request = slots_[lane];
    if (!request || !request->sequence || request->capture_pending ||
        request->cancelled.load(std::memory_order_acquire) ||
        instance_.session->context_blocks(*request->sequence) ||
        instance_.session->execution_pending(*request->sequence) ||
        !(request->is_control_ready() || request->is_decode_ready() ||
          request->is_prefilling() || request->is_replaying()))
        return ResidentReservation::Unavailable;

    const auto unit = execution_unit(*request);
    std::optional<typename ResourceManagement::ReclaimCursor> cursor;
    for (;;) {
        const auto result = instance_.session->reserve_units({&unit, 1});
        if (result) {
            reserved_units_[lane] = UnitPermit{unit, *request->residency};
            return ResidentReservation::Ready;
        }
        if (result.status == UnitReservationStatus::ExecutionUnavailable ||
            instance_.session->has_context_operation())
            return ResidentReservation::Unavailable;
        // Only the turn owner may change capacity ownership under pressure.
        if (scheduler_.progress_owner() && scheduler_.progress_owner() != request->id) {
            scheduler_.require_progress(request->id);
            return ResidentReservation::Unavailable;
        }
        if (!cursor) cursor.emplace(resources_.begin_reclaim(*instance_.session));
        const auto reclaimed = reclaim(result.assessment, false, {}, &*cursor);
        if (reclaimed == ReclaimProgress::Transferring || reclaimed == ReclaimProgress::Deferred)
            return ResidentReservation::Pending;
        if (reclaimed == ReclaimProgress::Changed || reclaimed == ReclaimProgress::Retry) continue;
        if (request->recovery_pending)
            throw std::logic_error("protected recovery exceeded its Native reservation");
        scheduler_.require_progress(request->id);
        if (pause_resident(lane, PauseGoal::RelocateOnly, &result.assessment) ||
            pause_reclaim_victim(result.assessment))
            return ResidentReservation::Pending;
        throw std::logic_error("progress owner cannot obtain its legal unit");
    }
}

template <class Instance>
void EngineCore<Instance>::reserve_resident_units(ReservationScope scope) {
    reserved_units_.fill(std::nullopt);
    const auto owner = scheduler_.progress_owner();
    std::optional<std::uint32_t> priority;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = slots_[lane];
        if (request && request->sequence && request->id == owner) priority = lane;
    }
    // Capacity released for a suspended owner cannot be borrowed by resident growth.
    if (owner && !priority) return;
    const auto priority_prefill =
        priority && (slots_[*priority]->is_prefilling() || slots_[*priority]->is_replaying())
            ? priority
            : std::nullopt;
    if (priority_prefill) {
        const auto continuation = instance_.session->prefill_continuation();
        if (continuation && slots_[*priority]->sequence != continuation)
            instance_.session->yield_prefill(*continuation);
    }
    const auto prefill = priority_prefill ? priority_prefill : next_prefill(slots_);
    std::vector<std::uint32_t> candidates;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = slots_[lane];
        if (!request) continue;
        if (prefill == lane || (scope == ReservationScope::Round &&
                                (request->is_control_ready() || request->is_decode_ready())))
            candidates.push_back(lane);
    }
    std::sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
        if (a == b) return false;
        if (priority == a || priority == b) return priority == a;
        return slots_[a]->id < slots_[b]->id;
    });
    const auto primary_count = candidates.size();
    if (prefill) {
        auto remaining = slots_;
        remaining[*prefill].reset();
        while (const auto fallback = scheduler_.next_prefill(remaining, max_concurrency_)) {
            candidates.push_back(*fallback);
            remaining[*fallback].reset();
        }
    }
    // Necessary Control/Decode capacity precedes optional capture reservations.
    for (std::size_t i = 0; i < primary_count; ++i) {
        const auto lane   = candidates[i];
        const auto result = reserve_resident_unit(lane);
        if (result == ResidentReservation::Pending ||
            (priority == lane && result != ResidentReservation::Ready))
            return;
    }
    const auto prepare_prefill = [&](std::uint32_t lane) {
        if (!reserved_units_[lane]) return false;
        if (prepare_semantic_capture(slots_[lane])) return true;
        reserved_units_[lane].reset();
        return false;
    };
    if (prefill && prepare_prefill(*prefill)) return;
    if (priority_prefill) return;
    // A reserved unit is not a selected Prefill until its capture preparation is ready.
    for (std::size_t i = primary_count; i < candidates.size(); ++i) {
        const auto lane   = candidates[i];
        const auto result = reserve_resident_unit(lane);
        if (result == ResidentReservation::Pending) return;
        if (result == ResidentReservation::Ready && prepare_prefill(lane)) return;
    }
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::unit_runnable(std::uint32_t lane) const {
    const auto& request = slots_[lane];
    const auto& permit  = reserved_units_[lane];
    if (!permit || !request || !request->sequence || !request->residency ||
        request->model_state == EngineRequestState::Pausing ||
        request->cancelled.load(std::memory_order_acquire) ||
        instance_.session->context_blocks(*request->sequence) ||
        instance_.session->execution_pending(*request->sequence))
        return false;
    const auto current = execution_unit(*request);
    return permit->residency == *request->residency && permit->unit.sequence == current.sequence &&
           permit->unit.kind == current.kind && permit->unit.tokens == current.tokens;
}

template <class Instance>
void EngineCore<Instance>::begin_cycle(HostPhaseMeasurement& boundary) {
    control_executed_.fill(false);
    (void)expire_pending_requests();
    if (!instance_.session->has_context_operation()) {
        for (const auto& request : slots_) {
            if (request && request->capture_pending &&
                !instance_.session->execution_pending(*request->sequence)) {
                reserve_active_capture(request, request->post_capture_state);
                break;
            }
        }
    }
    (void)settle_terminal_requests(boundary);
    cancel_active_requests(snapshot_cancellations(), boundary);
    for (auto it = paused_.begin();
         !instance_.session->has_context_operation() && it != paused_.end();) {
        const auto request = *it;
        if (!request->cancelled.load(std::memory_order_acquire)) {
            ++it;
            continue;
        }
        request->suspended.reset();
        it = paused_.erase(it);
        complete_cancelled(request);
    }
    const bool recovering = std::any_of(slots_.begin(), slots_.end(),
                                        [](const auto& r) { return r && r->recovery_pending; }) ||
                            (materializing_ && materializing_->resumed);
    if (!instance_.session->has_context_operation() &&
        scheduler_.advance_service(!paused_.empty(), recovering)) {
        // An unused permit belongs to the previous turn. Retained recovery capacity is
        // released only by Native after actual progress, never by this handoff.
        for (const auto& request : slots_)
            if (request && request->sequence)
                instance_.session->release_units({&*request->sequence, 1});
        admission_decision_.reset();
        request_admission_check();
    }
    if (!recovering && scheduler_.progress_owner()) {
        (void)try_admit_one(true);
        if (instance_.session->has_context_operation()) (void)poll_execution(boundary);
    }
    reserve_resident_units();
    if (!scheduler_.progress_owner() && paused_.empty() && !context_owner_ && !recovering) {
        if (!admission_decision_ && consume_admission_check()) {
            ++admission_generation_;
            scheduler_.begin_admission_scan();
        }
        if (admission_decision_ || scheduler_.admission_scan_pending()) {
            (void)try_admit_one(false);
        }
    }
    if (instance_.session->has_context_operation()) {
        (void)poll_execution(boundary);
        if (!instance_.session->has_context_operation()) reserve_resident_units();
    }
    cycle_step_ = CycleStep::Control;
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::advance_cycle() {
    if (compute_pending()) return false;
    set_host_work_class(HostWorkClass::Control);
    bool submitted = false;
    while (!compute_ && cycle_step_ != CycleStep::Boundary) {
        switch (cycle_step_) {
        case CycleStep::Control: {
            auto all = scheduler_.build_control_membership(slots_, max_concurrency_);
            ControlMembership selected;
            selected.row_stride = all.row_stride;
            for (std::size_t row = 0; row < all.size; ++row) {
                if (!unit_runnable(all.lanes[row])) continue;
                selected.lanes[selected.size]       = all.lanes[row];
                selected.sequences[selected.size++] = all.sequences[row];
                const auto begin                    = all.tokens.begin() + row * all.row_stride;
                selected.tokens.insert(selected.tokens.end(), begin, begin + all.row_stride);
            }
            if (!selected.empty()) {
                set_host_work_class(HostWorkClass::Control);
                if (!start_control(selected)) return submitted;
                submitted = true;
            }
            cycle_step_ = CycleStep::Decode;
            break;
        }
        case CycleStep::Decode: {
            const auto all = scheduler_.build_round_membership(slots_, max_concurrency_);
            RoundMembership selected;
            for (std::size_t row = 0; row < all.size; ++row) {
                if (!unit_runnable(all.lanes[row]) || control_executed_[all.lanes[row]]) continue;
                selected.lanes[selected.size]     = all.lanes[row];
                selected.sequences[selected.size] = all.sequences[row];
                selected.budgets[selected.size++] = all.budgets[row];
            }
            if (!selected.empty()) {
                set_host_work_class(HostWorkClass::Decode, selected.lane_span());
                if (!start_decode(selected)) return submitted;
                submitted = true;
            }
            cycle_step_ = CycleStep::Prefill;
            break;
        }
        case CycleStep::Prefill: {
            if (instance_.session->has_context_operation()) {
                auto boundary = begin_host_phase();
                (void)poll_execution(boundary);
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
            }
            reserve_resident_units(ReservationScope::Prefill);
            auto eligible = slots_;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (!unit_runnable(lane)) eligible[lane].reset();
            }
            if (const auto lane = next_prefill(eligible)) {
                set_host_work_class(HostWorkClass::Prefill);
                if (!start_prefill(*lane)) return submitted;
                submitted = true;
            }
            cycle_step_ = CycleStep::Complete;
            break;
        }
        case CycleStep::Complete:
            publish_runtime_stats();
            cycle_step_ = CycleStep::Boundary;
            break;
        case CycleStep::Boundary:
            break;
        }
    }
    return submitted;
}

template <class Instance>
bool EngineCore<Instance>::process_measurement() {
    if (!resident_empty() || !paused_.empty() || materializing_ || context_owner_ ||
        admission_decision_ || execution_outstanding())
        return false;
    MeasurementCommand command;
    {
        std::lock_guard lock(queue_mutex_);
        if (!measurement_ || !pending_.empty() || preparing_requests_) return false;
        command = measurement_->command;
    }
    EngineSnapshot result;
    std::exception_ptr error;
    try {
        instance_.executor->measure(command);
        publish_runtime_stats();
        result = snapshot();
    } catch (...) {
        error = std::current_exception();
        if (!instance_.executor->failed()) {
            try {
                instance_.executor->measure({MeasurementAction::Cancel});
            } catch (...) {
                instance_.executor->fail("failed to close measurement after a boundary error");
            }
        }
        if (instance_.executor->failed()) std::rethrow_exception(error);
    }
    std::unique_ptr<MeasurementRequest> completed;
    {
        std::lock_guard lock(queue_mutex_);
        completed = std::move(measurement_);
    }
    if (error)
        completed->completion.set_exception(error);
    else
        completed->completion.set_value(std::move(result));
    queue_cv_.notify_all();
    return true;
}

template <class Instance>
void EngineCore<Instance>::worker_loop() noexcept {
    std::uint64_t scheduled_revision = 0;
    for (;;) {
        {
            std::unique_lock lock(queue_mutex_);
            if (!stopping_ && pending_.empty() && resident_empty() && paused_.empty() &&
                !execution_outstanding()) {
                queue_cv_.wait(lock,
                               [&] { return stopping_ || !pending_.empty() || measurement_; });
            }
            if (stopping_) {
                lock.unlock();
                fail_all(std::make_exception_ptr(RequestError(
                    RequestErrorKind::Unavailable, "inference engine is shutting down")));
                return;
            }
        }
        bool progressed  = false;
        bool outstanding = false;
        try {
            if (compute_) {
                set_host_work_class(compute_->kind == UnitKind::Decode    ? HostWorkClass::Decode
                                    : compute_->kind == UnitKind::Control ? HostWorkClass::Control
                                                                          : HostWorkClass::Prefill,
                                    compute_->kind == UnitKind::Decode
                                        ? std::span<const std::uint32_t>(compute_->lanes)
                                        : std::span<const std::uint32_t>{});
            }
            auto boundary = begin_host_phase();
            progressed    = poll_execution(boundary);
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            const auto revision = schedule_revision_.load(std::memory_order_acquire);
            if (revision != published_revision_) publish_runtime_stats();
            const bool schedule = progressed || revision != scheduled_revision ||
                                  !execution_outstanding() || Clock::now() >= pending_deadline_;
            if (cycle_step_ == CycleStep::Boundary && !compute_pending() && schedule) {
                scheduled_revision = revision;
                set_host_work_class(HostWorkClass::Control);
                boundary = begin_host_phase();
                begin_cycle(boundary);
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
            }
            if (advance_cycle()) {
                progressed = true;
                publish_runtime_stats();
            }
            {
                ExecutionCallScope call(*this, CoordinatorCall::Flush);
                instance_.executor->flush();
                call.finish();
            }
            outstanding = execution_outstanding();
            if (!outstanding) progressed |= process_measurement();
        } catch (...) {
            log_worker_failure(std::current_exception());
            fail_all(std::current_exception());
            return;
        }
        if (outstanding) {
            const auto exposed = active_exposure_set();
            const auto started = Clock::now();
            std::this_thread::yield();
            record_coordinator_call(CoordinatorCall::Yield, elapsed_ns(started, Clock::now()),
                                    exposed);
        }
        if (!outstanding && !progressed) {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(1));
        }
    }
}

} // namespace ninfer::runtime
