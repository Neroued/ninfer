#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
using detail::seconds;

PrefillProgress Program::advance_prefill(SequenceHandle handle, runtime::ExecutionTiming* failed,
                                         runtime::TokenMaskProvider* masks) {
    auto& p = *impl_;
    auto& s = p.sequence(handle);
    if (p.pending_transaction || s.phase != detail::Phase::Prefill || !s.permit ||
        s.capture_pending)
        throw std::logic_error("prefill has no executable unit");
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    const auto started = detail::Clock::now();
    PrefillProgress out;
    out.summary = {s.request->summary.prompt_tokens, s.reused,
                   s.reused ? PrefixReusePath::Checkpoint : PrefixReusePath::Root};
    const std::array rows{&s};
    try {
        const auto f = s.continuation.metadata.main_frontier;
        Tensor sample;
        if (f < s.request->summary.prompt_tokens) {
            const auto end = p.window_end(s, f, s.request->summary.prompt_tokens);
            sample         = p.forward(rows, end - f, true);
            p.adopt_forward(rows, end - f);
            sample                      = sample.slice(1, end - f - 1, 1);
            out.processed_prompt_tokens = end - f;
            s.recovery_peak             = 0;
            s.capture_pending = s.capture && s.request->captures[s.next_capture].frontier == end;
        } else {
            sample = p.resources.states->sample_hidden().slice(1, p.selectors(s).source, 1);
        }
        out.complete = s.continuation.metadata.main_frontier == s.request->summary.prompt_tokens &&
                       !s.capture_pending;
        if (out.complete) {
            out.pending.emplace(p.sample(rows, sample, true, masks, timing));
        } else {
            timing.begin_wait();
            p.ple.wait_compute();
            timing.end_wait();
            p.settle(s);
        }
        if (out.processed_prompt_tokens)
            timing.include({.gpu_elapsed_ns = static_cast<std::uint64_t>(
                                double(p.prefill_timer.elapsed_ms()) * 1.0e6 + 0.5)});
        out.capture_ready         = s.capture_pending;
        const auto vision_seconds = p.collect_vision_timing(s);
        s.timings.prefill_seconds += std::max(0.0, seconds(started) - vision_seconds);
        out.timing = timing.finish();
        return out;
    } catch (...) {
        p.device.synchronize();
        p.clear(s);
        p.sequences[handle.lane].reset();
        throw;
    }
}

PendingBatch Program::decode(std::span<const SequenceHandle> members,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed, runtime::TokenMaskProvider* masks) {
    auto& p = *impl_;
    if (p.pending_transaction || members.empty() || members.size() > p.shape.max_concurrency ||
        budgets.size() != members.size())
        throw std::logic_error("invalid decode batch");
    std::vector<detail::Sequence*> rows;
    bool speculative = false;
    for (std::size_t i = 0; i < members.size(); ++i) {
        auto& s = p.sequence(members[i]);
        if (s.phase != detail::Phase::Active || !s.permit ||
            !budgets[i].generated_tokens_remaining ||
            s.identity.tokens.size() != s.continuation.metadata.main_frontier + 1 ||
            std::find(rows.begin(), rows.end(), &s) != rows.end())
            throw std::logic_error("row is not decode-ready");
        rows.push_back(&s);
        speculative |= p.shape.draft_tokens && budgets[i].generated_tokens_remaining > 1 &&
                       s.continuation.metadata.main_frontier + 1 < p.shape.max_context;
    }
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    const auto started = detail::Clock::now();
    try {
        if (speculative) {
            auto pending = p.mtp_decode(rows, budgets, masks, timing);
            for (auto* s : rows) s->timings.decode_seconds += seconds(started);
            pending.timing_ = timing.finish();
            return pending;
        }
        const auto hidden = p.forward(rows, 1, false, true);
        p.adopt_forward(rows, 1);
        auto pending = p.sample(rows, hidden, false, masks, timing);
        for (auto* s : rows) s->timings.decode_seconds += seconds(started);
        pending.timing_ = timing.finish();
        return pending;
    } catch (...) {
        p.device.synchronize();
        for (auto* s : rows) {
            const auto lane = s->lane;
            p.clear(*s);
            p.sequences[lane].reset();
        }
        throw;
    }
}

DiscardResult Program::abort_pending(PendingBatch&& batch) noexcept {
    auto& p = *impl_;
    if (batch.owner_ != &p || batch.transaction_ != p.pending_transaction ||
        batch.members_ != p.pending_members)
        return {};
    const auto count = batch.members_.size();
    for (const auto h : batch.members_)
        if (p.valid(h)) {
            p.clear(p.sequence(h));
            p.sequences[h.lane].reset();
        }
    batch.owner_          = nullptr;
    p.pending_transaction = 0;
    p.pending_members.clear();
    p.pending_verify.reset();
    return {runtime::ConsumeStatus::Consumed, count};
}

FinishResult Program::finish(SequenceHandle h) noexcept {
    auto& p = *impl_;
    if (!p.valid(h) || p.sequence(h).phase != detail::Phase::Finished) return {};
    try {
        auto& s = p.sequence(h);
        FinishResult out{runtime::ConsumeStatus::Consumed, s.timings, s.speculative, {}};
        if (s.request->summary.publish_continuation)
            out.checkpoint = p.detach(s, runtime::CheckpointRole::Continuation);
        p.clear(s);
        p.sequences[h.lane].reset();
        return out;
    } catch (...) { std::terminate(); }
}

AbortResult Program::abort(SequenceHandle h) noexcept {
    auto& p = *impl_;
    if (!p.valid(h)) return {};
    auto& s = p.sequence(h);
    AbortResult out{runtime::ConsumeStatus::Consumed, s.timings, s.speculative};
    p.clear(s);
    p.sequences[h.lane].reset();
    return out;
}

ReplayProgress Program::advance_replay(SequenceHandle h, runtime::ExecutionTiming* failed) {
    auto& p = *impl_;
    auto& s = p.sequence(h);
    if (s.phase != detail::Phase::Replay || !s.permit) throw std::logic_error("no Replay unit");
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    const auto f   = s.continuation.metadata.main_frontier;
    const auto end = p.window_end(s, f, s.replay_target);
    const std::array rows{&s};
    try {
        if (end > f) {
            (void)p.forward(rows, end - f, true);
            p.adopt_forward(rows, end - f);
            timing.begin_wait();
            p.ple.wait_compute();
            timing.end_wait();
            p.collect_vision_timing(s);
        }
        const bool complete = end == s.replay_target;
        if (complete) s.phase = s.resume_phase;
        p.settle(s);
        return {false, end - f, complete, timing.finish()};
    } catch (...) {
        p.device.synchronize();
        p.clear(s);
        p.sequences[h.lane].reset();
        throw;
    }
}

runtime::ExecutionTiming Program::append_forced_tokens(
    std::span<const SequenceHandle> members, std::span<const TokenId> forced, std::uint32_t stride,
    std::span<const std::optional<std::uint32_t>> splits, runtime::ExecutionTiming* failed) {
    auto& p = *impl_;
    if (p.pending_transaction || members.empty() || members.size() > p.shape.max_concurrency ||
        !stride || forced.size() != members.size() * stride || splits.size() != members.size())
        throw std::logic_error("invalid control window");
    std::vector<detail::Sequence*> rows;
    rows.reserve(members.size());
    for (std::size_t row = 0; row < members.size(); ++row) {
        auto& s = p.sequence(members[row]);
        if (s.phase != detail::Phase::Active || !s.permit ||
            s.permit->kind != ExecutionUnitKind::Control ||
            s.identity.tokens.size() != s.continuation.metadata.main_frontier + 1 ||
            std::uint64_t(s.continuation.metadata.main_frontier) + stride > s.permitted_frontier ||
            (splits[row] && (!*splits[row] || *splits[row] > stride)) ||
            std::find(rows.begin(), rows.end(), &s) != rows.end())
            throw std::logic_error("control row is not executable");
        rows.push_back(&s);
    }
    for (const auto id : forced)
        if (id < 0 || std::uint32_t(id) >= p.parameters.profile->public_tokens)
            throw std::invalid_argument("forced ID is outside vocabulary");
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    const auto started = detail::Clock::now();
    try {
        for (std::size_t row = 0; row < rows.size(); ++row) {
            auto& s        = *rows[row];
            const auto ids = forced.subspan(row * stride, stride);
            s.identity.tokens.insert(s.identity.tokens.end(), ids.begin(), ids.end());
            s.identity.positions.append_generated(stride, s.continuation.metadata.rope_delta,
                                                  splits[row]);
            s.identity.digests.append_generated(ids, s.continuation.metadata.rope_delta,
                                                splits[row]);
        }
        if (stride == 1) {
            (void)p.forward(rows, 1, false);
            p.adopt_forward(rows, 1);
            timing.begin_wait();
            p.ple.wait_compute();
            timing.end_wait();
        } else {
            for (auto* sequence : rows) {
                auto& s           = *sequence;
                const auto target = s.continuation.metadata.main_frontier + stride;
                const std::array single{&s};
                while (s.continuation.metadata.main_frontier < target) {
                    const auto f   = s.continuation.metadata.main_frontier;
                    const auto end = p.window_end(s, f, target);
                    (void)p.forward(single, end - f, true);
                    p.adopt_forward(single, end - f);
                    timing.begin_wait();
                    p.ple.wait_compute();
                    timing.end_wait();
                    timing.resume_submit();
                }
            }
        }
        for (std::size_t row = 0; row < rows.size(); ++row) {
            auto& s = *rows[row];
            p.increment_counts(s, forced.subspan(row * stride, stride));
            s.recovery_peak = 0;
            p.settle(s);
            s.timings.decode_seconds += seconds(started);
        }
        return timing.finish();
    } catch (...) {
        p.device.synchronize();
        for (auto* sequence : rows) {
            const auto lane = sequence->lane;
            p.clear(*sequence);
            p.sequences[lane].reset();
        }
        throw;
    }
}

std::vector<float> Program::causal_score(qwen::PreparedPrompt&& prompt,
                                         std::uint32_t first_target) {
    auto& p = *impl_;
    if (p.shape.purpose != EnginePurpose::CausalScoring)
        throw std::logic_error("not a scoring Program");
    runtime::ResolvedExecutionOptions o;
    o.allow_prefix_reuse = false;
    auto base            = plan_request(std::move(prompt), o);
    const auto total     = base.summary().prompt_tokens;
    if (!first_target || first_target >= total)
        throw std::invalid_argument("invalid scoring target range");
    if (!start_binding(base, {0}, {})) throw std::logic_error("scoring resources unavailable");
    std::optional<SequenceHandle> handle;
    while (has_context_operation()) {
        const auto progress = poll_context({});
        if (progress.sequence) handle = progress.sequence;
    }
    auto& s = p.sequence(*handle);
    std::vector<float> result;
    result.reserve(total - first_target);
    try {
        while (s.continuation.metadata.main_frontier + 1 < total) {
            const ExecutionUnit unit{*handle, ExecutionUnitKind::Prefill, 0};
            if (!reserve_units({&unit, 1})) throw std::logic_error("scoring KV capacity exhausted");
            const auto begin = s.continuation.metadata.main_frontier;
            const auto end   = p.window_end(s, begin, total - 1);
            const std::array rows{&s};
            const auto hidden = p.forward(rows, end - begin, true);
            p.adopt_forward(rows, end - begin);
            p.ple.wait_compute();
            for (auto at = std::max(begin, first_target - 1); at < end;) {
                auto scope   = p.resources.workspace.scope();
                const auto n = std::min(1024U, end - at);
                auto buffers = execution::workspace::score(
                    p.resources.workspace, p.parameters.model.config().text.vocabulary, n);
                auto* ids = static_cast<TokenId*>(
                    p.host_tensor(p.resources.plan.persistent.buffers.batch.ids).data);
                std::copy_n(s.identity.tokens.begin() + at + 1, n, ids);
                copy_bytes_async({buffers.targets.data, n * 4ULL, p.cuda.device()},
                                 {ids, n * 4ULL, Device::cpu(), true}, p.cuda);
                execution::project_head(hidden.slice(1, at - begin, n), p.parameters.text.head,
                                        buffers.logits, p.resources.workspace, p.cuda);
                ops::target_logprobs(buffers.logits, buffers.targets,
                                     p.parameters.profile->public_tokens, buffers.logprobs, p.cuda);
                copy_bytes_async(p.resources.egress().subspan(0, n * 4ULL),
                                 {buffers.logprobs.data, n * 4ULL, p.cuda.device()}, p.cuda);
                p.ple.wait_compute();
                const auto* values = static_cast<const float*>(p.resources.egress().data);
                result.insert(result.end(), values, values + n);
                at += n;
            }
            p.settle(s);
        }
        p.clear(s);
        p.sequences[0].reset();
        return result;
    } catch (...) {
        p.device.synchronize();
        p.clear(s);
        p.sequences[0].reset();
        throw;
    }
}

} // namespace ninfer::models::qwen4_exp
