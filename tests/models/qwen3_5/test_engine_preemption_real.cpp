#include "ninfer/engine.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kPromptTokens    = 192;
constexpr std::uint32_t kOutputTokens    = 256;
constexpr std::uint32_t kCapacity        = 512;
constexpr std::size_t kSnapshotHostBytes = 512ULL << 20;
const ninfer::GenerationObservationOptions kObservations{
    .phase_timings = true, .live_timings = true, .prompt_progress = true};

enum class CancelStage {
    None,
    Paused,
    Replay,
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string_view setting(const char* name, std::string_view fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

ninfer::SpeculativeBackend backend(std::string_view name) {
    if (name == "none") { return ninfer::SpeculativeBackend::None; }
    if (name == "mtp") { return ninfer::SpeculativeBackend::Mtp; }
    if (name == "dflash") { return ninfer::SpeculativeBackend::DFlash; }
    if (name == "dflash2") { return ninfer::SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("NINFER_TEST_BACKEND must be none, mtp, dflash or dflash2");
}

std::string grammar_prefix() {
    std::string prefix;
    for (char c = 'a'; c <= 'z'; ++c) prefix.append(17 + c - 'a', c);
    return prefix;
}

ninfer::RequestOptions request(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    if (setting("NINFER_TEST_CONSTRAINT", "none") != "none") {
        std::string source = "root ::= ";
        for (char c = 'a'; c <= 'z'; ++c) {
            source += "\"" + std::string(1, c) + "\"{" + std::to_string(17 + c - 'a') + "} ";
        }
        if (setting("NINFER_TEST_CONSTRAINT", "none") == "json_schema") {
            std::string pattern = "^";
            for (char c = 'a'; c <= 'z'; ++c)
                pattern += std::string(1, c) + "{" + std::to_string(17 + c - 'a') + "}";
            pattern += "z{65536}$";
            options.constraint = ninfer::OutputConstraint::json_schema(
                nlohmann::json{{"type", "string"}, {"pattern", pattern}}.dump());
        } else {
            require(setting("NINFER_TEST_CONSTRAINT", "none") == "grammar",
                    "unknown test constraint");
            options.constraint = ninfer::OutputConstraint::grammar(source + "\"z\"{65536}");
        }
        options.stop.include_model_defaults = true;
        options.output.raw                  = false;
    }
    return options;
}

ninfer::EngineOptions engine_options(const std::filesystem::path& artifact,
                                     ninfer::SpeculativeBackend selected, bool snapshot) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = kCapacity;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk        = 128;
    options.max_concurrency      = 2;
    options.max_pending_requests = 2;
    options.speculative.backend  = selected;
    if (selected != ninfer::SpeculativeBackend::None) {
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    // History is deliberately disabled: the two cases exercise only the requests' own recovery.
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = snapshot ? kSnapshotHostBytes : 0;
    return options;
}

struct Timeline {
    void start(std::size_t row) {
        std::lock_guard lock(mutex);
        started[row] = ++ordinal;
    }

    void output(std::size_t row, bool terminal) {
        std::lock_guard lock(mutex);
        if (first_output[row] == 0) { first_output[row] = ++ordinal; }
        if (terminal && finished[row] == 0) { finished[row] = ++ordinal; }
    }

    void complete(std::size_t row) {
        std::lock_guard lock(mutex);
        if (finished[row] == 0) { finished[row] = ++ordinal; }
    }

    void validate() const {
        const auto first_finish = std::min(finished[0], finished[1]);
        require(first_finish != 0, "a streaming request never reached its terminal boundary");
        for (std::size_t row = 0; row < 2; ++row) {
            require(started[row] != 0 && first_output[row] > started[row] &&
                        first_output[row] < first_finish,
                    "fixture did not observe both starts and first outputs before a terminal");
        }
    }

    std::mutex mutex;
    std::uint64_t ordinal = 0;
    std::array<std::uint64_t, 2> started{};
    std::array<std::uint64_t, 2> first_output{};
    std::array<std::uint64_t, 2> finished{};
};

class ObservationSink final : public ninfer::OutputSink {
public:
    ObservationSink(Timeline& timeline, std::size_t row, std::function<void()> on_start = {})
        : timeline_(timeline), row_(row), on_start_(std::move(on_start)) {}

    void start(ninfer::GenerationStart start) override {
        require(++starts_ == 1, "resume published GenerationStart more than once");
        require(start.prompt.prompt_tokens == kPromptTokens && start.reused_prompt_tokens == 0,
                "initial admission reported an unexpected prompt or history hit");
        timeline_.start(row_);
        if (on_start_) { on_start_(); }
    }

    void progress(ninfer::PromptProgress progress) override {
        require(starts_ == 1 && !timing_seen_ && progress.total_prompt_tokens == kPromptTokens &&
                    progress.reused_prompt_tokens == 0 &&
                    progress.processed_prompt_tokens >= processed_ &&
                    progress.processed_prompt_tokens <= kPromptTokens &&
                    progress.elapsed_ns >= progress_elapsed_ns_,
                "pause/replay repeated or regressed initial prompt progress");
        processed_           = progress.processed_prompt_tokens;
        progress_elapsed_ns_ = progress.elapsed_ns;
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        require(starts_ == 1 && processed_ == kPromptTokens &&
                    timing.generated_tokens > last_timing_.generated_tokens &&
                    timing.generated_tokens <= kOutputTokens &&
                    (!timing_seen_ ||
                     (timing.prompt_elapsed_ns == last_timing_.prompt_elapsed_ns &&
                      timing.generation_elapsed_ns >= last_timing_.generation_elapsed_ns)),
                "pause/replay duplicated output accounting or reset request timing");
        timing_seen_ = true;
        last_timing_ = timing;
        timeline_.output(row_, timing.generated_tokens == kOutputTokens);
    }

    void publish(ninfer::OutputDelta delta) override {
        require(timing_seen_, "output delta preceded its committed-token observation");
        (delta.channel == ninfer::OutputChannel::Reasoning ? reasoning_ : content_) += delta.text;
    }

    [[nodiscard]] std::uint32_t committed_tokens() const noexcept {
        return last_timing_.generated_tokens;
    }

    void validate(const ninfer::GenerationResult& result, ninfer::SpeculativeBackend selected,
                  bool cancelled = false) const {
        require(starts_ == 1 && timing_seen_ && processed_ == kPromptTokens &&
                    last_timing_.generated_tokens == result.generated_token_ids.size(),
                "stream did not publish one complete request lifetime");
        require(result.prompt.prompt_tokens == kPromptTokens &&
                    (cancelled ? result.generated_token_ids.size() > 1 &&
                                     result.generated_token_ids.size() < kOutputTokens &&
                                     result.finish_reason == ninfer::FinishReason::Cancelled
                               : result.generated_token_ids.size() == kOutputTokens &&
                                     result.finish_reason == ninfer::FinishReason::OutputLimit),
                "resumed request did not honor its original output budget");
        // Compare only the two publication views of this request, never different math paths.
        if (setting("NINFER_TEST_CONSTRAINT", "none") != "none") {
            const auto expected =
                (setting("NINFER_TEST_CONSTRAINT", "none") == "json_schema" ? std::string("\"")
                                                                            : std::string{}) +
                grammar_prefix();
            require(!content_.empty() && reasoning_.empty() &&
                        (content_.size() <= expected.size()
                             ? expected.starts_with(content_)
                             : content_.starts_with(expected) &&
                                   content_.find_first_not_of('z', expected.size()) ==
                                       std::string::npos),
                    "recovery or cancellation changed the grammar position");
        }
        require(content_ == result.content && reasoning_ == result.reasoning,
                "stream and terminal response disagree after recovery");
        require(result.reused_prompt_tokens == 0 &&
                    result.prefix_reuse_path == ninfer::PrefixReusePath::Root,
                "self recovery was incorrectly reported as an initial history hit");
        require(result.speculative.backend == selected &&
                    (selected == ninfer::SpeculativeBackend::None ||
                     (result.speculative.enabled && result.speculative.draft_window == 3 &&
                      result.speculative.rounds + result.speculative.fallback_steps != 0)),
                "request did not execute the selected backend");
        require(
            result.timings.first_token_seconds > 0.0 && result.timings.prefill_seconds > 0.0 &&
                result.timings.decode_seconds > 0.0 && result.timings.prompt_wall_seconds > 0.0 &&
                result.timings.generation_wall_seconds > 0.0 &&
                result.engine_timing.prefill_units > 0 && result.engine_timing.decode_rounds > 0,
            "request lost accumulated prefill/decode timings or work when it stopped");
        require((!content_.empty() || !reasoning_.empty()) && result.first_output_timing,
                "nonempty streamed output has no first-output snapshot");
        const auto& first           = *result.first_output_timing;
        constexpr double epsilon    = 1.0e-9;
        const double engine_elapsed = result.timings.total_seconds - result.timings.prepare_seconds;
        const double partitioned = first.engine.queue_wait_seconds + first.initial_binding_seconds +
                                   static_cast<double>(first.scheduling.paused_ns) * 1.0e-9;
        require(first.elapsed_seconds > 0.0 && first.elapsed_seconds < engine_elapsed + epsilon &&
                    partitioned <= first.elapsed_seconds + epsilon,
                "first-output wall-time boundaries overlap or exceed the completed lifetime");
        require(first.computed_prefill_tokens <= result.computed_prefill_tokens &&
                    first.computed_prefill_tokens == kPromptTokens &&
                    first.prefill.gpu_seconds > 0.0,
                "first-output snapshot omitted completed initial prefill or GPU work");
        require(first.engine.decode_rounds < result.engine_timing.decode_rounds &&
                    first.engine.program_submit_exposed_seconds <=
                        result.engine_timing.program_submit_exposed_seconds + epsilon &&
                    first.engine.device_wait_exposed_seconds <=
                        result.engine_timing.device_wait_exposed_seconds + epsilon,
                "continued decoding rewrote the frozen first-output work counters");
        require(first.scheduling.preemptions == 0 && first.scheduling.replay_restores == 0 &&
                    first.scheduling.snapshot_restores == 0 &&
                    first.scheduling.replayed_tokens == 0,
                "post-output pressure recovery leaked into the first-output snapshot");
        std::uint64_t first_d2h = 0, first_h2d = 0;
        for (const auto& resource : first.context_transfers) {
            first_d2h += resource[0].bytes;
            first_h2d += resource[1].bytes;
        }
        require(first_d2h <= result.scheduling.device_to_host_bytes &&
                    first_h2d <= result.scheduling.host_to_device_bytes,
                "first-output transfers exceed the request's completed transfer accounting");
    }

private:
    Timeline& timeline_;
    std::size_t row_;
    std::function<void()> on_start_;
    unsigned starts_                   = 0;
    bool timing_seen_                  = false;
    std::uint32_t processed_           = 0;
    std::uint64_t progress_elapsed_ns_ = 0;
    ninfer::GenerationTimingObservation last_timing_;
    std::string content_;
    std::string reasoning_;
};

void settled(const ninfer::RuntimeStats& stats, const ninfer::MemorySummary& memory) {
    require(stats.running_requests == 0 && stats.waiting_requests == 0 &&
                stats.paused_requests == 0 && stats.replaying_requests == 0 &&
                stats.prefilling_requests == 0 && stats.decode_ready_requests == 0 &&
                stats.materializing_requests == 0 && stats.capture_pending_requests == 0 &&
                stats.terminal_pending_requests == 0,
            "completed recovery left live scheduling membership");
    require(stats.device_state_occupied_slots == 0 && stats.device_main_kv_occupied_pages == 0 &&
                stats.device_backend_kv_occupied_pages == 0 &&
                stats.host_context_occupied_bytes == 0 && stats.host_context_reserved_bytes == 0 &&
                memory.host_context_occupied_bytes == 0 && memory.host_context_reserved_bytes == 0,
            "completed recovery retained physical resources with history disabled");
}

void exercise(const std::filesystem::path& artifact, ninfer::SpeculativeBackend selected,
              std::string_view backend_name, bool snapshot,
              CancelStage cancel_stage = CancelStage::None) {
    ninfer::Engine engine(engine_options(artifact, selected, snapshot));
    const auto memory = engine.memory_summary();
    require(memory.max_context == kCapacity && memory.kv_capacity == kCapacity,
            "Engine changed the fixed pressure workload's context capacity");
    require(memory.host_context_capacity_bytes <= (snapshot ? kSnapshotHostBytes : 0),
            "Engine exceeded the fixture's fixed Host quota");
    const auto before = engine.runtime_stats();
    std::array<unsigned, 2> first_token_counts{};
    std::array<ninfer::GenerationObservationOptions, 2> observations{kObservations, kObservations};
    for (std::size_t i = 0; i < observations.size(); ++i) {
        observations[i].first_token = [&, i](const ninfer::GenerationFirstTokenObservation& first) {
            require(first.elapsed_since_submit_seconds > 0.0 && first.prepare_seconds >= 0.0,
                    "first-token observation has invalid time boundaries");
            ++first_token_counts[i];
        };
    }

    // Main KV pages hold 64 tokens. Both 192-token prompts fit together in six of eight pages;
    // both can begin decoding, but their continued growth cannot remain resident together.
    // Each 448-token complete request fits alone, including the K=3 speculative unit.
    std::vector<ninfer::TokenId> first_prompt(kPromptTokens, 198);
    std::vector<ninfer::TokenId> second_prompt(kPromptTokens, 198);
    first_prompt.front()  = 1000;
    second_prompt.front() = 1001;
    auto first_prepared   = engine.prepare_tokens(std::move(first_prompt));
    auto second_prepared  = engine.prepare_tokens(std::move(second_prompt));

    Timeline timeline;
    std::mutex handoff_mutex;
    std::condition_variable handoff;
    std::optional<ninfer::GenerationHandle> second_handle;
    bool first_finished = false;
    std::exception_ptr first_error;
    std::optional<ninfer::GenerationResult> first_result;
    ObservationSink first_sink(timeline, 0, [&] {
        // OutputSink runs in wait()'s consumer thread. This gate submits already prepared input
        // at the observed first admission without holding up the Engine or a model callback.
        auto handle = engine.submit(std::move(second_prepared), request(kOutputTokens),
                                    ninfer::OutputConsumerMode::Streaming, observations[1]);
        {
            std::lock_guard lock(handoff_mutex);
            second_handle.emplace(std::move(handle));
        }
        handoff.notify_one();
    });
    ObservationSink second_sink(timeline, 1);
    auto first_handle = engine.submit(std::move(first_prepared), request(kOutputTokens),
                                      ninfer::OutputConsumerMode::Streaming, observations[0]);
    std::jthread first_consumer([&] {
        try {
            first_result = first_handle.wait(&first_sink);
            timeline.complete(0);
        } catch (...) { first_error = std::current_exception(); }
        {
            std::lock_guard lock(handoff_mutex);
            first_finished = true;
        }
        handoff.notify_one();
    });

    ninfer::GenerationHandle second;
    {
        std::unique_lock lock(handoff_mutex);
        handoff.wait(lock, [&] { return second_handle.has_value() || first_finished; });
        if (second_handle) { second = std::move(*second_handle); }
    }
    if (!second) {
        first_consumer.join();
        if (first_error) { std::rethrow_exception(first_error); }
        throw std::runtime_error("first admission never submitted the second request");
    }
    bool cancellation_requested = false;
    ninfer::RuntimeStats cancellation_observed;
    std::uint32_t committed_at_cancellation = 0;
    const ninfer::CancellationView cancellation([&] {
        if (cancellation_requested) { return true; }
        if (cancel_stage == CancelStage::None) { return false; }
        const auto stats    = engine.runtime_stats();
        const bool observed = cancel_stage == CancelStage::Paused
                                  ? stats.paused_requests == 1
                                  : stats.replaying_requests != 0 && stats.replayed_tokens != 0;
        if (!observed) { return false; }
        // The second request's ticket is strictly newer: its submit follows first.start().
        // With only these two requests, protecting the oldest resident makes it the only victim.
        // CancellationView is checked by the consumer, including while its stream is idle; it
        // observes actual Engine state without delaying a model callback or guessing a sleep.
        cancellation_observed     = stats;
        committed_at_cancellation = second_sink.committed_tokens();
        cancellation_requested    = true;
        return true;
    });
    const auto second_result = second.wait(&second_sink, cancellation);
    timeline.complete(1);
    first_consumer.join();
    if (first_error) { std::rethrow_exception(first_error); }
    require(first_result.has_value(), "first request did not return a result");
    timeline.validate();
    first_sink.validate(*first_result, selected);
    if (cancel_stage != CancelStage::None) {
        require(cancellation_requested,
                "cancellation fixture did not observe the requested paused/replaying state");
    }
    second_sink.validate(second_result, selected, cancel_stage != CancelStage::None);
    require(first_result->engine_request_id != second_result.engine_request_id,
            "concurrent requests lost their independent identity");

    // The physical snapshot also waits for the completed worker boundary before we inspect its
    // published counters; wait() may return as soon as the terminal response is available.
    const auto completed_memory = engine.memory_summary();
    const auto after            = engine.runtime_stats();
    settled(after, completed_memory);
    require(first_token_counts[0] == 1 && first_token_counts[1] == 1,
            "recovery or cancellation duplicated/lost first-token observations");
    require(after.prompt_tokens - before.prompt_tokens == 2 * kPromptTokens &&
                after.generated_tokens - before.generated_tokens ==
                    first_result->generated_token_ids.size() +
                        second_result.generated_token_ids.size(),
            "metrics duplicated recovered inputs or omitted first generated tokens");
    require(after.speculative_rounds - before.speculative_rounds ==
                    first_result->speculative.rounds + second_result.speculative.rounds &&
                after.speculative_draft_tokens - before.speculative_draft_tokens ==
                    first_result->speculative.drafted_tokens +
                        second_result.speculative.drafted_tokens &&
                after.speculative_accepted_tokens - before.speculative_accepted_tokens ==
                    first_result->speculative.accepted_tokens +
                        second_result.speculative.accepted_tokens,
            "live speculative counters disagree with settled work across recovery");
    require(after.decode_row_rounds - before.decode_row_rounds >
                after.decode_rounds - before.decode_rounds,
            "fixture never executed a real two-request decode batch");
    require(after.computed_prefill_tokens - before.computed_prefill_tokens == 2 * kPromptTokens &&
                after.committed_decode_tokens - before.committed_decode_tokens ==
                    first_result->generated_token_ids.size() +
                        second_result.generated_token_ids.size() - 2,
            "replay was charged to initial prefill or delivered decode tokens");

    ninfer::GenerationSchedulingStats totals;
    for (const auto* result :
         std::array<const ninfer::GenerationResult*, 2>{&*first_result, &second_result}) {
        totals.preemptions += result->scheduling.preemptions;
        totals.replay_restores += result->scheduling.replay_restores;
        totals.snapshot_restores += result->scheduling.snapshot_restores;
        totals.replayed_tokens += result->scheduling.replayed_tokens;
        totals.paused_ns += result->scheduling.paused_ns;
        totals.device_to_host_bytes += result->scheduling.device_to_host_bytes;
        totals.host_to_device_bytes += result->scheduling.host_to_device_bytes;
    }
    const char* label = cancel_stage == CancelStage::Paused   ? "cancel-paused"
                        : cancel_stage == CancelStage::Replay ? "cancel-replay"
                        : snapshot                            ? "snapshot"
                                                              : "replay";
    std::cout << label << " backend=" << backend_name << " preemptions=" << totals.preemptions
              << " snapshot_restores=" << totals.snapshot_restores
              << " replay_restores=" << totals.replay_restores
              << " replayed_tokens=" << totals.replayed_tokens << " paused_ns=" << totals.paused_ns
              << " d2h_bytes=" << totals.device_to_host_bytes
              << " h2d_bytes=" << totals.host_to_device_bytes;
    if (cancel_stage != CancelStage::None) {
        std::cout << " observed_paused=" << cancellation_observed.paused_requests
                  << " observed_replaying=" << cancellation_observed.replaying_requests
                  << " observed_replayed_tokens=" << cancellation_observed.replayed_tokens
                  << " published_at_cancel=" << committed_at_cancellation
                  << " returned_tokens=" << second_result.generated_token_ids.size();
    }
    std::cout << '\n';
    require(totals.preemptions != 0, "growth pressure never preempted a request");
    require(after.preemptions - before.preemptions == totals.preemptions &&
                after.snapshot_restores - before.snapshot_restores == totals.snapshot_restores &&
                after.replay_restores - before.replay_restores == totals.replay_restores &&
                after.replayed_tokens - before.replayed_tokens == totals.replayed_tokens,
            "request and Engine scheduling counters disagree");
    if (cancel_stage != CancelStage::None) {
        require(first_result->scheduling.preemptions == 0 &&
                    second_result.scheduling.preemptions != 0 &&
                    second_result.scheduling.paused_ns != 0,
                "cancellation pressure did not preserve the oldest resident");
        // The consumer may still have committed events queued when it observes paused state.
        // validate() checks the complete drained stream against the terminal result as well.
        require(second_result.generated_token_ids.size() >= committed_at_cancellation,
                "cancellation discarded tokens already committed to the stream");
        require(second_result.scheduling.snapshot_restores == 0 &&
                    totals.device_to_host_bytes == 0 && totals.host_to_device_bytes == 0,
                "Host=0 cancellation unexpectedly used a physical snapshot");
        if (cancel_stage == CancelStage::Paused) {
            require(second_result.scheduling.replay_restores == 0 &&
                        second_result.scheduling.replayed_tokens == 0,
                    "paused cancellation unexpectedly resumed execution");
        } else {
            require(second_result.scheduling.replayed_tokens >=
                            cancellation_observed.replayed_tokens &&
                        second_result.scheduling.replayed_tokens != 0,
                    "replay cancellation lost work performed before cancellation");
        }
    } else if (snapshot) {
        require(totals.snapshot_restores != 0 && totals.device_to_host_bytes != 0 &&
                    totals.host_to_device_bytes != 0,
                "fixed 512 MiB Host case did not save and restore a real request snapshot");
    } else {
        require(totals.snapshot_restores == 0 && totals.replay_restores != 0 &&
                    totals.replayed_tokens > kPromptTokens && totals.device_to_host_bytes == 0 &&
                    totals.host_to_device_bytes == 0,
                "Host=0 did not rebuild prompt and already published output without a snapshot");
    }
    if (cancel_stage == CancelStage::None) {
        require(totals.snapshot_restores + totals.replay_restores == totals.preemptions,
                "growth pressure did not restore every affected request");
    }

    const auto probe = engine.generate(engine.prepare_tokens({198, 1002, 198}), request(4));
    require(probe.generated_token_ids.size() == 4 &&
                probe.finish_reason == ninfer::FinishReason::OutputLimit,
            "request after recovery could not reuse the released execution resources");
    const auto settled_memory = engine.memory_summary();
    settled(engine.runtime_stats(), settled_memory);
    require(engine.is_available(), "Engine became unavailable after recovery");
}

} // namespace

int main() {
    const auto configured_artifact = setting("NINFER_TEST_ARTIFACT", "");
    if (configured_artifact.empty()) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const std::filesystem::path artifact(configured_artifact);
    try {
        require(std::filesystem::is_regular_file(artifact),
                "NINFER_TEST_ARTIFACT is not a regular file");
        const auto backend_name = setting("NINFER_TEST_BACKEND", "none");
        const auto selected     = backend(backend_name);
        const auto scenario     = setting("NINFER_PREEMPTION_REAL_SCENARIO", "all");
        require(scenario == "all" || scenario == "replay" || scenario == "snapshot" ||
                    scenario == "cancel-paused" || scenario == "cancel-replay",
                "NINFER_PREEMPTION_REAL_SCENARIO must be all, replay, snapshot, cancel-paused "
                "or cancel-replay");
        if (scenario == "all" || scenario == "replay") {
            exercise(artifact, selected, backend_name, false);
        }
        if (scenario == "all" || scenario == "snapshot") {
            exercise(artifact, selected, backend_name, true);
        }
        if (scenario == "all" || scenario == "cancel-paused") {
            exercise(artifact, selected, backend_name, false, CancelStage::Paused);
        }
        if (scenario == "all" || scenario == "cancel-replay") {
            exercise(artifact, selected, backend_name, false, CancelStage::Replay);
        }
    } catch (const std::exception& error) {
        std::cerr << "real Engine preemption test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
