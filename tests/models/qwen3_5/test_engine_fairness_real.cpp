#include "ninfer/engine.h"
#include "context_budget.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using namespace ninfer;
using Clock = std::chrono::steady_clock;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

RequestOptions request(std::uint32_t tokens) {
    RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

// Run the public Engine and the production Native allocator/executor. The observable
// acceptance condition is repeated new progress for BOTH requests before either ends.
// A short request arriving during pressure must retain its pending deadline.
void exercise(const char* artifact, bool snapshot) {
    EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_concurrency                   = 2;
    options.max_pending_requests              = 2;
    options.prefill_chunk                     = 128;
    options.kv_cache                          = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    options.speculative.backend               = SpeculativeBackend::Mtp;
    options.speculative.draft_tokens          = 3;
    options.context_cache.enabled             = false;
    options.context_cache.host_capacity_bytes = snapshot ? 512ULL << 20 : 0;
    const auto prompt_tokens = test::qwen3_5::configure_two_request_pressure(options, 256);
    Engine engine(options);
    const auto outputs = std::min(4096U, options.max_context - prompt_tokens - 1);
    require(outputs > 1024, "fairness fixture has no long generation budget");
    auto short_prompt = engine.prepare_tokens({198, 1002, 198});
    std::array<GenerationHandle, 2> handles;
    std::array<GenerationResult, 2> results;
    std::array<std::exception_ptr, 2> errors;
    std::array<unsigned, 2> recoveries{};
    std::array<std::uint64_t, 2> last_progress{};
    std::atomic<bool> stop{false};
    std::mutex mutex;
    bool ended_early    = false;
    const auto deadline = Clock::now() + std::chrono::seconds(120);
    for (std::size_t row = 0; row < 2; ++row) {
        std::vector<TokenId> tokens(prompt_tokens, 198);
        tokens.front() = 1000 + row;
        GenerationObservationOptions observations;
        observations.scheduling = [&, row](const GenerationSchedulingObservation& event) {
            if (event.transition != GenerationSchedulingTransition::RecoveryComplete) return;
            std::lock_guard lock(mutex);
            require(event.request_decode_tokens > last_progress[row],
                    "recovery completed without new committed decode progress");
            last_progress[row] = event.request_decode_tokens;
            ++recoveries[row];
            if (recoveries[0] >= 2 && recoveries[1] >= 2) stop = true;
        };
        handles[row] = engine.submit(engine.prepare_tokens(std::move(tokens)), request(outputs),
                                     OutputConsumerMode::Aggregate, observations);
    }
    std::array<std::jthread, 2> consumers;
    for (std::size_t row = 0; row < 2; ++row) {
        consumers[row] = std::jthread([&, row] {
            try {
                results[row] = handles[row].wait(nullptr, CancellationView([&] {
                                                     return stop.load() || Clock::now() >= deadline;
                                                 }));
                std::lock_guard lock(mutex);
                if (!stop.load()) ended_early = true;
            } catch (...) { errors[row] = std::current_exception(); }
        });
    }
    // The deadline bounds regressions as well as successful executions; never leave a
    // test client waiting indefinitely when the very behavior under test is starvation.
    bool saw_pause     = false;
    bool queue_timeout = false;
    while (Clock::now() < deadline && !stop.load()) {
        if (engine.snapshot().runtime.paused_requests) {
            saw_pause = true;
            auto probe =
                engine.submit(std::move(short_prompt), request(1), OutputConsumerMode::Aggregate,
                              {}, Clock::now() + std::chrono::milliseconds(100));
            try {
                (void)probe.wait();
            } catch (const RequestError& error) {
                queue_timeout = error.kind() == RequestErrorKind::QueueTimeout;
            }
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    for (auto& consumer : consumers) consumer.join();
    for (const auto& error : errors)
        if (error) std::rethrow_exception(error);
    require(saw_pause && queue_timeout, "fresh request bypassed paused work or lost its deadline");
    require(!ended_early && recoveries[0] >= 2 && recoveries[1] >= 2,
            "an admitted request waited for its peer to finish instead of receiving service");
    for (const auto& result : results)
        require(result.finish_reason == FinishReason::Cancelled &&
                    !result.generated_token_ids.empty() &&
                    result.generated_token_ids.size() < outputs,
                "fair service did not preserve cancellation and the generation budget");
    const auto settled = engine.snapshot().runtime;
    require(settled.running_requests == 0 && settled.paused_requests == 0 &&
                settled.waiting_requests == 0 && settled.materializing_requests == 0 &&
                settled.state_objects == 0 && settled.history_objects == 0,
            "cancelled pressure cohort retained membership or context objects");
    for (const auto& pool : settled.context_pools)
        require(test::qwen3_5::occupied_bytes(pool) == 0 && pool.future_reserved_bytes == 0,
                "cancelled pressure cohort leaked a reservation");
    const auto probe = engine.generate(engine.prepare_tokens({198, 1003, 198}), request(2));
    require(probe.generated_token_ids.size() == 2 && engine.is_available(),
            "pressure recovery did not reopen admission");
    std::cout << (snapshot ? "snapshot" : "replay") << " recoveries=" << recoveries[0] << ','
              << recoveries[1] << " preemptions=" << settled.preemptions
              << " replayed_tokens=" << settled.replayed_tokens << '\n';
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) return 77;
    try {
        exercise(artifact, true);
        exercise(artifact, false);
    } catch (const std::exception& error) {
        std::cerr << "Engine fairness test failed: " << error.what() << '\n';
        return 1;
    }
}
