#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <memory>
#include <string_view>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    // Waits for initial admission or successful completion without execution. Errors and
    // cancellation are reported before returning. Does not consume output or the handle;
    // wait() still delivers the stream start and result. Calls on one handle must be serial.
    void wait_for_admission(const CancellationView& cancellation = {});

    // A terminal result or request error is visible after logical scheduling membership is
    // released and the corresponding Engine snapshot is published.
    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

struct MeasurementOptions {
    bool profile = false;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()).
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink. Observation options request protocol-neutral publication facts without changing the
    // execution request.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] const EngineDescription& description() const;
    [[nodiscard]] const StartupReport& startup_report() const;
    [[nodiscard]] EngineSnapshot snapshot() const;
    [[nodiscard]] bool is_available() const;

    // Synchronous boundaries on the execution thread. Previously submitted requests and their
    // cleanup complete before the boundary; concurrent submissions wait until it finishes.
    // Begin resets window peaks and optionally starts the profiler on every execution rank.
    // End includes final retirement and returns the completed window's resource observations.
    [[nodiscard]] EngineSnapshot begin_measurement(MeasurementOptions options = {});
    [[nodiscard]] EngineSnapshot end_measurement();
    void reset_measurement_peaks();

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
