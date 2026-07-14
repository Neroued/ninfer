#pragma once

// Owns the resident engine + tokenizer and serializes all engine use behind one
// mutex (the engine is single-sequence and not thread-safe). Validation,
// tokenization, and CPU media preprocessing need no engine lock; execution
// (run) takes it for the whole generation.

#include "ninfer/runtime/engine.h"
#include "ninfer/model/processor.h"
#include "ninfer/serve/request.h"
#include "ninfer/serve/serve_options.h"
#include "ninfer/text/text_runner.h"
#include "ninfer/text/tokenizer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

// Per-request timing and speculative-decoding counters for console observability.
// Raw values only; the log layer derives tok/s, TTFT and acceptance rates.
struct GenerationMetrics {
    double render_tokenize_seconds   = 0.0;
    double prefill_seconds           = 0.0;
    double decode_seconds            = 0.0;
    double total_seconds             = 0.0;
    bool mtp_enabled                 = false;
    std::int64_t mtp_rounds          = 0;
    std::int64_t mtp_draft_tokens    = 0;
    std::int64_t mtp_accepted_tokens = 0;
    // Resident prefix tokens reused by the engine prefix cache this request (0 = full prefill).
    std::uint32_t prefix_cache_hit_tokens = 0;
};

struct GenerationOutcome {
    std::string text;      // answer text (non-streaming); empty when streamed via a sink
    std::string reasoning; // thinking text split out of the <think> block (non-streaming)
    std::vector<ToolCall> tool_calls;
    int prompt_tokens                     = 0;
    int completion_tokens                 = 0;
    ninfer::text::FinishReason finish_reason = ninfer::text::FinishReason::Length;
    GenerationMetrics metrics;
};

// Streaming hooks supplied by the transport layer. Reasoning (the <think> block)
// and answer content are delivered on separate channels so the transport can map
// them to reasoning_content vs content deltas.
struct StreamSink {
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<bool()> is_cancelled;
};

// A validated, tokenized request ready to execute. The prompt is rendered and
// tokenized once here in prepare(); run() reuses prompt_token_ids so a served
// request renders the chat template exactly once.
struct PreparedRequest {
    ninfer::text::TextGenerationOptions options;
    std::vector<std::string> stop_strings;
    std::vector<int> prompt_token_ids;
    std::optional<ninfer::model::ProcessedInput> multimodal;
    // Serializes memory-heavy media preprocessing. Released immediately after Vision prefill.
    std::unique_lock<std::mutex> media_permit;
    double render_tokenize_seconds = 0.0;
    int prompt_tokens              = 0;
    // Absolute assistant-content boundary (prompt_tokens - generation-prompt opener length)
    // threaded into the engine prefix cache so a later thinking-stripped turn can reuse up to it.
    std::uint32_t content_boundary = 0;
    bool include_usage             = false;
    bool tool_capable              = false;
};

// Resolve the maximum output that can fit beside an already-tokenized prompt.
// `requested_max_tokens` is an upper bound: it is reduced to the remaining KV
// capacity rather than causing an otherwise valid prompt to be rejected. The
// runner's prefill produces the first completion token without appending it to
// KV, hence the +1 in the available-output calculation. A prompt that cannot
// itself fit still raises context_length_exceeded.
struct ContextOutputBudget {
    int requested_max_tokens = 0;
    int effective_max_tokens = 0;

    [[nodiscard]] bool clamped() const noexcept {
        return effective_max_tokens != requested_max_tokens;
    }
};

[[nodiscard]] ContextOutputBudget resolve_context_output_budget(std::size_t prompt_tokens,
                                                                int requested_max_tokens,
                                                                std::uint32_t max_context);

// Which optional generation features the engine currently honors. Sampling is
// wired through translate.cpp -> TextGenerationOptions -> Engine::set_sampling;
// temperature 0 (or --greedy) selects the exact-argmax path.
struct ServerCapabilities {
    bool sampling = true;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options);

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ServerCapabilities& capabilities() const noexcept { return caps_; }

    [[nodiscard]] ninfer::EngineMemoryStats memory_stats() const;

    // Validate and prepare text or media input without taking the engine lock.
    // Throws ApiException on bad/oversized input.
    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req) const;

    // Return the fully expanded prompt token count for the Anthropic count_tokens
    // endpoint. Media is decoded and processed, but no GPU model work or engine
    // lock is needed. Never rejects solely because the expanded prompt is over context.
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req) const;

    // Execute under the engine lock. When sink != nullptr, deltas stream through
    // the sink and text is left empty; otherwise the full text is returned.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink);

    // One short generation to trigger CUDA-graph capture before serving traffic.
    void warmup();

private:
    ServeOptions options_;
    ServerCapabilities caps_;
    std::unique_ptr<ninfer::text::QwenTokenizer> tokenizer_;
    std::unique_ptr<ninfer::Engine> engine_;
    mutable std::mutex media_mutex_;
    std::mutex engine_mutex_;
};

} // namespace ninfer::serve
