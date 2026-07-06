#pragma once

// Owns the resident engine + tokenizer and serializes all engine use behind one
// mutex (the engine is single-sequence and not thread-safe). Validation and
// tokenization (prepare) need no engine lock; execution (run) takes it for the
// whole generation.

#include "qus/runtime/engine.h"
#include "qus/serve/request.h"
#include "qus/serve/serve_options.h"
#include "qus/text/text_runner.h"
#include "qus/text/tokenizer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace qus::serve {

// Per-request timing and speculative-decoding counters for console observability.
// Raw values only; the log layer derives tok/s, TTFT and acceptance rates.
struct GenerationMetrics {
    double render_tokenize_seconds = 0.0;
    double prefill_seconds         = 0.0;
    double decode_seconds          = 0.0;
    double total_seconds           = 0.0;
    bool mtp_enabled               = false;
    std::int64_t mtp_rounds          = 0;
    std::int64_t mtp_draft_tokens    = 0;
    std::int64_t mtp_accepted_tokens = 0;
};

struct GenerationOutcome {
    std::string text;       // answer text (non-streaming); empty when streamed via a sink
    std::string reasoning;  // thinking text split out of the <think> block (non-streaming)
    int prompt_tokens                     = 0;
    int completion_tokens                 = 0;
    qus::text::FinishReason finish_reason = qus::text::FinishReason::Length;
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

// A validated, tokenized request ready to execute.
struct PreparedRequest {
    std::vector<qus::text::ChatMessage> messages;
    qus::text::TextGenerationOptions options;
    std::vector<std::string> stop_strings;
    int prompt_tokens  = 0;
    bool include_usage = false;
};

// Which optional generation features the engine currently honors. Sampling is
// off (greedy) today; flipping this is the single seam the future sampler wires
// into, leaving the protocol/translation layers untouched.
struct ServerCapabilities {
    bool sampling = false;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options);

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }
    [[nodiscard]] const ServerCapabilities& capabilities() const noexcept { return caps_; }

    // Validate + tokenize (tokenizer only). Throws ApiException on bad/oversized input.
    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req) const;

    // Execute under the engine lock. When sink != nullptr, deltas stream through
    // the sink and text is left empty; otherwise the full text is returned.
    GenerationOutcome run(const PreparedRequest& prepared, const StreamSink* sink);

    // One short generation to trigger CUDA-graph capture before serving traffic.
    void warmup();

private:
    ServeOptions options_;
    ServerCapabilities caps_;
    std::unique_ptr<qus::text::QwenTokenizer> tokenizer_;
    std::unique_ptr<qus::Engine> engine_;
    std::mutex engine_mutex_;
};

} // namespace qus::serve
