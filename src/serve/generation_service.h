#pragma once

// Product-side adapter between HTTP protocol requests and the public NInfer
// engine. It owns one Engine and keeps protocol concerns (aliases, usage,
// streaming callbacks, and tool-call parsing) outside the target package.

#include "ninfer/engine.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationMetrics {
    double prepare_seconds = 0.0;
    double vision_seconds  = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds  = 0.0;
    double total_seconds   = 0.0;

    bool mtp_enabled                      = false;
    std::uint64_t mtp_rounds              = 0;
    std::uint64_t mtp_draft_tokens        = 0;
    std::uint64_t mtp_accepted_tokens     = 0;
    std::uint32_t prefix_cache_hit_tokens = 0;
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<ToolCall> tool_calls;
    int prompt_tokens                  = 0;
    int completion_tokens              = 0;
    ninfer::FinishReason finish_reason = ninfer::FinishReason::OutputLimit;
    GenerationMetrics metrics;
};

struct StreamSink {
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<bool()> is_cancelled;
};

// A prepared prompt is tied to the Engine that produced it and is consumed by
// run(). Protocol-only flags remain alongside it for response rendering.
struct PreparedRequest {
    ninfer::PreparedPrompt prompt;
    ninfer::RequestOptions options;
    double prepare_seconds = 0.0;
    int prompt_tokens      = 0;
    bool include_usage     = false;
    bool tool_capable      = false;

    // Limit concurrent ownership of media preprocessing buffers. Text requests
    // do not use this permit.
    std::unique_lock<std::mutex> media_permit;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options);

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req) const;

    // Consumes prepared.prompt. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink);

    void warmup();

private:
    ServeOptions options_;
    std::unique_ptr<ninfer::Engine> engine_;
    mutable std::mutex media_mutex_;
};

} // namespace ninfer::serve
