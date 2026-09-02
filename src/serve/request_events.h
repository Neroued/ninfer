#pragma once

#include "serve/generation_service.h"
#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ninfer::serve {

struct RequestLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    bool enable_thinking  = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<RequestedReasoningEffort> requested_reasoning_effort;
    std::optional<ninfer::ReasoningEffort> resolved_reasoning_effort;
    bool preserve_thinking                 = false;
    bool preserve_thinking_semantic_change = false;
    ninfer::ResolvedSamplingParameters sampling;
    double acquisition_seconds = 0.0;
    ninfer::PromptPreparationStats preparation;
};

struct RequestLogMetadata {
    std::string model;
    bool stream                            = false;
    bool output_tokens_explicit            = false;
    bool preserve_thinking_semantic_change = false;
};

// A parsed generation request that failed during synchronous preparation. It intentionally has a
// separate shape because sampler and prompt semantics may not have resolved.
struct RequestRejectionLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    std::optional<RequestedReasoningEffort> requested_reasoning_effort;
    ApiError error;
};

enum class RequestFailurePhase : std::uint8_t {
    Prepare,
    Generation,
    ResponseRender,
    ResponseStore,
    Transport,
    Http,
};

enum class RequestFailureClass : std::uint8_t {
    ClientInput,
    ClientDisconnected,
    Overload,
    Timeout,
    Unavailable,
    Upstream,
    Internal,
};

struct RequestFailure {
    RequestFailurePhase phase          = RequestFailurePhase::Generation;
    RequestFailureClass classification = RequestFailureClass::Internal;
    int http_status                    = 0;
    std::string error_type;
    std::string error_code;
    std::string param;
    // Used only by the independent JSONL measurement writer. Operational rendering never consumes
    // this field.
    std::string machine_message;
};

struct ThroughputReport {
    double interval_seconds               = 0.0;
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t committed_decode_tokens = 0;
    std::uint64_t decode_rounds           = 0;
    std::uint64_t decode_row_rounds       = 0;
    ninfer::RuntimeStats previous;
    ninfer::RuntimeStats current;
};

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const RequestLogMetadata& metadata,
                                           const PreparedRequest& prepared);
RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              const RequestLogMetadata& metadata,
                                                              ApiError error);

[[nodiscard]] RequestFailure make_request_failure(RequestFailurePhase phase, const ApiError& error);
[[nodiscard]] RequestFailure make_generation_request_failure(const ApiError& error);
[[nodiscard]] RequestFailure make_internal_request_failure(RequestFailurePhase phase,
                                                           std::string machine_message);
[[nodiscard]] RequestFailure make_client_disconnected_failure(RequestFailurePhase phase);

} // namespace ninfer::serve
