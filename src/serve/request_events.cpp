#include "serve/request_events.h"

#include <utility>

namespace ninfer::serve {

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const RequestLogMetadata& metadata,
                                           const PreparedRequest& prepared) {
    RequestLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = metadata.model;
    context.stream                             = metadata.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = metadata.output_tokens_explicit;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.enable_thinking                    = prepared.enable_thinking;
    context.thinking_budget                    = prepared.thinking_budget;
    context.requested_reasoning_effort         = request.reasoning_effort;
    context.resolved_reasoning_effort          = prepared.effective_reasoning_effort;
    context.preserve_thinking                  = prepared.preserve_thinking;
    context.preserve_thinking_semantic_change  = metadata.preserve_thinking_semantic_change;
    context.sampling                           = prepared.sampling;
    context.acquisition_seconds                = prepared.acquisition_seconds;
    context.preparation                        = prepared.preparation;
    return context;
}

RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              const RequestLogMetadata& metadata,
                                                              ApiError error) {
    RequestRejectionLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = metadata.model;
    context.stream                             = metadata.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = metadata.output_tokens_explicit;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.requested_reasoning_effort         = request.reasoning_effort;
    context.error                              = std::move(error);
    return context;
}

RequestFailure make_request_failure(RequestFailurePhase phase, const ApiError& error) {
    RequestFailureClass classification = RequestFailureClass::Internal;
    if (error.status == 499 || error.code == "client_disconnected") {
        classification = RequestFailureClass::ClientDisconnected;
    } else if (error.status == 429 || error.status == 529) {
        classification = RequestFailureClass::Overload;
    } else if (error.code == "request_queue_timeout" || error.code == "media_fetch_timeout" ||
               error.status == 504) {
        classification = RequestFailureClass::Timeout;
    } else if (error.code == "service_unavailable" || error.status == 503) {
        classification = RequestFailureClass::Unavailable;
    } else if (error.code == "media_fetch_failed" || error.status == 502) {
        classification = RequestFailureClass::Upstream;
    } else if (error.status >= 400 && error.status < 500) {
        classification = RequestFailureClass::ClientInput;
    }
    return RequestFailure{
        .phase           = phase,
        .classification  = classification,
        .http_status     = error.status,
        .error_type      = error.type,
        .error_code      = error.code,
        .param           = error.param,
        .machine_message = error.message,
    };
}

RequestFailure make_generation_request_failure(const ApiError& error) {
    RequestFailure failure = make_request_failure(RequestFailurePhase::Generation, error);
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        failure.phase = RequestFailurePhase::Transport;
    }
    return failure;
}

RequestFailure make_internal_request_failure(RequestFailurePhase phase,
                                             std::string machine_message) {
    return RequestFailure{
        .phase           = phase,
        .classification  = RequestFailureClass::Internal,
        .http_status     = 500,
        .error_type      = "internal_error",
        .machine_message = std::move(machine_message),
    };
}

RequestFailure make_client_disconnected_failure(RequestFailurePhase phase) {
    return RequestFailure{
        .phase           = phase,
        .classification  = RequestFailureClass::ClientDisconnected,
        .http_status     = 499,
        .error_type      = "request_cancelled",
        .error_code      = "client_disconnected",
        .machine_message = "client disconnected",
    };
}

} // namespace ninfer::serve
