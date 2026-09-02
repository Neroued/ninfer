#include "serve/operational_log.h"

#include "product/logging/logging.h"

#include <spdlog/logger.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <utility>

namespace ninfer::serve {
namespace {

const char* phase_name(RequestFailurePhase phase) noexcept {
    switch (phase) {
    case RequestFailurePhase::Prepare:
        return "prepare";
    case RequestFailurePhase::Generation:
        return "generation";
    case RequestFailurePhase::ResponseRender:
        return "response-render";
    case RequestFailurePhase::ResponseStore:
        return "response-store";
    case RequestFailurePhase::Transport:
        return "transport";
    case RequestFailurePhase::Http:
        return "http";
    }
    return "unknown";
}

const char* classification_name(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
        return "client-input";
    case RequestFailureClass::ClientDisconnected:
        return "client-disconnected";
    case RequestFailureClass::Overload:
        return "overload";
    case RequestFailureClass::Timeout:
        return "timeout";
    case RequestFailureClass::Unavailable:
        return "unavailable";
    case RequestFailureClass::Upstream:
        return "upstream";
    case RequestFailureClass::Internal:
        return "internal";
    }
    return "unknown";
}

OperationalSeverity failure_severity(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
    case RequestFailureClass::ClientDisconnected:
        return OperationalSeverity::Info;
    case RequestFailureClass::Overload:
    case RequestFailureClass::Timeout:
    case RequestFailureClass::Unavailable:
    case RequestFailureClass::Upstream:
        return OperationalSeverity::Warning;
    case RequestFailureClass::Internal:
        return OperationalSeverity::Error;
    }
    return OperationalSeverity::Error;
}

const char* finish_reason_name(ninfer::FinishReason reason) noexcept {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output_limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context_capacity";
    case ninfer::FinishReason::StopToken:
        return "stop_token";
    case ninfer::FinishReason::StopString:
        return "stop_string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) noexcept {
    switch (path) {
    case ninfer::PrefixReusePath::Root:
        return "root";
    case ninfer::PrefixReusePath::PrivateEndpoint:
        return "private_endpoint";
    case ninfer::PrefixReusePath::PrivateTurnClosure:
        return "private_turn_closure";
    case ninfer::PrefixReusePath::PrivateResponseReplay:
        return "private_response_replay";
    case ninfer::PrefixReusePath::PrivateLongAnchor:
        return "private_long_anchor";
    case ninfer::PrefixReusePath::SharedStablePrefix:
        return "shared_stable_prefix";
    }
    return "unknown";
}

const char* resolved_reasoning_effort_name(const RequestLogContext& context) noexcept {
    if (!context.enable_thinking) { return "none"; }
    if (!context.resolved_reasoning_effort) { return "unresolved"; }
    switch (*context.resolved_reasoning_effort) {
    case ninfer::ReasoningEffort::Low:
        return "low";
    case ninfer::ReasoningEffort::Medium:
        return "medium";
    case ninfer::ReasoningEffort::XHigh:
        return "xhigh";
    }
    return "unknown";
}

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

std::uint64_t host_active_ns(const ThroughputReport& report) noexcept {
    const ninfer::RuntimeHostWorkStats& previous = report.previous.host_work;
    const ninfer::RuntimeHostWorkStats& current  = report.current.host_work;
    return monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns) +
           monotonic_delta(previous.program_submit_ns, current.program_submit_ns) +
           monotonic_delta(previous.program_post_ns, current.program_post_ns) +
           monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns) +
           monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns);
}

void append_failure_fields(std::ostringstream& out, const RequestFailure& failure) {
    out << " phase=" << phase_name(failure.phase)
        << " classification=" << classification_name(failure.classification);
    if (failure.http_status != 0) { out << " http_status=" << failure.http_status; }
    if (!failure.error_type.empty()) {
        out << " error_type=" << product::quote_log_value(failure.error_type);
    }
    if (!failure.error_code.empty()) {
        out << " error_code=" << product::quote_log_value(failure.error_code);
    }
    if (!failure.param.empty()) { out << " param=" << product::quote_log_value(failure.param); }
}

} // namespace

OperationalRecord render_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "request id=" << context.id
        << " status=submitted protocol=" << product::quote_log_value(context.protocol)
        << " stream=" << (context.stream ? "true" : "false")
        << " messages=" << context.message_count << " media_items=" << context.media_item_count
        << " requested_output_tokens=" << context.requested_output_tokens
        << " tools=" << context.tool_count
        << " thinking=" << (context.enable_thinking ? "true" : "false")
        << " reasoning_effort=" << resolved_reasoning_effort_name(context)
        << " preserve_thinking=" << (context.preserve_thinking ? "true" : "false");
    if (context.thinking_budget) { out << " thinking_budget=" << *context.thinking_budget; }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

OperationalRecord render_request_rejected(const RequestRejectionLogContext& context) {
    const RequestFailure failure =
        make_request_failure(RequestFailurePhase::Prepare, context.error);
    std::ostringstream out;
    const char* status = "failed";
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        status = "cancelled";
    } else if (failure.classification == RequestFailureClass::ClientInput ||
               failure.classification == RequestFailureClass::Overload) {
        status = "rejected";
    }
    out << "request id=" << context.id << " status=" << status
        << " protocol=" << product::quote_log_value(context.protocol)
        << " stream=" << (context.stream ? "true" : "false")
        << " messages=" << context.message_count << " media_items=" << context.media_item_count
        << " tools=" << context.tool_count;
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_request_done(const RequestLogContext& context,
                                      const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics     = outcome.metrics;
    const double computed_prefill_tokens = static_cast<double>(
        std::max(0, outcome.prompt_tokens - static_cast<int>(metrics.prefix_cache_hit_tokens)));
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << "request id=" << context.id
        << " status=done finish_reason="
        << (outcome.tool_calls.empty() ? finish_reason_name(outcome.finish_reason) : "tool_calls")
        << " prompt_tokens=" << outcome.prompt_tokens
        << " completion_tokens=" << outcome.completion_tokens
        << " prefix_cache_hit_tokens=" << metrics.prefix_cache_hit_tokens
        << " prefix_reuse_path=" << prefix_reuse_path_name(metrics.prefix_reuse_path)
        << " ttft_ms=" << metrics.ttft_seconds * 1000.0
        << " duration_ms=" << metrics.total_seconds * 1000.0;
    if (metrics.prefill_seconds > 0.0) {
        out << " prefill_tokens_per_second=" << computed_prefill_tokens / metrics.prefill_seconds;
    }
    if (metrics.decode_seconds > 0.0) {
        out << " decode_tokens_per_second=" << decode_tokens / metrics.decode_seconds;
    }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

OperationalRecord render_request_failure(const RequestLogContext& context,
                                         const RequestFailure& failure) {
    std::ostringstream out;
    out << "request id=" << context.id
        << (failure.classification == RequestFailureClass::ClientDisconnected ? " status=cancelled"
                                                                              : " status=failed")
        << " protocol=" << product::quote_log_value(context.protocol);
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    std::ostringstream out;
    out << "response request_id=" << request_id << " status=failed";
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_throughput(const ThroughputReport& report) {
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "throughput interval_ms=" << report.interval_seconds * 1000.0
        << " computed_prefill_tokens=" << report.computed_prefill_tokens
        << " committed_decode_tokens=" << report.committed_decode_tokens
        << " prefill_tokens_per_second=" << prefill_rate
        << " decode_tokens_per_second=" << decode_rate
        << " running=" << report.current.running_requests
        << " prefilling=" << report.current.prefilling_requests
        << " decode_ready=" << report.current.decode_ready_requests
        << " waiting=" << report.current.waiting_requests
        << " materializing=" << report.current.materializing_requests
        << " capture_pending=" << report.current.capture_pending_requests
        << " terminal_pending=" << report.current.terminal_pending_requests
        << " host_active_ms=" << static_cast<double>(host_active_ns(report)) / 1.0e6;
    if (report.decode_rounds != 0) {
        out << " average_decode_batch="
            << static_cast<double>(report.decode_row_rounds) /
                   static_cast<double>(report.decode_rounds);
    }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

OperationalLog::OperationalLog(std::shared_ptr<spdlog::logger> logger)
    : logger_(std::move(logger)) {}

void OperationalLog::write(OperationalRecord record) const {
    switch (record.severity) {
    case OperationalSeverity::Info:
        logger_->info("{}", record.message);
        return;
    case OperationalSeverity::Warning:
        logger_->warn("{}", record.message);
        return;
    case OperationalSeverity::Error:
        logger_->error("{}", record.message);
        return;
    }
}

void OperationalLog::request_start(const RequestLogContext& context) const {
    write(render_request_start(context));
}

void OperationalLog::request_rejected(const RequestRejectionLogContext& context) const {
    write(render_request_rejected(context));
}

void OperationalLog::request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) const {
    write(render_request_done(context, outcome));
}

void OperationalLog::request_failure(const RequestLogContext& context,
                                     const RequestFailure& failure) const {
    write(render_request_failure(context, failure));
}

void OperationalLog::response_failure(std::uint64_t request_id,
                                      const RequestFailure& failure) const {
    write(render_response_failure(request_id, failure));
}

void OperationalLog::throughput(const ThroughputReport& report) const {
    write(render_throughput(report));
}

void OperationalLog::http_failure(std::string_view endpoint, const RequestFailure& failure,
                                  std::string_view request_id) const {
    std::ostringstream out;
    out << "http endpoint=" << product::quote_log_value(endpoint) << " status=failed";
    if (!request_id.empty()) { out << " request_id=" << product::quote_log_value(request_id); }
    append_failure_fields(out, failure);
    write({.severity = failure_severity(failure.classification), .message = out.str()});
}

} // namespace ninfer::serve
