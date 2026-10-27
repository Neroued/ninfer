#include "serve/operational_log.h"
#include "product/statistics.h"

#include "product/logging/pretty_format.h"
#include "product/speculative_options.h"

#include <spdlog/logger.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
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
        return "response rendering";
    case RequestFailurePhase::ResponseStore:
        return "response storage";
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
        return "client input";
    case RequestFailureClass::ClientDisconnected:
        return "client disconnected";
    case RequestFailureClass::Overload:
        return "overload";
    case RequestFailureClass::Timeout:
        return "timeout";
    case RequestFailureClass::Unavailable:
        return "unavailable";
    case RequestFailureClass::Upstream:
        return "upstream error";
    case RequestFailureClass::Internal:
        return "internal error";
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
        return "output limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context capacity";
    case ninfer::FinishReason::StopToken:
        return "stop token";
    case ninfer::FinishReason::StopString:
        return "stop string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) noexcept {
    switch (path) {
    case ninfer::PrefixReusePath::Root:
        return "root";
    case ninfer::PrefixReusePath::Checkpoint:
        return "checkpoint";
    }
    return "unknown";
}

std::string_view requested_effort_name(const RequestLogContext& context) noexcept {
    if (context.requested_reasoning_effort)
        return requested_reasoning_effort_name(*context.requested_reasoning_effort);
    return "template default";
}

const char* protocol_name(std::string_view protocol) noexcept {
    if (protocol == "openai_chat_completions") { return "openai-chat"; }
    if (protocol == "openai_responses") { return "openai-responses"; }
    if (protocol == "anthropic_messages") { return "anthropic"; }
    if (protocol == "openai_responses_input_tokens") { return "openai-input-tokens"; }
    if (protocol == "anthropic_count_tokens") { return "anthropic-count-tokens"; }
    return "http";
}

const char* kv_cache_name(ninfer::KvCacheStorage storage) noexcept {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

void append_clause(std::ostringstream& out, std::string_view clause) { out << " | " << clause; }

void append_counted_clause(std::ostringstream& out, std::string_view label, std::uint64_t count) {
    out << " | " << label << ' ' << product::format_pretty_count(count);
}

std::string pretty_code(std::string_view code) {
    std::string result;
    result.reserve(code.size());
    for (const char ch : code) { result.push_back(ch == '_' ? ' ' : ch); }
    return result;
}

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

std::uint64_t host_active_ns(const ThroughputReport& report) noexcept {
    const ninfer::RuntimeHostWorkStats& previous = report.previous.host_work;
    const ninfer::RuntimeHostWorkStats& current  = report.current.host_work;
    return monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns) +
           monotonic_delta(previous.session_call_ns, current.session_call_ns) +
           monotonic_delta(previous.executor_poll_ns, current.executor_poll_ns) +
           monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns) +
           monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns) +
           monotonic_delta(previous.executor_flush_ns, current.executor_flush_ns);
}

void append_failure_fields(std::ostringstream& out, const RequestFailure& failure) {
    if (failure.http_status != 0) { out << " | HTTP " << failure.http_status; }
    if (!failure.error_code.empty()) {
        append_clause(out, pretty_code(failure.error_code));
    } else {
        append_clause(out, classification_name(failure.classification));
    }
}

} // namespace

OperationalRecord render_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "req#" << context.id << " started | " << protocol_name(context.protocol) << ' '
        << (context.stream ? "stream" : "non-stream") << " | "
        << product::format_pretty_count(context.message_count)
        << (context.message_count == 1 ? " message" : " messages") << " | max output "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(context.requested_output_tokens, 0)));
    out << " | thinking ";
    if (context.enable_thinking) {
        out << requested_effort_name(context);
        if (context.thinking_budget) {
            out << ", budget " << product::format_pretty_count(*context.thinking_budget);
        }
    } else {
        out << "off";
    }
    if (context.media_item_count != 0) {
        append_counted_clause(out, "media", context.media_item_count);
        if (context.preparation.seconds > 0.0) {
            out << ", prepared " << product::format_pretty_duration(context.preparation.seconds);
        }
    }
    if (context.tool_count != 0) { append_counted_clause(out, "tools", context.tool_count); }
    if (context.preserve_thinking == true) { append_clause(out, "preserve thinking"); }
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
    out << "req#" << context.id << ' ' << status << " during prepare | "
        << protocol_name(context.protocol) << ' ' << (context.stream ? "stream" : "non-stream");
    append_failure_fields(out, failure);
    append_counted_clause(out, "messages", context.message_count);
    if (context.media_item_count != 0) {
        append_counted_clause(out, "media", context.media_item_count);
    }
    if (context.tool_count != 0) { append_counted_clause(out, "tools", context.tool_count); }
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_request_done(const RequestLogContext& context,
                                      const GenerationOutcome& outcome) {
    const RequestStats& metrics = outcome.stats;
    const auto rates            = product::generation_rates(metrics, outcome.completion_tokens);
    std::ostringstream out;
    out << "req#" << context.id << " done | " << protocol_name(context.protocol) << " | ";
    const auto complete_calls =
        std::count_if(outcome.tool_calls.begin(), outcome.tool_calls.end(),
                      [](const auto& call) { return call.status == ToolCallStatus::Complete; });
    if (complete_calls == 0) {
        out << finish_reason_name(outcome.finish_reason);
    } else {
        out << "tool calls " << product::format_pretty_count(complete_calls);
    }
    out << " | prompt "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0)))
        << " | output "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(outcome.completion_tokens, 0)));
    const double cache_ratio = outcome.prompt_tokens > 0
                                   ? static_cast<double>(metrics.reused_prompt_tokens) /
                                         static_cast<double>(outcome.prompt_tokens)
                                   : 0.0;
    out << " | cache " << product::format_pretty_count(metrics.reused_prompt_tokens) << " ("
        << product::format_pretty_percent(cache_ratio);
    if (metrics.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        out << ", " << prefix_reuse_path_name(metrics.prefix_reuse_path);
    }
    out << ") | TTFT "
        << product::format_pretty_duration(metrics.timings.first_token_seconds +
                                           outcome.service_prepare_seconds)
        << " | total "
        << product::format_pretty_duration(metrics.timings.total_seconds +
                                           outcome.service_prepare_seconds);
    if (metrics.engine_timing.queue_wait_seconds >= 0.01) {
        out << " | queue "
            << product::format_pretty_duration(metrics.engine_timing.queue_wait_seconds);
    }
    if (metrics.timings.prefill_seconds > 0.0) {
        out << " | prefill " << product::format_pretty_rate(rates.prefill_tokens_per_second, "tok");
    }
    if (metrics.timings.decode_seconds > 0.0) {
        out << " | decode " << product::format_pretty_rate(rates.output_tokens_per_second, "tok");
    }
    if (metrics.speculative.drafted_tokens != 0) {
        const double acceptance = static_cast<double>(metrics.speculative.accepted_tokens) /
                                  static_cast<double>(metrics.speculative.drafted_tokens);
        out << " | " << product::speculative_backend_name(metrics.speculative.backend)
            << " accepted " << product::format_pretty_count(metrics.speculative.accepted_tokens)
            << '/' << product::format_pretty_count(metrics.speculative.drafted_tokens) << " ("
            << product::format_pretty_percent(acceptance) << ')';
    }
    if (outcome.stats.thinking.configured_budget) {
        out << " | thinking "
            << product::format_pretty_count(outcome.stats.thinking.model_thinking_tokens) << '/'
            << product::format_pretty_count(*outcome.stats.thinking.configured_budget);
        if (outcome.stats.thinking.injected_tokens != 0) {
            out << ", control "
                << product::format_pretty_count(outcome.stats.thinking.injected_tokens);
        }
    }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

std::optional<OperationalRecord> render_tool_call_fallback(const RequestLogContext& context,
                                                           const GenerationOutcome& outcome) {
    const ninfer::ToolCallParseFallbackReason reason = outcome.tool_call_parse.fallback_reason;
    if (!outcome.tool_call_parse.marker_seen ||
        reason == ninfer::ToolCallParseFallbackReason::None) {
        return std::nullopt;
    }
    return OperationalRecord{
        .severity = OperationalSeverity::Warning,
        .message  = "req#" + std::to_string(context.id) + " tool markup returned as text | " +
                   pretty_code(ninfer::tool_call_parse_fallback_reason_name(reason)),
    };
}

OperationalRecord render_request_failure(const RequestLogContext& context,
                                         const RequestFailure& failure) {
    std::ostringstream out;
    out << "req#" << context.id
        << (failure.classification == RequestFailureClass::ClientDisconnected ? " cancelled during "
                                                                              : " failed during ")
        << phase_name(failure.phase) << " | " << protocol_name(context.protocol);
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    std::ostringstream out;
    out << "req#" << request_id << " response failed during " << phase_name(failure.phase);
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
    out << "throughput | " << product::format_pretty_duration(report.interval_seconds);
    if (report.computed_prefill_tokens != 0) {
        out << " | prefill " << product::format_pretty_rate(prefill_rate, "tok") << " ("
            << product::format_pretty_count(report.computed_prefill_tokens) << " tok)";
    }
    if (report.committed_decode_tokens != 0) {
        out << " | decode " << product::format_pretty_rate(decode_rate, "tok") << " ("
            << product::format_pretty_count(report.committed_decode_tokens) << " tok)";
    }
    out << " | running " << report.current.running_requests;
    if (report.current.prefilling_requests != 0 || report.current.decode_ready_requests != 0) {
        out << " (";
        bool separator = false;
        if (report.current.prefilling_requests != 0) {
            out << "prefill " << report.current.prefilling_requests;
            separator = true;
        }
        if (report.current.decode_ready_requests != 0) {
            if (separator) { out << ", "; }
            out << "decode-ready " << report.current.decode_ready_requests;
        }
        out << ')';
    }
    if (report.current.waiting_requests != 0) {
        out << " | waiting " << report.current.waiting_requests;
    }
    if (report.current.paused_requests != 0) {
        out << " | paused " << report.current.paused_requests;
    }
    if (report.current.replaying_requests != 0) {
        out << " | replaying " << report.current.replaying_requests;
    }
    if (report.current.materializing_requests != 0) {
        out << " | materializing " << report.current.materializing_requests;
    }
    if (report.current.capture_pending_requests != 0) {
        out << " | capture-pending " << report.current.capture_pending_requests;
    }
    if (report.current.terminal_pending_requests != 0) {
        out << " | terminal-pending " << report.current.terminal_pending_requests;
    }
    if (report.decode_rounds != 0) {
        out << " | batch " << std::fixed << std::setprecision(2)
            << static_cast<double>(report.decode_row_rounds) /
                   static_cast<double>(report.decode_rounds);
    }
    for (const auto kind : {ExecutionKind::Prefill, ExecutionKind::Decode}) {
        const auto i = static_cast<std::size_t>(kind);
        const auto value =
            product::execution_window(report.previous_execution[i], report.current_execution[i]);
        if (auto mean = value.mean_seconds())
            out << (kind == ExecutionKind::Prefill ? " | chunk " : " | step ")
                << product::format_pretty_duration(*mean);
    }
    const auto drafts =
        report.current.speculative_draft_tokens - report.previous.speculative_draft_tokens;
    const auto accepted =
        report.current.speculative_accepted_tokens - report.previous.speculative_accepted_tokens;
    if (drafts)
        out << " | draft accepted " << accepted << '/' << drafts << " ("
            << product::format_pretty_percent(double(accepted) / drafts) << ')';
    const double host_seconds = static_cast<double>(host_active_ns(report)) * 1.0e-9;
    const double host_ratio =
        report.interval_seconds > 0.0 ? host_seconds / report.interval_seconds : 0.0;
    out << " | host " << product::format_pretty_percent(host_ratio) << " ("
        << product::format_pretty_duration(host_seconds) << ')';
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
    if (std::optional<OperationalRecord> fallback = render_tool_call_fallback(context, outcome)) {
        write(std::move(*fallback));
    }
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
    out << "HTTP request failed during " << phase_name(failure.phase) << " | "
        << protocol_name(endpoint);
    if (!request_id.empty()) { out << " | request " << product::format_pretty_text(request_id); }
    append_failure_fields(out, failure);
    write({.severity = failure_severity(failure.classification), .message = out.str()});
}

void OperationalLog::engine_capacity(const GenerationService& service) const {
    const auto& description  = service.description();
    const auto& startup      = service.startup_report();
    const auto& engine       = service.engine_options();
    const auto& cache        = engine.context_cache;
    const auto& context_cost = startup.load.context_cost;
    logger_->info("capacity | {} | {} ranks | {} active lanes", kv_cache_name(description.kv_cache),
                  description.ranks.size(), engine.max_concurrency);
    for (const auto& pool : description.context_pools)
        logger_->info("context rank {} | {} {} | block {}", pool.rank,
                      pool.domain == ninfer::ContextMemoryDomain::Device ? "device" : "host",
                      product::format_pretty_bytes(pool.capacity_bytes),
                      product::format_pretty_bytes(pool.block_bytes));
    for (std::size_t i = 0; i < description.ranks.size(); ++i) {
        const auto& rank   = description.ranks[i];
        const auto& budget = startup.ranks.at(i);
        logger_->info("rank {} | node {} device {} | budget {} | runtime {} | startup free {}",
                      rank.rank, rank.node, rank.device,
                      product::format_pretty_bytes(budget.limit_bytes),
                      product::format_pretty_bytes(budget.runtime_reservation_bytes),
                      product::format_pretty_bytes(budget.available_after_startup_bytes));
        if (budget.graphs)
            logger_->debug("rank {} | graph rebuild estimate {} | preparation peak {} | delta {} B",
                           rank.rank,
                           product::format_pretty_bytes(budget.graphs->rebuild_estimate_bytes),
                           product::format_pretty_bytes(budget.graphs->preparation_peak_bytes),
                           budget.graphs->preparation_increment_bytes);
    }

    logger_->info("context | history {} | checkpoint interval {} tokens",
                  cache.enabled ? "on" : "off", cache.checkpoint_interval);

    if (service.options().launch.engine.enable_vision) {
        const ninfer::MediaCacheSummary media = service.snapshot().media;
        logger_->info("media | {} preprocess workers | cache {} | live {}",
                      media.preprocess_threads, product::format_pretty_bytes(media.capacity_bytes),
                      product::format_pretty_bytes(media.live_capacity_bytes));
    }

    logger_->debug("context cost | transfer {} | prefill {} | hardware {} | prefill signature {}",
                   ninfer::context_cost_preset_source_name(context_cost.transfer_source),
                   ninfer::context_cost_preset_source_name(context_cost.prefill_source),
                   product::format_pretty_text(context_cost.hardware_class),
                   product::format_pretty_text(context_cost.prefill_signature));
}

void OperationalLog::warmup_started() const { logger_->debug("warming up"); }

void OperationalLog::warmup_complete(double seconds) const {
    if (seconds >= 0.25) {
        logger_->info("warmup complete | {}", product::format_pretty_duration(seconds));
    } else {
        logger_->debug("warmup complete | {}", product::format_pretty_duration(seconds));
    }
}

void OperationalLog::warmup_failure(double seconds, std::string_view detail) const {
    logger_->critical("warmup failed | {} | {}", product::format_pretty_duration(seconds),
                      product::format_pretty_text(detail));
}

void OperationalLog::bind_failure(std::string_view host, int port) const {
    logger_->error("cannot bind {}:{}", product::format_pretty_text(host), port);
}

void OperationalLog::listen_failure(std::string_view host, int port) const {
    logger_->error("server listen failed | {}:{}", product::format_pretty_text(host), port);
}

void OperationalLog::server_ready(std::string_view host, int port, std::string_view model_id,
                                  bool auth_enabled) const {
    logger_->info("listening on http://{}:{} | model {} | auth {}",
                  product::format_pretty_text(host), port, product::format_pretty_text(model_id),
                  auth_enabled ? "bearer" : "disabled");
}

void OperationalLog::server_stopped() const { logger_->info("server stopped"); }

void OperationalLog::server_failure(bool serving, std::string_view detail) const {
    logger_->critical("server failed during {} | {}", serving ? "serving" : "startup",
                      product::format_pretty_text(detail));
}

} // namespace ninfer::serve
