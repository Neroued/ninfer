#include "serve/http_server.h"

#include "serve/anthropic_schema.h"
#include "serve/console_log.h"
#include "serve/openai_schema.h"
#include "serve/request_log.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

struct StreamingRequest {
    explicit StreamingRequest(PreparedRequest request) : prepared(std::move(request)) {}

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    bool started = false;
};

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

void write_stream_item(httplib::DataSink& sink, StreamingRequest& request,
                       const std::string& item) {
    if (request.cancelled.load(std::memory_order_acquire) ||
        (sink.is_writable && !sink.is_writable()) || !sink.write(item.data(), item.size())) {
        request.cancelled.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

void set_owned_content(httplib::Response& response, std::string body,
                       std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

void write_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(make_error_body(error), "application/json");
}


// Anthropic-shaped error body ({"type":"error","error":{...}}), used by the
// /v1/messages endpoints so Claude clients see the error format they expect.
void write_messages_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(make_messages_error_body(error), "application/json");
}

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_error(res, error);
}

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .scheduler         = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.scheduler.running_requests != 0 ||
           report.scheduler.waiting_requests != 0;
}

std::string_view unstreamed_content(const GenerationOutcome& outcome) {
    if (outcome.streamed_content_bytes > outcome.text.size()) {
        throw std::logic_error("streamed content exceeds terminal content");
    }
    return std::string_view(outcome.text).substr(outcome.streamed_content_bytes);
}

} // namespace

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    if (response.status != 413 || !response.body.empty()) {
        return httplib::Server::HandlerResponse::Unhandled;
    }

    ApiError error;
    error.status  = 413;
    error.type    = "invalid_request_error";
    error.code    = "request_too_large";
    error.message = "request body exceeds the configured payload limit of " +
                    std::to_string(options.max_request_bytes) + " bytes";
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_messages_error(response, error);
    } else {
        write_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

HttpServer::HttpServer(ServeOptions options)
    : options_(std::move(options)),
      response_store_(options_.response_store_max_records, options_.response_store_max_bytes),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, queued_requests);
    };
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

void HttpServer::log_line(const std::string& line) {
    write_console_log(ConsoleLogLevel::Info, line);
}

void HttpServer::log_request_start(const RequestLogContext& context) {
    log_line(format_request_start(context));
    request_jsonl_.write_request_start(context);
}

void HttpServer::log_request_rejected(const RequestRejectionLogContext& context) {
    log_line(format_request_rejected(context));
    request_jsonl_.write_request_rejected(context);
}

void HttpServer::log_request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) {
    log_line(format_request_done(context, outcome));
    request_jsonl_.write_request_done(context, outcome);
}

void HttpServer::log_request_error(const RequestLogContext& context, const std::string& message) {
    log_line(format_request_error(context, message));
    request_jsonl_.write_request_error(context, message);
}

void HttpServer::log_throughput(const ThroughputReport& report) {
    log_line(format_throughput(report));
    request_jsonl_.write_throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_for(lock, interval, [this] { return stats_stopping_; })) { break; }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { log_throughput(report); }
        previous      = current;
        previous_time = now;
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    if (tail.computed_prefill_tokens != 0 || tail.committed_decode_tokens != 0 ||
        tail.decode_rounds != 0) {
        log_throughput(tail);
    }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        // Echo the preflight's requested headers (Access-Control-Request-Headers)
        // so clients using custom headers (e.g. llama.cpp webui's x-conversation-id)
        // pass the browser's CORS check. The static default above remains the
        // floor for direct (non-browser) requests.
        server_.Options(R"(.*)", [](const httplib::Request& req, httplib::Response& res) {
            res.status = 204;
            auto it = req.get_header_value("Access-Control-Request-Headers");
            if (!it.empty()) {
                std::string joined;
                std::vector<std::string> seen;
                std::string part;
                const std::string& raw = it;
                for (size_t i = 0; i <= raw.size(); i++) {
                    const char c = i < raw.size() ? raw[i] : ',';
                    if (c == ',') {
                        size_t b = part.find_first_not_of(" \t");
                        if (b == std::string::npos) {
                            part.clear();
                            continue;
                        }
                        size_t e = part.find_last_not_of(" \t") + 1;
                        part = part.substr(b, e - b);
                        if (!part.empty()) {
                            std::string key = part;
                            for (auto& ch : key) {
                                if (ch >= 'A' && ch <= 'Z') {
                                    ch = char(ch - 'A' + 'a');
                                }
                            }
                            const bool dup = std::any_of(seen.begin(), seen.end(),
                                                         [&](const std::string& s) { return s == key; });
                            if (!dup) {
                                if (!joined.empty()) {
                                    joined += ", ";
                                }
                                joined += part;
                                seen.push_back(std::move(key));
                            }
                        }
                        part.clear();
                    } else {
                        part += c;
                    }
                }
                if (!joined.empty()) {
                    // Build the full header set explicitly: httplib's set_header
                    // uses emplace and cannot replace the statically seeded
                    // Access-Control-Allow-Headers default.
                    httplib::Headers headers;
                    for (const auto& h : res.headers) {
                        // The seeded default uses exactly this key spelling.
                        if (h.first == "Access-Control-Allow-Headers") {
                            headers.emplace(h.first, joined);
                        } else {
                            headers.emplace(h.first, h.second);
                        }
                    }
                    res.headers = std::move(headers);
                }
            }
        });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            req.get_header_value("Authorization") == ("Bearer " + options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_messages_error(res, error);
            } else {
                write_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                write_error(res, e.error());
            } catch (const std::exception& e) { write_exception(res, e); } catch (...) {
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                write_error(res, error);
            }
        });

    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Get("/props", [this](const httplib::Request& req, httplib::Response& res) {
        handle_props(req, res);
    });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

// llama.cpp webui dialect: /props is the client's server introspection endpoint
// (role detection, context size, default params, thinking-capability probe, api
// key validation). NInfer has no llama.cpp server behind it, so serve a faithful
// stub derived from the process configuration. Only process-level overrides are
// reported as parameter values; everything else stays at neutral zeros so client
// side defaults never swallow a user-set request parameter.
nlohmann::json make_props_stub(const ServeOptions& options, const std::string& model_id) {
    (void)model_id;
    const auto& ov = options.sampling_overrides;
    nlohmann::json params = nlohmann::json::object();
    params["n_predict"] = options.default_max_tokens;
    params["seed"] = ov.seed ? static_cast<double>(*ov.seed) : 0;
    params["temperature"] = ov.temperature ? static_cast<double>(*ov.temperature) : 0;
    params["dynatemp_range"] = 0;
    params["dynatemp_exponent"] = 0;
    params["top_k"] = ov.top_k ? *ov.top_k : 0;
    params["top_p"] = ov.top_p ? static_cast<double>(*ov.top_p) : 0;
    params["min_p"] = ov.min_p ? static_cast<double>(*ov.min_p) : 0;
    params["top_n_sigma"] = 0;
    params["xtc_probability"] = 0;
    params["xtc_threshold"] = 0;
    params["typ_p"] = 0;
    params["repeat_last_n"] = 0;
    params["repeat_penalty"] = 0;
    params["presence_penalty"] =
        ov.presence_penalty ? static_cast<double>(*ov.presence_penalty) : 0;
    params["frequency_penalty"] =
        ov.frequency_penalty ? static_cast<double>(*ov.frequency_penalty) : 0;
    params["dry_multiplier"] = 0;
    params["dry_base"] = 0;
    params["dry_allowed_length"] = 0;
    params["dry_penalty_last_n"] = 0;
    params["dry_sequence_breakers"] = nlohmann::json::array();
    params["mirostat"] = 0;
    params["mirostat_tau"] = 0;
    params["mirostat_eta"] = 0;
    params["stop"] = nlohmann::json::array();
    params["max_tokens"] = options.default_max_tokens;
    params["n_keep"] = 0;
    params["n_discard"] = 0;
    params["ignore_eos"] = false;
    params["stream"] = false;
    params["logit_bias"] = nlohmann::json::array();
    params["n_probs"] = 0;
    params["min_keep"] = 0;
    params["grammar"] = "";
    params["grammar_lazy"] = false;
    params["grammar_triggers"] = nlohmann::json::array();
    params["preserved_tokens"] = nlohmann::json::array();
    params["chat_format"] = "";
    params["reasoning_format"] = "";
    params["reasoning_in_content"] = false;
    params["generation_prompt"] = "";
    params["samplers"] = nlohmann::json::array();
    params["backend_sampling"] = false;
    params["speculative.n_max"] = 0;
    params["speculative.n_min"] = 0;
    params["speculative.p_min"] = 0.0;
    params["timings_per_token"] = false;
    params["post_sampling_probs"] = false;
    params["lora"] = nlohmann::json::array();

    nlohmann::json props = nlohmann::json::object();
    props["default_generation_settings"] = {
        {"id", 0},
        {"id_task", 0},
        {"n_ctx", static_cast<int>(options.max_context)},
        {"speculative", options.speculative.backend != SpeculativeBackend::None},
        {"is_processing", false},
        {"params", params},
        {"prompt", ""},
        {"next_token",
         {{"has_next_token", false},
          {"has_new_line", false},
          {"n_remain", 0},
          {"n_decoded", 0},
          {"stopping_word", ""}}},
    };
    props["total_slots"] = 1;
    props["model_path"] = options.artifact_path;
    props["role"] = "model";
    props["modalities"] = {{"vision", options.enable_vision}, {"audio", false}, {"video", false}};
    // Capability marker only: clients that probe the chat template (e.g. the
    // webui's thinking-support heuristic) need `enable_thinking` to appear; the
    // real template is embedded in the loaded artifact.
    props["chat_template"] =
        "{# ninfer-serve: capability marker; the real chat template is embedded in "
        "the loaded artifact #}\n"
        "{%- if enable_thinking is defined %}\n"
        "  {%- set thinking = enable_thinking %}\n"
        "{% endif %}";
    props["bos_token"] = "";
    props["eos_token"] = "";
    props["build_info"] = "ninfer-serve";
    return props;
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now()), "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now()), "application/json");
}

void HttpServer::handle_props(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_props_stub(options_, public_model_id_).dump(), "application/json");
}

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_error(res, error);
        return;
    }

    GenerationRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(body, limits, public_model_id_);
        if (request.model != public_model_id_) {
            ApiError error;
            error.status  = 404;
            error.type    = "invalid_request_error";
            error.code    = "model_not_found";
            error.message = "model '" + request.model + "' not found";
            throw ApiException(std::move(error));
        }
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
    } catch (const ApiException& e) {
        log_request_rejected(make_request_rejection_log_context(req_id, "openai_chat_completions",
                                                                request, e.error()));
        write_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        log_request_rejected(
            make_request_rejection_log_context(req_id, "openai_chat_completions", request, error));
        write_error(res, error);
        return;
    }

    const std::string id       = new_chat_completion_id();
    const std::int64_t created = unix_time_now();
    const std::string model    = request.model;

    const RequestLogContext log_context =
        make_request_log_context(req_id, "openai_chat_completions", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr, [&req] {
                return req.is_connection_alive && !req.is_connection_alive();
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            std::string response_body;
            if (!outcome.tool_calls.empty()) {
                response_body = make_chat_completion_tool_response(
                    id, model, created, outcome.text, outcome.reasoning, outcome.tool_calls, usage);
            } else {
                response_body = make_chat_completion_response(
                    id, model, created, outcome.text, outcome.reasoning,
                    finish_reason_wire(outcome.finish_reason), usage);
            }
            set_owned_content(res, std::move(response_body), prepared.lifetime);
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            throw;
        }
        return;
    }

    auto stream              = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool include_usage = stream->prepared.include_usage;
    const bool tool_capable  = stream->prepared.tool_capable;

    // SSE hints: disable client/proxy caching and reverse-proxy response buffering
    // so tokens flush immediately. Content-Type is set by the chunked provider.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, created, model, include_usage, tool_capable,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            try {
                write_stream_item(sink, *stream,
                                  make_chat_chunk_role(id, model, created, include_usage));
                StreamSink output;
                output.on_content = [&](const std::string& text) {
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_content(id, model, created, text, include_usage));
                };
                output.on_reasoning = [&](const std::string& text) {
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_reasoning(id, model, created, text, include_usage));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const std::string_view remaining = unstreamed_content(outcome);
                if (!outcome.tool_calls.empty()) {
                    if (!remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_tool_calls(
                                          id, model, created, outcome.tool_calls, include_usage));
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created, "tool_calls", include_usage));
                } else {
                    if (tool_capable && !remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created,
                                              finish_reason_wire(outcome.finish_reason),
                                              include_usage));
                }
                if (include_usage) {
                    const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_usage(id, model, created, usage));
                }
                write_stream_item(sink, *stream, sse_done());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }
    try {
        RequestLimits limits;
        limits.default_max_tokens       = options_.default_max_tokens;
        const GenerationRequest request = parse_messages_request(body, limits);
        const int input_tokens          = service_->count_prompt_tokens(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
        res.set_content(make_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_messages_error(res, error);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }

    GenerationRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        // The Anthropic endpoint accepts any `model` string (Claude Code sends real
        // Claude model names) and echoes it back; it never 404s on model id.
        request = parse_messages_request(body, limits);
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_messages_error(res, error);
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
    } catch (const ApiException& e) {
        log_request_rejected(
            make_request_rejection_log_context(req_id, "anthropic_messages", request, e.error()));
        write_messages_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        log_request_rejected(
            make_request_rejection_log_context(req_id, "anthropic_messages", request, error));
        write_messages_error(res, error);
        return;
    }

    const std::string id    = new_message_id();
    const std::string model = request.model; // echo the requested model
    const int input_tokens  = prepared.prompt_tokens;

    const RequestLogContext log_context =
        make_request_log_context(req_id, "anthropic_messages", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr, [&req] {
                return req.is_connection_alive && !req.is_connection_alive();
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            const char* stop_reason =
                messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
            set_owned_content(res,
                              make_messages_response(id, model, outcome.text, outcome.reasoning,
                                                     outcome.tool_calls, stop_reason, usage),
                              prepared.lifetime);
        } catch (const ApiException& e) {
            log_request_error(log_context, e.error().message);
            write_messages_error(res, e.error());
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = e.what();
            write_messages_error(res, error);
        }
        return;
    }

    auto stream             = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool tool_capable = stream->prepared.tool_capable;

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, model, input_tokens, tool_capable,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;

            int next_index     = 0;
            bool thinking_open = false;
            int thinking_index = -1;
            bool text_open     = false;
            int text_index     = -1;
            try {
                write_stream_item(sink, *stream, make_message_start(id, model, input_tokens));

                StreamSink output;
                output.on_reasoning = [&](const std::string& text) {
                    if (!thinking_open) {
                        thinking_index = next_index++;
                        thinking_open  = true;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_thinking(thinking_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_thinking(thinking_index, text));
                };
                output.on_content = [&](const std::string& text) {
                    if (thinking_open) {
                        write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                        thinking_open = false;
                    }
                    if (!text_open) {
                        text_index = next_index++;
                        text_open  = true;
                        write_stream_item(sink, *stream, make_content_block_start_text(text_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_text(text_index, text));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const std::string_view remaining = unstreamed_content(outcome);

                if (thinking_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                    thinking_open = false;
                }
                if (text_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(text_index));
                    text_open = false;
                }

                if (tool_capable) {
                    if (!remaining.empty()) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream, make_content_block_start_text(idx));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_text(idx, std::string(remaining)));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                    for (const ToolCall& call : outcome.tool_calls) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_tool_use(idx, call));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_tool_json(idx, call.arguments_json));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                }

                if (next_index == 0) {
                    const int idx = next_index++;
                    write_stream_item(sink, *stream, make_content_block_start_text(idx));
                    write_stream_item(sink, *stream, make_content_block_stop(idx));
                }

                const char* stop_reason =
                    messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
                write_stream_item(sink, *stream,
                                  make_message_delta(stop_reason, outcome.completion_tokens));
                write_stream_item(sink, *stream, make_message_stop());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_id);
    service_                       = &service;
    request_jsonl_.write_server_start(options_, service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
