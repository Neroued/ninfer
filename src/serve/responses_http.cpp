#include "serve/http_server.h"

#include "serve/openai_schema.h"
#include "serve/responses_schema.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

struct StreamingResponse {
    PreparedRequest prepared;
    std::vector<ChatTurn> input_turns;
    std::vector<Json> input_items;
    ResponseContext previous_context;
    std::string session_key;
    RequestLogContext log_context;
    std::unique_ptr<ResponsesEventStream> encoder;
    std::atomic<bool> cancelled{false};
    bool store   = false;
    bool started = false;
};

void write_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

ApiError responses_error(ApiError error) {
    if (error.param == "messages") { error.param = "input"; }
    return error;
}

ApiError internal_error(const std::exception& exception) {
    ApiError error;
    error.status  = 500;
    error.type    = "server_error";
    error.message = exception.what();
    return error;
}

ApiError response_not_found(const std::string& id) {
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "response_id";
    error.code    = "response_not_found";
    error.message = "response '" + id + "' not found";
    return error;
}

void validate_model(const std::string& requested, const std::string& available) {
    if (requested == available) { return; }
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "model";
    error.code    = "model_not_found";
    error.message = "model '" + requested + "' not found";
    throw ApiException(std::move(error));
}

Json parse_json_body(const httplib::Request& request) {
    try {
        return Json::parse(request.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.type    = "invalid_request_error";
        error.message = "request body is not valid JSON";
        throw ApiException(std::move(error));
    }
}

bool disconnected(const httplib::Request& request) {
    return request.is_connection_alive && !request.is_connection_alive();
}

void write_stream_item(httplib::DataSink& sink, StreamingResponse& request,
                       const std::string& item) {
    if (request.cancelled.load(std::memory_order_acquire) ||
        !sink.write(item.data(), item.size())) {
        request.cancelled.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

void write_stream_items(httplib::DataSink& sink, StreamingResponse& request,
                        std::vector<std::string> items) {
    for (const std::string& item : items) { write_stream_item(sink, request, item); }
}

void set_owned_content(httplib::Response& response, std::string body,
                       std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

ResponseContext terminal_context(ResponseContext previous, std::vector<ChatTurn> input_turns,
                                 std::vector<ChatTurn> output_history) {
    ResponseContext input = append_response_context(std::move(previous), std::move(input_turns));
    return append_response_context(std::move(input), std::move(output_history));
}

ResponsesRuntimeValues runtime_values(const PreparedRequest& prepared,
                                      const GenerationOutcome* outcome = nullptr) {
    ResponsesRuntimeValues runtime;
    runtime.temperature = prepared.sampling.temperature;
    runtime.top_p       = prepared.sampling.top_p;
    if (outcome != nullptr) {
        runtime.cached_input_tokens = static_cast<int>(outcome->metrics.prefix_cache_hit_tokens);
    }
    return runtime;
}

std::string path_response_id(const httplib::Request& request) {
    return request.matches.size() > 1 ? request.matches[1].str() : std::string();
}

int parse_limit(const httplib::Request& request) {
    if (!request.has_param("limit")) { return 20; }
    const std::string value = request.get_param_value("limit");
    int parsed              = 0;
    const auto result       = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 1 ||
        parsed > 100) {
        ApiError error;
        error.status  = 400;
        error.param   = "limit";
        error.code    = "invalid_pagination";
        error.message = "limit must be an integer in [1,100]";
        throw ApiException(std::move(error));
    }
    return parsed;
}

Json paginated_input_items(const httplib::Request& request, const std::vector<Json>& stored_items) {
    if (request.has_param("include") && !request.get_param_value("include").empty()) {
        ApiError error;
        error.status  = 400;
        error.param   = "include";
        error.code    = "include_not_supported";
        error.message = "additional input Item fields are not supported";
        throw ApiException(std::move(error));
    }
    const int limit   = parse_limit(request);
    std::string order = request.has_param("order") ? request.get_param_value("order") : "desc";
    if (order != "asc" && order != "desc") {
        ApiError error;
        error.status  = 400;
        error.param   = "order";
        error.code    = "invalid_pagination";
        error.message = "order must be 'asc' or 'desc'";
        throw ApiException(std::move(error));
    }

    std::vector<Json> ordered = stored_items;
    if (order == "desc") { std::reverse(ordered.begin(), ordered.end()); }
    std::size_t begin = 0;
    if (request.has_param("after")) {
        const std::string after = request.get_param_value("after");
        const auto found = std::find_if(ordered.begin(), ordered.end(), [&](const Json& item) {
            return item.contains("id") && item.at("id").is_string() &&
                   item.at("id").get<std::string>() == after;
        });
        if (found == ordered.end()) {
            ApiError error;
            error.status  = 400;
            error.param   = "after";
            error.code    = "invalid_pagination";
            error.message = "after does not identify an input Item in this response";
            throw ApiException(std::move(error));
        }
        begin = static_cast<std::size_t>(std::distance(ordered.begin(), found)) + 1;
    }
    const std::size_t end = std::min(ordered.size(), begin + static_cast<std::size_t>(limit));
    Json data             = Json::array();
    for (std::size_t index = begin; index < end; ++index) { data.push_back(ordered[index]); }
    return Json{{"object", "list"},
                {"data", data},
                {"first_id", data.empty() ? Json(nullptr) : data.front().at("id")},
                {"last_id", data.empty() ? Json(nullptr) : data.back().at("id")},
                {"has_more", end < ordered.size()}};
}

} // namespace

void HttpServer::handle_responses(const httplib::Request& req, httplib::Response& res) {
    ResponsesRequest request;
    ResponseContext previous_context;
    std::string session_key;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_responses_request(parse_json_body(req), limits);
        validate_model(request.generation.model, public_model_id_);
        if (request.previous_response_id) {
            const std::shared_ptr<const StoredResponse> previous =
                response_store_.get(*request.previous_response_id);
            if (!previous) {
                throw ApiException(response_not_found(*request.previous_response_id));
            }
            inherit_responses_preserve_thinking(request, previous->preserve_thinking);
            previous_context = previous->context;
            session_key      = previous->session_key;
        }
        compose_responses_generation_messages(request, flatten_response_context(previous_context));
    } catch (const ApiException& exception) {
        write_error(res, responses_error(exception.error()));
        return;
    } catch (const std::exception& exception) {
        write_error(res, internal_error(exception));
        return;
    }

    const std::string id = new_response_id();
    if (session_key.empty() && request.store) { session_key = id; }
    ContextCacheHints cache_hints;
    if (!session_key.empty()) { cache_hints.session_key = session_key; }
    cache_hints.retention =
        request.store ? CacheRetentionHint::LiveSession : CacheRetentionHint::Disposable;
    cache_hints.update_session_index = request.store;

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            request.generation, [&req] { return disconnected(req); }, std::move(cache_hints));
    } catch (const ApiException& exception) {
        const ApiError error = responses_error(exception.error());
        log_request_rejected(make_request_rejection_log_context(req_id, "openai_responses",
                                                                request.generation, error));
        write_error(res, error);
        return;
    } catch (const std::exception& exception) {
        const ApiError error = internal_error(exception);
        log_request_rejected(make_request_rejection_log_context(req_id, "openai_responses",
                                                                request.generation, error));
        write_error(res, error);
        return;
    }

    const std::int64_t created = unix_time_now();
    const RequestLogContext log_context =
        make_request_log_context(req_id, "openai_responses", request.generation, prepared);
    request.generation.messages.clear();
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome =
                service_->run(prepared, nullptr, [&req] { return disconnected(req); });
            const ResponsesRuntimeValues runtime = runtime_values(prepared, &outcome);
            BuiltResponse response = make_response_object(id, created, request, runtime, outcome);
            if (request.store) {
                StoredResponse stored;
                stored.id          = id;
                stored.session_key = session_key;
                stored.response    = response.body;
                stored.input_items = std::move(request.input_items);
                stored.context =
                    terminal_context(std::move(previous_context), std::move(request.input_turns),
                                     std::move(response.output_history));
                stored.preserve_thinking = prepared.preserve_thinking;
                response_store_.put(std::move(stored));
            }
            log_request_done(log_context, outcome);
            set_owned_content(res, response.body.dump(), prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = responses_error(exception.error());
            log_request_error(log_context, error.message);
            write_error(res, error);
        } catch (const std::exception& exception) {
            log_request_error(log_context, exception.what());
            write_error(res, internal_error(exception));
        }
        return;
    }

    auto stream              = std::make_shared<StreamingResponse>();
    stream->prepared         = std::move(prepared);
    stream->input_turns      = std::move(request.input_turns);
    stream->input_items      = std::move(request.input_items);
    stream->previous_context = std::move(previous_context);
    stream->session_key      = std::move(session_key);
    stream->log_context      = log_context;
    stream->store            = request.store;
    stream->encoder = std::make_unique<ResponsesEventStream>(id, created, std::move(request),
                                                             runtime_values(stream->prepared));

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            try {
                write_stream_items(sink, *stream, stream->encoder->start());
                StreamSink output;
                output.on_reasoning = [&](const std::string& text) {
                    write_stream_items(sink, *stream, stream->encoder->reasoning_delta(text));
                };
                output.on_content = [&](const std::string& text) {
                    write_stream_items(sink, *stream, stream->encoder->content_delta(text));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                ResponsesStreamFinish finished  = stream->encoder->finish(outcome);
                if (stream->store) {
                    StoredResponse stored;
                    stored.id                = finished.response.body.at("id").get<std::string>();
                    stored.session_key       = stream->session_key;
                    stored.response          = finished.response.body;
                    stored.input_items       = std::move(stream->input_items);
                    stored.context           = terminal_context(std::move(stream->previous_context),
                                                                std::move(stream->input_turns),
                                                                std::move(finished.response.output_history));
                    stored.preserve_thinking = stream->prepared.preserve_thinking;
                    response_store_.put(std::move(stored));
                }
                write_stream_items(sink, *stream, std::move(finished.events_before_terminal));
                log_request_done(stream->log_context, outcome);
                write_stream_item(sink, *stream, stream->encoder->terminal(finished.response));
                sink.done();
                return true;
            } catch (const ClientDisconnected& exception) {
                log_request_error(stream->log_context, exception.what());
                return false;
            } catch (const ApiException& exception) {
                const ApiError error = responses_error(exception.error());
                log_request_error(stream->log_context, error.message);
                try {
                    write_stream_item(sink, *stream, stream->encoder->failed(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& exception) {
                const ApiError error = internal_error(exception);
                log_request_error(stream->log_context, error.message);
                try {
                    write_stream_item(sink, *stream, stream->encoder->failed(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_response_input_tokens(const httplib::Request& req, httplib::Response& res) {
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        ResponsesRequest request =
            parse_response_input_tokens_request(parse_json_body(req), limits);
        validate_model(request.generation.model, public_model_id_);
        const int tokens =
            service_->count_prompt_tokens(request.generation, [&req] { return disconnected(req); });
        res.set_content(make_response_input_tokens_body(tokens), "application/json");
    } catch (const ApiException& exception) {
        write_error(res, responses_error(exception.error()));
    } catch (const std::exception& exception) { write_error(res, internal_error(exception)); }
}

void HttpServer::handle_response_get(const httplib::Request& req, httplib::Response& res) {
    const std::string id                               = path_response_id(req);
    const std::shared_ptr<const StoredResponse> stored = response_store_.get(id);
    if (!stored) {
        write_error(res, response_not_found(id));
        return;
    }
    res.set_content(stored->response.dump(), "application/json");
}

void HttpServer::handle_response_delete(const httplib::Request& req, httplib::Response& res) {
    const std::string id = path_response_id(req);
    if (!response_store_.erase(id)) {
        write_error(res, response_not_found(id));
        return;
    }
    res.set_content(Json{{"id", id}, {"object", "response.deleted"}, {"deleted", true}}.dump(),
                    "application/json");
}

void HttpServer::handle_response_input_items(const httplib::Request& req, httplib::Response& res) {
    const std::string id                               = path_response_id(req);
    const std::shared_ptr<const StoredResponse> stored = response_store_.get(id);
    if (!stored) {
        write_error(res, response_not_found(id));
        return;
    }
    try {
        res.set_content(paginated_input_items(req, stored->input_items).dump(), "application/json");
    } catch (const ApiException& exception) { write_error(res, exception.error()); }
}

void HttpServer::handle_response_cancel(const httplib::Request& req, httplib::Response& res) {
    const std::string id = path_response_id(req);
    if (!response_store_.get(id)) {
        write_error(res, response_not_found(id));
        return;
    }
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.code    = "background_not_supported";
    error.message = "only background responses can be cancelled; NInfer does not support "
                    "background execution";
    write_error(res, error);
}

void HttpServer::handle_response_compact(const httplib::Request&, httplib::Response& res) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.param   = "context_management";
    error.code    = "compaction_not_supported";
    error.message = "Responses compaction is not supported";
    write_error(res, error);
}

} // namespace ninfer::serve
