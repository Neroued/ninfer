#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/openai_responses.h"
#include "serve/request_validation.h"

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
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

struct PendingResponseStorage {
    std::vector<ChatTurn> input_turns;
    std::vector<Json> input_items;
    OpenAIResponseContext previous_context;
    std::string session_key;
    bool enabled = false;
};

struct StreamingResponse {
    PreparedRequest prepared;
    PendingResponseStorage storage;
    RequestLogContext log_context;
    std::unique_ptr<OpenAIResponsesEventStream> encoder;
    std::atomic<bool> cancelled{false};
    bool started = false;
};

ApiError responses_error(ApiError error) {
    if (error.param == "messages") { error.param = "input"; }
    if (error.param == "reasoning_effort") { error.param = "reasoning.effort"; }
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

OpenAIResponseContext terminal_context(OpenAIResponseContext previous,
                                       std::vector<ChatTurn> input_turns,
                                       std::vector<ChatTurn> output_history) {
    OpenAIResponseContext input =
        append_openai_response_context(std::move(previous), std::move(input_turns));
    return append_openai_response_context(std::move(input), std::move(output_history));
}

void commit_stored_response(OpenAIResponsesStore& store, PendingResponseStorage pending,
                            std::string id, const Json& response,
                            std::vector<ChatTurn> output_history, bool preserve_thinking) {
    if (!pending.enabled) { return; }
    if (pending.session_key.empty()) {
        throw std::logic_error("stored Response has no Engine session key");
    }
    StoredOpenAIResponse stored;
    stored.id                = std::move(id);
    stored.session_key       = std::move(pending.session_key);
    stored.response          = response;
    stored.input_items       = std::move(pending.input_items);
    stored.context           = terminal_context(std::move(pending.previous_context),
                                                std::move(pending.input_turns), std::move(output_history));
    stored.preserve_thinking = preserve_thinking;
    store.put(std::move(stored));
}

OpenAIResponsesRuntimeValues runtime_values(const PreparedRequest& prepared,
                                            const GenerationOutcome* outcome = nullptr) {
    OpenAIResponsesRuntimeValues runtime;
    runtime.temperature = prepared.sampling.temperature;
    runtime.top_p       = prepared.sampling.top_p;
    if (outcome != nullptr) {
        runtime.cached_input_tokens = static_cast<int>(outcome->metrics.prefix_cache_hit_tokens);
    }
    return runtime;
}

[[noreturn]] void invalid_query(std::string message, std::string param,
                                std::string code = "invalid_value") {
    bad_request(std::move(message), std::move(param), std::move(code));
}

void reject_unknown_query(const httplib::Request& request,
                          const std::vector<std::string_view>& allowed) {
    for (const auto& [key, value] : request.params) {
        (void)value;
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            invalid_query("unknown query parameter: " + key, key, "unknown_parameter");
        }
    }
}

bool query_bool(const httplib::Request& request, const char* key, bool fallback) {
    if (!request.has_param(key)) { return fallback; }
    const std::string value = request.get_param_value(key);
    if (value == "true") { return true; }
    if (value == "false") { return false; }
    invalid_query(std::string(key) + " must be 'true' or 'false'", key);
}

void validate_retrieve_query(const httplib::Request& request) {
    reject_unknown_query(request, {"include", "include_obfuscation", "starting_after", "stream"});
    if (request.has_param("include") && !request.get_param_value("include").empty()) {
        invalid_query("additional stored Response fields are not available", "include",
                      "include_not_supported");
    }
    (void)query_bool(request, "include_obfuscation", true);
    if (request.has_param("starting_after") && !request.get_param_value("starting_after").empty()) {
        invalid_query("stream event recovery is not available for stored Responses",
                      "starting_after", "stream_recovery_not_supported");
    }
    if (query_bool(request, "stream", false)) {
        invalid_query("stored Response event streams are not retained", "stream",
                      "retrieve_stream_not_supported");
    }
}

void redact_input_image_urls(Json& item) {
    if (!item.is_object()) { return; }
    const auto redact_parts = [](Json& parts) {
        if (!parts.is_array()) { return; }
        for (Json& part : parts) {
            if (part.is_object() && part.value("type", "") == "input_image") {
                part.erase("image_url");
            }
        }
    };
    if (item.contains("content")) { redact_parts(item["content"]); }
    if (item.value("type", "") == "function_call_output" && item.contains("output")) {
        redact_parts(item["output"]);
    }
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
    reject_unknown_query(request, {"after", "include", "limit", "order"});
    bool include_image_urls = false;
    if (request.has_param("include") && !request.get_param_value("include").empty()) {
        if (request.get_param_value("include") != "message.input_image.image_url") {
            invalid_query("only include=message.input_image.image_url is available", "include",
                          "include_not_supported");
        }
        include_image_urls = true;
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
    for (std::size_t index = begin; index < end; ++index) {
        Json item = ordered[index];
        if (!include_image_urls) { redact_input_image_urls(item); }
        data.push_back(std::move(item));
    }
    return Json{{"object", "list"},
                {"data", data},
                {"first_id", data.empty() ? Json(nullptr) : data.front().at("id")},
                {"last_id", data.empty() ? Json(nullptr) : data.back().at("id")},
                {"has_more", end < ordered.size()}};
}

} // namespace

void HttpServer::handle_responses(const httplib::Request& req, httplib::Response& res) {
    OpenAIResponsesCreateRequest request;
    OpenAIResponsesResolvedPrompt resolved;
    const std::string id = new_openai_response_id();
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request = parse_openai_responses_create_request(parse_json_body(req), limits);
        validate_openai_model(request.prompt.model, public_model_id_);
        resolved = resolve_openai_responses_prompt(request.prompt, openai_responses_store_, id,
                                                   request.store);
    } catch (const ApiException& exception) {
        write_openai_error(res, responses_error(exception.error()));
        return;
    } catch (const std::exception& exception) {
        write_openai_error(res, internal_error(exception));
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{
        .model                             = request.prompt.model,
        .stream                            = request.stream,
        .output_tokens_explicit            = request.requested_max_output_tokens.has_value(),
        .preserve_thinking_semantic_change = resolved.preserve_thinking_semantic_change,
    };
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            resolved.generation,
            request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate,
            [&req] { return client_disconnected(req); }, std::move(resolved.cache_hints));
    } catch (const ApiException& exception) {
        const ApiError error = responses_error(exception.error());
        log_request_rejected(make_request_rejection_log_context(
            req_id, "openai_responses", resolved.generation, metadata, error));
        write_openai_error(res, error);
        return;
    } catch (const std::exception& exception) {
        const ApiError error = internal_error(exception);
        log_request_rejected(make_request_rejection_log_context(
            req_id, "openai_responses", resolved.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const std::int64_t created          = unix_time_now();
    const RequestLogContext log_context = make_request_log_context(
        req_id, "openai_responses", resolved.generation, metadata, prepared);
    resolved.generation.messages.clear();
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome =
                service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            const OpenAIResponsesRuntimeValues runtime = runtime_values(prepared, &outcome);
            BuiltOpenAIResponse response =
                make_openai_response_object(id, created, request, runtime, outcome);
            PendingResponseStorage storage;
            storage.input_turns      = std::move(request.prompt.input_turns);
            storage.input_items      = std::move(request.prompt.input_items);
            storage.previous_context = std::move(resolved.parent);
            if (resolved.session_key) { storage.session_key = std::move(*resolved.session_key); }
            storage.enabled = request.store;
            commit_stored_response(openai_responses_store_, std::move(storage), id, response.body,
                                   std::move(response.output_history), prepared.preserve_thinking);
            log_request_done(log_context, outcome);
            set_owned_json_content(res, response.body.dump(), prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = responses_error(exception.error());
            log_request_error(log_context, error.message);
            write_openai_error(res, error);
        } catch (const std::exception& exception) {
            log_request_error(log_context, exception.what());
            write_openai_error(res, internal_error(exception));
        }
        return;
    }

    auto stream                      = std::make_shared<StreamingResponse>();
    stream->prepared                 = std::move(prepared);
    stream->storage.input_turns      = std::move(request.prompt.input_turns);
    stream->storage.input_items      = std::move(request.prompt.input_items);
    stream->storage.previous_context = std::move(resolved.parent);
    if (resolved.session_key) { stream->storage.session_key = std::move(*resolved.session_key); }
    stream->storage.enabled = request.store;
    stream->log_context     = log_context;
    stream->encoder         = std::make_unique<OpenAIResponsesEventStream>(
        id, created, std::move(request), runtime_values(stream->prepared));

    prepare_sse_response(res);
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            SseTransport transport(sink, stream->cancelled);
            try {
                transport.write(stream->encoder->start());
                StreamSink output;
                output.on_reasoning = [&](const std::string& text) {
                    transport.write(stream->encoder->reasoning_delta(text));
                };
                output.on_content = [&](const std::string& text) {
                    transport.write(stream->encoder->content_delta(text));
                };
                output.is_cancelled = [&] { return transport.poll(); };

                const GenerationOutcome outcome      = service_->run(stream->prepared, &output);
                OpenAIResponsesStreamFinish finished = stream->encoder->finish(outcome);
                commit_stored_response(openai_responses_store_, std::move(stream->storage), id,
                                       finished.response.body,
                                       std::move(finished.response.output_history),
                                       stream->prepared.preserve_thinking);
                transport.write(finished.events_before_terminal);
                log_request_done(stream->log_context, outcome);
                transport.write(stream->encoder->terminal(finished.response));
                sink.done();
                return true;
            } catch (const ClientDisconnected& exception) {
                log_request_error(stream->log_context, exception.what());
                return false;
            } catch (const ApiException& exception) {
                const ApiError error = responses_error(exception.error());
                log_request_error(stream->log_context, error.message);
                try {
                    transport.write(stream->encoder->failed(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& exception) {
                const ApiError error = internal_error(exception);
                log_request_error(stream->log_context, error.message);
                try {
                    transport.write(stream->encoder->failed(error));
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
        OpenAIResponsesPromptRequest request =
            parse_openai_responses_input_tokens_request(parse_json_body(req), limits);
        validate_openai_model(request.model, public_model_id_);
        OpenAIResponsesResolvedPrompt resolved =
            resolve_openai_responses_prompt(request, openai_responses_store_, std::nullopt, false);
        const int tokens = service_->count_prompt_tokens(
            resolved.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_openai_response_input_tokens_body(tokens), "application/json");
    } catch (const ApiException& exception) {
        write_openai_error(res, responses_error(exception.error()));
    } catch (const std::exception& exception) {
        write_openai_error(res, internal_error(exception));
    }
}

void HttpServer::handle_response_get(const httplib::Request& req, httplib::Response& res) {
    try {
        validate_retrieve_query(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    const std::string id                                     = path_response_id(req);
    const std::shared_ptr<const StoredOpenAIResponse> stored = openai_responses_store_.get(id);
    if (!stored) {
        write_openai_error(res, response_not_found(id));
        return;
    }
    res.set_content(stored->response.dump(), "application/json");
}

void HttpServer::handle_response_delete(const httplib::Request& req, httplib::Response& res) {
    const std::string id = path_response_id(req);
    if (!openai_responses_store_.erase(id)) {
        write_openai_error(res, response_not_found(id));
        return;
    }
    res.set_content(Json{{"id", id}, {"object", "response.deleted"}, {"deleted", true}}.dump(),
                    "application/json");
}

void HttpServer::handle_response_input_items(const httplib::Request& req, httplib::Response& res) {
    const std::string id                                     = path_response_id(req);
    const std::shared_ptr<const StoredOpenAIResponse> stored = openai_responses_store_.get(id);
    if (!stored) {
        write_openai_error(res, response_not_found(id));
        return;
    }
    try {
        res.set_content(paginated_input_items(req, stored->input_items).dump(), "application/json");
    } catch (const ApiException& exception) { write_openai_error(res, exception.error()); }
}

void HttpServer::handle_response_cancel(const httplib::Request& req, httplib::Response& res) {
    const std::string id = path_response_id(req);
    if (!openai_responses_store_.get(id)) {
        write_openai_error(res, response_not_found(id));
        return;
    }
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.code    = "background_not_supported";
    error.message = "only background responses can be cancelled; NInfer does not support "
                    "background execution";
    write_openai_error(res, error);
}

void HttpServer::handle_response_compact(const httplib::Request&, httplib::Response& res) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.param   = "context_management";
    error.code    = "compaction_not_supported";
    error.message = "Responses compaction is not supported";
    write_openai_error(res, error);
}

} // namespace ninfer::serve
