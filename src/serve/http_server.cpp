#include "serve/http_server.h"

#include "serve/anthropic_schema.h"
#include "serve/openai_schema.h"
#include "serve/request_log.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ninfer::serve {
namespace {

// Unbounded, non-blocking-producer SSE event queue. The producer (generation
// worker) never blocks; the consumer (httplib content provider) waits for items.
class SseQueue {
public:
    enum class PopStatus { Item, Timeout, Done };

    void push(std::string item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            items_.push_back(std::move(item));
        }
        cv_.notify_one();
    }

    // Waits up to `timeout` for an item. Returns Item when one is dequeued, Done
    // when the producer finished and the queue is drained, or Timeout otherwise.
    // The timeout lets the consumer poll for client disconnect while the worker
    // is still in a long prefill (or blocked on the engine mutex) and thus not
    // yet producing chunks.
    PopStatus pop(std::string& out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [&] { return !items_.empty() || producer_done_; })) {
            return PopStatus::Timeout;
        }
        if (items_.empty()) { return PopStatus::Done; }
        out = std::move(items_.front());
        items_.pop_front();
        return PopStatus::Item;
    }

    void mark_producer_done() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            producer_done_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> items_;
    bool producer_done_ = false;
};

// RAII wrapper that guarantees the generation worker is joined on every response
// termination path (normal completion or client disconnect), since httplib
// destroys the captured content-provider/releaser closures when the stream ends.
struct JoiningThread {
    std::thread thread;

    ~JoiningThread() {
        if (thread.joinable()) { thread.join(); }
    }
};

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

} // namespace

HttpServer::HttpServer(ServeOptions options)
    : options_(std::move(options)),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path) {
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

void HttpServer::log_line(const std::string& line) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::cerr << "ninfer-serve: " << line << '\n';
}

void HttpServer::log_request_start(const RequestLogContext& context) {
    log_line(format_request_start(context));
    request_jsonl_.write_request_start(context);
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

void HttpServer::register_routes() {
    server_.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (res.status != 413) { return; }
        ApiError error;
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit";
        if (req.path.rfind("/v1/messages", 0) == 0) {
            write_messages_error(res, error);
        } else {
            write_error(res, error);
        }
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
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
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(options_.model_id, unix_time_now()), "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != options_.model_id) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_error(res, error);
        return;
    }
    res.set_content(make_model_object(options_.model_id, unix_time_now()), "application/json");
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
    PreparedRequest prepared;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(body, limits);
        if (request.model != options_.model_id) {
            ApiError error;
            error.status  = 404;
            error.type    = "invalid_request_error";
            error.code    = "model_not_found";
            error.message = "model '" + request.model + "' not found";
            throw ApiException(std::move(error));
        }
        prepared = service_->prepare(request);
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::string id       = new_chat_completion_id();
    const std::int64_t created = unix_time_now();
    const std::string model    = request.model;

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogContext log_context =
        make_request_log_context(req_id, "openai_chat_completions", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr);
            log_request_done(log_context, outcome);
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            if (!outcome.tool_calls.empty()) {
                res.set_content(make_chat_completion_tool_response(id, model, created, outcome.text,
                                                                   outcome.reasoning,
                                                                   outcome.tool_calls, usage),
                                "application/json");
            } else {
                res.set_content(make_chat_completion_response(
                                    id, model, created, outcome.text, outcome.reasoning,
                                    finish_reason_wire(outcome.finish_reason), usage),
                                "application/json");
            }
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            throw;
        }
        return;
    }

    auto queue               = std::make_shared<SseQueue>();
    auto cancelled           = std::make_shared<std::atomic<bool>>(false);
    auto worker              = std::make_shared<JoiningThread>();
    auto prepared_ptr        = std::make_shared<PreparedRequest>(std::move(prepared));
    const bool include_usage = prepared_ptr->include_usage;
    const bool tool_capable  = prepared_ptr->tool_capable;

    worker->thread = std::thread([this, queue, cancelled, prepared_ptr, id, created, model,
                                  include_usage, tool_capable, log_context]() {
        try {
            queue->push(make_chat_chunk_role(id, model, created, include_usage));
            StreamSink sink;
            sink.on_content = [&](const std::string& text) {
                queue->push(make_chat_chunk_content(id, model, created, text, include_usage));
            };
            sink.on_reasoning = [&](const std::string& text) {
                queue->push(make_chat_chunk_reasoning(id, model, created, text, include_usage));
            };
            sink.is_cancelled = [&]() { return cancelled->load(); };

            const GenerationOutcome outcome = service_->run(*prepared_ptr, &sink);
            log_request_done(log_context, outcome);
            if (!outcome.tool_calls.empty()) {
                if (!outcome.text.empty()) {
                    queue->push(
                        make_chat_chunk_content(id, model, created, outcome.text, include_usage));
                }
                queue->push(make_chat_chunk_tool_calls(id, model, created, outcome.tool_calls,
                                                       include_usage));
                queue->push(make_chat_chunk_final(id, model, created, "tool_calls", include_usage));
            } else {
                if (tool_capable && !outcome.text.empty()) {
                    queue->push(
                        make_chat_chunk_content(id, model, created, outcome.text, include_usage));
                }
                queue->push(make_chat_chunk_final(
                    id, model, created, finish_reason_wire(outcome.finish_reason), include_usage));
            }
            if (include_usage) {
                const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
                queue->push(make_chat_chunk_usage(id, model, created, usage));
            }
            queue->push(sse_done());
        } catch (const ApiException& e) {
            log_request_error(log_context, e.error().message);
            queue->push(sse_error_event(e.error()));
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = e.what();
            queue->push(sse_error_event(error));
        }
        queue->mark_producer_done();
    });

    // SSE hints: disable client/proxy caching and reverse-proxy response buffering
    // so tokens flush immediately. Content-Type is set by the chunked provider.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [queue, cancelled](std::size_t, httplib::DataSink& sink) -> bool {
            using namespace std::chrono_literals;
            for (;;) {
                std::string item;
                const SseQueue::PopStatus status = queue->pop(item, 200ms);
                if (status == SseQueue::PopStatus::Done) {
                    sink.done();
                    return true;
                }
                if (status == SseQueue::PopStatus::Timeout) {
                    // No chunk yet (prefill/mutex wait). Detect a vanished client
                    // promptly so generation can be cancelled mid-prefill.
                    if (sink.is_writable && !sink.is_writable()) {
                        cancelled->store(true);
                        return false;
                    }
                    continue;
                }
                if (!sink.write(item.data(), item.size())) {
                    cancelled->store(true);
                    return false;
                }
                return true;
            }
        },
        [worker, cancelled](bool) {
            // Streaming ended (completed or client gone): ensure the worker stops
            // and is joined. The JoiningThread destructor joins when these
            // captured shared_ptrs are released by httplib.
            cancelled->store(true);
        });
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
        const int input_tokens          = service_->count_prompt_tokens(request);
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
    PreparedRequest prepared;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        // The Anthropic endpoint accepts any `model` string (Claude Code sends real
        // Claude model names) and echoes it back; it never 404s on model id.
        request  = parse_messages_request(body, limits);
        prepared = service_->prepare(request);
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

    const std::string id    = new_message_id();
    const std::string model = request.model; // echo the requested model
    const int input_tokens  = prepared.prompt_tokens;

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogContext log_context =
        make_request_log_context(req_id, "anthropic_messages", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr);
            log_request_done(log_context, outcome);
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            const char* stop_reason =
                messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
            res.set_content(make_messages_response(id, model, outcome.text, outcome.reasoning,
                                                   outcome.tool_calls, stop_reason, usage),
                            "application/json");
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

    auto queue              = std::make_shared<SseQueue>();
    auto cancelled          = std::make_shared<std::atomic<bool>>(false);
    auto worker             = std::make_shared<JoiningThread>();
    auto prepared_ptr       = std::make_shared<PreparedRequest>(std::move(prepared));
    const bool tool_capable = prepared_ptr->tool_capable;

    worker->thread = std::thread([this, queue, cancelled, prepared_ptr, id, model, input_tokens,
                                  tool_capable, log_context]() {
        // Anthropic content-block state machine: an optional thinking block (fed by
        // the reasoning channel) precedes an optional text block; tool_use blocks
        // are appended after generation. Block indices increase in emission order.
        int next_index     = 0;
        bool thinking_open = false;
        int thinking_index = -1;
        bool text_open     = false;
        int text_index     = -1;
        try {
            queue->push(make_message_start(id, model, input_tokens));

            StreamSink sink;
            sink.on_reasoning = [&](const std::string& text) {
                if (!thinking_open) {
                    thinking_index = next_index++;
                    thinking_open  = true;
                    queue->push(make_content_block_start_thinking(thinking_index));
                }
                queue->push(make_content_block_delta_thinking(thinking_index, text));
            };
            sink.on_content = [&](const std::string& text) {
                // Reasoning fully precedes the answer in Qwen output; close the
                // thinking block before the first text delta.
                if (thinking_open) {
                    queue->push(make_content_block_stop(thinking_index));
                    thinking_open = false;
                }
                if (!text_open) {
                    text_index = next_index++;
                    text_open  = true;
                    queue->push(make_content_block_start_text(text_index));
                }
                queue->push(make_content_block_delta_text(text_index, text));
            };
            sink.is_cancelled = [&]() { return cancelled->load(); };

            const GenerationOutcome outcome = service_->run(*prepared_ptr, &sink);
            log_request_done(log_context, outcome);

            // Close any block still open from live streaming.
            if (thinking_open) {
                queue->push(make_content_block_stop(thinking_index));
                thinking_open = false;
            }
            if (text_open) {
                queue->push(make_content_block_stop(text_index));
                text_open = false;
            }

            // In tool mode the service buffers the answer instead of streaming it;
            // emit the text and tool_use blocks now from the final outcome.
            if (tool_capable) {
                if (!outcome.text.empty()) {
                    const int idx = next_index++;
                    queue->push(make_content_block_start_text(idx));
                    queue->push(make_content_block_delta_text(idx, outcome.text));
                    queue->push(make_content_block_stop(idx));
                }
                for (const ToolCall& call : outcome.tool_calls) {
                    const int idx = next_index++;
                    queue->push(make_content_block_start_tool_use(idx, call));
                    queue->push(make_content_block_delta_tool_json(idx, call.arguments_json));
                    queue->push(make_content_block_stop(idx));
                }
            }

            if (next_index == 0) {
                // Nothing was produced; Anthropic content must not be empty.
                const int idx = next_index++;
                queue->push(make_content_block_start_text(idx));
                queue->push(make_content_block_stop(idx));
            }

            const char* stop_reason =
                messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
            queue->push(make_message_delta(stop_reason, outcome.completion_tokens));
            queue->push(make_message_stop());
        } catch (const ApiException& e) {
            log_request_error(log_context, e.error().message);
            queue->push(messages_sse_error_event(e.error()));
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = e.what();
            queue->push(messages_sse_error_event(error));
        }
        queue->mark_producer_done();
    });

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [queue, cancelled](std::size_t, httplib::DataSink& sink) -> bool {
            using namespace std::chrono_literals;
            for (;;) {
                std::string item;
                const SseQueue::PopStatus status = queue->pop(item, 200ms);
                if (status == SseQueue::PopStatus::Done) {
                    sink.done();
                    return true;
                }
                if (status == SseQueue::PopStatus::Timeout) {
                    if (sink.is_writable && !sink.is_writable()) {
                        cancelled->store(true);
                        return false;
                    }
                    continue;
                }
                if (!sink.write(item.data(), item.size())) {
                    cancelled->store(true);
                    return false;
                }
                return true;
            }
        },
        [worker, cancelled](bool) { cancelled->store(true); });
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    service_ = &service;
    request_jsonl_.write_server_start(options_, service.load_summary(), service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    return server_.listen_after_bind();
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
