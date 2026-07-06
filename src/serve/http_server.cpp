#include "qus/serve/http_server.h"

#include "qus/serve/openai_schema.h"
#include "qus/serve/request_log.h"
#include "qus/serve/translate.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace qus::serve {
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

HttpServer::HttpServer(GenerationService& service, ServeOptions options)
    : service_(service), options_(std::move(options)) {
    register_routes();
}

void HttpServer::log_line(const std::string& line) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::cerr << "qus-serve: " << line << '\n';
}

void HttpServer::register_routes() {
    if (options_.enable_cors) {
        server_.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                                     {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
                                     {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
            res.status = 204;
        });
    }

    server_.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
            if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            if (req.get_header_value("Authorization") != ("Bearer " + options_.api_key)) {
                ApiError error;
                error.status  = 401;
                error.type    = "invalid_request_error";
                error.code    = "invalid_api_key";
                error.message = "missing or invalid API key";
                write_error(res, error);
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
            } catch (const std::exception& e) {
                write_exception(res, e);
            } catch (...) {
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
    server_.Get("/v1/models",
                [this](const httplib::Request& req, httplib::Response& res) { handle_models(req, res); });
    server_.Get(R"(/v1/models/(.+))",
                [this](const httplib::Request& req, httplib::Response& res) { handle_model(req, res); });
    server_.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handle_chat_completions(req, res);
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
        limits.max_context        = options_.max_context;
        request                   = parse_chat_completion_request(body, limits);
        if (request.model != options_.model_id) {
            ApiError error;
            error.status  = 404;
            error.type    = "invalid_request_error";
            error.code    = "model_not_found";
            error.message = "model '" + request.model + "' not found";
            throw ApiException(std::move(error));
        }
        prepared = service_.prepare(request);
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::string id       = new_chat_completion_id();
    const std::int64_t created = unix_time_now();
    const std::string model    = request.model;

    const std::uint64_t req_id = ++request_seq_;
    log_line(format_request_start(req_id, request.stream, request.messages.size(),
                                  prepared.options.max_new_tokens, request.max_tokens_set,
                                  request.tools.size(), request.tool_choice,
                                  request.has_tool_history()));

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_.run(prepared, nullptr);
            log_line(format_request_done(req_id, outcome));
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            if (!outcome.tool_calls.empty()) {
                res.set_content(make_chat_completion_tool_response(id, model, created, outcome.text,
                                                                   outcome.reasoning,
                                                                   outcome.tool_calls, usage),
                                "application/json");
            } else {
                res.set_content(make_chat_completion_response(id, model, created, outcome.text,
                                                              outcome.reasoning,
                                                              finish_reason_wire(outcome.finish_reason),
                                                              usage),
                                "application/json");
            }
        } catch (const std::exception& e) {
            log_line(format_request_error(req_id, e.what()));
            throw;
        }
        return;
    }

    auto queue     = std::make_shared<SseQueue>();
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    auto worker    = std::make_shared<JoiningThread>();
    auto prepared_ptr = std::make_shared<PreparedRequest>(std::move(prepared));
    const bool include_usage = prepared_ptr->include_usage;
    const bool tool_capable  = prepared_ptr->tool_capable;

    worker->thread = std::thread([this, queue, cancelled, prepared_ptr, id, created, model,
                                  include_usage, tool_capable, req_id]() {
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

            const GenerationOutcome outcome = service_.run(*prepared_ptr, &sink);
            log_line(format_request_done(req_id, outcome));
            if (!outcome.tool_calls.empty()) {
                if (!outcome.text.empty()) {
                    queue->push(make_chat_chunk_content(id, model, created, outcome.text,
                                                        include_usage));
                }
                queue->push(make_chat_chunk_tool_calls(id, model, created, outcome.tool_calls,
                                                       include_usage));
                queue->push(make_chat_chunk_final(id, model, created, "tool_calls", include_usage));
            } else {
                if (tool_capable && !outcome.text.empty()) {
                    queue->push(make_chat_chunk_content(id, model, created, outcome.text,
                                                        include_usage));
                }
                queue->push(make_chat_chunk_final(id, model, created,
                                                  finish_reason_wire(outcome.finish_reason),
                                                  include_usage));
            }
            if (include_usage) {
                const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
                queue->push(make_chat_chunk_usage(id, model, created, usage));
            }
            queue->push(sse_done());
        } catch (const ApiException& e) {
            log_line(format_request_error(req_id, e.error().message));
            queue->push(sse_error_event(e.error()));
        } catch (const std::exception& e) {
            log_line(format_request_error(req_id, e.what()));
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

bool HttpServer::listen() { return server_.listen(options_.host, options_.port); }

void HttpServer::stop() { server_.stop(); }

} // namespace qus::serve
