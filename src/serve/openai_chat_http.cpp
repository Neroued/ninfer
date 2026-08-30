#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(parse_json_body(req), limits);
        validate_openai_model(request.model, public_model_id_);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(request.generation,
                                     request.stream ? GenerationConsumerMode::Streaming
                                                    : GenerationConsumerMode::Aggregate,
                                     [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        log_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        log_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    const RequestLogContext log_context       = make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome =
                service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            log_request_done(log_context, outcome);
            set_owned_json_content(res, make_chat_completion_response(identity, outcome),
                                   prepared.lifetime);
        } catch (const std::exception& exception) {
            log_request_error(log_context, exception.what());
            throw;
        }
        return;
    }

    auto stream  = std::make_shared<HttpGenerationStream>(std::move(prepared));
    auto encoder = std::make_shared<OpenAIChatStream>(identity, request.include_usage);

    prepare_sse_response(res);
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, encoder, log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            SseTransport transport(sink, stream->cancelled);
            try {
                transport.write(encoder->start());
                StreamSink output;
                output.on_content = [&](const std::string& text) {
                    transport.write(encoder->content_delta(text));
                };
                output.on_reasoning = [&](const std::string& text) {
                    transport.write(encoder->reasoning_delta(text));
                };
                output.is_cancelled = [&] { return transport.poll(); };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                transport.write(encoder->finish(outcome));
                sink.done();
                return true;
            } catch (const ClientDisconnected& exception) {
                log_request_error(log_context, exception.what());
                return false;
            } catch (const ApiException& exception) {
                log_request_error(log_context, exception.error().message);
                try {
                    transport.write(sse_error_event(exception.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& exception) {
                log_request_error(log_context, exception.what());
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = exception.what();
                try {
                    transport.write(sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

} // namespace ninfer::serve
