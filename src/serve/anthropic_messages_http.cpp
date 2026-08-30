#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace ninfer::serve {

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);
    try {
        const AnthropicCountTokensRequest request =
            parse_anthropic_count_tokens_request(parse_json_body(req), anthropic_thinking_signer_);
        const int input_tokens = service_->count_prompt_tokens(
            request.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_anthropic_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);

    AnthropicMessagesRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_anthropic_messages_request(parse_json_body(req), limits,
                                                                     anthropic_thinking_signer_);
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
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
        const ApiError error = normalize_anthropic_error(exception.error());
        log_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        log_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    }

    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity(request_id, request.model);
    const int input_tokens = prepared.prompt_tokens;

    const RequestLogContext log_context = make_request_log_context(
        req_id, "anthropic_messages", request.generation, metadata, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome =
                service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            log_request_done(log_context, outcome);
            set_owned_json_content(
                res,
                make_anthropic_messages_response(identity, outcome, anthropic_thinking_signer_),
                prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            log_request_error(log_context, error.message);
            write_anthropic_error(res, error, request_id);
        } catch (const std::exception& exception) {
            log_request_error(log_context, exception.what());
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
        }
        return;
    }

    auto stream  = std::make_shared<HttpGenerationStream>(std::move(prepared));
    auto encoder = std::make_shared<AnthropicMessagesStream>(identity, input_tokens,
                                                             anthropic_thinking_signer_);

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
                StreamSink output;
                output.on_start = [&](const ninfer::GenerationStart& start) {
                    transport.write(encoder->start(start));
                };
                output.on_reasoning = [&](const std::string& text) {
                    transport.write(encoder->reasoning_delta(text));
                };
                output.on_content = [&](const std::string& text) {
                    transport.write(encoder->content_delta(text));
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
                const ApiError error = normalize_anthropic_error(exception.error());
                log_request_error(log_context, error.message);
                try {
                    if (!encoder->started()) { transport.write(encoder->start()); }
                    transport.write(encoder->error(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& exception) {
                log_request_error(log_context, exception.what());
                ApiError error;
                error.status  = 500;
                error.message = exception.what();
                try {
                    if (!encoder->started()) { transport.write(encoder->start()); }
                    transport.write(encoder->error(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

} // namespace ninfer::serve
