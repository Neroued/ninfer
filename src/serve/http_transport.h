#pragma once

#include "serve/generation_service.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

class ResponseRenderFailure final : public std::runtime_error {
public:
    explicit ResponseRenderFailure(const std::string& message) : std::runtime_error(message) {}
};

struct HttpGenerationStream {
    explicit HttpGenerationStream(PreparedRequest request) : prepared(std::move(request)) {}

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
};

class SseTransport final {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr std::chrono::seconds kHeartbeatInterval{5};
    static constexpr std::string_view kHeartbeatComment = ": keep-alive\n\n";

    explicit SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                          Clock::duration heartbeat_interval = kHeartbeatInterval,
                          Clock::time_point now              = Clock::now());

    void write(std::string_view item, Clock::time_point now = Clock::now());
    void write(const std::vector<std::string>& items, Clock::time_point now = Clock::now());

    // Called by the Engine wait loop. Besides observing an already-closed socket, a quiet stream
    // periodically writes an SSE comment so TCP_USER_TIMEOUT has traffic with which to detect an
    // unacknowledged peer. A failed probe marks the request for cancellation.
    [[nodiscard]] bool poll(Clock::time_point now = Clock::now());

private:
    bool mark_cancelled() noexcept;

    httplib::DataSink& sink_;
    std::atomic<bool>& cancelled_;
    Clock::duration heartbeat_interval_;
    Clock::time_point last_write_;
};

template <class Render>
void render_and_write(SseTransport& transport, Render&& render) {
    try {
        auto payload = std::forward<Render>(render)();
        transport.write(payload);
    } catch (const ClientDisconnected&) { throw; } catch (const ResponseRenderFailure&) {
        throw;
    } catch (const std::exception& exception) { throw ResponseRenderFailure(exception.what()); }
}

nlohmann::json parse_json_body(const httplib::Request& request);
[[nodiscard]] bool client_disconnected(const httplib::Request& request);

void prepare_sse_response(httplib::Response& response);
void configure_http_server_socket(socket_t socket) noexcept;
void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime);

} // namespace ninfer::serve
