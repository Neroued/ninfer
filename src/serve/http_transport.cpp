#include "serve/http_transport.h"

#include "serve/request_validation.h"
#include "text/json_input.h"

#if defined(__linux__)
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#endif

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

#if defined(__linux__)
constexpr int kKeepAliveIdleSeconds                = 10;
constexpr int kKeepAliveIntervalSeconds            = 3;
constexpr int kKeepAliveProbeCount                 = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, &value, sizeof(value));
}
#endif

} // namespace

RequestJson parse_json_body(const httplib::Request& request) {
    // Before the parser: the prompt path reads client JSON recursively (capy-flash).
    check_request_json_nesting(request.body, {});
    RequestJson body;
    try {
        body = RequestJson::parse(request.body);
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
    // A body with an output-format schema is read a second time keeping the numeric spellings
    // (Neroued/ninfer 81c8ce09): a bound such as 1e400 would otherwise reach the grammar compiler
    // as a different number. Every other body is parsed once, exactly as before.
    if (has_output_format_schema(body)) {
        validate_schema_number_input(text::parse_json_numbers(request.body));
    }
    return body;
}

std::optional<std::string> parse_client_label(const httplib::Request& request) {
    const std::size_t count = request.get_header_value_count(kClientLabelHeader);
    if (count == 0) { return std::nullopt; }
    const auto allowed = [](char value) {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
               (value >= '0' && value <= '9') || value == '.' || value == '_' || value == ':' ||
               value == '/' || value == '@' || value == '+' || value == '-';
    };
    std::string label = count == 1 ? request.get_header_value(kClientLabelHeader) : std::string();
    if (label.empty() || label.size() > kClientLabelMaxLength ||
        !std::all_of(label.begin(), label.end(), allowed)) {
        bad_request(std::string(kClientLabelHeader) + " must be one value of 1 to " +
                        std::to_string(kClientLabelMaxLength) +
                        " characters from A-Z a-z 0-9 . _ : / @ + -",
                    kClientLabelHeader, "invalid_client_label");
    }
    return label;
}

bool client_disconnected(const httplib::Request& request) { return request.is_connection_closed(); }

void prepare_sse_response(httplib::Response& response) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
}

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                           Clock::duration heartbeat_interval, Clock::time_point now)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval),
      last_write_(now) {
    if (heartbeat_interval_ <= Clock::duration::zero()) {
        throw std::invalid_argument("SSE heartbeat interval must be positive");
    }
}

bool SseTransport::mark_cancelled() noexcept {
    cancelled_.store(true, std::memory_order_release);
    return true;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire) || !sink_.write(item.data(), item.size())) {
        mark_cancelled();
        throw ClientDisconnected();
    }
    last_write_ = now;
}

void SseTransport::write(const std::vector<std::string>& items, Clock::time_point now) {
    for (const std::string& item : items) { write(item, now); }
}

bool SseTransport::poll(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    if (sink_.is_writable && !sink_.is_writable()) { return mark_cancelled(); }
    if (now - last_write_ < heartbeat_interval_) { return false; }
    if (!sink_.write(kHeartbeatComment.data(), kHeartbeatComment.size())) {
        return mark_cancelled();
    }
    last_write_ = now;
    return false;
}

void configure_http_server_socket(socket_t socket) noexcept {
#if defined(__linux__)
    const int enabled = 1;
    // Issue #24: SO_REUSEADDR instead of httplib's default SO_REUSEPORT. SO_REUSEPORT let a second
    // server bind the same port and silently share the connections with the first, and binds over
    // TIME-WAIT only when the sockets that left it had SO_REUSEPORT too. With SO_REUSEADDR a second
    // server is refused and a restart binds over the TIME-WAIT its predecessor left.
    set_socket_option(socket, SOL_SOCKET, SO_REUSEADDR, enabled);
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#else
    httplib::default_socket_options(socket);
#endif
}

void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.user_data.set("ninfer.request_lifetime", std::move(lifetime));
}

} // namespace ninfer::serve
