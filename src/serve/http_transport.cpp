#include "serve/http_transport.h"

#include "core/cpu/task_pool.h"
#include "serve/request_validation.h"
#include "text/json_input.h"

#if defined(__linux__)
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#endif

#include <stdexcept>
#include <algorithm>
#include <limits>
#include <utility>

namespace ninfer::serve {
namespace {

class HttpTaskQueue final : public httplib::TaskQueue {
public:
    HttpTaskQueue(std::uint32_t workers, std::size_t capacity)
        : pool_({.threads = workers, .queue_capacity = capacity}) {}

    bool enqueue(std::function<void()> task) override {
        return pool_.try_enqueue(task) == EnqueueResult::Accepted;
    }

    void shutdown() override { pool_.shutdown(); }

private:
    TaskPool pool_;
};

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

std::unique_ptr<httplib::TaskQueue> make_http_task_queue(std::size_t workers,
                                                         std::size_t queued_requests) {
    if (workers > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("HTTP worker count exceeds uint32");
    return std::make_unique<HttpTaskQueue>(static_cast<std::uint32_t>(workers), queued_requests);
}

std::unique_ptr<httplib::TaskQueue> make_http_task_queue(const product::ServeOptions& options,
                                                       std::size_t reserved_workers) {
    const std::size_t outstanding =
        static_cast<std::size_t>(options.launch.engine.max_concurrency) +
        options.launch.engine.max_pending_requests +
        (options.launch.engine.enable_decisions ? kDecisionRequestCapacity : 0);
    // Long-lived monitor subscribers must not consume Jev's independent admission budget.
    return make_http_task_queue(outstanding + 1 + reserved_workers, outstanding);
}

RequestJson parse_json_body(const httplib::Request& request) {
    text::ParsedJsonNumbers parsed;
    try {
        parsed = text::parse_json_numbers(request.body);
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
    validate_schema_number_input(parsed);
    return std::move(parsed.value);
}

bool client_disconnected(const httplib::Request& request) { return request.is_connection_closed(); }

bool permits_local_media(const httplib::Request& request) {
    if (request.remote_addr != "127.0.0.1" && request.remote_addr != "::1" &&
        request.remote_addr != "::ffff:127.0.0.1")
        return false;
    for (const auto& [name, value] : request.headers) {
        std::string lower = name;
        for (auto& c : lower)
            if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        if (lower == "origin" || lower == "forwarded" || lower.starts_with("x-forwarded-"))
            return false;
    }
    const auto host = request.get_header_value("Host");
    for (const std::string_view local : {"localhost", "127.0.0.1", "[::1]"})
        if (host == local || (host.starts_with(local) && host.size() > local.size() + 1 &&
                              host[local.size()] == ':' &&
                              std::all_of(host.begin() + local.size() + 1, host.end(),
                                          [](char c) { return c >= '0' && c <= '9'; })))
            return true;
    return false;
}

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
    // Reuse closed connections' addresses while keeping the active listener exclusive.
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
