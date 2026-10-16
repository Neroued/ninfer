#include "serve/http_transport.h"
#include "serve/request.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <unistd.h>
#endif

namespace {

using ninfer::serve::ClientDisconnected;
using ninfer::serve::SseTransport;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int test_decision_transport_capacity() {
    using namespace std::chrono_literals;
    int failures = 0;
    for (const bool decisions : {false, true}) {
        ninfer::serve::ServeOptions options;
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.jev                  = decisions;
        const std::size_t outstanding = 2 + (decisions ? 17 : 0);
        std::unique_ptr<httplib::TaskQueue> queue(ninfer::serve::make_http_task_queue(options));
        std::mutex mutex;
        std::condition_variable changed;
        std::size_t started = 0;
        std::size_t finished = 0;
        bool release         = false;
        const auto hold = [&] {
            std::unique_lock lock(mutex);
            ++started;
            changed.notify_all();
            changed.wait(lock, [&] { return release; });
            ++finished;
        };
        bool ready = true;
        // Occupy every generation and decision lifetime, including response delivery.
        for (std::size_t i = 0; i < outstanding; ++i) {
            if (!queue->enqueue(hold)) { ready = false; break; }
            std::unique_lock lock(mutex);
            if (!changed.wait_for(lock, 2s, [&] { return started == i + 1; })) {
                ready = false;
                break;
            }
        }
        failures += check(ready, "transport cannot service both admission budgets concurrently");
        std::promise<void> responsive;
        auto response = responsive.get_future();
        if (ready) {
            failures += check(queue->enqueue([&] { responsive.set_value(); }) &&
                                  response.wait_for(2s) == std::future_status::ready,
                              "admitted inference stranded health/overload handling");
            // Fill the spare worker, then verify that the transport queue remains bounded.
            ready = queue->enqueue(hold);
            if (ready) {
                std::unique_lock lock(mutex);
                ready = changed.wait_for(lock, 2s, [&] { return started == outstanding + 1; });
            }
            failures += check(ready, "transport lost its spare handler");
            if (ready) {
                for (std::size_t i = 0; i < outstanding; ++i) {
                    failures += check(queue->enqueue(hold),
                                      "transport queue omitted admission capacity");
                }
                failures += check(!queue->enqueue(hold), "transport queue is unbounded");
            }
        }
        {
            std::lock_guard lock(mutex);
            release = true;
        }
        changed.notify_all();
        queue->shutdown();
        if (ready) {
            failures += check(finished == 2 * outstanding + 1,
                              "transport did not drain its admitted and queued work");
        }
    }
    return failures;
}

int test_task_queue() {
    std::atomic<unsigned> executed{0};
    auto queue = ninfer::serve::make_http_task_queue(1, 1);
    std::promise<void> started, release;
    const auto gate = release.get_future().share();
    int failures    = check(queue->enqueue([&, gate] {
        started.set_value();
        gate.wait();
        ++executed;
    }),
                            "HTTP queue rejected first request");
    started.get_future().get();
    failures += check(queue->enqueue([&] { ++executed; }), "HTTP queue rejected pending request");
    failures += check(!queue->enqueue([&] { ++executed; }), "HTTP queue failed to reject overflow");
    release.set_value();
    queue->shutdown();
    failures += check(executed == 2, "HTTP shutdown did not drain accepted requests");
    failures += check(!queue->enqueue([] {}), "HTTP queue accepted after shutdown");
    return failures;
}

int test_sse_transport() {
    using namespace std::chrono_literals;
    int failures = 0;

    std::vector<std::string> writes;
    bool writable = true;
    bool write_ok = true;
    httplib::DataSink sink;
    sink.write = [&](const char* data, std::size_t size) {
        if (!write_ok) { return false; }
        writes.emplace_back(data, size);
        return true;
    };
    sink.is_writable = [&] { return writable; };

    std::atomic<bool> cancelled{false};
    const SseTransport::Clock::time_point start{100s};
    SseTransport transport(sink, cancelled, 5s, start);

    failures += check(!transport.poll(start + 4999ms) && writes.empty() && !cancelled,
                      "SSE transport emitted a heartbeat before the quiet interval");
    failures +=
        check(!transport.poll(start + 5s) &&
                  writes == std::vector<std::string>{std::string(SseTransport::kHeartbeatComment)},
              "SSE transport did not emit the standard comment heartbeat on time");
    failures += check(!transport.poll(start + 9999ms) && writes.size() == 1,
                      "SSE heartbeat did not reset the quiet interval");

    transport.write("data: token\n\n", start + 10s);
    failures += check(!transport.poll(start + 14999ms) && writes.size() == 2,
                      "normal SSE output did not reset the heartbeat interval");
    failures += check(!transport.poll(start + 15s) && writes.size() == 3 &&
                          writes.back() == SseTransport::kHeartbeatComment,
                      "quiet SSE output did not resume heartbeats after a data event");

    transport.write(std::vector<std::string>{"data: one\n\n", "data: two\n\n"}, start + 16s);
    failures +=
        check(writes.size() == 5 && writes[3] == "data: one\n\n" && writes[4] == "data: two\n\n",
              "SSE transport changed ordered multi-event output");

    writable = false;
    failures += check(transport.poll(start + 16s) && cancelled,
                      "unwritable SSE transport did not cancel its request");
    bool cancelled_write_threw = false;
    try {
        transport.write("data: unreachable\n\n", start + 16s);
    } catch (const ClientDisconnected&) { cancelled_write_threw = true; }
    failures += check(cancelled_write_threw && writes.size() == 5,
                      "cancelled SSE transport accepted another event");

    httplib::DataSink failed_sink;
    failed_sink.write       = [](const char*, std::size_t) { return false; };
    failed_sink.is_writable = [] { return true; };
    std::atomic<bool> heartbeat_cancelled{false};
    SseTransport failed_heartbeat(failed_sink, heartbeat_cancelled, 5s, start);
    failures += check(failed_heartbeat.poll(start + 5s) && heartbeat_cancelled,
                      "failed SSE heartbeat did not cancel its request");

    write_ok = false;
    writable = true;
    std::atomic<bool> event_cancelled{false};
    SseTransport failed_event(sink, event_cancelled, 5s, start);
    bool failed_event_threw = false;
    try {
        failed_event.write("data: failed\n\n", start);
    } catch (const ClientDisconnected&) { failed_event_threw = true; }
    failures += check(failed_event_threw && event_cancelled,
                      "failed SSE event write did not cancel its request");

    std::atomic<bool> releaser_cancelled{true};
    SseTransport released(sink, releaser_cancelled, 5s, start);
    failures += check(released.poll(start),
                      "response resource cancellation was not observed by SSE transport");
    return failures;
}

int test_sse_response_headers() {
    httplib::Response response;
    ninfer::serve::prepare_sse_response(response);
    return check(response.get_header_value("Cache-Control") == "no-cache" &&
                     response.get_header_value("X-Accel-Buffering") == "no",
                 "SSE response lost its no-cache or anti-buffering header");
}

int test_prompt_json_member_order() {
    httplib::Request request;
    request.body =
        R"({"model":"claude-local","tools":[{"name":"probe","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}],"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"probe","input":{"zeta":"last","alpha":{"yankee":2,"bravo":true}}}]}],"max_tokens":32})";

    const auto parsed = ninfer::serve::parse_json_body(request);
    return check(
        parsed.at("tools").at(0).at("input_schema").at("properties").dump() ==
                R"({"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}})" &&
            parsed.at("messages").at(0).at("content").at(0).at("input").dump() ==
                R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})",
        "HTTP request JSON changed prompt-bearing object member order");
}

int test_schema_number_precision() {
    int failures = 0;
    const std::vector<std::pair<std::string, std::string>> cases{
        {R"({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"number","minimum":0.10000000000000001}}}})",
         "response_format.json_schema.schema/minimum"},
        {R"({"text":{"format":{"type":"json_schema","name":"x","schema":{"type":"number","minimum":1e-400}}}})",
         "text.format.schema/minimum"},
        {R"({"output_config":{"format":{"type":"json_schema","schema":{"type":"number","maximum":0.10000000000000001}}}})",
         "output_config.format.schema/maximum"},
        {R"({"tools":[{"type":"function","function":{"name":"x","strict":true,"parameters":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}}]})",
         "tools/0/function/parameters/properties/x/minimum"},
        {R"({"tools":[{"type":"namespace","name":"n","tools":[{"type":"function","name":"x","strict":true,"parameters":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}]}]})",
         "tools/0/tools/0/parameters/properties/x/minimum"},
        {R"({"tools":[{"name":"x","strict":true,"input_schema":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}]})",
         "tools/0/input_schema/properties/x/minimum"},
        {R"({"temperature":0.10000000000000001,"tools":[{"name":"x","strict":false,"input_schema":{"minimum":0.10000000000000001}}]})",
         ""},
        {R"({"response_format":{"json_schema":{"schema":{"type":"number","default":0.10000000000000001,"minimum":0.1}}}})",
         ""},
    };
    for (const auto& [body, expected] : cases) {
        httplib::Request request;
        request.body = body;
        try {
            (void)ninfer::serve::parse_json_body(request);
            failures += check(expected.empty(), "HTTP rounded a schema number before validation");
        } catch (const ninfer::serve::ApiException& error) {
            failures +=
                check(!expected.empty() && error.error().code == "unsupported_json_schema" &&
                          error.error().param == expected,
                      "schema precision error lost its protocol location");
        }
    }
    return failures;
}

#if defined(__linux__)
class Socket final {
public:
    explicit Socket(int descriptor = -1) : descriptor_(descriptor) {}

    ~Socket() {
        if (descriptor_ >= 0) { ::close(descriptor_); }
    }

    Socket(const Socket&)            = delete;
    Socket& operator=(const Socket&) = delete;

    [[nodiscard]] int get() const noexcept { return descriptor_; }

private:
    int descriptor_ = -1;
};

template <class T>
bool socket_option_equals(int socket, int level, int option, T expected) {
    T actual{};
    socklen_t size = sizeof(actual);
    return ::getsockopt(socket, level, option, &actual, &size) == 0 && size == sizeof(actual) &&
           actual == expected;
}

int test_inherited_socket_liveness() {
    int failures = 0;
    Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
    if (listener.get() < 0) { return check(false, "failed to create HTTP listener test socket"); }
    ninfer::serve::configure_http_server_socket(listener.get());

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener.get(), 1) != 0) {
        return check(false, "failed to bind HTTP listener test socket");
    }
    socklen_t address_size = sizeof(address);
    if (::getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
        return check(false, "failed to inspect HTTP listener test address");
    }

    Socket client(::socket(AF_INET, SOCK_STREAM, 0));
    if (client.get() < 0 || ::connect(client.get(), reinterpret_cast<const sockaddr*>(&address),
                                      sizeof(address)) != 0) {
        return check(false, "failed to connect HTTP liveness test socket");
    }
    Socket accepted(::accept(listener.get(), nullptr, nullptr));
    if (accepted.get() < 0) { return check(false, "failed to accept HTTP liveness test socket"); }

    failures += check(socket_option_equals(accepted.get(), SOL_SOCKET, SO_KEEPALIVE, 1),
                      "accepted HTTP socket did not inherit SO_KEEPALIVE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPIDLE, 10),
                      "accepted HTTP socket did not inherit TCP_KEEPIDLE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPINTVL, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPINTVL");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPCNT, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPCNT");
    failures += check(
        socket_option_equals<unsigned int>(accepted.get(), IPPROTO_TCP, TCP_USER_TIMEOUT, 15000U),
        "accepted HTTP socket did not inherit TCP_USER_TIMEOUT");
    return failures;
}
#endif

} // namespace

int main() {
    int failures = test_sse_transport() + test_sse_response_headers() +
                   test_prompt_json_member_order() + test_schema_number_precision() +
                   test_decision_transport_capacity() + test_task_queue();
#if defined(__linux__)
    failures += test_inherited_socket_liveness();
#endif
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
