#include "serve/http_transport.h"
#include "serve/request_validation.h"

#include <atomic>
#include <chrono>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
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

int test_client_label() {
    using ninfer::serve::parse_client_label;
    const auto with = [](std::initializer_list<std::pair<const char*, std::string>> headers) {
        httplib::Request request;
        for (const auto& [name, value] : headers) { request.headers.emplace(name, value); }
        return request;
    };
    const auto rejected = [](const httplib::Request& request) {
        try {
            (void)parse_client_label(request);
        } catch (const ninfer::serve::ApiException& exception) {
            return exception.error().status == 400 &&
                   exception.error().code == "invalid_client_label" &&
                   exception.error().param == "X-Ninfer-Client";
        }
        return false;
    };
    int failures = check(!parse_client_label(httplib::Request{}),
                         "a request without X-Ninfer-Client produced a client label");
    failures += check(parse_client_label(with({{"X-Ninfer-Client", "bench/serve-corpus"}})) ==
                          std::optional<std::string>("bench/serve-corpus"),
                      "a valid X-Ninfer-Client label was not returned verbatim");
    failures += check(parse_client_label(with({{"x-ninfer-client", "gate/tp2"}})) ==
                          std::optional<std::string>("gate/tp2"),
                      "the X-Ninfer-Client header name is not case-insensitive");
    const std::string longest(64, 'a');
    failures += check(parse_client_label(with({{"X-Ninfer-Client", longest}})) ==
                          std::optional<std::string>(longest),
                      "a 64-character client label was rejected");
    failures += check(rejected(with({{"X-Ninfer-Client", std::string(65, 'a')}})),
                      "an over-long client label was accepted");
    failures +=
        check(rejected(with({{"X-Ninfer-Client", ""}})), "an empty client label was accepted");
    failures += check(rejected(with({{"X-Ninfer-Client", "gate v045"}})),
                      "a client label with a space was accepted");
    failures += check(rejected(with({{"X-Ninfer-Client", "a"}, {"X-Ninfer-Client", "b"}})),
                      "a repeated X-Ninfer-Client header was accepted");
    return failures;
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

// Structured output: a schema number whose spelling the double cannot keep is rejected at the
// body boundary with its location (Neroued/ninfer 81c8ce09); the same number outside a schema, or
// a schema without such numbers, parses as before.
// capy-flash: a body nested deeper than --max-request-json-depth is a 400 before parsing (a tool
// schema nested 100 000 levels used to overflow the stack of the template renderer).
int test_request_json_depth() {
    const auto error_of = [](std::string body) {
        httplib::Request request;
        request.body = std::move(body);
        try {
            (void)ninfer::serve::parse_json_body(request);
        } catch (const ninfer::serve::ApiException& exception) { return exception.error(); }
        return ninfer::serve::ApiError{.status = 0};
    };
    const auto body = [](std::size_t levels) {
        return R"({"model":"m","messages":[],"x":)" + std::string(levels, '[') +
               std::string(levels, ']') + "}";
    };
    int failures = 0;
    failures +=
        check(error_of(body(999)).status == 0, "a body at the default depth (1000) was refused");
    for (const std::size_t levels : {1000U, 100000U}) {
        const auto error = error_of(body(levels));
        failures += check(error.status == 400 && error.code == "request_json_too_deep" &&
                              error.message.find("--max-request-json-depth") != std::string::npos,
                          "a body nested past the default depth was not refused");
    }
    failures += check(error_of(R"({"s":")" + std::string(5000, '[') + R"("})").status == 0,
                      "brackets inside a string counted as nesting");
    ninfer::serve::configure_max_request_json_depth(4);
    failures += check(error_of(body(4)).status == 400 && error_of(body(3)).status == 0,
                      "the configured depth was not applied");
    ninfer::serve::configure_max_request_json_depth(1000);
    return failures;
}

int test_schema_number_spelling() {
    const auto error_of = [](std::string body) {
        httplib::Request request;
        request.body = std::move(body);
        try {
            (void)ninfer::serve::parse_json_body(request);
        } catch (const ninfer::serve::ApiException& exception) { return exception.error(); }
        return ninfer::serve::ApiError{.status = 0};
    };
    int failures            = 0;
    const auto inexact_chat = error_of(
        R"({"model":"m","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"object","properties":{"x":{"type":"number","maximum":0.30000000000000001}}}}}})");
    failures +=
        check(inexact_chat.status == 400 && inexact_chat.code == "unsupported_json_schema" &&
                  inexact_chat.param == "response_format.json_schema.schema/properties/x/maximum",
              "inexact Chat schema bound was not rejected with its location");
    const auto inexact_anthropic = error_of(
        R"({"model":"m","messages":[],"output_config":{"format":{"type":"json_schema","schema":{"enum":[1, 0.10000000000000001]}}}})");
    failures += check(inexact_anthropic.param == "output_config.format.schema/enum/1",
                      "inexact Anthropic schema enum was not located");
    const auto inexact_responses = error_of(
        R"({"model":"m","input":"x","text":{"format":{"type":"json_schema","name":"n","schema":{"const":0.30000000000000001}}}})");
    failures += check(inexact_responses.param == "text.format.schema/const",
                      "inexact Responses schema const was not located");
    failures += check(
        error_of(
            R"({"model":"m","messages":[],"temperature":0.30000000000000001,"response_format":{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"number","maximum":0.5}}}})")
                .status == 0,
        "an exact schema or a non-schema number was rejected");
    failures += check(error_of(R"({"model":"m","temperature":0.30000000000000001})").status == 0,
                      "an unconstrained body was checked for number spelling");
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
                   test_prompt_json_member_order() + test_client_label() +
                   test_schema_number_spelling() + test_request_json_depth();
#if defined(__linux__)
    failures += test_inherited_socket_liveness();
#endif
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
