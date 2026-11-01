// Real artifacts on two devices: a captured tensor-parallel exchange that times out while serving
// fails only the request it ran for, and the server keeps serving on the stepped-down transport
// with the same tokens. When the step-down itself fails, the Engine fails and the request is not
// told that the server keeps serving. Runs on NINFER_TEST_ARTIFACT (Qwen3.8-27B) and on
// NINFER_TEST_QWEN4_ARTIFACT (Qwen4Exp, the Flash-Next family), each when set.
//
// Under WSL2 the pinned-host mailbox occasionally hangs mid-service: the waiting rank gives up
// after a bounded spin and sets the mailbox's sticky hang word, and the check after the step
// throws. The hang is injected the same way, by a link-time wrapper around
// PeerMailbox::hang_reported() that sets the real hang word once when armed: once during a
// streamed answer's decode, once at the first check of a later request. Each is armed
// only after the previous step-down finished (a completed request), since an injection during the
// capture itself is not a hang an exchange can produce. A second server injects one hang and makes
// the step-down fail, through a link-time wrapper around Program::recover_peer_transport(). The
// wrappers affect only this test executable; the production path has no fault hooks.
#include "ninfer/ops/peer_mailbox.h"
#include "product/logging/logging.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <cuda_runtime.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

std::atomic<bool> g_armed{false};
std::atomic<int> g_injected{0};
std::atomic<bool> g_fail_step_down{false};
std::atomic<int> g_step_downs_failed{0};

} // namespace

extern "C" {
bool __real__ZNK6ninfer3ops11PeerMailbox13hang_reportedEv(const ninfer::ops::PeerMailbox* self);
// std::string ninfer::models::qwen3_5::Program::recover_peer_transport() (Itanium ABI: the
// returned string's address, then `this`, as for a free function of `this`).
std::string __real__ZN6ninfer6models7qwen3_57Program22recover_peer_transportB5cxx11Ev(void* self);

std::string __wrap__ZN6ninfer6models7qwen3_57Program22recover_peer_transportB5cxx11Ev(void* self) {
    if (g_fail_step_down.exchange(false)) {
        g_step_downs_failed.fetch_add(1);
        throw std::runtime_error("injected: the transport could not step down");
    }
    return __real__ZN6ninfer6models7qwen3_57Program22recover_peer_transportB5cxx11Ev(self);
}

// ninfer::ops::PeerMailbox::hang_reported() const
bool __wrap__ZNK6ninfer3ops11PeerMailbox13hang_reportedEv(const ninfer::ops::PeerMailbox* self) {
    if (g_armed.exchange(false)) {
        // What a poller that gave up does: the sticky word, seen by every later check.
        const_cast<ninfer::ops::PeerMailbox*>(self)->report_hang();
        g_injected.fetch_add(1);
    }
    return __real__ZNK6ninfer3ops11PeerMailbox13hang_reportedEv(self);
}
}

namespace {

using Json = nlohmann::json;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

Json chat_body(const std::string& model, bool stream) {
    return {{"model", model},
            {"messages", Json::array({{{"role", "user"},
                                       {"content", "Write a short poem about the sea, then explain "
                                                   "its imagery in a few sentences."}}})},
            {"max_tokens", 96},
            {"stream", stream}};
}

// Stops the server, then joins its accept loop and the watchdog, however the case ends.
class ServerThreads {
public:
    ServerThreads(ninfer::serve::HttpServer& server, std::atomic<bool>& done)
        : server_(server), done_(done) {}

    ServerThreads(const ServerThreads&)            = delete;
    ServerThreads& operator=(const ServerThreads&) = delete;

    ~ServerThreads() { join(); }

    // Waits until the accept loop ends by itself (an Engine failure stops it; the watchdog stops
    // the server after 300 s).
    void await_listener() {
        if (listener.joinable()) { listener.join(); }
    }

    void join() noexcept {
        done_.store(true);
        try {
            server_.stop();
        } catch (...) {}
        if (listener.joinable()) { listener.join(); }
        if (watchdog.joinable()) { watchdog.join(); }
    }

    std::thread listener;
    std::thread watchdog;

private:
    ninfer::serve::HttpServer& server_;
    std::atomic<bool>& done_;
};

// What a case injects: two hangs, each followed by a step-down (Recovery), or one hang whose
// step-down fails (FailedStepDown).
enum class Scenario { Recovery, FailedStepDown };

// One server on `artifact`: 0 PASS, 1 FAIL, 77 when its captured all-reduces do not start on the
// mailbox.
int run_case(const std::string& label, const std::string& artifact, int port_offset, bool qwen4,
             Scenario scenario) {
    std::cout << "== " << label << ": " << artifact << '\n';
    g_armed.store(false);
    g_injected.store(0);
    g_fail_step_down.store(false);
    g_step_downs_failed.store(0);
    int port_number = 18091;
    if (const char* env = std::getenv("NINFER_TEST_PORT"); env != nullptr && *env != '\0') {
        port_number = std::atoi(env);
    }
    std::string port = std::to_string(port_number + port_offset);
    // MTP, so the transport has two steps below the mailbox: the draft phase on the copies,
    // then the copies everywhere. The Engine starts on the mailbox where the startup probe
    // passes (native Linux); elsewhere the test has nothing to check.
    std::vector<std::string> args = {"ninfer-serve",
                                     artifact,
                                     "--host",
                                     "127.0.0.1",
                                     "--port",
                                     port,
                                     "--tp",
                                     "2",
                                     "--devices",
                                     "0,1",
                                     "--max-context",
                                     "4096",
                                     "--kv-capacity",
                                     "4096",
                                     "--max-concurrency",
                                     "1",
                                     "--spec",
                                     "mtp",
                                     "--draft-tokens",
                                     "3",
                                     "--log-stats-interval-ms",
                                     "0",
                                     "--greedy"};
    std::vector<char*> argv;
    for (std::string& arg : args) { argv.push_back(arg.data()); }
    ninfer::serve::ServeOptions options =
        ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = std::string("ninfer-serve-test-") + label,
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});

    int failures = 0;
    try {
        ninfer::serve::HttpServer server(options, logging.logger());
        if (!server.bind()) {
            std::cerr << "FAIL: could not bind 127.0.0.1:" << options.port << '\n';
            return 1;
        }
        // Qwen4Exp: which routed experts are resident on the GPUs depends on copy timing, and a
        // resident and a missing expert are computed on different devices; the documented
        // validation switch freezes the residency so that answers compare exactly.
        if (qwen4) {
            ::setenv("NINFER_QWEN4_VALIDATION", "1", 1);
            ::setenv("NINFER_QWEN4_EXPERT_FREEZE", "all", 1);
        }
        ninfer::serve::GenerationService service(options);
        service.warmup();
        server.attach(service);
        const std::string transport = service.load_summary().tp_transport;
        if (transport != "mailbox") {
            std::cout << "skip: the captured all-reduces start on '" << transport
                      << "', not on the mailbox\n";
            server.stop();
            return 77;
        }

        std::atomic<bool> listened{false};
        std::atomic<bool> timed_out{false};
        std::atomic<bool> done{false};
        ServerThreads threads(server, done);
        threads.listener = std::thread([&] { listened.store(server.listen()); });
        threads.watchdog = std::thread([&] {
            for (int i = 0; i < 3000 && !done.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!done.load()) {
                timed_out.store(true);
                server.stop();
            }
        });

        httplib::Client client("127.0.0.1", options.port);
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(180, 0);
        bool healthy = false;
        for (int i = 0; i < 100 && !healthy; ++i) {
            if (auto res = client.Get("/health"); res && res->status == 200) {
                healthy = Json::parse(res->body).value("status", "") == "ok";
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        failures += check(healthy, "/health did not report ok before the hangs");
        const std::string model = server.public_model_id();

        const auto complete = [&](const char* stage) -> std::string {
            auto res = client.Post("/v1/chat/completions", chat_body(model, false).dump(),
                                   "application/json");
            if (!res || res->status != 200) {
                failures +=
                    check(false, std::string(stage) + ": request did not complete (" +
                                     (res ? std::to_string(res->status) : "no response") + ")");
                return {};
            }
            const Json reply    = Json::parse(res->body);
            const Json& message = reply.at("choices").at(0).at("message");
            return message.value("reasoning_content", "") + "\n--\n" + message.value("content", "");
        };

        if (scenario == Scenario::FailedStepDown) {
            // One hang during a streamed answer's decode, and its step-down fails: the Engine
            // fails, and the stream ends with that failure, not with the promise of a server that
            // keeps serving.
            std::string streamed;
            int chunks = 0;
            (void)client.Post("/v1/chat/completions", {}, chat_body(model, true).dump(),
                              "application/json", [&](const char* data, std::size_t length) {
                                  streamed.append(data, length);
                                  if (++chunks == 4) {
                                      g_fail_step_down.store(true);
                                      g_armed.store(true);
                                  }
                                  return true;
                              });
            const std::string tail =
                streamed.substr(streamed.size() > 400 ? streamed.size() - 400 : 0);
            failures += check(g_injected.load() == 1, "the hang was never injected");
            failures += check(g_step_downs_failed.load() == 1, "the step-down was never attempted");
            failures +=
                check(streamed.find("\"error\"") != std::string::npos,
                      "the stream whose step-down failed did not end with an error: " + tail);
            failures += check(streamed.find("keeps serving") == std::string::npos,
                              "the stream whose step-down failed was told that the server keeps "
                              "serving: " +
                                  tail);
            threads.await_listener();
            failures += check(server.engine_failed(), "the Engine did not fail");
        } else {
            // The answer on the mailbox.
            const std::string reference = complete("reference on the mailbox");
            failures += check(reference.size() > 4, "the reference answer is empty");

            // 1. Decode: the hang word is set after the streamed answer's first tokens arrived.
            std::string streamed;
            int chunks = 0;
            auto stream =
                client.Post("/v1/chat/completions", {}, chat_body(model, true).dump(),
                            "application/json", [&](const char* data, std::size_t length) {
                                streamed.append(data, length);
                                if (++chunks == 4) { g_armed.store(true); }
                                return true;
                            });
            failures += check(g_injected.load() == 1, "the decode hang was never injected");
            failures +=
                check(static_cast<bool>(stream), "the streamed request lost its connection");
            failures +=
                check(streamed.find("service_unavailable") != std::string::npos,
                      "the hung stream did not end with a service_unavailable error: " +
                          streamed.substr(streamed.size() > 400 ? streamed.size() - 400 : 0));
            failures += check(streamed.find("\"finish_reason\":\"length\"") == std::string::npos &&
                                  streamed.find("\"finish_reason\":\"stop\"") == std::string::npos,
                              "the hung stream finished as a normal answer");
            failures += check(service.is_available() && !server.engine_failed(),
                              "the Engine failed after the decode hang");
            // The failed request completes before the graphs are captured again; the next one is
            // served after it, on the mailbox with the MTP draft phase on the copies.
            failures +=
                check(complete("answer after the first step-down") == reference,
                      "the answer after the first step-down differs from the mailbox answer");

            // 2. The next request: the hang word is set at its first synchronization. Its prompt
            // runs eager on the staged copies, which never use the mailbox, so a real hang can
            // surface there only from earlier captured work; the first check of its own comes in
            // its first speculative round.
            g_armed.store(true);
            auto hung = client.Post("/v1/chat/completions", chat_body(model, false).dump(),
                                    "application/json");
            failures += check(g_injected.load() == 2, "the second hang was never injected");
            failures +=
                check(hung && hung->status == 503,
                      "the request that hung did not fail with 503 (" +
                          (hung ? std::to_string(hung->status) : std::string("no response")) + ")");
            failures += check(service.is_available() && !server.engine_failed(),
                              "the Engine failed after the second hang");

            // 3. Served again, now on the copies everywhere (no mailbox is left to hang), with the
            // tokens of the mailbox: the transports combine with the same arithmetic.
            g_armed.store(true);
            const std::string after = complete("answer after the two step-downs");
            failures +=
                check(g_injected.load() == 2, "a mailbox was still attached after two hangs");
            g_armed.store(false);
            failures += check(after == reference,
                              "the answer after the step-downs differs from the mailbox answer");
            failures += check(complete("second answer after the step-downs") == reference,
                              "a second answer after the step-downs differs");
        }
        threads.join();
        failures += check(!timed_out.load(), "the test did not finish within 300 s");
        failures += check(listened.load(), "listen() reported an accept-loop failure");
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: exception: " << exception.what() << '\n';
        ++failures;
    }
    std::cout << (failures == 0 ? "PASS " : "FAIL ") << label << '\n';
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: tensor parallelism needs two CUDA devices\n";
        return 77;
    }
    // Qwen3.8-27B (NINFER_TEST_ARTIFACT) and the Qwen4Exp (Flash-Next) family
    // (NINFER_TEST_QWEN4_ARTIFACT, the synthetic artifact in CTest): each runs when set.
    int ran = 0, failed = 0;
    for (const auto& [label, variable, offset] :
         {std::tuple{"qwen3.8-27b", "NINFER_TEST_ARTIFACT", 0},
          std::tuple{"qwen4exp", "NINFER_TEST_QWEN4_ARTIFACT", 1}}) {
        const bool qwen4     = offset == 1;
        const char* artifact = std::getenv(variable);
        if (artifact == nullptr || *artifact == '\0') {
            std::cout << "skip " << label << ": " << variable << " is not set\n";
            continue;
        }
        for (const Scenario scenario : {Scenario::Recovery, Scenario::FailedStepDown}) {
            const bool recovery = scenario == Scenario::Recovery;
            const int result =
                run_case(recovery ? std::string(label) : std::string(label) + " failed step-down",
                         artifact, offset + (recovery ? 0 : 2), qwen4, scenario);
            if (result == 77) { break; }
            ++ran;
            failed += result != 0 ? 1 : 0;
        }
    }
    if (ran == 0) { return 77; }
    std::cout << (failed == 0 ? "PASS" : "FAIL") << " tensor-parallel transport recovery\n";
    return failed == 0 ? 0 : 1;
}
