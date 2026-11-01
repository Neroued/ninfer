// Real artifact on two devices: an Engine-wide failure while serving must stop the accept loop and
// leave HttpServer::engine_failed() set, so ninfer-serve can exit non-zero for its supervisor.
//
// The failure is injected with a link-time wrapper around cudaStreamSynchronize, which the Engine
// worker calls on every decode round (ProgramImpl::synchronize_devices). The wrapper throws: a CUDA
// error returned to CUDA_CHECK aborts the process by design, which a supervisor already sees. No
// exchange reported a hang, so the Engine cannot recover by stepping the transport down (issue
// #20, ninfer_serve_tp_transport_recovery_real_test) and fails as a whole. The wrapper affects
// only this test executable; the production path has no fault hooks.
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
#include <vector>

namespace {

std::atomic<bool> g_inject_failure{false};
std::atomic<bool> g_injected{false};

} // namespace

extern "C" {
cudaError_t CUDARTAPI __real_cudaStreamSynchronize(cudaStream_t);

cudaError_t CUDARTAPI __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    if (g_inject_failure.exchange(false)) {
        g_injected.store(true);
        throw std::runtime_error("injected: decode round synchronization failed (test)");
    }
    return __real_cudaStreamSynchronize(stream);
}
}

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: tensor parallelism needs two CUDA devices\n";
        return 77;
    }

    // Built through the CLI parser so the two-GPU defaults (no host tiers at tp 2) apply as in
    // ninfer-serve itself.
    std::string port = "18089";
    if (const char* env = std::getenv("NINFER_TEST_PORT"); env != nullptr && *env != '\0') {
        port = env;
    }
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
                                     "2048",
                                     "--kv-capacity",
                                     "2048",
                                     "--max-concurrency",
                                     "1",
                                     "--log-stats-interval-ms",
                                     "0",
                                     "--greedy"};
    std::vector<char*> argv;
    for (std::string& arg : args) { argv.push_back(arg.data()); }
    ninfer::serve::ServeOptions options =
        ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve-test",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});

    int failures = 0;
    try {
        ninfer::serve::HttpServer server(options, logging.logger());
        if (!server.bind()) {
            std::cerr << "FAIL: could not bind 127.0.0.1:" << options.port << '\n';
            return 1;
        }
        ninfer::serve::GenerationService service(options);
        service.warmup();
        server.attach(service);

        std::atomic<bool> listened{false};
        std::thread listener([&] { listened.store(server.listen()); });
        // Safety net: a watch that never fires would otherwise hang the test forever.
        std::atomic<bool> timed_out{false};
        std::atomic<bool> done{false};
        std::thread watchdog([&] {
            for (int i = 0; i < 600 && !done.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!done.load()) {
                timed_out.store(true);
                server.stop();
            }
        });

        httplib::Client client("127.0.0.1", options.port);
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(120, 0);

        bool healthy = false;
        for (int i = 0; i < 100 && !healthy; ++i) {
            if (auto res = client.Get("/health"); res && res->status == 200) {
                healthy = Json::parse(res->body).value("status", "") == "ok";
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        failures += check(healthy, "/health did not report ok before the failure");
        failures += check(service.is_available(), "engine not available before the failure");
        failures += check(!server.engine_failed(), "engine_failed() set before the failure");

        g_inject_failure.store(true);
        const Json body = {{"model", server.public_model_id()},
                           {"messages", Json::array({{{"role", "user"}, {"content", "Say hi."}}})},
                           {"max_tokens", 16}};
        auto res        = client.Post("/v1/chat/completions", body.dump(), "application/json");
        // The request is failed by the Engine (5xx), or the connection is torn down by the stop:
        // either way it must not have produced a normal completion.
        failures += check(!res || res->status >= 500,
                          "request completed normally although the Engine failed");

        listener.join();
        done.store(true);
        watchdog.join();

        failures += check(!timed_out.load(), "engine watch did not stop listen() within 60 s");
        failures += check(g_injected.load(), "the failure was never injected");
        failures += check(!service.is_available(), "engine still available after the failure");
        failures += check(server.engine_failed(), "engine_failed() not set after the failure");
        failures += check(listened.load(), "listen() reported an accept-loop failure");
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: exception: " << exception.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " serve engine failure\n";
    return failures == 0 ? 0 : 1;
}
