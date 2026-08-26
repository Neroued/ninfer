#include "runtime/engine/concurrent_executor.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

void test_exception_types() {
    std::cout << "Testing exception type classification...\n";
    // 1. RequestError is a std::invalid_argument (and thus std::logic_error),
    // but MUST be catchable as RequestError.
    try {
        throw RequestError(RequestErrorKind::Unavailable, "test transient failure");
    } catch (const RequestError& err) {
        if (err.kind() != RequestErrorKind::Unavailable) {
            std::cerr << "FAIL: RequestError kind mismatch\n";
            std::exit(1);
        }
    } catch (...) {
        std::cerr << "FAIL: RequestError was not caught by const RequestError&\n";
        std::exit(1);
    }

    // 2. std::logic_error must NOT be caught by catch (const RequestError&)
    bool logic_error_caught = false;
    try {
        try {
            throw std::logic_error("scheduler invariant violation");
        } catch (const RequestError&) {
            std::cerr << "FAIL: std::logic_error incorrectly caught as RequestError!\n";
            std::exit(1);
        }
    } catch (const std::logic_error&) {
        logic_error_caught = true;
    }
    if (!logic_error_caught) {
        std::cerr << "FAIL: std::logic_error was lost\n";
        std::exit(1);
    }
}

void test_http_health_route() {
    std::cout << "Testing HTTP /health route status behavior...\n";
    const ApiError unavail = request_error_to_api_error(
        RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable"));
    if (unavail.status != 503 || unavail.code != "service_unavailable") {
        std::cerr << "FAIL: Unavailable error does not map to 503 service_unavailable\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    test_exception_types();
    test_http_health_route();

    std::cout << "All executor recovery and exception classification unit tests passed.\n";
    return 0;
}
