// Observable serving contract for fitting an explicit/default output upper
// bound into the model context after chat-template tokenization.

#include "ninfer/serve/generation_service.h"
#include "ninfer/serve/request_log.h"

#include <iostream>
#include <string>

namespace {

using namespace ninfer::serve;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

int test_context_output_budget() {
    int failures = 0;

    const ContextOutputBudget fitting = resolve_context_output_budget(512, 1024, 2048);
    failures += check(fitting.effective_max_tokens == 1024 && !fitting.clamped(),
                      "fitting output limit is unchanged");

    const ContextOutputBudget clamped = resolve_context_output_budget(546, 65000, 65536);
    failures += check(clamped.requested_max_tokens == 65000, "requested limit retained");
    failures += check(clamped.effective_max_tokens == 64991 && clamped.clamped(),
                      "output limit uses remaining context including prefill token");

    const ContextOutputBudget full_prompt = resolve_context_output_budget(65536, 65000, 65536);
    failures += check(full_prompt.effective_max_tokens == 1,
                      "full prompt can return the token sampled by prefill");

    try {
        (void)resolve_context_output_budget(65537, 1, 65536);
        failures += fail("oversized prompt accepted");
    } catch (const ApiException& exception) {
        failures += check(exception.error().status == 400, "oversized prompt status");
        failures += check(exception.error().code == "context_length_exceeded",
                          "oversized prompt error code");
        failures += check(exception.error().param == "messages", "oversized prompt parameter");
    }
    return failures;
}

int test_clamped_request_log() {
    ninfer::kernels::SamplingConfig sampling;
    const std::string line =
        format_request_start(7, false, 1, 65000, 64991, true, 0, ToolChoice{}, false, sampling);
    int failures = 0;
    failures +=
        check(line.find("max_tokens=65000") != std::string::npos, "log contains requested limit");
    failures += check(line.find("effective_max_tokens=64991 (context clamp)") != std::string::npos,
                      "log contains effective context limit");
    return failures;
}

} // namespace

int main() {
    const int failures = test_context_output_budget() + test_clamped_request_log();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
