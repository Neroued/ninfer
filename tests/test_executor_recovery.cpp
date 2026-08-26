#include "runtime/engine/concurrent_executor.h"
#include "serve/console_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::runtime;
using namespace ninfer::serve;
using FrontendTestAccess = ninfer::targets::qwen3_6::FrontendTestAccess;
using FrontendResources  = ninfer::targets::qwen3_6::FrontendResources;

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("failed to open test resource: ") + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::string read_template_fixture(const char* path) {
    std::string source = read_file(path);
    std::string normalized;
    normalized.reserve(source.size());
    for (char c : source) {
        if (c != '\r') { normalized.push_back(c); }
    }
    if (!normalized.empty() && normalized.back() == '\n') { normalized.pop_back(); }
    return normalized;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

FrontendResources minimal_resources() {
    FrontendResources result;
    result.chat_template_jinja  = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
    const nlohmann::json tokens = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(30, "user\n"), added(31, "assistant\n"), added(32, "\n"),
         added(248045, "<|im_start|>", true), added(248046, "<|im_end|>", true),
         added(248053, "<|vision_start|>", true), added(248054, "<|vision_end|>", true),
         added(248056, "<|image_pad|>", true), added(248057, "<|video_pad|>", true),
         added(248068, "<think>"), added(248069, "</think>")});
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}}},
          {"merges", nlohmann::json::array()}}},
        {"added_tokens",
         tokens}}.dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

// Minimal fake types to instantiate a real ConcurrentExecutor in-process with zero CUDA dependencies.
struct FakeProgram;
struct FakeBasePlan;
struct FakePlan;

struct FakePackage {
    using Program          = FakeProgram;
    using RequestBasePlan  = FakeBasePlan;
    using RequestPlan      = FakePlan;
};

struct FakeBasePlan {
    RequestPlanSummary summary_{};
    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return summary_; }
};

struct FakePlan {
    RequestPlanSummary summary_{};
    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return summary_; }
};

struct FakeLoaded {
    targets::qwen3_6::Frontend frontend = FrontendTestAccess::create_component(minimal_resources(), false);
};

struct FakeRequestMemory {
    ArenaMemorySummary summary() const { return {}; }
    void activate(std::size_t, std::size_t) {}
    void deactivate() {}
    TransientRegion region() const { return {}; }
    void reset_peak() {}
};

struct FakeProgram {
    AdmissionResources admission_capacity_{
        .active_lanes     = 1,
        .main_kv_pages    = 100,
        .backend_kv_pages = 0,
    };
    bool throw_logic_error   = false;
    bool throw_runtime_error = false;
    bool throw_request_error = false;
    TokenId token            = 6; // eos token (<eos>)

    [[nodiscard]] AdmissionResources admission_capacity() const noexcept { return admission_capacity_; }

    FakeBasePlan plan_request_base(const targets::qwen3_6::PreparedPrompt&,
                                   const ResolvedExecutionOptions&) {
        FakeBasePlan p;
        p.summary_.prompt_tokens           = 2;
        p.summary_.reusable_prompt_tokens  = 0;
        p.summary_.effective_output_tokens = 1;
        p.summary_.effective_limit_reason  = FinishReason::OutputLimit;
        p.summary_.service_work_quanta     = 1;
        p.summary_.admission.active_lanes  = 1;
        p.summary_.admission.main_kv_pages = 1;
        return p;
    }

    FakePlan plan_request_for_lane(std::uint32_t,
                                   const targets::qwen3_6::PreparedPrompt&,
                                   const FakeBasePlan&) {
        FakePlan p;
        p.summary_.prompt_tokens           = 2;
        p.summary_.reusable_prompt_tokens  = 0;
        p.summary_.effective_output_tokens = 1;
        p.summary_.effective_limit_reason  = FinishReason::OutputLimit;
        p.summary_.service_work_quanta     = 1;
        p.summary_.admission.active_lanes  = 1;
        p.summary_.admission.main_kv_pages = 1;
        return p;
    }

    bool can_admit_lane(std::uint32_t, const FakePlan&) const { return true; }
    bool can_admit_lane_after_retained_eviction(std::uint32_t, const FakePlan&) const { return true; }
    bool has_retained_lane(std::uint32_t) const { return false; }
    void evict_retained_lane(std::uint32_t) {}
    void abort_lane(std::uint32_t) {}

    PrefillStepResult start_prefill_lane(std::uint32_t, const targets::qwen3_6::PreparedPrompt&,
                                         const FakePlan&, TransientRegion) {
        if (throw_logic_error) {
            throw std::logic_error("scheduler invariant test failure");
        }
        if (throw_runtime_error) {
            throw std::runtime_error("CUDA driver context wedged");
        }
        if (throw_request_error) {
            throw RequestError(RequestErrorKind::Unavailable,
                               "planned prefix seed is no longer available");
        }
        return PrefillStepResult{
            .round    = GeneratedRound{.tokens = std::span<const TokenId>(&token, 1)},
            .complete = true,
        };
    }

    PrefillStepResult advance_prefill_lane(std::uint32_t) {
        return PrefillStepResult{
            .round    = GeneratedRound{.tokens = std::span<const TokenId>(&token, 1)},
            .complete = true,
        };
    }

    void resolve_prefill_lane(std::uint32_t, bool) {}
    BatchedGeneratedRound decode_batch(std::span<const std::uint32_t>,
                                       std::span<const RoundBudget>) {
        return {};
    }
    void resolve_pending_batch(std::span<const std::uint32_t>, std::span<const std::uint32_t>,
                               std::span<const std::uint8_t>, std::span<const std::uint8_t>) {}
    MemorySummary memory_summary() const { return {}; }
    void reset_memory_peaks() {}
    std::size_t prefix_seed_held_bytes() const { return 0; }
    std::size_t release_prefix_seeds() { return 0; }
    bool reclaim_prefix_seeds() { return false; }
    GenerationTimings generation_timings_lane(std::uint32_t) const { return {}; }
    SpeculativeStats speculative_stats_lane(std::uint32_t) const { return {}; }
};

struct FakeInstance {
    using Package = FakePackage;
    std::unique_ptr<FakeProgram> program = std::make_unique<FakeProgram>();
    FakeRequestMemory request_memory;
    std::shared_ptr<FakeLoaded> loaded = std::make_shared<FakeLoaded>();
    KvCapacityResolution kv_capacity_resolution{};
};

void test_exception_types() {
    std::cout << "Testing exception type classification...\n";
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

void test_real_path_fatal_classification() {
    std::cout << "Testing real ConcurrentExecutor fatal error classification with probe...\n";
    FakeInstance instance;
    instance.program->throw_logic_error = true;

    bool probe_fired = false;
    std::string captured_message;
    std::string captured_detail;

    EngineOptions options;
    options.max_concurrency      = 1;
    options.max_pending_requests = 4;
    options.pending_timeout_ms   = 5000;
    options.on_fatal_error       = [&](std::exception_ptr err, const std::string& msg) {
        probe_fired          = true;
        captured_message     = msg;
        if (err) {
            try {
                std::rethrow_exception(err);
            } catch (const std::exception& e) {
                captured_detail = e.what();
            }
        }
    };

    ConcurrentExecutor<FakeInstance> executor(instance, options);
    targets::qwen3_6::PreparedPrompt prompt = instance.loaded->frontend.prepare_tokens({1, 2});
    PromptSummary summary{.prompt_tokens = 2};
    ResolvedRequestOptions req_options{};

    auto submission = executor.submit(std::move(prompt), summary, 0.0, std::move(req_options),
                                      std::chrono::steady_clock::now() + std::chrono::seconds(5));

    bool submission_threw = false;
    try {
        submission.wait(nullptr, CancellationView());
    } catch (const std::logic_error&) {
        submission_threw = true;
    } catch (...) {}

    if (!submission_threw) {
        std::cerr << "FAIL: submission.wait did not receive the fatal logic_error\n";
        std::exit(1);
    }
    if (!probe_fired) {
        std::cerr << "FAIL: on_fatal_error probe was NOT invoked on real executor fatal error\n";
        std::exit(1);
    }
    if (captured_detail != "scheduler invariant test failure") {
        std::cerr << "FAIL: captured exception detail mismatch: " << captured_detail << '\n';
        std::exit(1);
    }
    if (captured_message.find("fatal executor failure") == std::string::npos ||
        captured_message.find("scheduler invariant test failure") == std::string::npos ||
        captured_message.find("terminating process") == std::string::npos) {
        std::cerr << "FAIL: captured formatted message malformed: " << captured_message << '\n';
        std::exit(1);
    }
    if (executor.is_healthy()) {
        std::cerr << "FAIL: executor must report unhealthy (!is_healthy()) after fatal error\n";
        std::exit(1);
    }
}

void test_real_path_negative_request_error_does_not_exit() {
    std::cout << "Testing real ConcurrentExecutor NEGATIVE test (RequestError must NOT reach on_fatal_error)...\n";
    FakeInstance instance;
    instance.program->throw_request_error = true;

    bool probe_fired = false;
    EngineOptions options;
    options.max_concurrency      = 1;
    options.max_pending_requests = 4;
    options.pending_timeout_ms   = 5000;
    options.on_fatal_error       = [&](std::exception_ptr, const std::string&) {
        probe_fired = true;
    };

    ConcurrentExecutor<FakeInstance> executor(instance, options);
    targets::qwen3_6::PreparedPrompt prompt1 = instance.loaded->frontend.prepare_tokens({1, 2});
    PromptSummary summary1{.prompt_tokens = 2};
    ResolvedRequestOptions req_options1{};

    // Request 1: Fails with RequestError
    auto submission1 = executor.submit(std::move(prompt1), summary1, 0.0, std::move(req_options1),
                                       std::chrono::steady_clock::now() + std::chrono::seconds(5));

    bool req1_threw = false;
    try {
        submission1.wait(nullptr, CancellationView());
    } catch (const RequestError& err) {
        req1_threw = true;
        if (err.kind() != RequestErrorKind::Unavailable) {
            std::cerr << "FAIL: Request 1 threw wrong RequestErrorKind\n";
            std::exit(1);
        }
    }

    if (!req1_threw) {
        std::cerr << "FAIL: Request 1 did not throw expected RequestError\n";
        std::exit(1);
    }

    // THE CRITICAL GUARD: on_fatal_error MUST NOT have been called!
    if (probe_fired) {
        std::cerr << "FAIL: on_fatal_error was incorrectly called for a request-scoped RequestError!\n";
        std::exit(1);
    }
    if (!executor.is_healthy()) {
        std::cerr << "FAIL: executor must remain healthy after a request-scoped failure\n";
        std::exit(1);
    }

    // Request 2: Normal subsequent request must succeed on the recovered lane
    instance.program->throw_request_error = false;
    targets::qwen3_6::PreparedPrompt prompt2 = instance.loaded->frontend.prepare_tokens({1, 2});
    PromptSummary summary2{.prompt_tokens = 2};
    ResolvedRequestOptions req_options2{};

    auto submission2 = executor.submit(std::move(prompt2), summary2, 0.0, std::move(req_options2),
                                       std::chrono::steady_clock::now() + std::chrono::seconds(5));

    GenerationResult result = submission2.wait(nullptr, CancellationView());
    (void)result;

    if (probe_fired) {
        std::cerr << "FAIL: on_fatal_error was triggered during recovered execution\n";
        std::exit(1);
    }
    if (!executor.is_healthy()) {
        std::cerr << "FAIL: executor is not healthy after successful recovered request\n";
        std::exit(1);
    }
}

[[noreturn]] void run_child_real_fatal_exit() {
    FakeInstance instance;
    instance.program->throw_logic_error = true;

    EngineOptions options;
    options.max_concurrency      = 1;
    options.max_pending_requests = 4;
    options.pending_timeout_ms   = 5000;
    // No custom on_fatal_error: uses default ConcurrentExecutor hard exit to stderr

    ConcurrentExecutor<FakeInstance> executor(instance, options);
    targets::qwen3_6::PreparedPrompt prompt = instance.loaded->frontend.prepare_tokens({1, 2});
    PromptSummary summary{.prompt_tokens = 2};
    ResolvedRequestOptions req_options{};

    auto submission = executor.submit(std::move(prompt), summary, 0.0, std::move(req_options),
                                      std::chrono::steady_clock::now() + std::chrono::seconds(5));
    try {
        submission.wait(nullptr, CancellationView());
    } catch (...) {}

    // Wait for the worker thread to catch the fatal error and call std::_Exit(1)
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::exit(2);
}

[[noreturn]] void run_child_real_serve_fatal_exit() {
    FakeInstance instance;
    instance.program->throw_runtime_error = true;

    EngineOptions options;
    options.max_concurrency      = 1;
    options.max_pending_requests = 4;
    options.pending_timeout_ms   = 5000;
    options.on_fatal_error       = [](std::exception_ptr, const std::string& message) {
        write_console_log(ConsoleLogLevel::Error, message);
        std::cerr.flush();
        std::cout.flush();
        std::_Exit(1);
    };

    ConcurrentExecutor<FakeInstance> executor(instance, options);
    targets::qwen3_6::PreparedPrompt prompt = instance.loaded->frontend.prepare_tokens({1, 2});
    PromptSummary summary{.prompt_tokens = 2};
    ResolvedRequestOptions req_options{};

    auto submission = executor.submit(std::move(prompt), summary, 0.0, std::move(req_options),
                                      std::chrono::steady_clock::now() + std::chrono::seconds(5));
    try {
        submission.wait(nullptr, CancellationView());
    } catch (...) {}

    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::exit(2);
}

void test_subprocess_fatal_exit(const char* binary_path, const char* flag,
                                const std::string& expected_needle) {
    std::cout << "Testing real subprocess exit on fatal executor failure (" << flag << ")...\n";
    std::string command = std::string(binary_path) + " " + flag + " 2>&1";
#if defined(_WIN32)
    FILE* pipe = _popen(command.c_str(), "r");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) {
        std::cerr << "FAIL: failed to open pipe for child test process\n";
        std::exit(1);
    }
    char buffer[256];
    std::string output;
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
#if defined(_WIN32)
    const int raw_status = _pclose(pipe);
    if (raw_status == -1) {
        std::cerr << "FAIL: _pclose failed on child process\n";
        std::exit(1);
    }
    const int exit_code = raw_status;
#else
    const int raw_status = pclose(pipe);
    if (raw_status == -1) {
        std::cerr << "FAIL: pclose failed on child process\n";
        std::exit(1);
    }
    const int exit_code = WIFEXITED(raw_status) ? WEXITSTATUS(raw_status) : -1;
#endif

    if (exit_code != 1) {
        std::cerr << "FAIL: child process exited with code " << exit_code << ", expected 1\n";
        std::cerr << "Output was:\n" << output << '\n';
        std::exit(1);
    }
    if (output.find(expected_needle) == std::string::npos) {
        std::cerr << "FAIL: child process output missing expected needle: " << expected_needle
                  << "\nActual output was:\n"
                  << output << '\n';
        std::exit(1);
    }
    if (output.find("fatal executor failure") == std::string::npos ||
        output.find("terminating process") == std::string::npos) {
        std::cerr << "FAIL: child process output missing expected fatal marker\n"
                  << "Actual output was:\n"
                  << output << '\n';
        std::exit(1);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::string_view arg = argv[1];
        if (arg == "--child-fatal-exit") {
            run_child_real_fatal_exit();
        } else if (arg == "--child-serve-fatal-exit") {
            run_child_real_serve_fatal_exit();
        }
    }

    test_exception_types();
    test_http_health_route();
    test_real_path_fatal_classification();
    test_real_path_negative_request_error_does_not_exit();
    test_subprocess_fatal_exit(argv[0], "--child-fatal-exit",
                               "scheduler invariant test failure");
    test_subprocess_fatal_exit(argv[0], "--child-serve-fatal-exit",
                               "CUDA driver context wedged");

    std::cout << "All executor recovery, fatal error logging, and process exit tests passed.\n";
    return 0;
}
