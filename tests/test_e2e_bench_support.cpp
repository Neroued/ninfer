#include "e2e_bench_support.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

int expect_bool(bool value, std::string_view label) {
    return value ? 0 : fail(label);
}

int expect_u64(std::uint64_t actual, std::uint64_t expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_string(const std::string& actual, std::string_view expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected `" << expected << "`, got `" << actual << "`\n";
    return 1;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, std::string_view label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw\n";
    return 1;
}

std::filesystem::path temp_file(std::string_view stem, std::string_view contents) {
    const auto path = std::filesystem::temp_directory_path() / std::string(stem);
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to create temp file"); }
    out << contents;
    return path;
}

int test_parse_ids_file() {
    int failures = 0;
    const auto path = temp_file("qus_e2e_ids_ok.ids", "1 2\n3\t4\n");
    const std::vector<int> ids = qus::bench::e2e::parse_ids_file(path.string());
    failures += expect_u64(ids.size(), 4, "ids size");
    failures += expect_bool(ids[0] == 1 && ids[1] == 2 && ids[2] == 3 && ids[3] == 4,
                            "ids values");

    const auto empty = temp_file("qus_e2e_ids_empty.ids", "\n\t ");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::bench::e2e::parse_ids_file(empty.string()); }, "empty ids");

    const auto negative = temp_file("qus_e2e_ids_negative.ids", "1 -2 3");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::bench::e2e::parse_ids_file(negative.string()); }, "negative ids");

    const auto comment = temp_file("qus_e2e_ids_comment.ids", "1 2 # 3");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::bench::e2e::parse_ids_file(comment.string()); }, "comment ids");
    return failures;
}

int test_parse_case_arg() {
    int failures = 0;
    const qus::bench::e2e::CaseSpec spec =
        qus::bench::e2e::parse_case_arg("cn_short:bench/fixtures/prompts/cn_short.ids:128");
    failures += expect_string(spec.name, "cn_short", "case name");
    failures += expect_string(spec.prompt_ids_path, "bench/fixtures/prompts/cn_short.ids",
                              "case path");
    failures += expect_u64(spec.max_new_tokens, 128, "case max_new_tokens");

    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::bench::e2e::parse_case_arg(":x.ids:8"); }, "empty case name");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::bench::e2e::parse_case_arg("x::8"); }, "empty case path");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::bench::e2e::parse_case_arg("x:y.ids:0"); }, "zero max_new_tokens");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::bench::e2e::parse_case_arg("x:y.ids:not_int"); },
        "bad max_new_tokens");
    return failures;
}

int test_context_validation() {
    int failures = 0;
    qus::bench::e2e::CaseRunInput ok;
    ok.name = "ok";
    ok.prompt_ids = {1, 2, 3};
    ok.requested_max_new_tokens = 4;
    ok.max_context = 6;
    qus::bench::e2e::validate_case_context(ok);
    failures += expect_u64(ok.decode_loop_tokens_requested(), 3, "decode requested");
    failures += expect_u64(ok.required_max_context(), 6, "required max_ctx");

    qus::bench::e2e::CaseRunInput one = ok;
    one.requested_max_new_tokens = 1;
    one.max_context = 3;
    qus::bench::e2e::validate_case_context(one);
    failures += expect_u64(one.decode_loop_tokens_requested(), 0, "single token decode requested");
    failures += expect_u64(one.required_max_context(), 3, "single token required max_ctx");

    qus::bench::e2e::CaseRunInput too_small = ok;
    too_small.max_context = 5;
    failures += expect_throws<std::invalid_argument>(
        [&] { qus::bench::e2e::validate_case_context(too_small); }, "max_ctx too small");

    qus::bench::e2e::CaseRunInput no_prompt = ok;
    no_prompt.prompt_ids.clear();
    failures += expect_throws<std::invalid_argument>(
        [&] { qus::bench::e2e::validate_case_context(no_prompt); }, "empty prompt");
    return failures;
}

int test_json_helpers() {
    int failures = 0;
    failures += expect_string(qus::bench::e2e::json_escape("a\"b\\c\n"),
                              "a\\\"b\\\\c\\n", "json escape");
    failures += expect_string(qus::bench::e2e::json_escape(std::string("x") + char(0x01) + "y"),
                              "x\\u0001y", "json control escape");
    failures += qus::bench::e2e::median({3.0, 1.0, 2.0}) == 2.0
                    ? 0
                    : fail("odd median mismatch");
    failures += qus::bench::e2e::median({4.0, 1.0, 2.0, 3.0}) == 2.5
                    ? 0
                    : fail("even median mismatch");
    return failures;
}

int test_error_report_json_is_valid() {
    const auto path = std::filesystem::temp_directory_path() / "qus_e2e_error_report.json";
    qus::bench::e2e::write_error_report(path.string(), "load", "missing \"file\"");
    const std::string command = "python3 -m json.tool \"" + path.string() + "\" > /dev/null";
    const int rc = std::system(command.c_str());
    if (rc != 0) { return fail("error report is not valid JSON"); }
    std::ifstream in(path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    int failures = 0;
    failures += text.find("\"status\": \"error\"") != std::string::npos
                    ? 0
                    : fail("error report missing status");
    failures += text.find("\"phase\": \"load\"") != std::string::npos
                    ? 0
                    : fail("error report missing phase");
    failures += text.find("missing \\\"file\\\"") != std::string::npos
                    ? 0
                    : fail("error report did not escape message");
    return failures;
}

int test_raw_report_json_is_valid() {
    qus::EngineMemoryStats memory;
    memory.loaded = true;
    memory.device = 0;
    memory.max_context = 8;
    memory.position = 0;
    memory.weights = qus::ArenaMemoryStats{true, 1000, 800, 900};
    memory.cache = qus::ArenaMemoryStats{true, 2000, 300, 400};
    memory.workspace = qus::ArenaMemoryStats{true, 3000, 0, 0};
    memory.q5090_loaded_payload_bytes = 700;
    memory.q5090_tensor_count = 10;
    memory.q5090_quant_count = 6;

    qus::bench::e2e::RepeatReport repeat;
    repeat.repeat_index = 0;
    repeat.prefill_time_s = 0.01;
    repeat.decode_time_s = 0.02;
    repeat.prompt_tokens = 3;
    repeat.decode_loop_tokens = 2;
    repeat.generated_token_ids = {10, 11, 12};
    repeat.memory = memory;

    qus::bench::e2e::RepeatReport zero_decode_repeat = repeat;
    zero_decode_repeat.repeat_index = 1;
    zero_decode_repeat.decode_time_s = 0.0;
    zero_decode_repeat.decode_loop_tokens = 0;
    zero_decode_repeat.generated_token_ids = {42};

    qus::bench::e2e::CaseReport case_report;
    case_report.input.name = "cn_short";
    case_report.input.prompt_ids_path = "prompt.ids";
    case_report.input.prompt_ids = {1, 2, 3};
    case_report.input.requested_max_new_tokens = 3;
    case_report.input.max_context = 8;
    case_report.prompt_ids_sha256 = "ids-sha";
    case_report.fixture_manifest_sha256 = "manifest-sha";
    case_report.measured_repeats = 2;
    case_report.repeats.push_back(repeat);
    case_report.repeats.push_back(zero_decode_repeat);

    qus::bench::e2e::RawReport report;
    report.command = "qus_e2e_bench --case cn_short:prompt.ids:3";
    report.git_commit = "abc";
    report.load_time_s = 1.25;
    report.environment.cuda_runtime_version = "12.4";
    report.environment.cuda_driver_version = "12.4";
    report.environment.gpu_name = "test gpu";
    report.environment.device_id = 0;
    report.q5090_path = "weights.qus";
    report.q5090_file_size_bytes = 1234;
    report.q5090_sha256 = "weights-sha";
    report.max_context = 8;
    report.post_load_memory = memory;
    report.cases.push_back(case_report);

    const auto path = std::filesystem::temp_directory_path() / "qus_e2e_raw_report.json";
    qus::bench::e2e::write_raw_report(path.string(), report);
    const std::string command = "python3 -m json.tool \"" + path.string() + "\" > /dev/null";
    const int rc = std::system(command.c_str());
    if (rc != 0) { return fail("raw report is not valid JSON"); }
    std::ifstream in(path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    int failures = 0;
    failures += text.find("\"artifact_type\": \"qus_e2e_benchmark_report\"") != std::string::npos
                    ? 0
                    : fail("raw report artifact_type missing");
    failures += text.find("\"run\"") != std::string::npos &&
                        text.find("\"environment\"") != std::string::npos &&
                        text.find("\"engine\"") != std::string::npos &&
                        text.find("\"weights\"") != std::string::npos &&
                        text.find("\"memory\"") != std::string::npos &&
                        text.find("\"summary\"") != std::string::npos &&
                        text.find("\"cases\"") != std::string::npos
                    ? 0
                    : fail("raw report required sections missing");
    failures += text.find("\"gpu_name\": \"test gpu\"") != std::string::npos
                    ? 0
                    : fail("raw report environment missing");
    failures += text.find("\"workspace_lifetime_policy\": \"step_reset\"") != std::string::npos &&
                        text.find("\"token_readback\": \"per_step_sync_d2h\"") !=
                            std::string::npos &&
                        text.find("\"includes_token_readback\": true") != std::string::npos
                    ? 0
                    : fail("raw report engine constants missing");
    failures += text.find("\"q5090_conv1d_layout\": \"runtime_native_conv_dim_by_kernel\"") !=
                            std::string::npos &&
                        text.find("\"text_core\": true") != std::string::npos &&
                        text.find("\"mtp\": false") != std::string::npos &&
                        text.find("\"vision\": false") != std::string::npos
                    ? 0
                    : fail("raw report weights constants missing");
    failures += text.find("\"accounting_scope\": \"engine_owned_device_arenas_complete\"") !=
                            std::string::npos &&
                        text.find("\"hidden_device_allocations\": false") != std::string::npos
                    ? 0
                    : fail("raw report memory constants missing");
    failures += text.find("\"fixture_set\": \"m2.8-v1\"") != std::string::npos &&
                        text.find("\"prompt_ids_sha256\": \"ids-sha\"") != std::string::npos
                    ? 0
                    : fail("raw report fixture identity missing");
    failures += text.find("\"prefill_time_s_median\"") != std::string::npos &&
                        text.find("\"decode_eager_tok_s_median\"") != std::string::npos &&
                        text.find("\"deterministic_token_ids\"") != std::string::npos &&
                        text.find("\"max_workspace_arena_peak_used_bytes\"") != std::string::npos
                    ? 0
                    : fail("raw report summary fields missing");
    failures += text.find("\"decode_eager_tok_s_valid\": true") != std::string::npos
                    ? 0
                    : fail("decode validity missing");
    failures += text.find("\"decode_eager_tok_s\": null") != std::string::npos &&
                        text.find("\"decode_eager_tok_s_valid\": false") != std::string::npos
                    ? 0
                    : fail("zero decode throughput null handling missing");
    failures += text.find("\"generated_token_ids\": [10, 11, 12]") != std::string::npos
                    ? 0
                    : fail("generated ids missing");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_ids_file();
    failures += test_parse_case_arg();
    failures += test_context_validation();
    failures += test_json_helpers();
    failures += test_error_report_json_is_valid();
    failures += test_raw_report_json_is_valid();
    return failures == 0 ? 0 : fail("e2e bench support test failed");
}
