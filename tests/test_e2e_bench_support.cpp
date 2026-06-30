#include "e2e_bench_support.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Json = nlohmann::json;

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

int expect_int(int actual, int expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_double(double actual, double expected, std::string_view label) {
    if (std::abs(actual - expected) < 1e-9) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_string(const std::string& actual, std::string_view expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected `" << expected << "`, got `" << actual << "`\n";
    return 1;
}

int expect_int_vector(const std::vector<int>& actual, const std::vector<int>& expected,
                      std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected [";
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (i != 0) { std::cerr << ", "; }
        std::cerr << expected[i];
    }
    std::cerr << "], got [";
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (i != 0) { std::cerr << ", "; }
        std::cerr << actual[i];
    }
    std::cerr << "]\n";
    return 1;
}

int expect_contains(const std::string& actual, std::string_view expected, std::string_view label) {
    if (actual.find(expected) != std::string::npos) { return 0; }
    std::cerr << label << " expected to find `" << expected << "` in:\n" << actual << '\n';
    return 1;
}

int expect_json_value(const Json& actual, const Json& expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected.dump() << ", got " << actual.dump() << '\n';
    return 1;
}

int expect_json_null(const Json& actual, std::string_view label) {
    if (actual.is_null()) { return 0; }
    std::cerr << label << " expected null, got " << actual.dump() << '\n';
    return 1;
}

int expect_json_int_vector(const Json& actual, const std::vector<int>& expected,
                           std::string_view label) {
    if (!actual.is_array()) {
        std::cerr << label << " expected array, got " << actual.dump() << '\n';
        return 1;
    }
    std::vector<int> parsed;
    parsed.reserve(actual.size());
    for (const Json& item : actual) {
        if (!item.is_number_integer()) {
            std::cerr << label << " expected integer entries, got " << actual.dump() << '\n';
            return 1;
        }
        parsed.push_back(item.get<int>());
    }
    return expect_int_vector(parsed, expected, label);
}

bool json_contains_key_recursive(const Json& value, const std::string& key) {
    if (value.is_object()) {
        if (value.contains(key)) { return true; }
        for (const auto& item : value.items()) {
            if (json_contains_key_recursive(item.value(), key)) { return true; }
        }
    } else if (value.is_array()) {
        for (const Json& item : value) {
            if (json_contains_key_recursive(item, key)) { return true; }
        }
    }
    return false;
}

int expect_json_not_contains_key(const Json& actual, std::string_view key,
                                 std::string_view label) {
    const std::string key_string(key);
    if (!json_contains_key_recursive(actual, key_string)) { return 0; }
    std::cerr << label << " expected no `" << key << "` key in:\n" << actual.dump(2) << '\n';
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

Json parse_json_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("failed to open JSON file: " + path.string()); }
    try {
        return Json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("invalid JSON in " + path.string() + ": " + e.what());
    }
}

std::filesystem::path repo_file(std::string_view relative) {
#ifdef QUS_SOURCE_DIR
    return std::filesystem::path(QUS_SOURCE_DIR) / std::string(relative);
#else
    return std::filesystem::path(relative);
#endif
}

qus::bench::e2e::RunOptions parse_args_for_test(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) { argv.push_back(arg.data()); }
    return qus::bench::e2e::parse_args(static_cast<int>(argv.size()), argv.data());
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

int test_parse_args_help() {
    int failures = 0;
    const qus::bench::e2e::RunOptions long_help =
        parse_args_for_test({"qus_e2e_bench", "--help"});
    failures += expect_bool(long_help.help_requested, "--help sets help_requested");
    failures += expect_bool(long_help.weights_path.empty(), "--help does not require weights");
    failures += expect_bool(long_help.output_json_path.empty(), "--help does not require output json");
    failures += expect_bool(long_help.cases.empty(), "--help does not require cases");

    const qus::bench::e2e::RunOptions short_help =
        parse_args_for_test({"qus_e2e_bench", "-h"});
    failures += expect_bool(short_help.help_requested, "-h sets help_requested");
    failures += expect_u64(short_help.max_ctx, qus::EngineOptions{}.max_ctx,
                           "help max_ctx default");
    failures += expect_u64(short_help.repeats, 1, "help repeats default");
    failures += expect_u64(short_help.warmup_repeats, 0, "help warmup default");
    failures += expect_u64(short_help.device, 0, "help device default");
    return failures;
}

int test_parse_args_stop_tokens() {
    int failures = 0;
    const std::vector<std::string> base = {"qus_e2e_bench",
                                           "--weights",
                                           "weights.q5090",
                                           "--output-json",
                                           "out.json",
                                           "--case",
                                           "cn_short:bench/fixtures/prompts/cn_short.ids:8"};

    const qus::bench::e2e::RunOptions defaults = parse_args_for_test(base);
    failures += expect_int_vector(defaults.stop_token_ids, {248046, 248044},
                                  "default stop token ids");

    std::vector<std::string> repeated = base;
    repeated.insert(repeated.end(), {"--stop-token-id", "7", "--stop-token-id", "9",
                                     "--stop-token-id", "7"});
    const qus::bench::e2e::RunOptions parsed_repeated = parse_args_for_test(repeated);
    failures += expect_int_vector(parsed_repeated.stop_token_ids, {7, 9},
                                  "explicit repeated stop token ids");

    failures += expect_throws<std::invalid_argument>(
        [&] {
            std::vector<std::string> old_flag = base;
            old_flag.insert(old_flag.end(), {"--eos-token-id", "123"});
            (void)parse_args_for_test(old_flag);
        },
        "old eos flag is not accepted");

    failures += expect_throws<std::invalid_argument>(
        [&] {
            std::vector<std::string> bad = base;
            bad.insert(bad.end(), {"--stop-token-id", "-1"});
            (void)parse_args_for_test(bad);
        },
        "negative stop token id");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            std::vector<std::string> bad = base;
            bad.insert(bad.end(), {"--stop-token-id", "abc"});
            (void)parse_args_for_test(bad);
        },
        "malformed stop token id");
    return failures;
}

int test_parse_args_errors_still_apply_without_help() {
    int failures = 0;
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)parse_args_for_test({"qus_e2e_bench"}); }, "missing weights");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)parse_args_for_test({"qus_e2e_bench", "--unknown"}); },
        "unknown argument");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)parse_args_for_test({"qus_e2e_bench", "--weights"}); },
        "missing weights value");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)parse_args_for_test(
                {"qus_e2e_bench", "--weights", "weights.qus", "--output-json", "out.json"});
        },
        "missing case");
    return failures;
}

int test_usage_text() {
    const std::string usage = qus::bench::e2e::usage_text("qus_e2e_bench");
    int failures = 0;
    failures += expect_contains(usage, "Usage:", "usage heading");
    failures += expect_contains(usage, "--weights", "usage weights");
    failures += expect_contains(usage, "--output-json", "usage output-json");
    failures += expect_contains(usage, "--case <name>:<prompt-ids-path>:<max-new-tokens>",
                                "usage case");
    failures += expect_contains(usage, "--warmup-repeats", "usage warmup");
    failures += expect_contains(usage, "--repeats", "usage repeats");
    failures += expect_contains(usage, "--max-ctx", "usage max-ctx");
    failures += expect_contains(usage, "--device", "usage device");
    failures += expect_contains(usage, "--stop-token-id", "usage stop token");
    failures += expect_contains(usage, "[248046, 248044]", "usage stop token default");
    failures += expect_contains(usage, "default: " + std::to_string(qus::EngineOptions{}.max_ctx),
                                "usage max_ctx default");
    failures += expect_contains(usage, "default: 1", "usage repeats default");
    failures += expect_contains(usage, "default: 0", "usage warmup or device default");
    return failures;
}

int test_fixture_manifest_reader() {
    int failures = 0;
    const auto manifest_path = repo_file("bench/fixtures/prompts/m2.8-v1.manifest.json");
    const auto metadata = qus::bench::e2e::load_fixture_metadata_for_case(
        manifest_path.string(), "cn_short", "bench/fixtures/prompts/cn_short.ids");
    failures += expect_string(metadata.fixture_set, "m2.8-v1", "manifest fixture_set");
    failures += expect_int_vector(metadata.stop_token_ids, {248046, 248044},
                                  "manifest stop token ids");
    failures += expect_string(metadata.case_metadata.messages_path,
                              "bench/fixtures/prompts/cn_short.messages.json",
                              "manifest messages path");
    failures += expect_string(metadata.case_metadata.prompt_ids_path,
                              "bench/fixtures/prompts/cn_short.ids",
                              "manifest ids path");
    failures += expect_string(metadata.case_metadata.prompt_format, "qwen3.6-chat-template",
                              "manifest prompt format");
    failures += expect_bool(metadata.case_metadata.add_generation_prompt,
                            "manifest add_generation_prompt");
    failures += expect_bool(!metadata.case_metadata.add_special_tokens,
                            "manifest add_special_tokens false");
    failures += expect_bool(!metadata.case_metadata.enable_thinking,
                            "manifest enable_thinking false");
    failures += expect_bool(!metadata.case_metadata.messages_sha256.empty(),
                            "manifest messages sha");
    failures += expect_bool(!metadata.case_metadata.rendered_prompt_sha256.empty(),
                            "manifest rendered prompt sha");
    failures += expect_bool(!metadata.case_metadata.prompt_ids_sha256.empty(),
                            "manifest ids sha");
    failures += expect_int(metadata.case_metadata.prompt_tokens, 31, "manifest prompt tokens");

    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)qus::bench::e2e::load_fixture_metadata_for_case(
                manifest_path.string(), "cn_short", "/tmp/cn_short.ids");
        },
        "manifest rejects basename-only ids match");

    const auto missing_prompt_format = temp_file(
        "qus_e2e_manifest_missing_prompt_format.json",
        R"json({
  "fixture_set": "m2.8-v1",
  "generation": {"stop_token_ids": [248046, 248044]},
  "cases": [{
    "name": "bad",
    "messages": "bad.messages.json",
    "ids": "bad.ids",
    "prompt_tokens": 1,
    "messages_sha256": "m",
    "rendered_prompt_sha256": "r",
    "ids_sha256": "i",
    "add_generation_prompt": true,
    "add_special_tokens": false,
    "chat_template_kwargs": {"enable_thinking": false}
  }]
})json");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)qus::bench::e2e::load_fixture_metadata_for_case(
                missing_prompt_format.string(), "bad", "bad.ids");
        },
        "manifest missing prompt_format");

    const auto wrong_enable_thinking = temp_file(
        "qus_e2e_manifest_wrong_enable_thinking.json",
        R"json({
  "fixture_set": "m2.8-v1",
  "generation": {"stop_token_ids": [248046, 248044]},
  "cases": [{
    "name": "bad",
    "messages": "bad.messages.json",
    "ids": "bad.ids",
    "prompt_tokens": 1,
    "messages_sha256": "m",
    "rendered_prompt_sha256": "r",
    "ids_sha256": "i",
    "prompt_format": "qwen3.6-chat-template",
    "add_generation_prompt": true,
    "add_special_tokens": false,
    "chat_template_kwargs": {"enable_thinking": true}
  }]
})json");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)qus::bench::e2e::load_fixture_metadata_for_case(
                wrong_enable_thinking.string(), "bad", "bad.ids");
        },
        "manifest wrong enable_thinking");
    return failures;
}

int test_prompt_ids_fixture_validation() {
    int failures = 0;
    const auto manifest_path = repo_file("bench/fixtures/prompts/m2.8-v1.manifest.json");
    const auto metadata = qus::bench::e2e::load_fixture_metadata_for_case(
        manifest_path.string(), "cn_short", "bench/fixtures/prompts/cn_short.ids");

    const std::vector<int> ids = qus::bench::e2e::load_verified_prompt_ids(
        "bench/fixtures/prompts/cn_short.ids", metadata.case_metadata);
    failures += expect_int(static_cast<int>(ids.size()), metadata.case_metadata.prompt_tokens,
                           "verified prompt ids size");

    const auto wrong_hash = temp_file("qus_e2e_wrong_hash.ids", "1 2 3\n");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)qus::bench::e2e::load_verified_prompt_ids(wrong_hash.string(),
                                                            metadata.case_metadata);
        },
        "prompt ids sha mismatch");

    qus::bench::e2e::FixtureCaseMetadata wrong_count = metadata.case_metadata;
    wrong_count.prompt_ids_sha256 = qus::bench::e2e::sha256_file_or_empty(wrong_hash.string());
    wrong_count.prompt_tokens = 4;
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)qus::bench::e2e::load_verified_prompt_ids(wrong_hash.string(), wrong_count);
        },
        "prompt ids token count mismatch");
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
    const Json report = parse_json_file(path);
    const Json& error = report.at("error");
    int failures = 0;
    failures += expect_json_value(report.at("schema_version"), 1, "error report schema version");
    failures += expect_json_value(report.at("artifact_type"), "qus_e2e_benchmark_report",
                                  "error report artifact type");
    failures += expect_json_value(report.at("status"), "error", "error report status");
    failures += expect_bool(error.is_object(), "error report error object");
    failures += expect_json_value(error.at("phase"), "load", "error report phase");
    failures += expect_json_value(error.at("message"), "missing \"file\"", "error report message");
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
    case_report.input.prompt_ids_path = "bench/fixtures/prompts/cn_short.ids";
    case_report.input.prompt_ids = {1, 2, 3};
    case_report.input.requested_max_new_tokens = 3;
    case_report.input.stop_token_ids = {248046, 248044};
    case_report.input.max_context = 8;
    case_report.prompt_format = "qwen3.6-chat-template";
    case_report.messages_path = "bench/fixtures/prompts/cn_short.messages.json";
    case_report.messages_sha256 = "messages-sha";
    case_report.rendered_prompt_sha256 = "rendered-sha";
    case_report.prompt_ids_sha256 = "ids-sha";
    case_report.fixture_manifest_sha256 = "manifest-sha";
    case_report.measured_repeats = 2;
    case_report.repeats.push_back(repeat);
    case_report.repeats.push_back(zero_decode_repeat);

    qus::bench::e2e::CaseReport zero_decode_case = case_report;
    zero_decode_case.input.name = "long_2k";
    zero_decode_case.input.prompt_ids_path = "long_2k.ids";
    zero_decode_case.input.prompt_ids = {1, 2, 3, 4};
    zero_decode_case.input.requested_max_new_tokens = 1;
    zero_decode_case.input.stop_token_ids = {248046, 248044};
    zero_decode_case.prompt_format = "qwen3.6-chat-template";
    zero_decode_case.messages_path = "bench/fixtures/prompts/long_2k.messages.json";
    zero_decode_case.messages_sha256 = "long-messages-sha";
    zero_decode_case.rendered_prompt_sha256 = "long-rendered-sha";
    zero_decode_case.prompt_ids_sha256 = "long-ids-sha";
    zero_decode_case.measured_repeats = 1;
    zero_decode_case.repeats.clear();
    zero_decode_case.repeats.push_back(zero_decode_repeat);

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
    report.decode_path = "cuda_graph";
    report.post_load_memory = memory;
    report.cases.push_back(case_report);
    report.cases.push_back(zero_decode_case);

    const auto path = std::filesystem::temp_directory_path() / "qus_e2e_raw_report.json";
    qus::bench::e2e::write_raw_report(path.string(), report);
    const Json raw = parse_json_file(path);
    const Json& run = raw.at("run");
    const Json& environment = raw.at("environment");
    const Json& engine = raw.at("engine");
    const Json& weights = raw.at("weights");
    const Json& memory_report = raw.at("memory");
    const Json& summary = raw.at("summary");
    const Json& cases = raw.at("cases");
    const Json& cn_case = cases.at(0);
    const Json& long_case = cases.at(1);
    const Json& cn_summary = cn_case.at("summary");
    const Json& long_summary = long_case.at("summary");
    const Json& cn_repeat = cn_case.at("repeats").at(0);
    const Json& zero_decode_repeat_json = cn_case.at("repeats").at(1);
    int failures = 0;
    failures += expect_json_value(raw.at("schema_version"), 1, "raw report schema version");
    failures += expect_json_value(raw.at("artifact_type"), "qus_e2e_benchmark_report",
                                  "raw report artifact_type");
    failures += expect_json_value(raw.at("status"), "ok", "raw report status");
    failures += expect_bool(run.is_object(), "raw report run section");
    failures += expect_bool(environment.is_object(), "raw report environment section");
    failures += expect_bool(engine.is_object(), "raw report engine section");
    failures += expect_bool(weights.is_object(), "raw report weights section");
    failures += expect_bool(memory_report.is_object(), "raw report memory section");
    failures += expect_bool(summary.is_object(), "raw report summary section");
    failures += expect_bool(cases.is_array(), "raw report cases section");
    failures += expect_u64(cases.size(), 2, "raw report case count");

    failures += expect_json_value(run.at("binary"), "qus_e2e_bench", "raw report binary");
    failures += expect_json_value(run.at("command"), "qus_e2e_bench --case cn_short:prompt.ids:3",
                                  "raw report command");
    failures += expect_json_value(run.at("git_commit"), "abc", "raw report git commit");
    failures += expect_json_value(run.at("worktree_dirty"), false, "raw report dirty flag");
    failures += expect_double(run.at("load_time_s").get<double>(), 1.25,
                              "raw report run load time");

    failures += expect_json_value(environment.at("cuda_runtime_version"), "12.4",
                                  "raw report cuda runtime");
    failures += expect_json_value(environment.at("cuda_driver_version"), "12.4",
                                  "raw report cuda driver");
    failures += expect_json_value(environment.at("gpu_name"), "test gpu",
                                  "raw report environment gpu");
    failures += expect_json_value(environment.at("device_id"), 0, "raw report device id");

    failures += expect_json_value(engine.at("max_context"), 8, "raw report max context");
    failures += expect_json_value(engine.at("workspace_lifetime_policy"),
                                  "block_scoped_mixer_mlp_rewind",
                                  "raw report workspace lifetime policy");
    failures += expect_json_value(engine.at("decode_metric"), "decode_tok_s",
                                  "raw report decode metric");
    failures += expect_json_value(engine.at("decode_path"), "cuda_graph",
                                  "raw report decode path");
    failures += expect_json_value(engine.at("sampling_location"), "device_argmax",
                                  "raw report sampling location");
    failures += expect_json_value(engine.at("token_readback"), "per_step_sync_d2h",
                                  "raw report token readback");
    failures += expect_json_value(engine.at("includes_token_readback"), true,
                                  "raw report includes token readback");
    failures += expect_json_value(engine.at("timing_boundary"), "host_visible_phase_end",
                                  "raw report timing boundary");

    failures += expect_json_value(weights.at("q5090_path"), "weights.qus",
                                  "raw report weights path");
    failures += expect_json_value(weights.at("q5090_file_size_bytes"), 1234,
                                  "raw report weights file size");
    failures += expect_json_value(weights.at("q5090_sha256"), "weights-sha",
                                  "raw report weights sha");
    failures += expect_json_value(weights.at("q5090_conv1d_layout"),
                                  "runtime_native_conv_dim_by_kernel",
                                  "raw report conv1d layout");
    failures += expect_json_value(weights.at("selected_modules").at("text_core"), true,
                                  "raw report text core selected");
    failures += expect_json_value(weights.at("selected_modules").at("mtp"), false,
                                  "raw report mtp not selected");
    failures += expect_json_value(weights.at("selected_modules").at("vision"), false,
                                  "raw report vision not selected");
    failures += expect_json_value(weights.at("q5090_loaded_payload_bytes"), 700,
                                  "raw report loaded payload");
    failures += expect_json_value(weights.at("weight_arena_slack_bytes"), 200,
                                  "raw report weight arena slack");
    failures += expect_json_value(weights.at("weight_payload_to_arena_used_overhead_bytes"), 100,
                                  "raw report weight payload overhead");

    failures += expect_json_value(memory_report.at("accounting_scope"),
                                  "engine_owned_device_arenas_complete",
                                  "raw report memory accounting scope");
    failures += expect_json_value(memory_report.at("hidden_device_allocations"), false,
                                  "raw report hidden allocation flag");
    failures += expect_json_value(memory_report.at("loaded"), true, "raw report memory loaded");
    failures += expect_json_value(memory_report.at("device"), 0, "raw report memory device");
    failures += expect_json_value(memory_report.at("max_context"), 8,
                                  "raw report memory max context");
    failures += expect_u64(memory_report.at("arenas").size(), 3, "raw report arena count");
    failures += expect_json_value(memory_report.at("arenas").at(0).at("name"), "weights",
                                  "raw report weight arena");
    failures += expect_json_value(memory_report.at("arenas").at(1).at("name"), "cache",
                                  "raw report cache arena");
    failures += expect_json_value(memory_report.at("arenas").at(2).at("name"), "workspace",
                                  "raw report workspace arena");
    failures += expect_json_value(memory_report.at("q5090_tensor_count"), 10,
                                  "raw report tensor count");
    failures += expect_json_value(memory_report.at("q5090_quant_count"), 6,
                                  "raw report quant count");

    failures += expect_json_value(summary.at("case_count"), 2, "raw report summary case count");
    failures += expect_double(summary.at("load_time_s").get<double>(), 1.25,
                              "raw report summary load time");

    failures += expect_json_value(cn_case.at("name"), "cn_short", "raw report case name");
    failures += expect_json_value(cn_case.at("fixture_set"), "m2.8-v1",
                                  "raw report fixture set");
    failures += expect_json_value(cn_case.at("fixture_manifest_sha256"), "manifest-sha",
                                  "raw report fixture manifest sha");
    failures += expect_json_value(cn_case.at("prompt_ids_sha256"), "ids-sha",
                                  "raw report fixture ids sha");
    failures += expect_json_value(cn_case.at("prompt_format"), "qwen3.6-chat-template",
                                  "raw report prompt format");
    failures += expect_json_value(cn_case.at("messages_path"),
                                  "bench/fixtures/prompts/cn_short.messages.json",
                                  "raw report messages path");
    failures += expect_json_value(cn_case.at("messages_sha256"), "messages-sha",
                                  "raw report messages sha");
    failures += expect_json_value(cn_case.at("rendered_prompt_sha256"), "rendered-sha",
                                  "raw report rendered prompt sha");
    failures += expect_json_value(cn_case.at("add_generation_prompt"), true,
                                  "raw report add generation prompt");
    failures += expect_json_value(cn_case.at("add_special_tokens"), false,
                                  "raw report add special tokens");
    failures += expect_json_value(cn_case.at("chat_template_kwargs").at("enable_thinking"), false,
                                  "raw report enable thinking");
    failures += expect_json_int_vector(cn_case.at("stop_token_ids"), {248046, 248044},
                                       "raw report stop token ids");
    failures += expect_json_not_contains_key(raw, "eos_token_id", "raw report eos token id removed");

    failures += expect_bool(cn_summary.contains("prefill_time_s_median"),
                            "raw report prefill median field");
    failures += expect_double(cn_summary.at("prefill_prompt_tok_s_median").get<double>(), 300.0,
                              "raw report summary prefill prompt throughput");
    failures += expect_double(cn_summary.at("decode_tok_s_median").get<double>(), 100.0,
                              "raw report summary decode throughput");
    failures += expect_json_value(cn_summary.at("deterministic_token_ids"), true,
                                  "raw report deterministic summary");
    failures += expect_json_value(cn_summary.at("max_workspace_arena_peak_used_bytes"), 0,
                                  "raw report max workspace peak summary");

    failures += expect_double(cn_repeat.at("prefill_prompt_tok_s").get<double>(), 300.0,
                              "raw report repeat prefill prompt throughput");
    failures += expect_json_value(cn_repeat.at("decode_tok_s_valid"), true,
                                  "raw report decode validity");
    failures += expect_double(cn_repeat.at("decode_tok_s").get<double>(), 100.0,
                              "raw report repeat decode throughput");
    failures += expect_json_int_vector(cn_repeat.at("generated_token_ids"), {10, 11, 12},
                                       "raw report generated ids");

    failures += expect_json_null(zero_decode_repeat_json.at("decode_tok_s"),
                                 "zero decode throughput null");
    failures += expect_json_value(zero_decode_repeat_json.at("decode_tok_s_valid"), false,
                                  "zero decode throughput validity");
    failures += expect_json_null(long_summary.at("decode_tok_s_median"),
                                 "zero decode summary throughput null");
    failures += expect_json_value(long_case.at("name"), "long_2k", "zero decode case name");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_ids_file();
    failures += test_parse_case_arg();
    failures += test_parse_args_help();
    failures += test_parse_args_stop_tokens();
    failures += test_parse_args_errors_still_apply_without_help();
    failures += test_usage_text();
    failures += test_fixture_manifest_reader();
    failures += test_prompt_ids_fixture_validation();
    failures += test_context_validation();
    failures += test_json_helpers();
    failures += test_error_report_json_is_valid();
    failures += test_raw_report_json_is_valid();
    return failures == 0 ? 0 : fail("e2e bench support test failed");
}
