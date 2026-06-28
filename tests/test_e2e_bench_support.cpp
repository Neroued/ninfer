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

int expect_int(int actual, int expected, std::string_view label) {
    if (actual == expected) { return 0; }
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

int expect_not_contains(const std::string& actual, std::string_view unexpected,
                        std::string_view label) {
    if (actual.find(unexpected) == std::string::npos) { return 0; }
    std::cerr << label << " expected not to find `" << unexpected << "` in:\n" << actual << '\n';
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

    std::vector<std::string> eos_alias = base;
    eos_alias.insert(eos_alias.end(), {"--eos-token-id", "123"});
    const qus::bench::e2e::RunOptions parsed_alias = parse_args_for_test(eos_alias);
    failures += expect_int_vector(parsed_alias.stop_token_ids, {123},
                                  "deprecated eos alias stop token ids");

    failures += expect_throws<std::invalid_argument>(
        [&] {
            std::vector<std::string> mixed = base;
            mixed.insert(mixed.end(), {"--stop-token-id", "7", "--eos-token-id", "123"});
            (void)parse_args_for_test(mixed);
        },
        "stop token then eos alias mixed flags");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            std::vector<std::string> mixed = base;
            mixed.insert(mixed.end(), {"--eos-token-id", "123", "--stop-token-id", "7"});
            (void)parse_args_for_test(mixed);
        },
        "eos alias then stop token mixed flags");

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
    failures += expect_contains(usage, "--eos-token-id", "usage eos");
    failures += expect_contains(usage, "deprecated", "usage eos deprecated");
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
    failures += expect_int(metadata.case_metadata.prompt_tokens, 60, "manifest prompt tokens");

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
    report.post_load_memory = memory;
    report.cases.push_back(case_report);
    report.cases.push_back(zero_decode_case);

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
    failures += text.find("\"workspace_lifetime_policy\": \"block_scoped_mixer_mlp_rewind\"") !=
                        std::string::npos
                    ? 0
                    : fail("raw report workspace lifetime policy missing");
    failures += text.find("\"token_readback\": \"per_step_sync_d2h\"") != std::string::npos &&
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
    failures += text.find("\"prompt_format\": \"qwen3.6-chat-template\"") != std::string::npos &&
                        text.find("\"messages_path\": \"bench/fixtures/prompts/cn_short.messages.json\"") !=
                            std::string::npos &&
                        text.find("\"messages_sha256\": \"messages-sha\"") != std::string::npos &&
                        text.find("\"rendered_prompt_sha256\": \"rendered-sha\"") !=
                            std::string::npos
                    ? 0
                    : fail("raw report chat identity missing");
    failures += text.find("\"add_generation_prompt\": true") != std::string::npos &&
                        text.find("\"add_special_tokens\": false") != std::string::npos &&
                        text.find("\"chat_template_kwargs\"") != std::string::npos &&
                        text.find("\"enable_thinking\": false") != std::string::npos
                    ? 0
                    : fail("raw report chat template kwargs missing");
    failures += text.find("\"stop_token_ids\": [248046, 248044]") != std::string::npos
                    ? 0
                    : fail("case stop_token_ids missing");
    failures += expect_not_contains(text, "\"eos_token_id\"", "case eos_token_id removed");
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
    failures += text.find("\"decode_eager_tok_s_median\": null") != std::string::npos
                    ? 0
                    : fail("zero decode summary throughput null handling missing");
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
