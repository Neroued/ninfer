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

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_ids_file();
    failures += test_parse_case_arg();
    failures += test_context_validation();
    return failures == 0 ? 0 : fail("e2e bench support test failed");
}
