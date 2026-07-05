#include "qus_bench_support.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace qb = qus::bench;

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

int expect_bool(bool value, std::string_view label) { return value ? 0 : fail(label); }

int expect_u32(std::uint32_t actual, std::uint32_t expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_string(const std::string& actual, std::string_view expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected `" << expected << "`, got `" << actual << "`\n";
    return 1;
}

int expect_double_near(double actual, double expected, std::string_view label) {
    if (std::abs(actual - expected) < 1e-6) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
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

qb::BenchOptions parse_args_for_test(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) { argv.push_back(arg.data()); }
    return qb::parse_args(static_cast<int>(argv.size()), argv.data());
}

int test_expand_tests_defaults_and_labels() {
    int failures = 0;
    qb::BenchOptions defaults;
    defaults.weights_path                          = "w.qus";
    const std::vector<qb::BenchTest> default_tests = qb::expand_tests(defaults);
    failures += expect_size(default_tests.size(), 2, "default test count");
    failures += expect_string(default_tests[0].label, "pp512", "default prefill label");
    failures += expect_string(default_tests[1].label, "tg128", "default decode label");

    qb::BenchOptions options;
    options.n_prompt                       = {128, 2048};
    options.n_gen                          = {64};
    options.prompt_gen                     = {{2048, 128}};
    const std::vector<qb::BenchTest> tests = qb::expand_tests(options);
    failures += expect_size(tests.size(), 4, "expanded test count");
    failures += expect_string(tests[0].label, "pp128", "prefill label 0");
    failures += expect_string(tests[1].label, "pp2048", "prefill label 1");
    failures += expect_string(tests[2].label, "tg64", "decode label");
    failures += expect_string(tests[3].label, "pp2048+tg128", "combined label");
    failures += expect_bool(tests[0].kind == qb::TestKind::Prefill, "prefill kind");
    failures += expect_bool(tests[2].kind == qb::TestKind::Decode, "decode kind");
    failures += expect_bool(tests[3].kind == qb::TestKind::PrefillDecode, "combined kind");
    return failures;
}

int test_required_context_and_max_ctx() {
    int failures = 0;
    const qb::BenchTest pp{qb::TestKind::Prefill, 512, 0, "pp512"};
    const qb::BenchTest tg{qb::TestKind::Decode, 0, 128, "tg128"};
    const qb::BenchTest pgt{qb::TestKind::PrefillDecode, 2048, 128, "pp2048+tg128"};
    failures += expect_u32(pp.required_context(), 512, "pp required context");
    failures +=
        expect_u32(tg.required_context(), 128 + qb::kDecodeSeedTokens, "tg required context");
    failures += expect_u32(pgt.required_context(), 2176, "pp+tg required context");

    const std::vector<qb::BenchTest> tests = {pp, tg, pgt};
    failures += expect_u32(qb::resolve_max_ctx(tests, std::nullopt), 2176, "auto max_ctx");
    failures += expect_u32(qb::resolve_max_ctx(tests, std::optional<std::uint32_t>(4096)), 4096,
                           "override max_ctx honored");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qb::resolve_max_ctx(tests, std::optional<std::uint32_t>(2000)); },
        "override below requirement throws");
    return failures;
}

int test_corpus_load_and_slice() {
    int failures               = 0;
    const auto ok              = temp_file("qus_bench_corpus_ok.ids", "10 11 12\n13\t14\n15\n");
    const std::vector<int> ids = qb::load_corpus_ids(ok.string());
    failures += expect_size(ids.size(), 6, "corpus size");
    failures += expect_bool(ids[0] == 10 && ids[5] == 15, "corpus values");

    const std::vector<int> slice = qb::prompt_slice(ids, 3);
    failures += expect_size(slice.size(), 3, "slice size");
    failures +=
        expect_bool(slice[0] == 10 && slice[1] == 11 && slice[2] == 12, "slice exact prefix");

    failures += expect_throws<std::invalid_argument>([&] { (void)qb::prompt_slice(ids, 7); },
                                                     "slice beyond corpus throws");
    failures += expect_throws<std::invalid_argument>([&] { (void)qb::prompt_slice(ids, 0); },
                                                     "slice zero throws");

    const auto empty = temp_file("qus_bench_corpus_empty.ids", "\n \t");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qb::load_corpus_ids(empty.string()); }, "empty corpus throws");
    const auto negative = temp_file("qus_bench_corpus_negative.ids", "1 -2 3");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qb::load_corpus_ids(negative.string()); }, "negative id throws");
    const auto nondigit = temp_file("qus_bench_corpus_nondigit.ids", "1 a 3");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qb::load_corpus_ids(nondigit.string()); }, "non-digit id throws");
    return failures;
}

int test_validate_prompt_lengths() {
    int failures                           = 0;
    const std::vector<qb::BenchTest> tests = {
        qb::BenchTest{qb::TestKind::Prefill, 50, 0, "pp50"},
        qb::BenchTest{qb::TestKind::PrefillDecode, 80, 20, "pp80+tg20"}};
    qb::validate_prompt_lengths(tests, 100); // ok

    failures += expect_throws<std::invalid_argument>(
        [&] { qb::validate_prompt_lengths(tests, 60); }, "prefill exceeding corpus throws");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            const std::vector<qb::BenchTest> decode_only = {
                qb::BenchTest{qb::TestKind::Decode, 0, 8, "tg8"}};
            qb::validate_prompt_lengths(decode_only, 0);
        },
        "empty corpus for decode throws");
    return failures;
}

int test_parse_args() {
    int failures = 0;
    const qb::BenchOptions parsed =
        parse_args_for_test({"qus_bench", "--weights", "w.qus", "-p",
                             "128,512",   "-n",        "64",    "-pg",
                             "2048,128",  "-r",        "3",     "--warmup",
                             "2",         "--max-ctx", "4096",  "--prefill-chunk",
                             "128",       "--mtp-draft-tokens",
                             "5",         "-o",        "json",   "--no-cuda-graph"});
    failures += expect_string(parsed.weights_path, "w.qus", "weights path");
    failures += expect_size(parsed.n_prompt.size(), 2, "n_prompt list size");
    failures +=
        expect_bool(parsed.n_prompt[0] == 128 && parsed.n_prompt[1] == 512, "n_prompt values");
    failures += expect_size(parsed.n_gen.size(), 1, "n_gen list size");
    failures += expect_size(parsed.prompt_gen.size(), 1, "prompt_gen size");
    failures +=
        expect_bool(parsed.prompt_gen[0].first == 2048 && parsed.prompt_gen[0].second == 128,
                    "prompt_gen values");
    failures += expect_bool(parsed.repetitions == 3, "repetitions parsed");
    failures += expect_bool(parsed.warmup == 2, "warmup parsed");
    failures +=
        expect_bool(parsed.max_ctx.has_value() && *parsed.max_ctx == 4096, "max_ctx parsed");
    failures += expect_bool(parsed.prefill_chunk == 128, "prefill_chunk parsed");
    failures += expect_bool(parsed.mtp_draft_tokens == 5, "mtp draft tokens parsed");
    failures += expect_bool(parsed.output == qb::OutputFormat::Json, "output json parsed");
    failures += expect_bool(!parsed.use_cuda_graph, "no-cuda-graph parsed");

    const qb::BenchOptions help = parse_args_for_test({"qus_bench", "--help"});
    failures += expect_bool(help.help_requested, "help flag");

    failures += expect_throws<std::invalid_argument>(
        [&] { (void)parse_args_for_test({"qus_bench"}); }, "missing weights throws");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)parse_args_for_test({"qus_bench", "--weights", "w.qus", "--unknown"}); },
        "unknown arg throws");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)parse_args_for_test({"qus_bench", "--weights", "w.qus", "-o", "yaml"}); },
        "bad output format throws");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)parse_args_for_test({"qus_bench", "--weights", "w.qus", "-p", "512,0"}); },
        "non-positive n_prompt throws");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)parse_args_for_test({"qus_bench", "--weights", "w.qus", "-pg", "2048"}); },
        "malformed prompt-gen throws");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)parse_args_for_test(
                {"qus_bench", "--weights", "w.qus", "--prefill-chunk", "129"});
        },
        "non-128 prefill chunk throws");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)parse_args_for_test(
                {"qus_bench", "--weights", "w.qus", "--mtp-draft-tokens", "6"});
        },
        "too many mtp draft tokens throws");
    return failures;
}

int test_stats() {
    int failures          = 0;
    const qb::Stats three = qb::compute_stats({1.0, 2.0, 3.0});
    failures += expect_double_near(three.mean, 2.0, "stats mean");
    failures += expect_double_near(three.stddev, 1.0, "stats sample stddev");
    failures += expect_bool(three.count == 3, "stats count");

    const qb::Stats single = qb::compute_stats({5.0});
    failures += expect_double_near(single.mean, 5.0, "single stats mean");
    failures += expect_double_near(single.stddev, 0.0, "single stats stddev zero");
    return failures;
}

std::vector<qb::TestResult> sample_results() {
    qb::TestResult pp;
    pp.test                 = qb::BenchTest{qb::TestKind::Prefill, 512, 0, "pp512"};
    pp.reps                 = {qb::RepTiming{0.5, 0.0}, qb::RepTiming{0.25, 0.0}};
    pp.reps[0].mtp          = qb::BenchMtpStats{true, 5, 10, 4, 2, 1, {2, 1, 1, 0, 0}};
    pp.reps[1].mtp          = qb::BenchMtpStats{true, 5, 5, 2, 1, 0, {1, 1, 0, 0, 0}};
    pp.workspace_peak_bytes = 5368709120ULL; // 5 GiB

    qb::TestResult tg;
    tg.test                 = qb::BenchTest{qb::TestKind::Decode, 0, 128, "tg128"};
    tg.reps                 = {qb::RepTiming{0.0, 0.5}, qb::RepTiming{0.0, 1.0}};
    tg.reps[0].mtp          = qb::BenchMtpStats{true, 5, 5, 0, 1, 1, {0, 0, 0, 0, 0}};
    tg.reps[1].mtp          = qb::BenchMtpStats{true, 5, 0, 0, 0, 2, {0, 0, 0, 0, 0}};
    tg.workspace_peak_bytes = 1048576ULL; // 1 MiB
    return {pp, tg};
}

int test_format_json_schema() {
    int failures = 0;
    qb::BenchEnvironment env;
    env.gpu_name      = "test gpu";
    env.git_commit    = "abc";
    env.max_ctx       = 640;
    env.decode_path   = "cuda_graph";
    env.repetitions   = 2;
    env.prefill_chunk = qus::model::kDefaultPrefillChunk;
    env.mtp_draft_tokens = 5;
    env.corpus_path   = "bench/fixtures/bench_corpus.ids";
    env.corpus_tokens = 10241;
    env.weights_path  = "w.qus";

    const std::string json_text =
        qb::format_json(env, "qus_bench --weights w.qus", sample_results());
    Json report;
    try {
        report = Json::parse(json_text);
    } catch (const nlohmann::json::exception& e) {
        return fail(std::string("format_json produced invalid JSON: ") + e.what());
    }
    failures += expect_bool(report.at("schema_version").get<int>() == 3, "json schema version");
    failures += expect_string(report.at("artifact_type").get<std::string>(), "qus_bench_report",
                              "json artifact type");
    failures += expect_bool(report.at("config").at("prefill_chunk").get<int>() ==
                                static_cast<int>(qus::model::kDefaultPrefillChunk),
                            "json config prefill_chunk");
    failures += expect_bool(report.at("config").at("mtp_draft_tokens").get<int>() == 5,
                            "json config mtp_draft_tokens");
    const Json& tests = report.at("tests");
    failures += expect_bool(tests.is_array() && tests.size() == 2, "json tests array");

    const Json& pp = tests.at(0);
    failures += expect_string(pp.at("label").get<std::string>(), "pp512", "json pp label");
    failures += expect_string(pp.at("kind").get<std::string>(), "pp", "json pp kind");
    failures += expect_bool(pp.at("n_gen").get<int>() == 0, "json pp n_gen");
    // 512 / 0.5 = 1024, 512 / 0.25 = 2048, mean 1536.
    failures += expect_double_near(pp.at("prefill_tok_s_mean").get<double>(), 1536.0,
                                   "json pp prefill mean");
    failures += expect_bool(pp.at("decode_tok_s_mean").is_null(), "json pp decode null");
    failures += expect_bool(pp.at("workspace_peak_bytes").get<std::uint64_t>() == 5368709120ULL,
                            "json pp workspace peak");
    failures += expect_bool(pp.at("mtp").at("enabled").get<bool>(), "json pp mtp enabled");
    failures += expect_bool(pp.at("mtp").at("k").get<int>() == 5, "json pp mtp k");
    failures += expect_bool(pp.at("mtp").at("draft_tokens").get<int>() == 15,
                            "json pp mtp draft tokens");
    failures += expect_bool(pp.at("mtp").at("accepted_tokens").get<int>() == 6,
                            "json pp mtp accepted tokens");
    failures += expect_double_near(pp.at("mtp").at("acceptance_rate").get<double>(), 0.4,
                                   "json pp mtp acceptance rate");
    failures += expect_double_near(pp.at("mtp").at("acceptance_length").get<double>(), 3.0,
                                   "json pp mtp acceptance length");
    failures += expect_bool(pp.at("mtp").at("rounds").get<int>() == 3, "json pp mtp rounds");
    failures += expect_bool(pp.at("mtp").at("fallback_steps").get<int>() == 1,
                            "json pp mtp fallback steps");
    failures += expect_bool(pp.at("mtp").at("accepted_per_pos").is_array() &&
                                pp.at("mtp").at("accepted_per_pos").size() == 5,
                            "json pp mtp accepted per pos");
    failures += expect_bool(pp.at("mtp").at("accepted_per_pos").at(0).get<int>() == 3,
                            "json pp mtp accepted per pos 0");
    failures += expect_bool(pp.at("reps").is_array() && pp.at("reps").size() == 2, "json pp reps");

    const Json& tg = tests.at(1);
    failures += expect_string(tg.at("kind").get<std::string>(), "tg", "json tg kind");
    // 128 / 0.5 = 256, 128 / 1.0 = 128, mean 192.
    failures +=
        expect_double_near(tg.at("decode_tok_s_mean").get<double>(), 192.0, "json tg decode mean");
    failures += expect_bool(tg.at("prefill_tok_s_mean").is_null(), "json tg prefill null");
    return failures;
}

int test_format_table_and_csv() {
    int failures                              = 0;
    const std::vector<qb::TestResult> results = sample_results();
    qb::BenchEnvironment env;
    env.decode_path         = "cuda_graph";
    env.prefill_chunk       = qus::model::kDefaultPrefillChunk;
    env.mtp_draft_tokens    = 0;
    const std::string table = qb::format_table(env, results);
    failures += expect_bool(table.find("pp512") != std::string::npos, "table has pp512");
    failures += expect_bool(table.find("tg128") != std::string::npos, "table has tg128");
    failures += expect_bool(table.find("prefill t/s") != std::string::npos, "table has header");
    failures += expect_bool(
        table.find("prefill_chunk=" + std::to_string(qus::model::kDefaultPrefillChunk)) !=
            std::string::npos,
        "table has prefill_chunk config");
    failures += expect_bool(table.find("mtp_k=0") != std::string::npos, "table has mtp config");
    failures += expect_bool(table.find("work peak") != std::string::npos, "table has work peak");
    failures += expect_bool(table.find("mtp acc") != std::string::npos, "table has mtp acc");
    failures += expect_bool(table.find("GiB") != std::string::npos, "table shows GiB peak");

    failures +=
        expect_string(qb::decode_path_name(true, 0), "cuda_graph", "decode path cuda graph");
    failures += expect_string(qb::decode_path_name(false, 0), "eager", "decode path eager");
    failures += expect_string(qb::decode_path_name(true, 5), "mtp_cuda_graph",
                              "decode path mtp cuda graph");
    failures += expect_string(qb::decode_path_name(false, 5), "mtp_eager",
                              "decode path mtp eager");

    const std::string csv = qb::format_csv(env, results);
    failures += expect_bool(csv.find("label,kind,n_prompt") == 0, "csv header first");
    failures +=
        expect_bool(csv.find("prefill_chunk") != std::string::npos, "csv has prefill_chunk");
    failures += expect_bool(csv.find("mtp_draft_tokens") != std::string::npos,
                            "csv has mtp_draft_tokens");
    failures += expect_bool(csv.find("mtp_acceptance_rate") != std::string::npos,
                            "csv has mtp_acceptance_rate");
    failures += expect_bool(csv.find("workspace_peak_bytes") != std::string::npos,
                            "csv has workspace_peak_bytes");
    std::size_t lines = 0;
    for (const char c : csv) {
        if (c == '\n') { ++lines; }
    }
    failures += expect_size(lines, 3, "csv line count (header + 2 rows)");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_expand_tests_defaults_and_labels();
    failures += test_required_context_and_max_ctx();
    failures += test_corpus_load_and_slice();
    failures += test_validate_prompt_lengths();
    failures += test_parse_args();
    failures += test_stats();
    failures += test_format_json_schema();
    failures += test_format_table_and_csv();
    return failures == 0 ? 0 : fail("qus_bench support test failed");
}
