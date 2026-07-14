#include "ninfer_bench_support.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace qb = ninfer::bench;

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

int expect(bool value, std::string_view label) { return value ? 0 : fail(label); }

int expect_u32(std::uint32_t actual, std::uint32_t expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_string(std::string_view actual, std::string_view expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected `" << expected << "`, got `" << actual << "`\n";
    return 1;
}

int expect_near(double actual, double expected, std::string_view label) {
    if (std::abs(actual - expected) < 1e-6) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, std::string_view label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    return fail(std::string(label) + " did not throw");
}

qb::BenchOptions parse_for_test(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return qb::parse_args(static_cast<int>(argv.size()), argv.data());
}

int test_cli_contract() {
    int failures                  = 0;
    const qb::BenchOptions parsed = parse_for_test({
        "ninfer_bench",
        "--weights",
        "model.ninfer",
        "-p",
        "128,512",
        "-n",
        "64",
        "-pg",
        "2048,128",
        "-r",
        "3",
        "--warmup",
        "2",
        "--max-ctx",
        "4096",
        "--prefill-chunk",
        "128",
        "--kv-dtype",
        "int8",
        "--mtp-draft-tokens",
        "5",
        "--lm-head-draft",
        "--device",
        "1",
        "--no-cuda-graph",
        "--output",
        "json",
        "--output-file",
        "report.json",
    });

    failures += expect_string(parsed.artifact_path, "model.ninfer", "artifact path");
    failures += expect(parsed.n_prompt == std::vector<int>({128, 512}), "prompt list");
    failures += expect(parsed.n_gen == std::vector<int>({64}), "generation list");
    failures += expect(parsed.prompt_gen == std::vector<std::pair<int, int>>({{2048, 128}}),
                       "combined list");
    failures += expect(parsed.repetitions == 3 && parsed.warmup == 2, "repetition settings");
    failures += expect(parsed.max_context == std::optional<std::uint32_t>(4096), "max context");
    failures += expect(parsed.prefill_chunk == 128, "prefill chunk");
    failures += expect(parsed.kv_cache == ninfer::KvCacheStorage::Int8Group64, "INT8 KV");
    failures += expect(parsed.mtp_draft_tokens == 5, "MTP window");
    failures +=
        expect(parsed.proposal_head == ninfer::ProposalHead::Optimized, "optimized proposal head");
    failures += expect(parsed.device == 1 && !parsed.use_cuda_graph, "device and graph settings");
    failures +=
        expect(parsed.output == qb::OutputFormat::Json && parsed.output_file == "report.json",
               "output settings");

    const auto defaults = qb::expand_tests(qb::BenchOptions{});
    failures +=
        expect(defaults.size() == 2 && defaults[0].label == "pp512" && defaults[1].label == "tg128",
               "default pp/tg matrix");
    failures += expect(qb::usage_text("ninfer_bench").find("artifact.ninfer") != std::string::npos,
                       "help names native artifact");
    failures += expect(parse_for_test({"ninfer_bench", "--help"}).help_requested, "help flag");

    failures += expect_throws<std::invalid_argument>([] { (void)parse_for_test({"ninfer_bench"}); },
                                                     "missing artifact");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--lm-head-draft"});
        },
        "optimized head without MTP");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--mtp-draft-tokens", "6"});
        },
        "unsupported MTP window");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--prefill-chunk", "129"});
        },
        "misaligned prefill chunk");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--kv-dtype", "fp8"});
        },
        "unsupported KV storage");
    return failures;
}

int test_measurement_contract() {
    int failures = 0;
    const qb::BenchTest pp{qb::TestKind::Prefill, 512, 0, "pp512"};
    const qb::BenchTest tg{qb::TestKind::Decode, 0, 128, "tg128"};
    const qb::BenchTest combined{qb::TestKind::PrefillDecode, 2048, 128, "pp2048+tg128"};

    failures += expect_u32(pp.requested_output_tokens(), 1, "pp begin output");
    failures += expect_u32(tg.requested_output_tokens(), 129, "tg begin plus G outputs");
    failures +=
        expect_u32(combined.requested_output_tokens(), 129, "combined begin plus G outputs");
    failures += expect_u32(pp.required_context(0), 512, "pp context");
    failures += expect_u32(pp.required_context(5), 522, "MTP pp context");
    failures += expect_u32(tg.required_context(0), 129, "tg context");
    failures += expect_u32(tg.required_context(5), 139, "MTP tg context");
    failures += expect_u32(combined.required_context(5), 2186, "MTP combined context");
    failures += expect_u32(qb::decode_graph_prime_output_tokens(5), 13, "MTP graph-prime outputs");
    failures +=
        expect_u32(qb::decode_graph_prime_required_context(5), 23, "MTP graph-prime context");

    const std::vector<qb::BenchTest> matrix = {pp, tg, combined};
    failures +=
        expect_u32(qb::resolve_max_context(matrix, std::nullopt, 5, true), 2186, "auto context");
    failures +=
        expect_u32(qb::resolve_max_context(matrix, std::optional<std::uint32_t>(4096), 5, true),
                   4096, "explicit context");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qb::resolve_max_context(matrix, std::optional<std::uint32_t>(2048), 5, true); },
        "undersized context");
    return failures;
}

ninfer::GenerationTimings timings(double prepare, double prefill, double decode, double total) {
    return {.prepare_seconds = prepare,
            .vision_seconds  = 0.0,
            .prefill_seconds = prefill,
            .decode_seconds  = decode,
            .total_seconds   = total};
}

ninfer::SpeculativeStats speculative(std::uint64_t rounds, std::uint64_t drafted,
                                     std::uint64_t accepted, std::uint64_t fallback,
                                     std::vector<std::uint64_t> per_position) {
    return {.enabled               = true,
            .draft_window          = 5,
            .rounds                = rounds,
            .drafted_tokens        = drafted,
            .accepted_tokens       = accepted,
            .fallback_steps        = fallback,
            .accepted_per_position = std::move(per_position)};
}

std::vector<qb::TestResult> sample_results() {
    qb::TestResult pp;
    pp.test = {qb::TestKind::Prefill, 512, 0, "pp512"};
    pp.reps = {{timings(0.01, 0.5, 0.0, 0.52), speculative(0, 0, 0, 0, {0, 0, 0, 0, 0}), 1},
               {timings(0.02, 0.25, 0.0, 0.28), speculative(0, 0, 0, 0, {0, 0, 0, 0, 0}), 1}};
    pp.workspace_peak_bytes = 5ULL * 1024ULL * 1024ULL * 1024ULL;

    qb::TestResult tg;
    tg.test = {qb::TestKind::Decode, 0, 3, "tg3"};
    tg.reps = {{timings(0.01, 0.1, 0.5, 0.62), speculative(1, 5, 5, 0, {1, 1, 1, 1, 1}), 4},
               {timings(0.02, 0.1, 1.0, 1.13), speculative(0, 0, 0, 3, {0, 0, 0, 0, 0}), 4}};
    tg.workspace_peak_bytes = 1024ULL * 1024ULL;
    return {std::move(pp), std::move(tg)};
}

qb::BenchEnvironment sample_environment() {
    qb::BenchEnvironment env;
    env.gpu_name                         = "RTX 5090";
    env.cuda_runtime_version             = "13.1";
    env.cuda_driver_version              = "590.1";
    env.device_id                        = 0;
    env.artifact_path                    = "model.ninfer";
    env.artifact_file_size_bytes         = 17500000000ULL;
    env.load                             = {.target               = "qwen3_6_27b_rtx5090",
                                            .load_seconds         = 2.5,
                                            .upload_seconds       = 2.0,
                                            .artifact_bytes_read  = 17500000000ULL,
                                            .host_to_device_bytes = 17400000000ULL,
                                            .peak_staging_bytes   = 134217728ULL,
                                            .tensor_count         = 1166,
                                            .resource_count       = 6};
    env.memory.device                    = 0;
    env.memory.max_context               = 4096;
    env.memory.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    env.memory.weights                   = {17400000000ULL, 17400000000ULL, 17400000000ULL};
    env.memory.sequence                  = {2000000000ULL, 1900000000ULL, 1900000000ULL};
    env.memory.workspace                 = {100000000ULL, 0, 0};
    env.memory.kv_payload_bytes          = 123456ULL;
    env.max_context                      = 4096;
    env.prefill_chunk                    = 1024;
    env.kv_cache                         = ninfer::KvCacheStorage::Int8Group64;
    env.mtp_draft_tokens                 = 5;
    env.proposal_head                    = ninfer::ProposalHead::Optimized;
    env.use_cuda_graph                   = true;
    env.decode_graph_primed              = true;
    env.decode_graph_prime_output_tokens = 13;
    env.repetitions                      = 2;
    env.warmup                           = 1;
    env.corpus_path                      = "bench/fixtures/bench_corpus.ids";
    env.corpus_tokens                    = 65536;
    return env;
}

int test_report_contract() {
    int failures                   = 0;
    const qb::BenchEnvironment env = sample_environment();
    const auto results             = sample_results();
    Json report;
    try {
        report = Json::parse(qb::format_json(
            env, "ninfer_bench --weights model.ninfer --mtp-draft-tokens 5", results));
    } catch (const nlohmann::json::exception& error) {
        return fail(std::string("invalid benchmark JSON: ") + error.what());
    }

    failures += expect(report.at("schema_version") == 8, "report schema v8");
    failures += expect(report.at("artifact_type") == "ninfer_bench_report", "report identity");
    failures += expect(report.at("artifact").at("path") == "model.ninfer", "artifact path");
    failures += expect(report.at("load").at("target") == "qwen3_6_27b_rtx5090", "load target");
    failures +=
        expect(report.at("load").at("host_to_device_bytes") == 17400000000ULL, "load H2D bytes");
    failures += expect(report.at("memory").at("kv_cache") == "int8-group64", "memory KV");
    failures += expect(report.at("memory").at("workspace").at("capacity_bytes") == 100000000ULL,
                       "workspace capacity");
    failures += expect(report.at("memory").at("kv_payload_bytes") == 123456ULL, "KV payload");
    failures += expect(report.at("config").at("proposal_head") == "optimized", "proposal head");
    failures += expect(report.at("config").at("decode_graph_prime").at("output_tokens") == 13,
                       "graph prime output count");

    const Json& pp = report.at("tests").at(0);
    failures +=
        expect(pp.at("kind") == "pp" && pp.at("requested_output_tokens") == 1, "pp request shape");
    failures += expect_near(pp.at("prefill_tok_s_mean").get<double>(), 1536.0, "pp throughput");
    failures += expect(pp.at("decode_output_tok_s_mean").is_null(), "pp decode is null");
    failures += expect(pp.at("workspace_peak_bytes") == 5ULL * 1024ULL * 1024ULL * 1024ULL,
                       "pp workspace peak");

    const Json& tg = report.at("tests").at(1);
    failures +=
        expect(tg.at("kind") == "tg" && tg.at("requested_output_tokens") == 4, "tg request shape");
    failures += expect_near(tg.at("decode_output_tok_s_mean").get<double>(), 4.5,
                            "decode output throughput");
    failures += expect_near(tg.at("decode_engine_tok_s_mean").get<double>(), 7.5,
                            "decode engine throughput");
    failures += expect(tg.at("speculative").at("rounds") == 1, "speculative rounds");
    failures += expect(tg.at("speculative").at("fallback_steps") == 3, "speculative fallbacks");
    failures += expect_near(tg.at("speculative").at("acceptance_rate").get<double>(), 1.0,
                            "speculative acceptance");
    failures += expect(tg.at("speculative").at("accepted_per_position").size() == 5,
                       "per-position acceptance");
    failures +=
        expect(tg.at("reps").at(0).at("generated_output_tokens") == 4, "rep generated tokens");
    failures += expect(tg.at("reps").at(0).at("decode_engine_tokens") == 6, "rep engine tokens");
    failures += expect_near(tg.at("reps").at(0).at("timings").at("decode_seconds").get<double>(),
                            0.5, "rep GenerationTimings");
    failures += expect(tg.at("reps").at(0).at("speculative").at("drafted_tokens") == 5,
                       "rep SpeculativeStats");
    return failures;
}

int test_human_and_csv_reports() {
    int failures                   = 0;
    const qb::BenchEnvironment env = sample_environment();
    const auto results             = sample_results();
    const std::string table        = qb::format_table(env, results);
    failures += expect(table.find("qwen3_6_27b_rtx5090") != std::string::npos, "table target");
    failures += expect(table.find("model.ninfer") != std::string::npos, "table artifact");
    failures +=
        expect(table.find("proposal_head=optimized") != std::string::npos, "table proposal head");
    failures +=
        expect(table.find("decode eng t/s") != std::string::npos, "table engine throughput");
    failures += expect(table.find("work peak") != std::string::npos, "table workspace peak");

    const std::string csv = qb::format_csv(env, results);
    failures += expect(csv.starts_with("label,kind,n_prompt,n_gen,target"), "CSV identity columns");
    for (const std::string_view field :
         {"proposal_head", "kv_payload_bytes", "load_host_to_device_bytes", "workspace_peak_bytes",
          "spec_acceptance_rate", "decode_output_tok_s_mean", "decode_engine_tok_s_mean",
          "total_seconds_mean"}) {
        failures += expect(csv.find(field) != std::string::npos,
                           std::string("CSV field ") + std::string(field));
    }
    failures += expect(std::count(csv.begin(), csv.end(), '\n') == 3, "CSV header plus two rows");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_cli_contract();
    failures += test_measurement_contract();
    failures += test_report_contract();
    failures += test_human_and_csv_reports();
    return failures == 0 ? 0 : fail("ninfer_bench support contract failed");
}
