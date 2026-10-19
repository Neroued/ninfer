#include "options.h"
#include "report.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

using namespace ninfer;
using Json = nlohmann::json;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, const char* message) {
    expect(std::abs(actual - expected) < 1e-8, message);
}

bench::Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& argument : args) argv.push_back(argument.data());
    return bench::parse_options(static_cast<int>(argv.size()), argv.data());
}

void rejects(std::vector<std::string> args) {
    try {
        (void)parse(std::move(args));
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid benchmark arguments");
}

void options_test() {
    expect(parse({"bench", "--help"}).help_requested, "help without model");
    const auto mixed = parse({"bench", "model.ninfer", "--messages", "first.json", "--prompt",
                              "--messages", "--kv-dtype", "k8v4", "--spec", "mtp", "--draft-tokens",
                              "3", "--max-tokens", "64", "--no-cuda-graph", "--gpu-timing"});
    expect(mixed.inputs.size() == 2 && mixed.inputs[0].kind == bench::InputKind::Messages &&
               mixed.inputs[1].value == "--messages",
           "ordered inputs and literal flag-like values");
    expect(mixed.concurrency == 2 && mixed.launch.engine.max_concurrency == 2,
           "inferred concurrency");
    expect(mixed.launch.engine.kv_cache == KvCacheStorage::Fp8KeyNvfp4Value &&
               mixed.launch.engine.speculative.backend == SpeculativeBackend::Mtp &&
               mixed.launch.engine.speculative.draft_tokens == 3 &&
               !mixed.launch.engine.use_cuda_graph &&
               mixed.launch.engine.observability.gpu_timing &&
               mixed.launch.generation.max_tokens == 64,
           "shared engine and generation options");
    const auto copied = parse({"bench", "model.ninfer", "--prompt=x", "--concurrency=4",
                               "--warmup=0", "--repetitions=1", "--profile-measured"});
    expect(copied.inputs.size() == 1 && copied.concurrency == 4 && copied.warmup == 0,
           "one input copied across requests");
    const auto capacity = parse(
        {"bench", "model.ninfer", "--prompt", "x", "--concurrency", "4", "--max-concurrency", "2"});
    expect(capacity.launch.engine.max_concurrency == 2, "explicit engine capacity preserved");
    expect(!capacity.launch.engine.context_cache.enabled, "prefix reuse defaults off");
    expect(bench::usage_text("bench").find("--kv-dtype") != std::string::npos, "shared help");
    rejects({"bench", "model.ninfer"});
    rejects({"bench", "model.ninfer", "--prompt", "a", "--prompt", "b", "--concurrency", "3"});
    rejects({"bench", "model.ninfer", "--prompt", "a", "--repetitions", "0"});
    rejects({"bench", "model.ninfer", "--prompt", "a", "--profile-measured"});
    rejects({"bench", "model.ninfer", "--prompt", "a", "--max-tokens", "0"});
    rejects({"bench", "model.ninfer", "--prompt", "a", "--unknown"});
}

void config_test() {
    const auto directory =
        std::filesystem::path(NINFER_SOURCE_DIR) / ".agent-tmp/ninfer-bench-test";
    std::filesystem::create_directories(directory);
    const auto path = directory / "input.toml";
    {
        std::ofstream file(path);
        file << "[model]\npath = 'model.ninfer'\n[engine]\nmax_context = 2048\n"
                "kv_dtype = 'int8'\nmax_concurrency = 2\n[speculative]\nbackend = 'mtp'\n"
                "draft_tokens = 3\n[generation]\nmax_tokens = 32\n[cli]\nprompt = 'ignored'\n";
    }
    const auto options = parse({"bench", "--config", path.string(), "--prompt", "input",
                                "--kv-dtype", "bf16", "--max-tokens", "48", "--concurrency", "3"});
    expect(options.launch.engine.artifact_path == directory / "model.ninfer",
           "config-relative artifact");
    expect(options.launch.engine.kv_cache == KvCacheStorage::BFloat16 &&
               options.launch.engine.max_context == 2048 &&
               options.launch.engine.max_concurrency == 2 &&
               options.launch.generation.max_tokens == 48,
           "native TOML plus command-line precedence");
    expect(options.inputs.size() == 1 && options.inputs[0].value == "input",
           "CLI section is not bench input");
    std::filesystem::remove_all(directory);
}

bench::Repetition sample(std::uint64_t steps, std::uint64_t ns, std::uint64_t rows) {
    bench::Repetition rep;
    rep.wall_seconds      = ns * 1e-9;
    const auto decode     = static_cast<std::size_t>(ExecutionKind::Decode);
    auto& before          = rep.before.executions[decode];
    before.units          = 100;
    before.rows           = 200;
    before.exposure.total = {100, 10000000000, 1000000000};
    auto& after           = rep.after.executions[decode];
    after                 = before;
    after.units += steps;
    after.rows += rows;
    after.exposure.total.count += steps;
    after.exposure.total.total_ns += ns;
    rep.before.runtime.committed_decode_tokens = 200;
    rep.after.runtime.committed_decode_tokens  = 200 + rows;
    // Both requests saw the full batch duration. Summing them would double the measurement.
    rep.requests.resize(2);
    for (auto& request : rep.requests) {
        request.prompt.prompt_tokens          = 128;
        request.stats.computed_prefill_tokens = 64;
        request.stats.reused_prompt_tokens    = 64;
        request.generated_token_ids.resize(4);
        request.finish_reason                     = FinishReason::StopToken;
        request.stats.executions[decode].total    = {steps, ns, ns / steps};
        request.stats.speculative.drafted_tokens  = 10;
        request.stats.speculative.accepted_tokens = 6;
    }
    return rep;
}

void report_test() {
    std::vector<bench::Repetition> repetitions{sample(2, 20000000, 4), sample(3, 60000000, 5)};
    const auto summary = bench::summarize(repetitions);
    const auto& decode = summary["executions"]["decode"];
    expect(decode["units"] == 5 && decode["rows"] == 9, "actual global steps and rows");
    near(decode["total"]["mean_ms"], 16,
         "step-weighted mean, not request exposure or repetition mean");
    near(decode["total"]["max_ms"], 20, "window maximum excludes warmup/lifetime maximum");
    near(decode["mean_batch"], 1.8, "shrinking batch is observable");
    expect(summary["computed_prefill_tokens"] == 256 && summary["reused_prompt_tokens"] == 256,
           "cache counters");
    near(summary["speculative"]["acceptance"], 0.6, "token-weighted acceptance");
    expect(summary["executions"]["prefill"]["total"]["mean_ms"].is_null(),
           "missing timing is null");
    std::vector<bench::Repetition> prefill_only(1);
    expect(bench::summarize(prefill_only)["executions"]["decode"]["total"]["mean_ms"].is_null(),
           "prefill-only has no decode latency");

    auto options = parse({"bench", "model.ninfer", "--prompt", "input", "--concurrency", "2"});
    const std::uint64_t seeds[] = {0, 1};
    const auto report           = bench::make_report(options, {}, {}, {}, seeds, repetitions);
    expect(report["inputs"][1]["seed"] == 1, "resolved per-row seeds");
    expect(report["repetitions"][0]["requests"][0]["finish_reason"] == "stop_token",
           "early EOS retained");
    expect(Json::parse(bench::format_report(report, bench::OutputFormat::Json)) == report,
           "JSON round trip");
    expect(bench::format_report(report, bench::OutputFormat::Table).find("16.000") !=
               std::string::npos,
           "table main metric");
    expect(bench::format_report(report, bench::OutputFormat::Csv).find("decode_mean_ms") !=
               std::string::npos,
           "CSV main metric");
}

} // namespace

int main() {
    try {
        options_test();
        config_test();
        report_test();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
