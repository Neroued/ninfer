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
    expect(capacity.launch.engine.context_cache.enabled &&
               !capacity.launch.engine.context_cache.extra_state_slots,
           "decode defaults to native prefix reuse and extra state capacity");
    const auto prefill = parse({"bench", "model.ninfer", "--prompt", "x", "--max-tokens", "1"});
    expect(!prefill.launch.engine.context_cache.enabled &&
               prefill.launch.engine.context_cache.extra_state_slots == 0,
           "prefill-only retains its uncached defaults");
    const auto configured = parse({"bench", "model.ninfer", "--prompt", "x", "--no-prefix-reuse",
                                   "--extra-state-slots", "0"});
    expect(!configured.launch.engine.context_cache.enabled &&
               configured.launch.engine.context_cache.extra_state_slots == 0,
           "explicit cache settings are preserved");
    const auto defaults = parse({"bench", "model.ninfer", "--prompt", "x", "--prefix-reuse=default",
                                 "--extra-state-slots", "default"});
    expect(defaults.launch.engine.context_cache.enabled &&
               !defaults.launch.engine.context_cache.extra_state_slots,
           "explicit default uses decode defaults");
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

bench::Repetition sample(std::initializer_list<std::pair<std::uint32_t, std::uint64_t>> steps) {
    bench::Repetition rep;
    const auto decode     = static_cast<std::size_t>(ExecutionKind::Decode);
    auto& before          = rep.before.executions[decode];
    before.units          = 100;
    before.rows           = 200;
    before.exposure.total = {100, 10000000000, 1000000000};
    auto& after           = rep.after.executions[decode];
    after                 = before;
    rep.requests.resize(4);
    for (auto& request : rep.requests) {
        request.prompt.prompt_tokens          = 128;
        request.stats.computed_prefill_tokens = 64;
        request.stats.reused_prompt_tokens    = 64;
        request.generated_token_ids.resize(4);
        request.finish_reason = FinishReason::StopToken;
        request.stats.decode_by_batch.resize(4);
        request.stats.speculative.drafted_tokens  = 10;
        request.stats.speculative.accepted_tokens = 6;
    }
    for (const auto& [batch, ns] : steps) {
        const ExecutionLatency latency{.result_ns     = ns * 3 / 4,
                                       .resolution_ns = ns / 8,
                                       .retirement_ns = ns / 8,
                                       .total_ns      = ns};
        ++after.units;
        after.rows += batch;
        after.exposure.observe(latency);
        rep.wall_seconds += ns * 1e-9;
        for (std::uint32_t row = 0; row < batch; ++row) {
            rep.requests[row].stats.executions[decode].observe(latency);
            rep.requests[row].stats.decode_by_batch[batch - 1].observe(latency);
        }
    }
    return rep;
}

void report_test() {
    std::vector<bench::Repetition> repetitions{
        sample({{2, 8000000}, {2, 12000000}, {1, 6000000}}),
        sample({{2, 20000000}, {1, 4000000}, {1, 10000000}})};
    const auto summary = bench::summarize(repetitions, 4);
    const auto& decode = summary["executions"]["decode"];
    expect(decode["units"] == 6 && decode["rows"] == 9, "actual global steps and rows");
    near(decode["total"]["mean_ms"], 10, "global step mean");
    near(decode["total"]["max_ms"], 20, "window maximum excludes warmup/lifetime maximum");
    expect(summary["computed_prefill_tokens"] == 512, "computed prefill tokens");
    const auto& batches = summary["decode_by_batch"];
    expect(batches.size() == 3 && batches[0]["batch"] == 1 && batches[1]["batch"] == 2,
           "only observed batches and requested target are reported");
    expect(batches[0]["steps"] == 3 && batches[1]["steps"] == 3, "shared steps counted once");
    near(batches[0]["total"]["mean_ms"], 20.0 / 3, "B1 mean");
    near(batches[1]["total"]["mean_ms"], 40.0 / 3, "B2 step-weighted mean across repetitions");
    near(batches[1]["result"]["mean_ms"], 10, "grouped result interval");
    near(batches[1]["total"]["max_ms"], 20, "maximum is not divided by batch");
    expect(batches[2]["batch"] == 4 && batches[2]["target"] == true && batches[2]["steps"] == 0 &&
               batches[2]["total"]["mean_ms"].is_null(),
           "missing target batch has no latency sample");
    near(summary["speculative"]["acceptance"], 0.6, "token-weighted acceptance");
    expect(summary["executions"]["prefill"]["total"]["mean_ms"].is_null(),
           "missing timing is null");
    std::vector<bench::Repetition> prefill_only(1);
    expect(bench::summarize(prefill_only, 1)["decode_by_batch"][0]["total"]["mean_ms"].is_null(),
           "prefill-only has no decode latency");

    auto options = parse({"bench", "model.ninfer", "--prompt", "input", "--concurrency", "4"});
    const std::uint64_t seeds[] = {0, 1, 2, 3};
    const auto report           = bench::make_report(options, {}, {}, {}, seeds, repetitions);
    expect(report["inputs"][1]["seed"] == 1, "resolved per-row seeds");
    expect(report["repetitions"][0]["requests"][0]["finish_reason"] == "stop_token",
           "early EOS retained");
    expect(Json::parse(bench::format_report(report, bench::OutputFormat::Json)) == report,
           "JSON round trip");
    expect(bench::format_report(report, bench::OutputFormat::Table).find("13.333") !=
               std::string::npos,
           "table main metric");
    expect(bench::format_report(report, bench::OutputFormat::Table).find("no samples") !=
               std::string::npos,
           "missing target is shown in table");
    expect(bench::format_report(report, bench::OutputFormat::Csv).find("all,decode,2,false,3,") !=
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
