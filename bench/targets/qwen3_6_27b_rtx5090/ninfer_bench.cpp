#include "ninfer_bench_support.h"

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string command_line(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) {
        if (i != 0) { out << ' '; }
        out << argv[i];
    }
    return out.str();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

void fill_cuda_environment(ninfer::bench::BenchEnvironment& env, int device) {
    env.device_id       = device;
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        env.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        env.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        env.gpu_name = properties.name;
    }
}

bool has_decode_tests(const std::vector<ninfer::bench::BenchTest>& tests) {
    for (const auto& test : tests) {
        if (test.has_decode()) { return true; }
    }
    return false;
}

ninfer::RequestOptions benchmark_request(const ninfer::bench::BenchTest& test) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = test.requested_output_tokens();
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    options.output.preserve_special_tokens    = true;
    return options;
}

ninfer::bench::RepTiming run_repetition(ninfer::Engine& engine,
                                        const ninfer::bench::BenchTest& test,
                                        const std::vector<ninfer::TokenId>& corpus) {
    const int prompt_tokens = test.kind == ninfer::bench::TestKind::Decode
                                  ? ninfer::bench::kDecodeSeedTokens
                                  : test.n_prompt;
    auto prompt = engine.prepare_tokens(ninfer::bench::prompt_slice(corpus, prompt_tokens), false);
    ninfer::GenerationResult generated =
        engine.generate(std::move(prompt), benchmark_request(test));

    const std::uint32_t expected = test.requested_output_tokens();
    if (generated.generated_token_ids.size() != expected) {
        throw std::runtime_error(test.label + " generated " +
                                 std::to_string(generated.generated_token_ids.size()) +
                                 " tokens; expected " + std::to_string(expected));
    }
    if (generated.finish_reason != ninfer::FinishReason::OutputLimit) {
        throw std::runtime_error(test.label + " did not finish at the requested output limit");
    }

    ninfer::bench::RepTiming timing;
    timing.timings                 = generated.timings;
    timing.speculative             = std::move(generated.speculative);
    timing.generated_output_tokens = expected;
    return timing;
}

void prime_decode_graph(ninfer::Engine& engine, ninfer::bench::BenchEnvironment& env,
                        const std::vector<ninfer::TokenId>& corpus) {
    if (!env.use_cuda_graph || env.decode_graph_prime_output_tokens == 0) { return; }
    const int decode_tokens = static_cast<int>(env.decode_graph_prime_output_tokens - 1);
    const ninfer::bench::BenchTest prime{ninfer::bench::TestKind::Decode, 0, decode_tokens,
                                         "decode-graph-prime"};
    (void)run_repetition(engine, prime, corpus);
    env.decode_graph_primed = true;
}

void write_output(const ninfer::bench::BenchOptions& options, const std::string& text) {
    if (options.output_file.empty()) {
        std::cout << text;
        return;
    }
    const std::filesystem::path path(options.output_file);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open output file: " + options.output_file); }
    output << text;
    std::cout << "wrote " << options.output_file << '\n';
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::BenchOptions options;
    try {
        options = ninfer::bench::parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 2;
    }
    if (options.help_requested) {
        std::cout << ninfer::bench::usage_text(argc > 0 ? argv[0] : "ninfer_bench");
        return 0;
    }

    try {
        const std::vector<ninfer::TokenId> corpus =
            ninfer::bench::load_corpus_ids(options.corpus_path);
        const std::vector<ninfer::bench::BenchTest> tests = ninfer::bench::expand_tests(options);
        ninfer::bench::validate_prompt_lengths(tests, corpus.size());
        const std::uint32_t max_context = ninfer::bench::resolve_max_context(
            tests, options.max_context, options.mtp_draft_tokens, options.use_cuda_graph);

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path             = options.artifact_path;
        engine_options.device                    = options.device;
        engine_options.max_context               = max_context;
        engine_options.prefill_chunk             = options.prefill_chunk;
        engine_options.kv_cache                  = options.kv_cache;
        engine_options.speculative.draft_tokens  = options.mtp_draft_tokens;
        engine_options.speculative.proposal_head = options.proposal_head;
        engine_options.use_cuda_graph            = options.use_cuda_graph;

        ninfer::bench::BenchEnvironment env;
        env.artifact_path            = options.artifact_path;
        env.artifact_file_size_bytes = ninfer::bench::file_size_or_zero(options.artifact_path);
        env.max_context              = max_context;
        env.prefill_chunk            = options.prefill_chunk;
        env.kv_cache                 = options.kv_cache;
        env.mtp_draft_tokens         = options.mtp_draft_tokens;
        env.proposal_head            = options.proposal_head;
        env.use_cuda_graph           = options.use_cuda_graph;
        env.repetitions              = options.repetitions;
        env.warmup                   = options.warmup;
        env.corpus_path              = options.corpus_path;
        env.corpus_tokens            = corpus.size();
        if (options.use_cuda_graph && has_decode_tests(tests)) {
            env.decode_graph_prime_output_tokens =
                ninfer::bench::decode_graph_prime_output_tokens(options.mtp_draft_tokens);
        }

        std::cerr << "[ninfer_bench] loading " << options.artifact_path
                  << " (max_context=" << max_context
                  << ", kv_cache=" << ninfer::bench::kv_cache_name(options.kv_cache) << ")\n";
        ninfer::Engine engine(std::move(engine_options));
        fill_cuda_environment(env, options.device);
        env.load   = engine.load_summary();
        env.memory = engine.memory_summary();

        prime_decode_graph(engine, env, corpus);

        std::vector<ninfer::bench::TestResult> results;
        results.reserve(tests.size());
        for (std::size_t i = 0; i < tests.size(); ++i) {
            const auto& test = tests[i];
            std::cerr << "[ninfer_bench] test " << (i + 1) << '/' << tests.size() << ' '
                      << test.label << ": warmup=" << options.warmup
                      << " reps=" << options.repetitions << '\n';

            ninfer::bench::TestResult result;
            result.test = test;
            engine.reset_memory_peaks();
            for (int warmup = 0; warmup < options.warmup; ++warmup) {
                (void)run_repetition(engine, test, corpus);
            }
            result.reps.reserve(static_cast<std::size_t>(options.repetitions));
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                result.reps.push_back(run_repetition(engine, test, corpus));
            }
            result.workspace_peak_bytes = engine.memory_summary().workspace.peak_used_bytes;
            results.push_back(std::move(result));
        }

        std::string report;
        switch (options.output) {
        case ninfer::bench::OutputFormat::Table:
            report = ninfer::bench::format_table(env, results);
            break;
        case ninfer::bench::OutputFormat::Json:
            report = ninfer::bench::format_json(env, command_line(argc, argv), results);
            break;
        case ninfer::bench::OutputFormat::Csv:
            report = ninfer::bench::format_csv(env, results);
            break;
        }
        write_output(options, report);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 1;
    }
}
