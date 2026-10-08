#include "ninfer_bench_support.h"

#include "ninfer/engine.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool has_decode_tests(const std::vector<ninfer::bench::BenchTest>& tests) {
    for (const auto& test : tests) {
        if (test.has_decode()) { return true; }
    }
    return false;
}

ninfer::RequestOptions
benchmark_request(const ninfer::bench::BenchTest& test,
                  const std::optional<ninfer::OutputConstraint>& constraint) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = test.requested_output_tokens();
    options.execution.allow_prefix_reuse      = false;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    options.output.preserve_special_tokens    = true;
    if (constraint) {
        options.constraint                     = constraint;
        options.stop.include_model_defaults    = true;
        options.output.raw                     = false;
        options.output.preserve_special_tokens = false;
    }
    return options;
}

void run_repetition(ninfer::Engine& engine, const ninfer::bench::BenchEnvironment& env,
                    const ninfer::bench::BenchTest& test,
                    const std::vector<ninfer::TokenId>& corpus,
                    ninfer::bench::TestResult* measured = nullptr) {
    const int prompt_tokens = test.kind == ninfer::bench::TestKind::Decode
                                  ? ninfer::bench::kDecodeSeedTokens
                                  : test.n_prompt;
    std::vector<ninfer::PreparedPrompt> prompts;
    std::vector<ninfer::GenerationHandle> handles;
    std::vector<ninfer::GenerationResult> generated;
    prompts.reserve(env.concurrency);
    handles.reserve(env.concurrency);
    generated.reserve(env.concurrency);
    for (std::uint32_t row = 0; row < env.concurrency; ++row) {
        prompts.push_back(
            engine.prepare_tokens(ninfer::bench::prompt_slice(corpus, prompt_tokens), false));
    }
    const auto started = std::chrono::steady_clock::now();
    for (std::uint32_t row = 0; row < env.concurrency; ++row) {
        const bool constrained =
            env.constraint.has_value() && (!env.mixed_constraints || row % 2 == 0);
        handles.push_back(
            engine.submit(std::move(prompts[row]),
                          benchmark_request(test, constrained ? env.constraint : std::nullopt),
                          ninfer::OutputConsumerMode::Aggregate, {.phase_timings = true}));
    }
    for (auto& handle : handles) { generated.push_back(handle.wait()); }
    const double wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    for (std::uint32_t row = 0; row < env.concurrency; ++row) {
        auto& result           = generated[row];
        const bool constrained =
            env.constraint.has_value() && (!env.mixed_constraints || row % 2 == 0);
        const auto count       = static_cast<std::uint32_t>(result.generated_token_ids.size());
        const bool at_limit    = result.finish_reason == ninfer::FinishReason::OutputLimit &&
                              count == test.requested_output_tokens();
        const bool grammar_done = constrained &&
                                  result.finish_reason == ninfer::FinishReason::StopToken &&
                                  count <= test.requested_output_tokens();
        if (!at_limit && !grammar_done) {
            throw std::runtime_error(test.label +
                                     " did not finish at its output limit or grammar EOS");
        }
        if (measured) {
            measured->reps.push_back({result.timings, std::move(result.speculative), count,
                                      result.constraint,
                                      result.engine_timing.constraint_draft_wait_exposed_seconds});
        }
    }
    if (measured) { measured->repetition_wall_seconds.push_back(wall_seconds); }
}

void prime_decode_graph(ninfer::Engine& engine, ninfer::bench::BenchEnvironment& env,
                        const std::vector<ninfer::TokenId>& corpus) {
    if (!env.use_cuda_graph || env.decode_graph_prime_output_tokens == 0) { return; }
    const int decode_tokens = static_cast<int>(env.decode_graph_prime_output_tokens - 1);
    const ninfer::bench::BenchTest prime{ninfer::bench::TestKind::Decode, 0, decode_tokens,
                                         "decode-graph-prime"};
    run_repetition(engine, env, prime, corpus);
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
        if (options.profile_measured && (tests.size() != 1 || options.repetitions != 1)) {
            throw std::invalid_argument(
                "--profile-measured requires exactly one benchmark test and -r 1");
        }
        ninfer::bench::validate_prompt_lengths(tests, corpus.size());
        const std::uint32_t max_context = ninfer::bench::resolve_max_context(
            tests, options.max_context, options.speculative, options.use_cuda_graph);

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path   = options.artifact_path;
        engine_options.device          = options.device;
        engine_options.max_context     = max_context;
        engine_options.max_concurrency = options.concurrency;
        const std::uint64_t kv_capacity =
            ((static_cast<std::uint64_t>(max_context) + 63U) / 64U) * 64U * options.concurrency;
        if (kv_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("concurrent benchmark KV capacity exceeds uint32");
        }
        engine_options.kv_capacity =
            ninfer::KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(kv_capacity));
        engine_options.prefill_chunk                     = options.prefill_chunk;
        engine_options.kv_cache                          = options.kv_cache;
        engine_options.context_cache.enabled             = false;
        engine_options.context_cache.device_state_slots  = 0;
        engine_options.context_cache.host_capacity_bytes = 0;
        engine_options.speculative                       = options.speculative;
        engine_options.use_cuda_graph                    = options.use_cuda_graph;

        ninfer::bench::BenchEnvironment env;
        env.artifact_path            = options.artifact_path;
        env.artifact_file_size_bytes = ninfer::bench::file_size_or_zero(options.artifact_path);
        env.max_context              = max_context;
        env.prefill_chunk            = options.prefill_chunk;
        env.kv_cache                 = options.kv_cache;
        env.speculative              = options.speculative;
        env.use_cuda_graph           = options.use_cuda_graph;
        env.repetitions              = options.repetitions;
        env.warmup                   = options.warmup;
        env.corpus_path              = options.corpus_path;
        env.corpus_tokens            = corpus.size();
        env.concurrency              = options.concurrency;
        env.constraint_file          = options.constraint_file;
        env.constraint               = options.constraint;
        env.mixed_constraints        = options.mixed_constraints;
        if (env.constraint && (env.constraint->kind == ninfer::OutputConstraintKind::Grammar ||
                               env.constraint->kind == ninfer::OutputConstraintKind::JsonSchema)) {
            std::ifstream input(options.constraint_file, std::ios::binary);
            if (!input)
                throw std::runtime_error("cannot read constraint: " + options.constraint_file);
            env.constraint->source.assign(std::istreambuf_iterator<char>(input), {});
            if (input.bad()) throw std::runtime_error("failed to read constraint file");
        }
        if (options.use_cuda_graph && has_decode_tests(tests)) {
            env.decode_graph_prime_output_tokens =
                ninfer::bench::decode_graph_prime_output_tokens(options.speculative);
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
            result.test        = test;
            result.concurrency = env.concurrency;
            engine.reset_memory_peaks();
            for (int warmup = 0; warmup < options.warmup; ++warmup) {
                run_repetition(engine, env, test, corpus);
            }
            result.reps.reserve(static_cast<std::size_t>(options.repetitions) * env.concurrency);
            if (options.profile_measured) {
                require_cuda(cudaDeviceSynchronize(), "profile pre-boundary synchronize");
                require_cuda(cudaProfilerStart(), "cudaProfilerStart");
            }
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                run_repetition(engine, env, test, corpus, &result);
            }
            if (options.profile_measured) {
                require_cuda(cudaDeviceSynchronize(), "profile post-boundary synchronize");
                require_cuda(cudaProfilerStop(), "cudaProfilerStop");
            }
            const ninfer::MemorySummary memory    = engine.memory_summary();
            result.workspace_peak_bytes           = memory.workspace_logical_peak_bytes;
            result.workspace_allocator_peak_bytes = memory.workspace.peak_used_bytes;
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
