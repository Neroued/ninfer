#include "ninfer_bench_support.h"

#include "ninfer/core/nvtx_range.h"
#include "ninfer/runtime/engine.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

ninfer::bench::BenchMtpStats to_bench_mtp_stats(const ninfer::EngineMtpStats& stats) {
    ninfer::bench::BenchMtpStats out;
    out.enabled         = stats.enabled;
    out.k               = stats.k;
    out.draft_tokens    = stats.draft_tokens;
    out.accepted_tokens = stats.accepted_tokens;
    out.rounds          = stats.rounds;
    out.fallback_steps  = stats.fallback_steps;
    for (std::size_t i = 0; i < out.accepted_per_pos.size(); ++i) {
        out.accepted_per_pos[i] = stats.accepted_per_pos[i];
    }
    return out;
}

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
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
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
    for (const ninfer::bench::BenchTest& test : tests) {
        if (test.has_decode()) { return true; }
    }
    return false;
}

void prime_decode_graphs(ninfer::Engine& engine, ninfer::bench::BenchEnvironment& env,
                         const std::vector<int>& corpus) {
    if (!env.decode_graph_requested) { return; }
    const std::vector<int> seed = ninfer::bench::prompt_slice(corpus, ninfer::bench::kDecodeSeedTokens);
    engine.prefill(seed);
    const int steps = ninfer::bench::decode_graph_prime_steps(env.mtp_draft_tokens);
    for (int step = 0; step < steps; ++step) { engine.decode_step(); }
    engine.reset_mtp_stats();
    env.decode_graph_primed      = true;
    env.decode_graph_prime_steps = steps;
}

// Run one repetition of a test, returning its timing. Prefill resets KV/state, so repetitions
// are independent. Decode always runs exactly n_gen steps (decode_step ignores stop tokens).
ninfer::bench::RepTiming run_repetition(ninfer::Engine& engine, const ninfer::bench::BenchTest& test,
                                     const std::vector<int>& corpus) {
    ninfer::bench::RepTiming timing;
    switch (test.kind) {
    case ninfer::bench::TestKind::Prefill: {
        const std::vector<int> slice = ninfer::bench::prompt_slice(corpus, test.n_prompt);
        engine.reset_mtp_stats();
        const auto start = Clock::now();
        const ninfer::NvtxRange range("ninfer_bench.prefill." + test.label);
        engine.prefill(slice);
        timing.prefill_time_s = seconds_between(start, Clock::now());
        timing.mtp            = to_bench_mtp_stats(engine.mtp_stats());
        break;
    }
    case ninfer::bench::TestKind::Decode: {
        const std::vector<int> seed =
            ninfer::bench::prompt_slice(corpus, ninfer::bench::kDecodeSeedTokens);
        engine.prefill(seed); // untimed setup
        engine.reset_mtp_stats();
        const auto start = Clock::now();
        const ninfer::NvtxRange range("ninfer_bench.decode." + test.label);
        for (int step = 0; step < test.n_gen; ++step) { engine.decode_step(); }
        timing.decode_time_s = seconds_between(start, Clock::now());
        timing.mtp           = to_bench_mtp_stats(engine.mtp_stats());
        break;
    }
    case ninfer::bench::TestKind::PrefillDecode: {
        const std::vector<int> slice = ninfer::bench::prompt_slice(corpus, test.n_prompt);
        engine.reset_mtp_stats();
        const auto start = Clock::now();
        {
            const ninfer::NvtxRange range("ninfer_bench.prefill." + test.label);
            engine.prefill(slice);
        }
        const auto after_prefill = Clock::now();
        {
            const ninfer::NvtxRange range("ninfer_bench.decode." + test.label);
            for (int step = 0; step < test.n_gen; ++step) { engine.decode_step(); }
        }
        const auto end        = Clock::now();
        timing.prefill_time_s = seconds_between(start, after_prefill);
        timing.decode_time_s  = seconds_between(after_prefill, end);
        timing.mtp            = to_bench_mtp_stats(engine.mtp_stats());
        break;
    }
    }
    return timing;
}

void write_output(const ninfer::bench::BenchOptions& options, const std::string& text) {
    if (options.output_file.empty()) {
        std::cout << text;
        return;
    }
    const std::filesystem::path path(options.output_file);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open output file: " + options.output_file); }
    out << text;
    std::cout << "wrote " << options.output_file << '\n';
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::BenchOptions options;
    try {
        options = ninfer::bench::parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "ninfer_bench: " << e.what() << '\n';
        return 2;
    }
    if (options.help_requested) {
        std::cout << ninfer::bench::usage_text(argc > 0 ? argv[0] : "ninfer_bench");
        return 0;
    }

    try {
        const std::vector<int> corpus = ninfer::bench::load_corpus_ids(options.corpus_path);
        const std::vector<ninfer::bench::BenchTest> tests = ninfer::bench::expand_tests(options);
        ninfer::bench::validate_prompt_lengths(tests, corpus.size());
        const bool needs_decode     = has_decode_tests(tests);
        const std::uint32_t max_ctx = ninfer::bench::resolve_max_ctx(
            tests, options.max_ctx, options.mtp_draft_tokens, options.use_cuda_graph);

        ninfer::EngineOptions engine_options;
        engine_options.device            = options.device;
        engine_options.max_ctx           = max_ctx;
        engine_options.prefill_chunk     = options.prefill_chunk;
        engine_options.kv_dtype          = options.kv_dtype;
        engine_options.mtp_draft_tokens  = options.mtp_draft_tokens;
        engine_options.use_cuda_graph    = options.use_cuda_graph;
        engine_options.use_lm_head_draft = options.use_lm_head_draft;
        if (options.work_bytes.has_value()) { engine_options.work_bytes = *options.work_bytes; }

        ninfer::bench::BenchEnvironment env;
        env.git_commit              = ninfer::bench::current_git_commit_or_empty();
        env.worktree_dirty          = ninfer::bench::current_git_worktree_dirty();
        env.weights_path            = options.weights_path;
        env.weights_file_size_bytes = ninfer::bench::file_size_or_zero(options.weights_path);
        env.max_ctx                 = max_ctx;
        env.prefill_chunk           = options.prefill_chunk;
        env.kv_dtype                = ninfer::bench::kv_dtype_name(options.kv_dtype);
        env.kv_quant_group          = options.kv_dtype == ninfer::DType::I8 ? ninfer::kKvQuantGroup : 0;
        env.mtp_draft_tokens        = options.mtp_draft_tokens;
        env.use_lm_head_draft       = options.use_lm_head_draft;
        env.decode_path =
            ninfer::bench::decode_path_name(options.use_cuda_graph, options.mtp_draft_tokens);
        env.decode_graph_requested = options.use_cuda_graph && needs_decode;
        env.repetitions            = options.repetitions;
        env.warmup                 = options.warmup;
        env.corpus_path            = options.corpus_path;
        env.corpus_tokens          = corpus.size();

        std::cerr << "[ninfer_bench] loading weights " << options.weights_path
                  << " (max_ctx=" << max_ctx
                  << ", kv_dtype=" << ninfer::bench::kv_dtype_name(options.kv_dtype) << ")\n";
        ninfer::Engine engine(engine_options);
        engine.load(options.weights_path);
        fill_cuda_environment(env, options.device);
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) {
            throw std::runtime_error(std::string("cuda sync after load: ") +
                                     cudaGetErrorString(status));
        }
        const ninfer::EngineMemoryStats memory   = engine.memory_stats();
        env.work_bytes                        = memory.workspace.capacity_bytes;
        env.kv_dtype                          = ninfer::bench::kv_dtype_name(memory.kv_dtype);
        env.kv_quant_group                    = memory.kv_quant_group;
        env.kv_cache_payload_bytes            = memory.kv_cache_payload_bytes;
        const ninfer::Q5090LoadStats& load_stats = engine.q5090_load_stats();
        env.q5090_h2d_bytes                   = load_stats.h2d_bytes;
        env.q5090_resident_bytes              = load_stats.device_resident_bytes;
        constexpr std::array<std::string_view, 4> module_names = {
            "TEXT_CORE", "MTP_DRAFT", "VISION_ENCODER", "LM_HEAD_DRAFT"};
        for (std::size_t i = 0; i < load_stats.modules.size(); ++i) {
            if (load_stats.modules[i].selected) {
                env.q5090_resident_modules.emplace_back(module_names[i]);
            }
        }
        prime_decode_graphs(engine, env, corpus);

        std::vector<ninfer::bench::TestResult> results;
        results.reserve(tests.size());
        for (std::size_t i = 0; i < tests.size(); ++i) {
            const ninfer::bench::BenchTest& test = tests[i];
            std::cerr << "[ninfer_bench] test " << (i + 1) << '/' << tests.size() << ' ' << test.label
                      << ": warmup=" << options.warmup << " reps=" << options.repetitions << '\n';
            ninfer::bench::TestResult result;
            result.test = test;
            engine.reset_memory_peaks();
            for (int w = 0; w < options.warmup; ++w) { run_repetition(engine, test, corpus); }
            for (int m = 0; m < options.repetitions; ++m) {
                result.reps.push_back(run_repetition(engine, test, corpus));
            }
            result.workspace_peak_bytes = engine.memory_stats().workspace.peak_used_bytes;
            results.push_back(std::move(result));
        }

        std::string text;
        switch (options.output) {
        case ninfer::bench::OutputFormat::Table:
            text = ninfer::bench::format_table(env, results);
            break;
        case ninfer::bench::OutputFormat::Json:
            text = ninfer::bench::format_json(env, command_line(argc, argv), results);
            break;
        case ninfer::bench::OutputFormat::Csv:
            text = ninfer::bench::format_csv(env, results);
            break;
        }
        write_output(options, text);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ninfer_bench: " << e.what() << '\n';
        return 1;
    }
}
