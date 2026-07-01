#include "qus_bench_support.h"

#include "qus/core/nvtx_range.h"
#include "qus/runtime/engine.h"

#include <cuda_runtime.h>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
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

void fill_cuda_environment(qus::bench::BenchEnvironment& env, int device) {
    env.device_id = device;
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

// Run one repetition of a test, returning its timing. Prefill resets KV/state, so repetitions
// are independent. Decode always runs exactly n_gen steps (decode_step ignores stop tokens).
qus::bench::RepTiming run_repetition(qus::Engine& engine, const qus::bench::BenchTest& test,
                                     const std::vector<int>& corpus) {
    qus::bench::RepTiming timing;
    switch (test.kind) {
    case qus::bench::TestKind::Prefill: {
        const std::vector<int> slice = qus::bench::prompt_slice(corpus, test.n_prompt);
        const auto start = Clock::now();
        const qus::NvtxRange range("qus_bench.prefill." + test.label);
        engine.prefill(slice);
        timing.prefill_time_s = seconds_between(start, Clock::now());
        break;
    }
    case qus::bench::TestKind::Decode: {
        const std::vector<int> seed = qus::bench::prompt_slice(corpus, qus::bench::kDecodeSeedTokens);
        engine.prefill(seed); // untimed setup
        const auto start = Clock::now();
        const qus::NvtxRange range("qus_bench.decode." + test.label);
        for (int step = 0; step < test.n_gen; ++step) { engine.decode_step(); }
        timing.decode_time_s = seconds_between(start, Clock::now());
        break;
    }
    case qus::bench::TestKind::PrefillDecode: {
        const std::vector<int> slice = qus::bench::prompt_slice(corpus, test.n_prompt);
        const auto start = Clock::now();
        {
            const qus::NvtxRange range("qus_bench.prefill." + test.label);
            engine.prefill(slice);
        }
        const auto after_prefill = Clock::now();
        {
            const qus::NvtxRange range("qus_bench.decode." + test.label);
            for (int step = 0; step < test.n_gen; ++step) { engine.decode_step(); }
        }
        const auto end = Clock::now();
        timing.prefill_time_s = seconds_between(start, after_prefill);
        timing.decode_time_s = seconds_between(after_prefill, end);
        break;
    }
    }
    return timing;
}

void write_output(const qus::bench::BenchOptions& options, const std::string& text) {
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
    qus::bench::BenchOptions options;
    try {
        options = qus::bench::parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "qus_bench: " << e.what() << '\n';
        return 2;
    }
    if (options.help_requested) {
        std::cout << qus::bench::usage_text(argc > 0 ? argv[0] : "qus_bench");
        return 0;
    }

    try {
        const std::vector<int> corpus = qus::bench::load_corpus_ids(options.corpus_path);
        const std::vector<qus::bench::BenchTest> tests = qus::bench::expand_tests(options);
        qus::bench::validate_prompt_lengths(tests, corpus.size());
        const std::uint32_t max_ctx = qus::bench::resolve_max_ctx(tests, options.max_ctx);

        qus::EngineOptions engine_options;
        engine_options.device = options.device;
        engine_options.max_ctx = max_ctx;
        engine_options.use_cuda_graph = options.use_cuda_graph;
        if (options.work_bytes.has_value()) { engine_options.work_bytes = *options.work_bytes; }

        qus::bench::BenchEnvironment env;
        fill_cuda_environment(env, options.device);
        env.git_commit = qus::bench::current_git_commit_or_empty();
        env.worktree_dirty = qus::bench::current_git_worktree_dirty();
        env.weights_path = options.weights_path;
        env.weights_file_size_bytes = qus::bench::file_size_or_zero(options.weights_path);
        env.max_ctx = max_ctx;
        env.work_bytes = engine_options.work_bytes;
        env.decode_path = options.use_cuda_graph ? "cuda_graph" : "eager";
        env.repetitions = options.repetitions;
        env.warmup = options.warmup;
        env.corpus_path = options.corpus_path;
        env.corpus_tokens = corpus.size();

        std::cerr << "[qus_bench] loading weights " << options.weights_path << " (max_ctx="
                  << max_ctx << ")\n";
        qus::Engine engine(engine_options);
        engine.load(options.weights_path);
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) {
            throw std::runtime_error(std::string("cuda sync after load: ") +
                                     cudaGetErrorString(status));
        }

        std::vector<qus::bench::TestResult> results;
        results.reserve(tests.size());
        for (std::size_t i = 0; i < tests.size(); ++i) {
            const qus::bench::BenchTest& test = tests[i];
            std::cerr << "[qus_bench] test " << (i + 1) << '/' << tests.size() << ' ' << test.label
                      << ": warmup=" << options.warmup << " reps=" << options.repetitions << '\n';
            qus::bench::TestResult result;
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
        case qus::bench::OutputFormat::Table:
            text = qus::bench::format_table(env, results);
            break;
        case qus::bench::OutputFormat::Json:
            text = qus::bench::format_json(env, command_line(argc, argv), results);
            break;
        case qus::bench::OutputFormat::Csv:
            text = qus::bench::format_csv(results);
            break;
        }
        write_output(options, text);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "qus_bench: " << e.what() << '\n';
        return 1;
    }
}
