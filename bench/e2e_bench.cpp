#include "e2e_bench_support.h"
#include "qus/runtime/engine.h"

#include <cuda_runtime.h>

#include <chrono>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start, Clock::time_point end) {
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

std::optional<std::string> output_json_arg(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "--output-json") { return std::string(argv[i + 1]); }
    }
    return std::nullopt;
}

void write_error_if_possible(const std::optional<std::string>& path, std::string_view phase,
                             std::string_view message) {
    if (path.has_value()) { qus::bench::e2e::write_error_report(*path, phase, message); }
}

std::string stop_reason_for(bool eos, int generated, int requested) {
    if (eos) { return "eos_token"; }
    if (generated >= requested) { return "max_new_tokens"; }
    return "unknown";
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

qus::bench::e2e::EnvironmentReport collect_environment(int device) {
    qus::bench::e2e::EnvironmentReport environment;
    environment.device_id = device;
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        environment.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name = properties.name;
    }
    return environment;
}

void synchronize_cuda() {
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

bool generated_ids_equal(const std::vector<int>& a, const std::vector<int>& b) { return a == b; }

} // namespace

int main(int argc, char** argv) {
    const std::optional<std::string> output_json_path = output_json_arg(argc, argv);
    qus::bench::e2e::RunOptions options;
    try {
        options = qus::bench::e2e::parse_args(argc, argv);
    } catch (const std::exception& e) {
        write_error_if_possible(output_json_path, "parse", e.what());
        std::cerr << "qus_e2e_bench: " << e.what() << '\n';
        return 2;
    }

    qus::EngineOptions engine_options;
    engine_options.device = options.device;
    engine_options.max_ctx = options.max_ctx;
    engine_options.eos_token_id = options.eos_token_id;

    std::vector<qus::bench::e2e::CaseRunInput> case_inputs;
    try {
        case_inputs.reserve(options.cases.size());
        for (const qus::bench::e2e::CaseSpec& spec : options.cases) {
            qus::bench::e2e::CaseRunInput input;
            input.name = spec.name;
            input.prompt_ids_path = spec.prompt_ids_path;
            input.prompt_ids = qus::bench::e2e::parse_ids_file(spec.prompt_ids_path);
            input.requested_max_new_tokens = spec.max_new_tokens;
            input.eos_token_id = options.eos_token_id;
            input.max_context = options.max_ctx;
            qus::bench::e2e::validate_case_context(input);
            case_inputs.push_back(std::move(input));
        }
    } catch (const std::exception& e) {
        qus::bench::e2e::write_error_report(options.output_json_path, "preflight", e.what());
        std::cerr << "qus_e2e_bench: " << e.what() << '\n';
        return 1;
    }

    qus::Engine engine(engine_options);
    double load_time_s = 0.0;
    try {
        const auto load_start = Clock::now();
        engine.load(options.weights_path);
        synchronize_cuda();
        const auto load_end = Clock::now();
        load_time_s = seconds_since(load_start, load_end);
    } catch (const std::exception& e) {
        qus::bench::e2e::write_error_report(options.output_json_path, "load", e.what());
        std::cerr << "qus_e2e_bench: " << e.what() << '\n';
        return 1;
    }

    try {
        qus::bench::e2e::RawReport report;
        report.command = command_line(argc, argv);
        report.git_commit = qus::bench::e2e::current_git_commit_or_empty();
        report.worktree_dirty = qus::bench::e2e::current_git_worktree_dirty();
        report.load_time_s = load_time_s;
        report.environment = collect_environment(options.device);
        report.q5090_path = options.weights_path;
        report.q5090_file_size_bytes = qus::bench::e2e::file_size_or_zero(options.weights_path);
        report.q5090_sha256 = qus::bench::e2e::sha256_file_or_empty(options.weights_path);
        report.max_context = options.max_ctx;
        report.post_load_memory = engine.memory_stats();

        for (const qus::bench::e2e::CaseRunInput& input : case_inputs) {
            qus::bench::e2e::CaseReport case_report;
            case_report.input = input;
            case_report.prompt_ids_sha256 =
                qus::bench::e2e::sha256_file_or_empty(input.prompt_ids_path);
            case_report.fixture_manifest_sha256 =
                qus::bench::e2e::sha256_file_or_empty(case_report.fixture_manifest_path);
            case_report.warmup_repeats = options.warmup_repeats;
            case_report.measured_repeats = options.repeats;

            auto run_one_repeat = [&](int repeat_index, bool measured) {
                engine.reset_memory_peaks();
                qus::bench::e2e::RepeatReport repeat;
                repeat.repeat_index = repeat_index;
                repeat.prompt_tokens = input.prompt_ids.size();

                const auto prefill_start = Clock::now();
                int token = engine.prefill(input.prompt_ids);
                const auto prefill_end = Clock::now();
                repeat.prefill_time_s = seconds_since(prefill_start, prefill_end);
                repeat.generated_token_ids.push_back(token);

                bool eos = options.eos_token_id >= 0 && token == options.eos_token_id;
                const auto decode_start = Clock::now();
                while (!eos &&
                       static_cast<int>(repeat.generated_token_ids.size()) <
                           input.requested_max_new_tokens) {
                    token = engine.decode_step();
                    repeat.generated_token_ids.push_back(token);
                    ++repeat.decode_loop_tokens;
                    eos = options.eos_token_id >= 0 && token == options.eos_token_id;
                }
                const auto decode_end = Clock::now();
                repeat.decode_time_s =
                    repeat.decode_loop_tokens == 0 ? 0.0 : seconds_since(decode_start, decode_end);
                repeat.stop_reason = stop_reason_for(
                    eos, static_cast<int>(repeat.generated_token_ids.size()),
                    input.requested_max_new_tokens);
                repeat.memory = engine.memory_stats();

                if (measured) { case_report.repeats.push_back(std::move(repeat)); }
            };

            for (int i = 0; i < options.warmup_repeats; ++i) { run_one_repeat(i, false); }
            for (int i = 0; i < options.repeats; ++i) { run_one_repeat(i, true); }

            if (!case_report.repeats.empty()) {
                const std::vector<int>& first = case_report.repeats.front().generated_token_ids;
                for (const auto& repeat : case_report.repeats) {
                    if (!generated_ids_equal(first, repeat.generated_token_ids)) {
                        case_report.deterministic = false;
                        break;
                    }
                }
            }
            report.cases.push_back(std::move(case_report));
        }

        qus::bench::e2e::write_raw_report(options.output_json_path, report);
        std::cout << "wrote " << options.output_json_path << '\n';
        return 0;
    } catch (const std::exception& e) {
        qus::bench::e2e::write_error_report(options.output_json_path, "run", e.what());
        std::cerr << "qus_e2e_bench: " << e.what() << '\n';
        return 1;
    }
}
