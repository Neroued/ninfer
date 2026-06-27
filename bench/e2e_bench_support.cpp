#include "e2e_bench_support.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace qus::bench::e2e {
namespace {

int parse_nonnegative_int(std::string_view text, const char* label) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    int value = 0;
    const auto* first = text.data();
    const auto* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last || value < 0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

int parse_positive_int(std::string_view text, const char* label) {
    const int value = parse_nonnegative_int(text, label);
    if (value <= 0) { throw std::invalid_argument(std::string(label) + " must be positive"); }
    return value;
}

std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for (const char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

std::string run_command_capture(const std::string& command) {
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) { return {}; }
    std::string out;
    char buffer[256];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) { out += buffer; }
    const int rc = pclose(pipe);
    if (rc != 0) { return {}; }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

std::size_t arena_slack(const ArenaMemoryStats& arena) {
    return arena.capacity_bytes > arena.used_bytes ? arena.capacity_bytes - arena.used_bytes : 0;
}

std::size_t payload_overhead(const EngineMemoryStats& memory) {
    return memory.weights.used_bytes > memory.q5090_loaded_payload_bytes
               ? memory.weights.used_bytes - memory.q5090_loaded_payload_bytes
               : 0;
}

void write_arena(std::ostream& out, const char* name, const ArenaMemoryStats& arena,
                 const std::string& indent) {
    out << indent << "{\n"
        << indent << "  \"name\": \"" << name << "\",\n"
        << indent << "  \"present\": " << json_bool(arena.present) << ",\n"
        << indent << "  \"capacity_bytes\": " << arena.capacity_bytes << ",\n"
        << indent << "  \"used_bytes\": " << arena.used_bytes << ",\n"
        << indent << "  \"peak_used_bytes\": " << arena.peak_used_bytes << "\n"
        << indent << "}";
}

void write_memory(std::ostream& out, const EngineMemoryStats& memory, const std::string& indent) {
    out << indent << "{\n"
        << indent << "  \"accounting_scope\": \"engine_owned_device_arenas_complete\",\n"
        << indent << "  \"hidden_device_allocations\": false,\n"
        << indent << "  \"loaded\": " << json_bool(memory.loaded) << ",\n"
        << indent << "  \"device\": " << memory.device << ",\n"
        << indent << "  \"max_context\": " << memory.max_context << ",\n"
        << indent << "  \"position\": " << memory.position << ",\n"
        << indent << "  \"arenas\": [\n";
    write_arena(out, "weights", memory.weights, indent + "    ");
    out << ",\n";
    write_arena(out, "cache", memory.cache, indent + "    ");
    out << ",\n";
    write_arena(out, "workspace", memory.workspace, indent + "    ");
    out << "\n"
        << indent << "  ],\n"
        << indent << "  \"q5090_loaded_payload_bytes\": " << memory.q5090_loaded_payload_bytes
        << ",\n"
        << indent << "  \"q5090_tensor_count\": " << memory.q5090_tensor_count << ",\n"
        << indent << "  \"q5090_quant_count\": " << memory.q5090_quant_count << ",\n"
        << indent
        << "  \"known_exclusions\": [\"host_q5090_file_buffer\", \"cuda_driver_runtime\", "
           "\"profiler_overhead\", \"os_process_rss\", \"profiles_artifacts\"]\n"
        << indent << "}";
}

void write_token_array(std::ostream& out, const std::vector<int>& ids) {
    out << "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) { out << ", "; }
        out << ids[i];
    }
    out << "]";
}

std::uint64_t max_peak(const std::vector<RepeatReport>& repeats,
                       ArenaMemoryStats EngineMemoryStats::*arena_member) {
    std::uint64_t max_value = 0;
    for (const RepeatReport& repeat : repeats) {
        const ArenaMemoryStats& arena = repeat.memory.*arena_member;
        max_value = std::max<std::uint64_t>(max_value, arena.peak_used_bytes);
    }
    return max_value;
}

void write_repeat(std::ostream& out, const RepeatReport& repeat, const std::string& indent) {
    out << indent << "{\n"
        << indent << "  \"repeat_index\": " << repeat.repeat_index << ",\n"
        << indent << "  \"prefill_time_s\": " << repeat.prefill_time_s << ",\n"
        << indent << "  \"decode_time_s\": " << repeat.decode_time_s << ",\n"
        << indent << "  \"e2e_excluding_load_time_s\": "
        << repeat.e2e_excluding_load_time_s() << ",\n"
        << indent << "  \"prompt_tokens\": " << repeat.prompt_tokens << ",\n"
        << indent << "  \"prefill_output_tokens\": " << repeat.prefill_output_tokens << ",\n"
        << indent << "  \"decode_loop_tokens\": " << repeat.decode_loop_tokens << ",\n"
        << indent << "  \"generated_tokens_total\": " << repeat.generated_tokens_total() << ",\n"
        << indent << "  \"decode_eager_tok_s\": ";
    if (repeat.decode_eager_tok_s_valid()) {
        out << repeat.decode_eager_tok_s();
    } else {
        out << "null";
    }
    out << ",\n"
        << indent << "  \"decode_eager_tok_s_valid\": "
        << json_bool(repeat.decode_eager_tok_s_valid()) << ",\n"
        << indent << "  \"e2e_excluding_load_tok_s\": "
        << repeat.e2e_excluding_load_tok_s() << ",\n"
        << indent << "  \"stop_reason\": \"" << json_escape(repeat.stop_reason) << "\",\n"
        << indent << "  \"generated_token_ids\": ";
    write_token_array(out, repeat.generated_token_ids);
    out << ",\n" << indent << "  \"memory\": ";
    write_memory(out, repeat.memory, indent + "  ");
    out << "\n" << indent << "}";
}

void write_case_summary(std::ostream& out, const CaseReport& case_report,
                        const std::string& indent) {
    std::vector<double> prefill_times;
    std::vector<double> decode_times;
    std::vector<double> decode_tok_s;
    std::vector<double> e2e_tok_s;
    prefill_times.reserve(case_report.repeats.size());
    decode_times.reserve(case_report.repeats.size());
    decode_tok_s.reserve(case_report.repeats.size());
    e2e_tok_s.reserve(case_report.repeats.size());
    for (const RepeatReport& repeat : case_report.repeats) {
        prefill_times.push_back(repeat.prefill_time_s);
        decode_times.push_back(repeat.decode_time_s);
        if (repeat.decode_eager_tok_s_valid()) { decode_tok_s.push_back(repeat.decode_eager_tok_s()); }
        e2e_tok_s.push_back(repeat.e2e_excluding_load_tok_s());
    }

    out << indent << "{\n"
        << indent << "  \"prefill_time_s_median\": " << median(prefill_times) << ",\n"
        << indent << "  \"decode_time_s_median\": " << median(decode_times) << ",\n"
        << indent << "  \"decode_eager_tok_s_median\": " << median(decode_tok_s) << ",\n"
        << indent << "  \"e2e_excluding_load_tok_s_median\": " << median(e2e_tok_s)
        << ",\n"
        << indent << "  \"deterministic_token_ids\": " << json_bool(case_report.deterministic)
        << ",\n"
        << indent << "  \"max_weight_arena_peak_used_bytes\": "
        << max_peak(case_report.repeats, &EngineMemoryStats::weights) << ",\n"
        << indent << "  \"max_cache_arena_peak_used_bytes\": "
        << max_peak(case_report.repeats, &EngineMemoryStats::cache) << ",\n"
        << indent << "  \"max_workspace_arena_peak_used_bytes\": "
        << max_peak(case_report.repeats, &EngineMemoryStats::workspace) << "\n"
        << indent << "}";
}

void write_case(std::ostream& out, const CaseReport& case_report, const std::string& indent) {
    out << indent << "{\n"
        << indent << "  \"name\": \"" << json_escape(case_report.input.name) << "\",\n"
        << indent << "  \"fixture_set\": \"" << json_escape(case_report.fixture_set) << "\",\n"
        << indent << "  \"fixture_manifest_path\": \""
        << json_escape(case_report.fixture_manifest_path) << "\",\n"
        << indent << "  \"fixture_manifest_sha256\": \""
        << json_escape(case_report.fixture_manifest_sha256) << "\",\n"
        << indent << "  \"prompt_ids_path\": \""
        << json_escape(case_report.input.prompt_ids_path) << "\",\n"
        << indent << "  \"prompt_ids_sha256\": \"" << json_escape(case_report.prompt_ids_sha256)
        << "\",\n"
        << indent << "  \"prompt_tokens\": " << case_report.input.prompt_tokens() << ",\n"
        << indent << "  \"requested_max_new_tokens\": "
        << case_report.input.requested_max_new_tokens << ",\n"
        << indent << "  \"eos_token_id\": " << case_report.input.eos_token_id << ",\n"
        << indent << "  \"max_context\": " << case_report.input.max_context << ",\n"
        << indent << "  \"decode_loop_tokens_requested\": "
        << case_report.input.decode_loop_tokens_requested() << ",\n"
        << indent << "  \"required_max_context\": " << case_report.input.required_max_context()
        << ",\n"
        << indent << "  \"warmup_repeats\": " << case_report.warmup_repeats << ",\n"
        << indent << "  \"measured_repeats\": " << case_report.measured_repeats << ",\n"
        << indent << "  \"deterministic_token_ids\": "
        << json_bool(case_report.deterministic) << ",\n"
        << indent << "  \"repeats\": [\n";
    for (std::size_t i = 0; i < case_report.repeats.size(); ++i) {
        if (i != 0) { out << ",\n"; }
        write_repeat(out, case_report.repeats[i], indent + "    ");
    }
    out << "\n" << indent << "  ],\n" << indent << "  \"summary\": ";
    write_case_summary(out, case_report, indent + "  ");
    out << "\n" << indent << "}";
}

} // namespace

std::size_t CaseRunInput::decode_loop_tokens_requested() const noexcept {
    return requested_max_new_tokens <= 1 ? 0 : static_cast<std::size_t>(requested_max_new_tokens - 1);
}

std::size_t CaseRunInput::required_max_context() const noexcept {
    return prompt_ids.size() + decode_loop_tokens_requested();
}

std::vector<int> parse_ids_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("failed to open prompt ids file: " + path); }

    std::vector<int> ids;
    std::string token;
    while (in >> token) { ids.push_back(parse_nonnegative_int(token, "token id")); }
    if (ids.empty()) { throw std::invalid_argument("prompt ids file is empty: " + path); }
    return ids;
}

CaseSpec parse_case_arg(const std::string& value) {
    const std::size_t first = value.find(':');
    const std::size_t last = value.rfind(':');
    if (first == std::string::npos || last == std::string::npos || first == last) {
        throw std::invalid_argument("case must be <name>:<prompt-ids-path>:<max-new-tokens>");
    }
    CaseSpec spec;
    spec.name = value.substr(0, first);
    spec.prompt_ids_path = value.substr(first + 1, last - first - 1);
    spec.max_new_tokens =
        parse_positive_int(std::string_view(value).substr(last + 1), "case max_new_tokens");
    if (spec.name.empty()) { throw std::invalid_argument("case name is empty"); }
    if (spec.prompt_ids_path.empty()) { throw std::invalid_argument("case prompt ids path is empty"); }
    return spec;
}

RunOptions parse_args(int argc, char** argv) {
    RunOptions options;
    bool saw_weights = false;
    bool saw_output = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(flag) + " requires value"); }
            return argv[++i];
        };
        if (arg == "--weights") {
            options.weights_path = require_value("--weights");
            saw_weights = true;
        } else if (arg == "--output-json") {
            options.output_json_path = require_value("--output-json");
            saw_output = true;
        } else if (arg == "--case") {
            options.cases.push_back(parse_case_arg(require_value("--case")));
        } else if (arg == "--warmup-repeats") {
            options.warmup_repeats =
                parse_nonnegative_int(require_value("--warmup-repeats"), "warmup_repeats");
        } else if (arg == "--repeats") {
            options.repeats = parse_positive_int(require_value("--repeats"), "repeats");
        } else if (arg == "--max-ctx") {
            options.max_ctx =
                static_cast<std::uint32_t>(parse_positive_int(require_value("--max-ctx"), "max_ctx"));
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--eos-token-id") {
            options.eos_token_id = parse_nonnegative_int(require_value("--eos-token-id"),
                                                         "eos_token_id");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!saw_weights) { throw std::invalid_argument("--weights is required"); }
    if (!saw_output) { throw std::invalid_argument("--output-json is required"); }
    if (options.cases.empty()) { throw std::invalid_argument("at least one --case is required"); }
    return options;
}

void validate_case_context(const CaseRunInput& input) {
    if (input.name.empty()) { throw std::invalid_argument("case name is empty"); }
    if (input.prompt_ids.empty()) { throw std::invalid_argument("case prompt is empty"); }
    if (input.requested_max_new_tokens <= 0) {
        throw std::invalid_argument("case max_new_tokens must be positive");
    }
    if (input.max_context < input.required_max_context()) {
        throw std::invalid_argument("max_ctx is too small for case " + input.name);
    }
}

std::string json_escape(std::string_view value) {
    std::string out;
    for (const char c : value) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default: {
            const auto uc = static_cast<unsigned char>(c);
            if (uc < 0x20) {
                std::ostringstream escaped;
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(uc);
                out += escaped.str();
            } else {
                out += c;
            }
            break;
        }
        }
    }
    return out;
}

double median(std::vector<double> values) {
    if (values.empty()) { return 0.0; }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if ((values.size() & 1U) != 0U) { return values[mid]; }
    return (values[mid - 1] + values[mid]) * 0.5;
}

std::string sha256_file_or_empty(const std::string& path) {
    if (!std::filesystem::exists(path)) { return {}; }
    const std::string command = "sha256sum " + shell_quote(path) + " | awk '{print $1}'";
    return run_command_capture(command);
}

std::uint64_t file_size_or_zero(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<std::uint64_t>(size);
}

std::string current_git_commit_or_empty() {
#ifdef QUS_SOURCE_DIR
    return run_command_capture("git -C " + shell_quote(QUS_SOURCE_DIR) + " rev-parse HEAD");
#else
    return {};
#endif
}

bool current_git_worktree_dirty() {
#ifdef QUS_SOURCE_DIR
    const std::string out =
        run_command_capture("git -C " + shell_quote(QUS_SOURCE_DIR) + " status --porcelain");
    return !out.empty();
#else
    return false;
#endif
}

void ensure_parent_dir(const std::string& path) {
    const std::filesystem::path p(path);
    const std::filesystem::path parent = p.parent_path();
    if (!parent.empty()) { std::filesystem::create_directories(parent); }
}

void write_error_report(const std::string& path, std::string_view phase, std::string_view message) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) { return; }
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"artifact_type\": \"qus_e2e_benchmark_report\",\n"
        << "  \"status\": \"error\",\n"
        << "  \"error\": {\n"
        << "    \"phase\": \"" << json_escape(phase) << "\",\n"
        << "    \"message\": \"" << json_escape(message) << "\"\n"
        << "  }\n"
        << "}\n";
}

void write_raw_report(const std::string& path, const RawReport& report) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open output JSON: " + path); }

    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"artifact_type\": \"qus_e2e_benchmark_report\",\n"
        << "  \"status\": \"ok\",\n"
        << "  \"run\": {\n"
        << "    \"binary\": \"" << json_escape(report.binary) << "\",\n"
        << "    \"command\": \"" << json_escape(report.command) << "\",\n"
        << "    \"git_commit\": \"" << json_escape(report.git_commit) << "\",\n"
        << "    \"worktree_dirty\": " << json_bool(report.worktree_dirty) << ",\n"
        << "    \"load_time_s\": " << report.load_time_s << "\n"
        << "  },\n"
        << "  \"environment\": {\n"
        << "    \"cuda_runtime_version\": \""
        << json_escape(report.environment.cuda_runtime_version) << "\",\n"
        << "    \"cuda_driver_version\": \""
        << json_escape(report.environment.cuda_driver_version) << "\",\n"
        << "    \"gpu_name\": \"" << json_escape(report.environment.gpu_name) << "\",\n"
        << "    \"device_id\": " << report.environment.device_id << "\n"
        << "  },\n"
        << "  \"engine\": {\n"
        << "    \"max_context\": " << report.max_context << ",\n"
        << "    \"workspace_lifetime_policy\": \"step_reset\",\n"
        << "    \"decode_metric\": \"decode_eager_tok_s\",\n"
        << "    \"sampling_location\": \"device_argmax\",\n"
        << "    \"token_readback\": \"per_step_sync_d2h\",\n"
        << "    \"includes_token_readback\": true,\n"
        << "    \"timing_boundary\": \"host_visible_phase_end\"\n"
        << "  },\n"
        << "  \"weights\": {\n"
        << "    \"q5090_path\": \"" << json_escape(report.q5090_path) << "\",\n"
        << "    \"q5090_file_size_bytes\": " << report.q5090_file_size_bytes << ",\n"
        << "    \"q5090_sha256\": \"" << json_escape(report.q5090_sha256) << "\",\n"
        << "    \"q5090_conv1d_layout\": \"runtime_native_conv_dim_by_kernel\",\n"
        << "    \"load_strategy\": \"full_file_host_vector_then_h2d_payload_upload\",\n"
        << "    \"default_weight_arena_policy\": \"q5090_file_size_plus_256MiB\",\n"
        << "    \"estimated_host_file_buffer_bytes\": " << report.q5090_file_size_bytes << ",\n"
        << "    \"selected_modules\": {\n"
        << "      \"text_core\": true,\n"
        << "      \"mtp\": false,\n"
        << "      \"vision\": false\n"
        << "    },\n"
        << "    \"q5090_loaded_payload_bytes\": "
        << report.post_load_memory.q5090_loaded_payload_bytes << ",\n"
        << "    \"weight_arena_capacity_bytes\": "
        << report.post_load_memory.weights.capacity_bytes << ",\n"
        << "    \"weight_arena_used_bytes\": " << report.post_load_memory.weights.used_bytes
        << ",\n"
        << "    \"weight_arena_peak_used_bytes\": "
        << report.post_load_memory.weights.peak_used_bytes << ",\n"
        << "    \"weight_arena_slack_bytes\": "
        << arena_slack(report.post_load_memory.weights) << ",\n"
        << "    \"weight_payload_to_arena_used_overhead_bytes\": "
        << payload_overhead(report.post_load_memory) << "\n"
        << "  },\n"
        << "  \"memory\": ";
    write_memory(out, report.post_load_memory, "  ");
    out << ",\n"
        << "  \"summary\": {\n"
        << "    \"case_count\": " << report.cases.size() << ",\n"
        << "    \"load_time_s\": " << report.load_time_s << "\n"
        << "  },\n"
        << "  \"cases\": [\n";
    for (std::size_t i = 0; i < report.cases.size(); ++i) {
        if (i != 0) { out << ",\n"; }
        write_case(out, report.cases[i], "    ");
    }
    out << "\n"
        << "  ]\n"
        << "}\n";
}

} // namespace qus::bench::e2e
