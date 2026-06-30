#include "e2e_bench_support.h"

#include <nlohmann/json.hpp>

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

using Json = nlohmann::json;

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

std::string path_display_string(const std::filesystem::path& path) {
    std::filesystem::path normalized = path.lexically_normal();
#ifdef QUS_SOURCE_DIR
    const std::filesystem::path source_dir = std::filesystem::path(QUS_SOURCE_DIR).lexically_normal();
    if (normalized.is_absolute()) {
        const std::filesystem::path relative = normalized.lexically_relative(source_dir);
        if (!relative.empty() && relative.native().find("..") != 0) { normalized = relative; }
    }
#endif
    return normalized.generic_string();
}

bool paths_match_requested(const std::filesystem::path& manifest_relative,
                           const std::filesystem::path& manifest_dir,
                           const std::string& requested_prompt_ids_path) {
    const std::filesystem::path requested =
        std::filesystem::path(requested_prompt_ids_path).lexically_normal();
    const std::filesystem::path joined = (manifest_dir / manifest_relative).lexically_normal();
    if (requested == joined) { return true; }
    const std::string joined_display = path_display_string(joined);
    if (!requested.is_absolute() && requested.generic_string() == joined_display) { return true; }
#ifdef QUS_SOURCE_DIR
    const std::filesystem::path source_joined =
        (std::filesystem::path(QUS_SOURCE_DIR) / joined_display).lexically_normal();
    if (requested.is_absolute() && requested == source_joined) { return true; }
#endif
    return false;
}

std::string existing_read_path(const std::string& path) {
    if (std::filesystem::exists(path)) { return path; }
#ifdef QUS_SOURCE_DIR
    const std::filesystem::path source_relative =
        (std::filesystem::path(QUS_SOURCE_DIR) / std::filesystem::path(path)).lexically_normal();
    if (std::filesystem::exists(source_relative)) { return source_relative.string(); }
#endif
    return path;
}

const Json& require_object_field(const Json& object, const char* key, const char* label) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument(std::string(label) + " missing " + key);
    }
    const Json& value = object.at(key);
    if (!value.is_object()) { throw std::invalid_argument(std::string(label) + "." + key + " must be object"); }
    return value;
}

const Json& require_array_field(const Json& object, const char* key, const char* label) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument(std::string(label) + " missing " + key);
    }
    const Json& value = object.at(key);
    if (!value.is_array()) { throw std::invalid_argument(std::string(label) + "." + key + " must be array"); }
    return value;
}

std::string require_string_field(const Json& object, const char* key, const char* label) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument(std::string(label) + " missing " + key);
    }
    const Json& value = object.at(key);
    if (!value.is_string()) { throw std::invalid_argument(std::string(label) + "." + key + " must be string"); }
    return value.get<std::string>();
}

std::string require_nonempty_string_field(const Json& object, const char* key, const char* label) {
    std::string value = require_string_field(object, key, label);
    if (value.empty()) { throw std::invalid_argument(std::string(label) + "." + key + " is empty"); }
    return value;
}

bool require_bool_field(const Json& object, const char* key, const char* label) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument(std::string(label) + " missing " + key);
    }
    const Json& value = object.at(key);
    if (!value.is_boolean()) { throw std::invalid_argument(std::string(label) + "." + key + " must be bool"); }
    return value.get<bool>();
}

int require_nonnegative_int_field(const Json& object, const char* key, const char* label) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument(std::string(label) + " missing " + key);
    }
    const Json& value = object.at(key);
    if (!value.is_number_integer()) {
        throw std::invalid_argument(std::string(label) + "." + key + " must be integer");
    }
    const int parsed = value.get<int>();
    if (parsed < 0) { throw std::invalid_argument(std::string(label) + "." + key + " must be nonnegative"); }
    return parsed;
}

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
        << indent << "  \"prefill_prompt_tok_s\": " << repeat.prefill_prompt_tok_s() << ",\n"
        << indent << "  \"decode_tok_s\": ";
    if (repeat.decode_tok_s_valid()) {
        out << repeat.decode_tok_s();
    } else {
        out << "null";
    }
    out << ",\n"
        << indent << "  \"decode_tok_s_valid\": "
        << json_bool(repeat.decode_tok_s_valid()) << ",\n"
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
    std::vector<double> prefill_tok_s;
    std::vector<double> decode_times;
    std::vector<double> decode_tok_s;
    std::vector<double> e2e_tok_s;
    prefill_times.reserve(case_report.repeats.size());
    prefill_tok_s.reserve(case_report.repeats.size());
    decode_times.reserve(case_report.repeats.size());
    decode_tok_s.reserve(case_report.repeats.size());
    e2e_tok_s.reserve(case_report.repeats.size());
    for (const RepeatReport& repeat : case_report.repeats) {
        prefill_times.push_back(repeat.prefill_time_s);
        prefill_tok_s.push_back(repeat.prefill_prompt_tok_s());
        decode_times.push_back(repeat.decode_time_s);
        if (repeat.decode_tok_s_valid()) { decode_tok_s.push_back(repeat.decode_tok_s()); }
        e2e_tok_s.push_back(repeat.e2e_excluding_load_tok_s());
    }

    out << indent << "{\n"
        << indent << "  \"prefill_time_s_median\": " << median(prefill_times) << ",\n"
        << indent << "  \"prefill_prompt_tok_s_median\": " << median(prefill_tok_s) << ",\n"
        << indent << "  \"decode_time_s_median\": " << median(decode_times) << ",\n"
        << indent << "  \"decode_tok_s_median\": ";
    if (decode_tok_s.empty()) {
        out << "null";
    } else {
        out << median(decode_tok_s);
    }
    out << ",\n"
        << indent << "  \"e2e_excluding_load_tok_s_median\": " << median(e2e_tok_s) << ",\n"
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
        << indent << "  \"prompt_format\": \"" << json_escape(case_report.prompt_format)
        << "\",\n"
        << indent << "  \"messages_path\": \"" << json_escape(case_report.messages_path)
        << "\",\n"
        << indent << "  \"messages_sha256\": \"" << json_escape(case_report.messages_sha256)
        << "\",\n"
        << indent << "  \"rendered_prompt_sha256\": \""
        << json_escape(case_report.rendered_prompt_sha256) << "\",\n"
        << indent << "  \"prompt_ids_path\": \""
        << json_escape(case_report.input.prompt_ids_path) << "\",\n"
        << indent << "  \"prompt_ids_sha256\": \"" << json_escape(case_report.prompt_ids_sha256)
        << "\",\n"
        << indent << "  \"prompt_tokens\": " << case_report.input.prompt_tokens() << ",\n"
        << indent << "  \"requested_max_new_tokens\": "
        << case_report.input.requested_max_new_tokens << ",\n"
        << indent << "  \"add_generation_prompt\": "
        << json_bool(case_report.add_generation_prompt) << ",\n"
        << indent << "  \"add_special_tokens\": "
        << json_bool(case_report.add_special_tokens) << ",\n"
        << indent << "  \"chat_template_kwargs\": {\n"
        << indent << "    \"enable_thinking\": "
        << json_bool(case_report.enable_thinking) << "\n"
        << indent << "  },\n"
        << indent << "  \"stop_token_ids\": ";
    write_token_array(out, case_report.input.stop_token_ids);
    out << ",\n"
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

std::string usage_text(std::string_view program) {
    const RunOptions defaults;
    if (program.empty()) { program = "qus_e2e_bench"; }

    std::ostringstream default_stop_tokens;
    write_token_array(default_stop_tokens, defaults.stop_token_ids);

    std::ostringstream out;
    out << "Usage: " << program
        << " --weights <q5090-path> --output-json <report-path>"
        << " --case <name>:<prompt-ids-path>:<max-new-tokens> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --weights <q5090-path>                         required q5090 weights path\n"
        << "  --output-json <report-path>                    required JSON report path\n"
        << "  --fixture-manifest <path>                      fixture manifest path (default: "
        << defaults.fixture_manifest_path << ")\n"
        << "  --case <name>:<prompt-ids-path>:<max-new-tokens>\n"
        << "                                                 benchmark case; repeat for multiple cases\n"
        << "  --warmup-repeats <n>                           warmup repeats (default: "
        << defaults.warmup_repeats << ")\n"
        << "  --repeats <n>                                  measured repeats (default: "
        << defaults.repeats << ")\n"
        << "  --max-ctx <tokens>                             max context tokens (default: "
        << defaults.max_ctx << ")\n"
        << "  --device <cuda-device>                         CUDA device ordinal (default: "
        << defaults.device << ")\n"
        << "  --stop-token-id <id>                           stop token id; repeat to append "
        << "(default: " << default_stop_tokens.str() << ")\n"
        << "  --quiet                                        suppress progress logging\n"
        << "  -h, --help                                     show this help\n";
    return out.str();
}

RunOptions parse_args(int argc, char** argv) {
    RunOptions options;
    bool saw_weights = false;
    bool saw_output = false;
    bool saw_explicit_stop_token = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(flag) + " requires value"); }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            options.help_requested = true;
            return options;
        } else if (arg == "--weights") {
            options.weights_path = require_value("--weights");
            saw_weights = true;
        } else if (arg == "--output-json") {
            options.output_json_path = require_value("--output-json");
            saw_output = true;
        } else if (arg == "--fixture-manifest") {
            options.fixture_manifest_path = require_value("--fixture-manifest");
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
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--stop-token-id") {
            if (!saw_explicit_stop_token) {
                options.stop_token_ids.clear();
                saw_explicit_stop_token = true;
            }
            options.stop_token_ids.push_back(
                parse_nonnegative_int(require_value("--stop-token-id"), "stop_token_id"));
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!saw_weights) { throw std::invalid_argument("--weights is required"); }
    if (!saw_output) { throw std::invalid_argument("--output-json is required"); }
    if (options.cases.empty()) { throw std::invalid_argument("at least one --case is required"); }
    options.stop_token_ids = normalize_stop_token_ids(options.stop_token_ids);
    if (options.stop_token_ids.empty()) { throw std::invalid_argument("stop token list is empty"); }
    return options;
}

std::vector<int> normalize_stop_token_ids(const std::vector<int>& ids) {
    std::vector<int> normalized;
    normalized.reserve(ids.size());
    for (const int id : ids) {
        if (id < 0) { throw std::invalid_argument("stop token id must be nonnegative"); }
        if (std::find(normalized.begin(), normalized.end(), id) == normalized.end()) {
            normalized.push_back(id);
        }
    }
    return normalized;
}

bool is_stop_token(const std::vector<int>& stop_token_ids, int token) {
    return std::find(stop_token_ids.begin(), stop_token_ids.end(), token) != stop_token_ids.end();
}

FixtureMetadata load_fixture_metadata_for_case(const std::string& manifest_path,
                                               const std::string& case_name,
                                               const std::string& prompt_ids_path) {
    std::ifstream in(manifest_path);
    if (!in) { throw std::runtime_error("failed to open fixture manifest: " + manifest_path); }

    Json manifest;
    try {
        in >> manifest;
    } catch (const nlohmann::json::exception& e) {
        throw std::invalid_argument(std::string("invalid fixture manifest JSON: ") + e.what());
    }

    FixtureMetadata metadata;
    metadata.fixture_set = require_string_field(manifest, "fixture_set", "manifest");
    if (metadata.fixture_set != "m2.8-v1") {
        throw std::invalid_argument("fixture_set must be m2.8-v1");
    }

    const Json& generation = require_object_field(manifest, "generation", "manifest");
    const Json& stop_ids = require_array_field(generation, "stop_token_ids", "manifest.generation");
    if (stop_ids.empty()) { throw std::invalid_argument("manifest generation.stop_token_ids is empty"); }
    for (const Json& id : stop_ids) {
        if (!id.is_number_integer()) {
            throw std::invalid_argument("manifest generation.stop_token_ids entries must be integers");
        }
        const int parsed = id.get<int>();
        if (parsed < 0) {
            throw std::invalid_argument("manifest generation.stop_token_ids entries must be nonnegative");
        }
        metadata.stop_token_ids.push_back(parsed);
    }
    metadata.stop_token_ids = normalize_stop_token_ids(metadata.stop_token_ids);
    if (metadata.stop_token_ids.empty()) {
        throw std::invalid_argument("manifest generation.stop_token_ids is empty");
    }

    const Json& cases = require_array_field(manifest, "cases", "manifest");
    const Json* selected = nullptr;
    for (const Json& candidate : cases) {
        if (!candidate.is_object()) { throw std::invalid_argument("manifest case must be object"); }
        if (require_string_field(candidate, "name", "manifest case") == case_name) {
            selected = &candidate;
            break;
        }
    }
    if (selected == nullptr) { throw std::invalid_argument("manifest case not found: " + case_name); }

    const std::filesystem::path manifest_dir = std::filesystem::path(manifest_path).parent_path();
    const std::string messages_relative = require_nonempty_string_field(*selected, "messages", "manifest case");
    const std::string ids_relative = require_nonempty_string_field(*selected, "ids", "manifest case");
    const std::filesystem::path ids_path(ids_relative);
    if (!paths_match_requested(ids_path, manifest_dir, prompt_ids_path)) {
        throw std::invalid_argument("manifest ids path does not match requested prompt ids path");
    }

    FixtureCaseMetadata case_metadata;
    case_metadata.name = case_name;
    case_metadata.messages_path =
        path_display_string(manifest_dir / std::filesystem::path(messages_relative));
    case_metadata.prompt_ids_path = path_display_string(manifest_dir / ids_path);
    case_metadata.prompt_tokens = require_nonnegative_int_field(*selected, "prompt_tokens", "manifest case");
    case_metadata.messages_sha256 =
        require_nonempty_string_field(*selected, "messages_sha256", "manifest case");
    case_metadata.rendered_prompt_sha256 =
        require_nonempty_string_field(*selected, "rendered_prompt_sha256", "manifest case");
    case_metadata.prompt_ids_sha256 =
        require_nonempty_string_field(*selected, "ids_sha256", "manifest case");
    case_metadata.prompt_format = require_string_field(*selected, "prompt_format", "manifest case");
    if (case_metadata.prompt_format != "qwen3.6-chat-template") {
        throw std::invalid_argument("manifest prompt_format must be qwen3.6-chat-template");
    }
    case_metadata.add_generation_prompt =
        require_bool_field(*selected, "add_generation_prompt", "manifest case");
    if (!case_metadata.add_generation_prompt) {
        throw std::invalid_argument("manifest add_generation_prompt must be true");
    }
    case_metadata.add_special_tokens =
        require_bool_field(*selected, "add_special_tokens", "manifest case");
    if (case_metadata.add_special_tokens) {
        throw std::invalid_argument("manifest add_special_tokens must be false");
    }
    const Json& kwargs = require_object_field(*selected, "chat_template_kwargs", "manifest case");
    case_metadata.enable_thinking =
        require_bool_field(kwargs, "enable_thinking", "manifest case.chat_template_kwargs");
    if (case_metadata.enable_thinking) {
        throw std::invalid_argument("manifest enable_thinking must be false");
    }

    metadata.case_metadata = std::move(case_metadata);
    return metadata;
}

std::vector<int> load_verified_prompt_ids(const std::string& prompt_ids_path,
                                          const FixtureCaseMetadata& case_metadata) {
    const std::string read_path = existing_read_path(prompt_ids_path);
    const std::string actual_sha256 = sha256_file_or_empty(read_path);
    if (actual_sha256.empty()) {
        throw std::runtime_error("failed to compute prompt ids sha256: " + prompt_ids_path);
    }
    if (actual_sha256 != case_metadata.prompt_ids_sha256) {
        throw std::invalid_argument("prompt ids sha256 does not match fixture manifest");
    }

    std::vector<int> ids = parse_ids_file(read_path);
    if (static_cast<int>(ids.size()) != case_metadata.prompt_tokens) {
        throw std::invalid_argument("prompt ids token count does not match fixture manifest");
    }
    return ids;
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
        << "    \"workspace_lifetime_policy\": \"" << json_escape(report.workspace_lifetime_policy)
        << "\",\n"
        << "    \"decode_metric\": \"decode_tok_s\",\n"
        << "    \"decode_path\": \"" << json_escape(report.decode_path) << "\",\n"
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
