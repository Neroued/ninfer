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

} // namespace qus::bench::e2e
