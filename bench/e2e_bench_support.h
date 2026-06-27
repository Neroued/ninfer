#pragma once

#include "qus/runtime/engine.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qus::bench::e2e {

struct CaseSpec {
    std::string name;
    std::string prompt_ids_path;
    int max_new_tokens = 0;
};

struct CaseRunInput {
    std::string name;
    std::string prompt_ids_path;
    std::vector<int> prompt_ids;
    int requested_max_new_tokens = 0;
    std::uint32_t max_context = 0;

    [[nodiscard]] std::size_t prompt_tokens() const noexcept { return prompt_ids.size(); }
    [[nodiscard]] std::size_t decode_loop_tokens_requested() const noexcept;
    [[nodiscard]] std::size_t required_max_context() const noexcept;
};

struct RunOptions {
    std::string weights_path;
    std::string output_json_path;
    std::vector<CaseSpec> cases;
    int warmup_repeats = 0;
    int repeats = 1;
    std::uint32_t max_ctx = qus::EngineOptions{}.max_ctx;
    int device = 0;
    int eos_token_id = -1;
};

std::vector<int> parse_ids_file(const std::string& path);
CaseSpec parse_case_arg(const std::string& value);
RunOptions parse_args(int argc, char** argv);
void validate_case_context(const CaseRunInput& input);

std::string json_escape(std::string_view value);
double median(std::vector<double> values);
std::string sha256_file_or_empty(const std::string& path);
std::uint64_t file_size_or_zero(const std::string& path);
std::string current_git_commit_or_empty();
bool current_git_worktree_dirty();
void ensure_parent_dir(const std::string& path);

} // namespace qus::bench::e2e
