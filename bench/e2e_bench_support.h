#pragma once

#include "qus/model/model.h"
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
    std::vector<int> stop_token_ids = {248046, 248044};
    std::uint32_t max_context = 0;

    [[nodiscard]] std::size_t prompt_tokens() const noexcept { return prompt_ids.size(); }
    [[nodiscard]] std::size_t decode_loop_tokens_requested() const noexcept;
    [[nodiscard]] std::size_t required_max_context() const noexcept;
};

struct FixtureCaseMetadata {
    std::string name;
    std::string messages_path;
    std::string prompt_ids_path;
    std::string messages_sha256;
    std::string rendered_prompt_sha256;
    std::string prompt_ids_sha256;
    std::string prompt_format;
    bool add_generation_prompt = false;
    bool add_special_tokens = true;
    bool enable_thinking = true;
    int prompt_tokens = 0;
};

struct FixtureMetadata {
    std::string fixture_set;
    std::vector<int> stop_token_ids;
    FixtureCaseMetadata case_metadata;
};

struct RunOptions {
    std::string weights_path;
    std::string output_json_path;
    std::string fixture_manifest_path = "bench/fixtures/prompts/m2.8-v1.manifest.json";
    std::vector<CaseSpec> cases;
    bool help_requested = false;
    int warmup_repeats = 0;
    int repeats = 1;
    std::uint32_t max_ctx = qus::EngineOptions{}.max_ctx;
    int device = 0;
    std::vector<int> stop_token_ids = {248046, 248044};
};

struct RepeatReport {
    int repeat_index = 0;
    double prefill_time_s = 0.0;
    double decode_time_s = 0.0;
    std::size_t prompt_tokens = 0;
    int prefill_output_tokens = 1;
    std::size_t decode_loop_tokens = 0;
    std::vector<int> generated_token_ids;
    std::string stop_reason = "max_new_tokens";
    EngineMemoryStats memory;

    [[nodiscard]] std::size_t generated_tokens_total() const noexcept {
        return static_cast<std::size_t>(prefill_output_tokens) + decode_loop_tokens;
    }
    [[nodiscard]] double e2e_excluding_load_time_s() const noexcept {
        return prefill_time_s + decode_time_s;
    }
    [[nodiscard]] bool decode_eager_tok_s_valid() const noexcept {
        return decode_loop_tokens > 0 && decode_time_s > 0.0;
    }
    [[nodiscard]] double decode_eager_tok_s() const noexcept {
        return decode_eager_tok_s_valid() ? static_cast<double>(decode_loop_tokens) / decode_time_s
                                          : 0.0;
    }
    [[nodiscard]] double prefill_prompt_tok_s() const noexcept {
        return prefill_time_s > 0.0 ? static_cast<double>(prompt_tokens) / prefill_time_s : 0.0;
    }
    [[nodiscard]] double e2e_excluding_load_tok_s() const noexcept {
        const double denom = e2e_excluding_load_time_s();
        return denom > 0.0 ? static_cast<double>(generated_tokens_total()) / denom : 0.0;
    }
};

struct CaseReport {
    CaseRunInput input;
    std::string prompt_format;
    std::string messages_path;
    std::string messages_sha256;
    std::string rendered_prompt_sha256;
    std::string prompt_ids_sha256;
    bool add_generation_prompt = true;
    bool add_special_tokens = false;
    bool enable_thinking = false;
    std::string fixture_set = "m2.8-v1";
    std::string fixture_manifest_path = "bench/fixtures/prompts/m2.8-v1.manifest.json";
    std::string fixture_manifest_sha256;
    int warmup_repeats = 0;
    int measured_repeats = 0;
    bool deterministic = true;
    std::vector<RepeatReport> repeats;
};

struct EnvironmentReport {
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
    std::string gpu_name;
    int device_id = 0;
};

struct RawReport {
    std::string command;
    std::string binary = "qus_e2e_bench";
    std::string git_commit;
    bool worktree_dirty = false;
    double load_time_s = 0.0;
    EnvironmentReport environment;
    std::string q5090_path;
    std::uint64_t q5090_file_size_bytes = 0;
    std::string q5090_sha256;
    std::uint32_t max_context = 0;
    std::string workspace_lifetime_policy = qus::model::kWorkspaceLifetimePolicy;
    EngineMemoryStats post_load_memory;
    std::vector<CaseReport> cases;
};

std::vector<int> parse_ids_file(const std::string& path);
CaseSpec parse_case_arg(const std::string& value);
RunOptions parse_args(int argc, char** argv);
std::string usage_text(std::string_view program);
void validate_case_context(const CaseRunInput& input);
std::vector<int> normalize_stop_token_ids(const std::vector<int>& ids);
bool is_stop_token(const std::vector<int>& stop_token_ids, int token);
FixtureMetadata load_fixture_metadata_for_case(const std::string& manifest_path,
                                               const std::string& case_name,
                                               const std::string& prompt_ids_path);
std::vector<int> load_verified_prompt_ids(const std::string& prompt_ids_path,
                                          const FixtureCaseMetadata& case_metadata);

std::string json_escape(std::string_view value);
double median(std::vector<double> values);
std::string sha256_file_or_empty(const std::string& path);
std::uint64_t file_size_or_zero(const std::string& path);
std::string current_git_commit_or_empty();
bool current_git_worktree_dirty();
void ensure_parent_dir(const std::string& path);
void write_error_report(const std::string& path, std::string_view phase, std::string_view message);
void write_raw_report(const std::string& path, const RawReport& report);

} // namespace qus::bench::e2e
