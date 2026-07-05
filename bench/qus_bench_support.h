#pragma once

// Pure (CUDA-free, Engine-free) support logic for the qus_bench throughput tool:
// CLI parsing, test-matrix expansion, max_ctx sizing, corpus .ids loading/slicing,
// statistics, and table/JSON/CSV formatting. The engine-driven measurement lives in
// qus_bench.cpp; everything here is host-only and unit-testable without a GPU.

#include "qus/model/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qus::bench {

inline constexpr int kSchemaVersion                  = 4;
inline constexpr std::string_view kArtifactType      = "qus_bench_report";
inline constexpr std::string_view kDefaultCorpusPath = "bench/fixtures/bench_corpus.ids";
// Seed tokens prefilled (untimed) before a pure decode (tg) test so the model has a valid
// context to decode from. Kept tiny so tg measures near-peak decode throughput.
inline constexpr int kDecodeSeedTokens   = 1;
inline constexpr int kDefaultNPrompt     = 512;
inline constexpr int kDefaultNGen        = 128;
inline constexpr int kDefaultRepetitions = 5;
inline constexpr int kDefaultWarmup      = 1;

enum class TestKind { Prefill, Decode, PrefillDecode };

// One benchmark test: pp{P} (Prefill), tg{G} (Decode), or pp{P}+tg{G} (PrefillDecode).
struct BenchTest {
    TestKind kind = TestKind::Prefill;
    int n_prompt  = 0; // P; prefill length (0 for pure decode)
    int n_gen     = 0; // G; decode steps (0 for pure prefill)
    std::string label;

    [[nodiscard]] bool has_prefill() const noexcept {
        return kind == TestKind::Prefill || kind == TestKind::PrefillDecode;
    }

    [[nodiscard]] bool has_decode() const noexcept {
        return kind == TestKind::Decode || kind == TestKind::PrefillDecode;
    }

    // Minimum EngineOptions.max_ctx that lets this test run without tripping the engine guards.
    // MTP sizing includes room for prefill draft preparation and full decode rounds, not only
    // caller-visible output tokens.
    [[nodiscard]] std::uint32_t required_context(int mtp_draft_tokens) const;
};

enum class OutputFormat { Table, Json, Csv };

struct BenchOptions {
    std::string weights_path;
    std::string corpus_path{kDefaultCorpusPath};
    std::vector<int> n_prompt;                   // -p
    std::vector<int> n_gen;                      // -n
    std::vector<std::pair<int, int>> prompt_gen; // -pg
    int repetitions = kDefaultRepetitions;
    int warmup      = kDefaultWarmup;
    std::optional<std::uint32_t> max_ctx;  // --max-ctx override
    std::optional<std::size_t> work_bytes; // --work-bytes override (prefill workspace)
    std::uint32_t prefill_chunk = model::kDefaultPrefillChunk;
    int mtp_draft_tokens        = 0;
    int device                  = 0;
    bool use_cuda_graph         = true;
    OutputFormat output         = OutputFormat::Table;
    std::string output_file; // empty => stdout
    bool help_requested = false;
};

// Timing for one measured repetition. Only the fields relevant to the test kind are set.
struct RepTiming {
    double prefill_time_s = 0.0;
    double decode_time_s  = 0.0;

    struct BenchMtpStats {
        bool enabled                 = false;
        int k                        = 0;
        std::int64_t draft_tokens    = 0;
        std::int64_t accepted_tokens = 0;
        std::int64_t rounds          = 0;
        std::int64_t fallback_steps  = 0;
        std::array<std::int64_t, model::kMaxMtpDraftTokens> accepted_per_pos{};
    } mtp;
};

using BenchMtpStats = RepTiming::BenchMtpStats;

struct TestResult {
    BenchTest test;
    std::vector<RepTiming> reps;
    std::size_t workspace_peak_bytes = 0; // high-water workspace-arena usage during this test
};

struct Stats {
    double mean   = 0.0;
    double stddev = 0.0; // sample stddev (n-1); 0 when fewer than 2 samples
    int count     = 0;
};

struct BenchEnvironment {
    std::string git_commit;
    bool worktree_dirty = false;
    std::string gpu_name;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
    int device_id = 0;
    std::string weights_path;
    std::uint64_t weights_file_size_bytes = 0;
    std::uint32_t max_ctx                 = 0;
    std::size_t work_bytes                = 0;
    std::uint32_t prefill_chunk           = model::kDefaultPrefillChunk;
    int mtp_draft_tokens                  = 0;
    std::string decode_path; // "cuda_graph", "eager", "mtp_cuda_graph", or "mtp_eager"
    bool decode_graph_requested  = false;
    bool decode_graph_primed     = false;
    int decode_graph_prime_steps = 0;
    int repetitions              = 0;
    int warmup                   = 0;
    std::string corpus_path;
    std::size_t corpus_tokens = 0;
};

// CLI ---------------------------------------------------------------------------------------
BenchOptions parse_args(int argc, char** argv);
std::string usage_text(std::string_view program);

// Test matrix -------------------------------------------------------------------------------
// Expand -p/-n/-pg into labeled tests. With none supplied, defaults to pp512 and tg128.
std::vector<BenchTest> expand_tests(const BenchOptions& options);
// Auto-size max_ctx to the largest test requirement, or validate the --max-ctx override.
std::uint32_t resolve_max_ctx(const std::vector<BenchTest>& tests,
                              std::optional<std::uint32_t> override_max_ctx, int mtp_draft_tokens,
                              bool use_cuda_graph);
// Reject prefill lengths that exceed the meaningful corpus (can't slice beyond it).
void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens);

// Corpus ------------------------------------------------------------------------------------
std::vector<int> load_corpus_ids(const std::string& path);
std::vector<int> prompt_slice(const std::vector<int>& corpus, int n_prompt);
std::string decode_path_name(bool use_cuda_graph, int mtp_draft_tokens);
int decode_graph_prime_steps(int mtp_draft_tokens);
std::uint32_t decode_graph_prime_required_context(int mtp_draft_tokens);

// Statistics --------------------------------------------------------------------------------
Stats compute_stats(const std::vector<double>& values);
// Per-rep throughput series for a result (empty if the test lacks that phase).
std::vector<double> prefill_tok_s_series(const TestResult& result);
std::vector<double> decode_output_tok_s_series(const TestResult& result);
std::vector<double> decode_engine_tok_s_series(const TestResult& result);
std::vector<double> prefill_time_series(const TestResult& result);
std::vector<double> decode_time_series(const TestResult& result);

// Formatting --------------------------------------------------------------------------------
std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results);
std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results);
std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results);

// Small shared helpers (also used by qus_bench.cpp for the report header).
std::string json_escape(std::string_view value);
std::string current_git_commit_or_empty();
bool current_git_worktree_dirty();
std::uint64_t file_size_or_zero(const std::string& path);

} // namespace qus::bench
