#pragma once

// Host-only support for the product throughput benchmark. The benchmark itself drives only the
// installed ninfer::Engine API; this file owns its CLI, matrix, statistics, and report schema.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::bench {

inline constexpr int kSchemaVersion                   = 8;
inline constexpr std::string_view kArtifactType       = "ninfer_bench_report";
inline constexpr std::string_view kDefaultCorpusPath  = "bench/fixtures/bench_corpus.ids";
inline constexpr int kDecodeSeedTokens                = 1;
inline constexpr int kDefaultNPrompt                  = 512;
inline constexpr int kDefaultNGen                     = 128;
inline constexpr int kDefaultRepetitions              = 5;
inline constexpr int kDefaultWarmup                   = 1;
inline constexpr std::uint32_t kDefaultPrefillChunk   = 1024;
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
inline constexpr std::uint32_t kMaxMtpDraftTokens     = 5;

enum class TestKind { Prefill, Decode, PrefillDecode };

// A pp test requests one output token because Engine::generate() deliberately performs no model
// work for a zero-token request. A tg or pp+tg test requests G+1 tokens: the first token belongs to
// Engine's begin/prefill round and the following G tokens are timed by GenerationTimings::decode.
struct BenchTest {
    TestKind kind = TestKind::Prefill;
    int n_prompt  = 0;
    int n_gen     = 0;
    std::string label;

    [[nodiscard]] bool has_prefill() const noexcept {
        return kind == TestKind::Prefill || kind == TestKind::PrefillDecode;
    }

    [[nodiscard]] bool has_decode() const noexcept {
        return kind == TestKind::Decode || kind == TestKind::PrefillDecode;
    }

    [[nodiscard]] std::uint32_t requested_output_tokens() const;
    [[nodiscard]] std::uint32_t required_context(std::uint32_t mtp_draft_tokens) const;
};

enum class OutputFormat { Table, Json, Csv };

struct BenchOptions {
    std::string artifact_path;
    std::string corpus_path{kDefaultCorpusPath};
    std::vector<int> n_prompt;
    std::vector<int> n_gen;
    std::vector<std::pair<int, int>> prompt_gen;
    int repetitions = kDefaultRepetitions;
    int warmup      = kDefaultWarmup;
    std::optional<std::uint32_t> max_context;
    std::uint32_t prefill_chunk    = kDefaultPrefillChunk;
    KvCacheStorage kv_cache        = KvCacheStorage::BFloat16;
    std::uint32_t mtp_draft_tokens = 0;
    ProposalHead proposal_head     = ProposalHead::Full;
    int device                     = 0;
    bool use_cuda_graph            = true;
    OutputFormat output            = OutputFormat::Table;
    std::string output_file;
    bool help_requested = false;
};

struct RepTiming {
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::uint32_t generated_output_tokens = 0;
};

struct TestResult {
    BenchTest test;
    std::vector<RepTiming> reps;
    std::size_t workspace_peak_bytes = 0;
};

struct Stats {
    double mean   = 0.0;
    double stddev = 0.0;
    int count     = 0;
};

struct BenchEnvironment {
    std::string gpu_name;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
    int device_id = 0;

    std::string artifact_path;
    std::uint64_t artifact_file_size_bytes = 0;
    LoadSummary load;
    MemorySummary memory;

    std::uint32_t max_context                      = 0;
    std::uint32_t prefill_chunk                    = kDefaultPrefillChunk;
    KvCacheStorage kv_cache                        = KvCacheStorage::BFloat16;
    std::uint32_t mtp_draft_tokens                 = 0;
    ProposalHead proposal_head                     = ProposalHead::Full;
    bool use_cuda_graph                            = true;
    bool decode_graph_primed                       = false;
    std::uint32_t decode_graph_prime_output_tokens = 0;
    int repetitions                                = 0;
    int warmup                                     = 0;
    std::string corpus_path;
    std::size_t corpus_tokens = 0;
};

BenchOptions parse_args(int argc, char** argv);
std::string usage_text(std::string_view program);

std::vector<BenchTest> expand_tests(const BenchOptions& options);
std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> override_max_context,
                                  std::uint32_t mtp_draft_tokens, bool use_cuda_graph);
void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens);

std::vector<TokenId> load_corpus_ids(const std::string& path);
std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt);
std::string decode_path_name(bool use_cuda_graph, std::uint32_t mtp_draft_tokens);
std::uint32_t decode_graph_prime_output_tokens(std::uint32_t mtp_draft_tokens);
std::uint32_t decode_graph_prime_required_context(std::uint32_t mtp_draft_tokens);

Stats compute_stats(const std::vector<double>& values);
std::vector<double> prefill_tok_s_series(const TestResult& result);
std::vector<double> decode_output_tok_s_series(const TestResult& result);
std::vector<double> decode_engine_tok_s_series(const TestResult& result);
std::vector<double> prepare_time_series(const TestResult& result);
std::vector<double> prefill_time_series(const TestResult& result);
std::vector<double> decode_time_series(const TestResult& result);
std::vector<double> total_time_series(const TestResult& result);

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results);
std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results);
std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results);

std::string json_escape(std::string_view value);
std::string kv_cache_name(KvCacheStorage storage);
std::string proposal_head_name(ProposalHead head);
std::uint64_t file_size_or_zero(const std::string& path);

} // namespace ninfer::bench
