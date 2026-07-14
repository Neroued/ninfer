#include "ninfer_bench_support.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace ninfer::bench {
namespace {

int parse_int(std::string_view text, const char* label) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    int value         = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

int parse_nonnegative(std::string_view text, const char* label) {
    const int value = parse_int(text, label);
    if (value < 0) { throw std::invalid_argument(std::string(label) + " must be nonnegative"); }
    return value;
}

int parse_positive(std::string_view text, const char* label) {
    const int value = parse_int(text, label);
    if (value <= 0) { throw std::invalid_argument(std::string(label) + " must be positive"); }
    return value;
}

std::uint32_t parse_u32(std::string_view text, const char* label, bool allow_zero = false) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    std::uint64_t value = 0;
    const auto result   = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value > std::numeric_limits<std::uint32_t>::max() || (!allow_zero && value == 0)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return static_cast<std::uint32_t>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    throw std::invalid_argument("--kv-dtype must be bf16 or int8");
}

std::vector<int> parse_int_list(std::string_view value, const char* label) {
    std::vector<int> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma      = value.find(',', start);
        const std::string_view piece = value.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        out.push_back(parse_positive(piece, label));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return out;
}

std::vector<std::pair<int, int>> parse_pair_list(std::string_view value, const char* label) {
    std::vector<std::pair<int, int>> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t semi       = value.find(';', start);
        const std::string_view piece = value.substr(
            start, semi == std::string_view::npos ? std::string_view::npos : semi - start);
        const std::size_t comma = piece.find(',');
        if (comma == std::string_view::npos ||
            piece.find(',', comma + 1) != std::string_view::npos) {
            throw std::invalid_argument(std::string(label) + " expects P,G pairs");
        }
        out.emplace_back(parse_positive(piece.substr(0, comma), "prompt-gen P"),
                         parse_positive(piece.substr(comma + 1), "prompt-gen G"));
        if (semi == std::string_view::npos) { break; }
        start = semi + 1;
    }
    return out;
}

std::string existing_read_path(const std::string& path) {
    if (std::filesystem::exists(path)) { return path; }
#ifdef NINFER_SOURCE_DIR
    const auto source_relative =
        (std::filesystem::path(NINFER_SOURCE_DIR) / path).lexically_normal();
    if (std::filesystem::exists(source_relative)) { return source_relative.string(); }
#endif
    return path;
}

const char* kind_string(TestKind kind) {
    switch (kind) {
    case TestKind::Prefill:
        return "pp";
    case TestKind::Decode:
        return "tg";
    case TestKind::PrefillDecode:
        return "pp+tg";
    }
    return "unknown";
}

std::uint32_t checked_context(std::uint64_t value, const char* label) {
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(label) + " exceeds the supported context range");
    }
    return static_cast<std::uint32_t>(value);
}

std::string number(double value) {
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string format_bytes(std::uint64_t bytes) {
    if (bytes == 0) { return "-"; }
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    constexpr double mib = 1024.0 * 1024.0;
    std::ostringstream out;
    if (static_cast<double>(bytes) >= gib) {
        out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / gib << " GiB";
    } else {
        out << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / mib << " MiB";
    }
    return out.str();
}

std::string rate_cell(const std::vector<double>& values) {
    if (values.empty()) { return "-"; }
    const Stats stats = compute_stats(values);
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << stats.mean;
    if (stats.count > 1) {
        out << " \u00b1 " << std::fixed << std::setprecision(2) << stats.stddev;
    }
    return out.str();
}

void append_stat(std::ostringstream& out, std::string_view name, const std::vector<double>& values,
                 std::string_view indent) {
    out << indent << '"' << name << "_mean\": ";
    if (values.empty()) {
        out << "null,\n" << indent << '"' << name << "_stddev\": null";
    } else {
        const Stats stats = compute_stats(values);
        out << number(stats.mean) << ",\n"
            << indent << '"' << name << "_stddev\": " << number(stats.stddev);
    }
}

SpeculativeStats aggregate_speculative(const TestResult& result) {
    SpeculativeStats out;
    for (const RepTiming& rep : result.reps) {
        const SpeculativeStats& in = rep.speculative;
        out.enabled                = out.enabled || in.enabled;
        out.draft_window           = std::max(out.draft_window, in.draft_window);
        out.rounds += in.rounds;
        out.drafted_tokens += in.drafted_tokens;
        out.accepted_tokens += in.accepted_tokens;
        out.fallback_steps += in.fallback_steps;
        if (out.accepted_per_position.size() < in.accepted_per_position.size()) {
            out.accepted_per_position.resize(in.accepted_per_position.size());
        }
        for (std::size_t i = 0; i < in.accepted_per_position.size(); ++i) {
            out.accepted_per_position[i] += in.accepted_per_position[i];
        }
    }
    return out;
}

std::uint64_t decode_engine_tokens(const TestResult& result, const RepTiming& rep) {
    if (!result.test.has_decode()) { return 0; }
    if (!rep.speculative.enabled) { return static_cast<std::uint64_t>(result.test.n_gen); }
    return rep.speculative.rounds + rep.speculative.accepted_tokens +
           rep.speculative.fallback_steps;
}

void append_arena_json(std::ostringstream& out, std::string_view name,
                       const ArenaMemorySummary& arena, std::string_view indent,
                       bool trailing_comma) {
    out << indent << '"' << name << "\": {\"capacity_bytes\": " << arena.capacity_bytes
        << ", \"used_bytes\": " << arena.used_bytes
        << ", \"peak_used_bytes\": " << arena.peak_used_bytes << '}';
    if (trailing_comma) { out << ','; }
    out << '\n';
}

void append_speculative_json(std::ostringstream& out, const SpeculativeStats& stats,
                             std::string_view indent) {
    out << indent << "\"speculative\": {\n"
        << indent << "  \"enabled\": " << (stats.enabled ? "true" : "false") << ",\n"
        << indent << "  \"draft_window\": " << stats.draft_window << ",\n"
        << indent << "  \"rounds\": " << stats.rounds << ",\n"
        << indent << "  \"drafted_tokens\": " << stats.drafted_tokens << ",\n"
        << indent << "  \"accepted_tokens\": " << stats.accepted_tokens << ",\n"
        << indent << "  \"fallback_steps\": " << stats.fallback_steps << ",\n"
        << indent << "  \"acceptance_rate\": ";
    if (stats.drafted_tokens == 0) {
        out << "null";
    } else {
        out << number(static_cast<double>(stats.accepted_tokens) /
                      static_cast<double>(stats.drafted_tokens));
    }
    out << ",\n" << indent << "  \"acceptance_length\": ";
    if (stats.rounds == 0) {
        out << "null";
    } else {
        out << number(1.0 + static_cast<double>(stats.accepted_tokens) /
                                static_cast<double>(stats.rounds));
    }
    out << ",\n" << indent << "  \"accepted_per_position\": [";
    for (std::size_t i = 0; i < stats.accepted_per_position.size(); ++i) {
        if (i != 0) { out << ", "; }
        out << stats.accepted_per_position[i];
    }
    out << "]\n" << indent << '}';
}

void append_timings_json(std::ostringstream& out, const GenerationTimings& timings,
                         std::string_view indent) {
    out << indent << "\"timings\": {" << "\"prepare_seconds\": " << number(timings.prepare_seconds)
        << ", " << "\"vision_seconds\": " << number(timings.vision_seconds) << ", "
        << "\"prefill_seconds\": " << number(timings.prefill_seconds) << ", "
        << "\"decode_seconds\": " << number(timings.decode_seconds) << ", "
        << "\"total_seconds\": " << number(timings.total_seconds) << '}';
}

} // namespace

std::uint32_t BenchTest::requested_output_tokens() const {
    return checked_context(static_cast<std::uint64_t>(has_decode() ? n_gen : 0) + 1ULL,
                           "requested benchmark output");
}

std::uint32_t BenchTest::required_context(std::uint32_t mtp_draft_tokens) const {
    const std::uint64_t prompt =
        static_cast<std::uint64_t>(kind == TestKind::Decode ? kDecodeSeedTokens : n_prompt);
    const std::uint64_t decode     = static_cast<std::uint64_t>(has_decode() ? n_gen : 0);
    const std::uint64_t mtp_margin = mtp_draft_tokens == 0 ? 0 : 2ULL * mtp_draft_tokens;
    return checked_context(prompt + decode + mtp_margin, "benchmark context requirement");
}

std::string usage_text(std::string_view program) {
    if (program.empty()) { program = "ninfer_bench"; }
    std::ostringstream out;
    out << "Usage: " << program << " --weights <artifact.ninfer> [options]\n\n"
        << "Product-route throughput benchmark over ninfer::Engine. pp measures Engine prefill;\n"
        << "tg measures G generated tokens after an untimed one-token seed prefill.\n\n"
        << "Options:\n"
        << "  --weights <path>            required .ninfer artifact\n"
        << "  --corpus <path>             token-id corpus (default: " << kDefaultCorpusPath << ")\n"
        << "  -p, --n-prompt <list>       pp lengths, for example 512,2048\n"
        << "  -n, --n-gen <list>          tg lengths, for example 128\n"
        << "  -pg, --prompt-gen <P,G;..>  combined pp+tg tests\n"
        << "  -r, --repetitions <n>       measured repetitions (default: " << kDefaultRepetitions
        << ")\n"
        << "  --warmup <n>                discarded repetitions (default: " << kDefaultWarmup
        << ")\n"
        << "  --max-ctx <tokens>          override auto-sized context capacity\n"
        << "  --prefill-chunk <tokens>    multiple of " << kPrefillChunkAlignment
        << " (default: " << kDefaultPrefillChunk << ")\n"
        << "  --kv-dtype <bf16|int8>      KV cache storage (default: bf16)\n"
        << "  --mtp-draft-tokens <0..5>   speculative draft window (default: 0)\n"
        << "  --lm-head-draft             use the optimized proposal head; requires MTP\n"
        << "  --device <id>               CUDA device ordinal (default: 0)\n"
        << "  --no-cuda-graph             use eager decode\n"
        << "  -o, --output <table|json|csv>  output format (default: table)\n"
        << "  --output-file <path>        write report to a file\n"
        << "  -h, --help                  show this help\n\n"
        << "With no -p/-n/-pg, the matrix is pp" << kDefaultNPrompt << " and tg" << kDefaultNGen
        << ".\n";
    return out.str();
}

BenchOptions parse_args(int argc, char** argv) {
    BenchOptions options;
    bool saw_artifact = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(flag) + " requires value");
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            options.help_requested = true;
            return options;
        }
        if (arg == "--weights") {
            options.artifact_path = value("--weights");
            saw_artifact          = true;
        } else if (arg == "--corpus") {
            options.corpus_path = value("--corpus");
        } else if (arg == "-p" || arg == "--n-prompt") {
            auto parsed = parse_int_list(value("--n-prompt"), "n-prompt");
            options.n_prompt.insert(options.n_prompt.end(), parsed.begin(), parsed.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            auto parsed = parse_int_list(value("--n-gen"), "n-gen");
            options.n_gen.insert(options.n_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-pg" || arg == "--prompt-gen") {
            auto parsed = parse_pair_list(value("--prompt-gen"), "prompt-gen");
            options.prompt_gen.insert(options.prompt_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-r" || arg == "--repetitions") {
            options.repetitions = parse_positive(value("--repetitions"), "repetitions");
        } else if (arg == "--warmup") {
            options.warmup = parse_nonnegative(value("--warmup"), "warmup");
        } else if (arg == "--max-ctx") {
            options.max_context = parse_u32(value("--max-ctx"), "max-ctx");
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value("--prefill-chunk"), "prefill-chunk");
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value("--kv-dtype"));
        } else if (arg == "--mtp-draft-tokens") {
            options.mtp_draft_tokens =
                parse_u32(value("--mtp-draft-tokens"), "mtp-draft-tokens", true);
            if (options.mtp_draft_tokens > kMaxMtpDraftTokens) {
                throw std::invalid_argument("--mtp-draft-tokens must be in [0,5]");
            }
        } else if (arg == "--lm-head-draft") {
            options.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--device") {
            options.device = parse_nonnegative(value("--device"), "device");
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "-o" || arg == "--output") {
            const std::string selected = value("--output");
            if (selected == "table") {
                options.output = OutputFormat::Table;
            } else if (selected == "json") {
                options.output = OutputFormat::Json;
            } else if (selected == "csv") {
                options.output = OutputFormat::Csv;
            } else {
                throw std::invalid_argument("--output must be table, json, or csv");
            }
        } else if (arg == "--output-file") {
            options.output_file = value("--output-file");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!saw_artifact) { throw std::invalid_argument("--weights is required"); }
    if (options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.proposal_head == ProposalHead::Optimized && options.mtp_draft_tokens == 0) {
        throw std::invalid_argument(
            "--lm-head-draft requires --mtp-draft-tokens greater than zero");
    }
    return options;
}

std::vector<BenchTest> expand_tests(const BenchOptions& options) {
    std::vector<int> prompts = options.n_prompt;
    std::vector<int> gens    = options.n_gen;
    auto combined            = options.prompt_gen;
    if (prompts.empty() && gens.empty() && combined.empty()) {
        prompts.push_back(kDefaultNPrompt);
        gens.push_back(kDefaultNGen);
    }

    std::vector<BenchTest> tests;
    for (const int p : prompts) {
        tests.push_back({TestKind::Prefill, p, 0, "pp" + std::to_string(p)});
    }
    for (const int g : gens) {
        tests.push_back({TestKind::Decode, 0, g, "tg" + std::to_string(g)});
    }
    for (const auto& [p, g] : combined) {
        tests.push_back(
            {TestKind::PrefillDecode, p, g, "pp" + std::to_string(p) + "+tg" + std::to_string(g)});
    }
    return tests;
}

std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> override_max_context,
                                  std::uint32_t mtp_draft_tokens, bool use_cuda_graph) {
    if (mtp_draft_tokens > kMaxMtpDraftTokens) {
        throw std::invalid_argument("mtp draft window must be in [0,5]");
    }
    std::uint32_t required = 0;
    std::string driver;
    bool has_decode = false;
    for (const BenchTest& test : tests) {
        const std::uint32_t candidate = test.required_context(mtp_draft_tokens);
        if (candidate > required) {
            required = candidate;
            driver   = test.label;
        }
        has_decode = has_decode || test.has_decode();
    }
    if (use_cuda_graph && has_decode) {
        const std::uint32_t candidate = decode_graph_prime_required_context(mtp_draft_tokens);
        if (candidate > required) {
            required = candidate;
            driver   = "decode graph prime";
        }
    }
    if (required == 0) { throw std::invalid_argument("no benchmark tests were selected"); }
    if (override_max_context && *override_max_context < required) {
        throw std::invalid_argument("--max-ctx " + std::to_string(*override_max_context) +
                                    " is too small; " + driver + " needs " +
                                    std::to_string(required));
    }
    return override_max_context.value_or(required);
}

void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens) {
    if (corpus_tokens < static_cast<std::size_t>(kDecodeSeedTokens)) {
        throw std::invalid_argument("corpus is too small to seed decode tests");
    }
    for (const BenchTest& test : tests) {
        if (test.has_prefill() && static_cast<std::size_t>(test.n_prompt) > corpus_tokens) {
            throw std::invalid_argument(test.label + " exceeds the token-id corpus");
        }
    }
}

std::vector<TokenId> load_corpus_ids(const std::string& path) {
    std::ifstream input(existing_read_path(path));
    if (!input) { throw std::runtime_error("failed to open corpus ids file: " + path); }
    std::vector<TokenId> ids;
    std::string word;
    while (input >> word) {
        const int parsed = parse_nonnegative(word, "corpus token id");
        ids.push_back(static_cast<TokenId>(parsed));
    }
    if (ids.empty()) { throw std::invalid_argument("corpus ids file is empty: " + path); }
    return ids;
}

std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt) {
    if (n_prompt <= 0 || static_cast<std::size_t>(n_prompt) > corpus.size()) {
        throw std::invalid_argument("invalid prompt slice length: " + std::to_string(n_prompt));
    }
    return {corpus.begin(), corpus.begin() + n_prompt};
}

std::string decode_path_name(bool use_cuda_graph, std::uint32_t mtp_draft_tokens) {
    if (mtp_draft_tokens != 0) { return use_cuda_graph ? "mtp_cuda_graph" : "mtp_eager"; }
    return use_cuda_graph ? "cuda_graph" : "eager";
}

std::uint32_t decode_graph_prime_output_tokens(std::uint32_t mtp_draft_tokens) {
    if (mtp_draft_tokens > kMaxMtpDraftTokens) {
        throw std::invalid_argument("mtp draft window must be in [0,5]");
    }
    return mtp_draft_tokens == 0 ? 3 : 2 * (mtp_draft_tokens + 1) + 1;
}

std::uint32_t decode_graph_prime_required_context(std::uint32_t mtp_draft_tokens) {
    const std::uint64_t outputs = decode_graph_prime_output_tokens(mtp_draft_tokens);
    return checked_context(outputs + (mtp_draft_tokens == 0 ? 0 : 2ULL * mtp_draft_tokens),
                           "decode graph prime context requirement");
}

Stats compute_stats(const std::vector<double>& values) {
    Stats out;
    out.count = static_cast<int>(values.size());
    if (values.empty()) { return out; }
    for (const double value : values) { out.mean += value; }
    out.mean /= static_cast<double>(values.size());
    if (values.size() > 1) {
        for (const double value : values) { out.stddev += (value - out.mean) * (value - out.mean); }
        out.stddev = std::sqrt(out.stddev / static_cast<double>(values.size() - 1));
    }
    return out;
}

std::vector<double> prefill_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_prefill()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.prefill_seconds > 0.0) {
            out.push_back(static_cast<double>(result.test.n_prompt) / rep.timings.prefill_seconds);
        }
    }
    return out;
}

std::vector<double> decode_output_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.decode_seconds > 0.0) {
            out.push_back(static_cast<double>(result.test.n_gen) / rep.timings.decode_seconds);
        }
    }
    return out;
}

std::vector<double> decode_engine_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.decode_seconds > 0.0) {
            out.push_back(static_cast<double>(decode_engine_tokens(result, rep)) /
                          rep.timings.decode_seconds);
        }
    }
    return out;
}

template <double GenerationTimings::*Field>
std::vector<double> timing_series(const TestResult& result) {
    std::vector<double> out;
    out.reserve(result.reps.size());
    for (const RepTiming& rep : result.reps) { out.push_back(rep.timings.*Field); }
    return out;
}

std::vector<double> prepare_time_series(const TestResult& result) {
    return timing_series<&GenerationTimings::prepare_seconds>(result);
}

std::vector<double> prefill_time_series(const TestResult& result) {
    return result.test.has_prefill() ? timing_series<&GenerationTimings::prefill_seconds>(result)
                                     : std::vector<double>{};
}

std::vector<double> decode_time_series(const TestResult& result) {
    return result.test.has_decode() ? timing_series<&GenerationTimings::decode_seconds>(result)
                                    : std::vector<double>{};
}

std::vector<double> total_time_series(const TestResult& result) {
    return timing_series<&GenerationTimings::total_seconds>(result);
}

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "ninfer_bench product throughput report\n"
        << "  target:     " << env.load.target << '\n'
        << "  gpu:        " << env.gpu_name << " (device " << env.device_id << ")\n"
        << "  cuda:       runtime " << env.cuda_runtime_version << ", driver "
        << env.cuda_driver_version << '\n'
        << "  artifact:   " << env.artifact_path << " (" << env.artifact_file_size_bytes
        << " bytes)\n"
        << "  load:       " << env.load.load_seconds << " s, upload " << env.load.upload_seconds
        << " s, read " << format_bytes(env.load.artifact_bytes_read) << ", H2D "
        << format_bytes(env.load.host_to_device_bytes) << ", staging peak "
        << format_bytes(env.load.peak_staging_bytes) << '\n'
        << "  memory:     weights " << format_bytes(env.memory.weights.capacity_bytes)
        << ", sequence " << format_bytes(env.memory.sequence.capacity_bytes) << ", workspace "
        << format_bytes(env.memory.workspace.capacity_bytes) << ", KV payload "
        << format_bytes(env.memory.kv_payload_bytes) << '\n'
        << "  corpus:     " << env.corpus_path << " (" << env.corpus_tokens << " tokens)\n"
        << "  config:     max_context=" << env.max_context << " prefill_chunk=" << env.prefill_chunk
        << " kv_cache=" << kv_cache_name(env.kv_cache) << " mtp_k=" << env.mtp_draft_tokens
        << " proposal_head=" << proposal_head_name(env.proposal_head)
        << " decode_path=" << decode_path_name(env.use_cuda_graph, env.mtp_draft_tokens)
        << " graph_prime="
        << (env.decode_graph_primed
                ? std::to_string(env.decode_graph_prime_output_tokens) + " outputs"
                : "n/a")
        << " repetitions=" << env.repetitions << " warmup=" << env.warmup << "\n\n";

    constexpr std::size_t cols                   = 9;
    const std::array<std::string, cols> headings = {
        "test",           "n_prompt", "n_gen",         "prefill t/s", "decode out t/s",
        "decode eng t/s", "spec acc", "spec round/fb", "work peak"};
    std::vector<std::array<std::string, cols>> rows;
    for (const TestResult& result : results) {
        const SpeculativeStats spec  = aggregate_speculative(result);
        const std::string acceptance = spec.drafted_tokens == 0
                                           ? "n/a"
                                           : number(static_cast<double>(spec.accepted_tokens) /
                                                    static_cast<double>(spec.drafted_tokens));
        const std::string rounds =
            spec.enabled ? std::to_string(spec.rounds) + "/" + std::to_string(spec.fallback_steps)
                         : "n/a";
        rows.push_back({result.test.label, std::to_string(result.test.n_prompt),
                        std::to_string(result.test.n_gen), rate_cell(prefill_tok_s_series(result)),
                        rate_cell(decode_output_tok_s_series(result)),
                        rate_cell(decode_engine_tok_s_series(result)), acceptance, rounds,
                        format_bytes(result.workspace_peak_bytes)});
    }

    std::array<std::size_t, cols> widths{};
    for (std::size_t i = 0; i < cols; ++i) { widths[i] = headings[i].size(); }
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < cols; ++i) { widths[i] = std::max(widths[i], row[i].size()); }
    }
    const auto print_row = [&](const auto& row) {
        for (std::size_t i = 0; i < cols; ++i) {
            if (i != 0) { out << "  "; }
            out << (i == 0 ? std::left : std::right) << std::setw(static_cast<int>(widths[i]))
                << row[i];
        }
        out << '\n';
    };
    print_row(headings);
    for (const auto& row : rows) { print_row(row); }
    return out.str();
}

std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": " << kSchemaVersion << ",\n"
        << "  \"artifact_type\": \"" << kArtifactType << "\",\n"
        << "  \"tool\": \"ninfer_bench\",\n"
        << "  \"command\": \"" << json_escape(command) << "\",\n"
        << "  \"environment\": {\"gpu_name\": \"" << json_escape(env.gpu_name)
        << "\", \"cuda_runtime_version\": \"" << json_escape(env.cuda_runtime_version)
        << "\", \"cuda_driver_version\": \"" << json_escape(env.cuda_driver_version)
        << "\", \"device_id\": " << env.device_id << "},\n"
        << "  \"artifact\": {\"path\": \"" << json_escape(env.artifact_path)
        << "\", \"file_size_bytes\": " << env.artifact_file_size_bytes << "},\n"
        << "  \"load\": {\n"
        << "    \"target\": \"" << json_escape(env.load.target) << "\",\n"
        << "    \"load_seconds\": " << number(env.load.load_seconds) << ",\n"
        << "    \"upload_seconds\": " << number(env.load.upload_seconds) << ",\n"
        << "    \"artifact_bytes_read\": " << env.load.artifact_bytes_read << ",\n"
        << "    \"host_to_device_bytes\": " << env.load.host_to_device_bytes << ",\n"
        << "    \"peak_staging_bytes\": " << env.load.peak_staging_bytes << ",\n"
        << "    \"tensor_count\": " << env.load.tensor_count << ",\n"
        << "    \"resource_count\": " << env.load.resource_count << "\n"
        << "  },\n"
        << "  \"memory\": {\n"
        << "    \"device\": " << env.memory.device << ",\n"
        << "    \"max_context\": " << env.memory.max_context << ",\n"
        << "    \"kv_cache\": \"" << kv_cache_name(env.memory.kv_cache) << "\",\n";
    append_arena_json(out, "weights", env.memory.weights, "    ", true);
    append_arena_json(out, "sequence", env.memory.sequence, "    ", true);
    append_arena_json(out, "workspace", env.memory.workspace, "    ", true);
    out << "    \"kv_payload_bytes\": " << env.memory.kv_payload_bytes << "\n"
        << "  },\n"
        << "  \"config\": {\n"
        << "    \"max_context\": " << env.max_context << ",\n"
        << "    \"prefill_chunk\": " << env.prefill_chunk << ",\n"
        << "    \"kv_cache\": \"" << kv_cache_name(env.kv_cache) << "\",\n"
        << "    \"mtp_draft_tokens\": " << env.mtp_draft_tokens << ",\n"
        << "    \"proposal_head\": \"" << proposal_head_name(env.proposal_head) << "\",\n"
        << "    \"use_cuda_graph\": " << (env.use_cuda_graph ? "true" : "false") << ",\n"
        << "    \"decode_path\": \"" << decode_path_name(env.use_cuda_graph, env.mtp_draft_tokens)
        << "\",\n"
        << "    \"decode_graph_prime\": {\"primed\": "
        << (env.decode_graph_primed ? "true" : "false")
        << ", \"output_tokens\": " << env.decode_graph_prime_output_tokens << "},\n"
        << "    \"repetitions\": " << env.repetitions << ",\n"
        << "    \"warmup\": " << env.warmup << ",\n"
        << "    \"corpus_path\": \"" << json_escape(env.corpus_path) << "\",\n"
        << "    \"corpus_tokens\": " << env.corpus_tokens << "\n"
        << "  },\n"
        << "  \"tests\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const TestResult& result = results[i];
        out << "    {\n"
            << "      \"label\": \"" << json_escape(result.test.label) << "\",\n"
            << "      \"kind\": \"" << kind_string(result.test.kind) << "\",\n"
            << "      \"n_prompt\": " << result.test.n_prompt << ",\n"
            << "      \"n_gen\": " << result.test.n_gen << ",\n"
            << "      \"requested_output_tokens\": " << result.test.requested_output_tokens()
            << ",\n";
        append_stat(out, "prefill_tok_s", prefill_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_output_tok_s", decode_output_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_engine_tok_s", decode_engine_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "prepare_seconds", prepare_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "prefill_seconds", prefill_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_seconds", decode_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "total_seconds", total_time_series(result), "      ");
        out << ",\n      \"workspace_peak_bytes\": " << result.workspace_peak_bytes << ",\n";
        append_speculative_json(out, aggregate_speculative(result), "      ");
        out << ",\n      \"reps\": [\n";
        for (std::size_t r = 0; r < result.reps.size(); ++r) {
            const RepTiming& rep = result.reps[r];
            out << "        {\n"
                << "          \"generated_output_tokens\": " << rep.generated_output_tokens << ",\n"
                << "          \"decode_output_tokens\": "
                << (result.test.has_decode() ? std::to_string(result.test.n_gen) : "null") << ",\n"
                << "          \"decode_engine_tokens\": "
                << (result.test.has_decode() ? std::to_string(decode_engine_tokens(result, rep))
                                             : "null")
                << ",\n";
            append_timings_json(out, rep.timings, "          ");
            out << ",\n";
            append_speculative_json(out, rep.speculative, "          ");
            out << "\n        }" << (r + 1 == result.reps.size() ? "" : ",") << '\n';
        }
        out << "      ]\n    }" << (i + 1 == results.size() ? "" : ",") << '\n';
    }
    out << "  ]\n}\n";
    return out.str();
}

std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "label,kind,n_prompt,n_gen,target,max_context,prefill_chunk,mtp_draft_tokens,"
           "proposal_head,decode_path,kv_cache,kv_payload_bytes,load_host_to_device_bytes,"
           "weights_capacity_bytes,sequence_capacity_bytes,workspace_capacity_bytes,"
           "workspace_peak_bytes,spec_rounds,spec_fallback_steps,spec_acceptance_rate,"
           "repetitions,prefill_tok_s_mean,prefill_tok_s_stddev,decode_output_tok_s_mean,"
           "decode_output_tok_s_stddev,decode_engine_tok_s_mean,decode_engine_tok_s_stddev,"
           "prepare_seconds_mean,prefill_seconds_mean,decode_seconds_mean,total_seconds_mean\n";
    const auto mean = [](const std::vector<double>& values) {
        return values.empty() ? std::string() : number(compute_stats(values).mean);
    };
    const auto stddev = [](const std::vector<double>& values) {
        return values.empty() ? std::string() : number(compute_stats(values).stddev);
    };
    for (const TestResult& result : results) {
        const SpeculativeStats spec  = aggregate_speculative(result);
        const std::string acceptance = spec.drafted_tokens == 0
                                           ? std::string()
                                           : number(static_cast<double>(spec.accepted_tokens) /
                                                    static_cast<double>(spec.drafted_tokens));
        out << result.test.label << ',' << kind_string(result.test.kind) << ','
            << result.test.n_prompt << ',' << result.test.n_gen << ',' << env.load.target << ','
            << env.max_context << ',' << env.prefill_chunk << ',' << env.mtp_draft_tokens << ','
            << proposal_head_name(env.proposal_head) << ','
            << decode_path_name(env.use_cuda_graph, env.mtp_draft_tokens) << ','
            << kv_cache_name(env.kv_cache) << ',' << env.memory.kv_payload_bytes << ','
            << env.load.host_to_device_bytes << ',' << env.memory.weights.capacity_bytes << ','
            << env.memory.sequence.capacity_bytes << ',' << env.memory.workspace.capacity_bytes
            << ',' << result.workspace_peak_bytes << ',' << spec.rounds << ','
            << spec.fallback_steps << ',' << acceptance << ',' << result.reps.size() << ','
            << mean(prefill_tok_s_series(result)) << ',' << stddev(prefill_tok_s_series(result))
            << ',' << mean(decode_output_tok_s_series(result)) << ','
            << stddev(decode_output_tok_s_series(result)) << ','
            << mean(decode_engine_tok_s_series(result)) << ','
            << stddev(decode_engine_tok_s_series(result)) << ','
            << mean(prepare_time_series(result)) << ',' << mean(prefill_time_series(result)) << ','
            << mean(decode_time_series(result)) << ',' << mean(total_time_series(result)) << '\n';
    }
    return out.str();
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
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                std::ostringstream escaped;
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                out += escaped.str();
            } else {
                out += c;
            }
        }
    }
    return out;
}

std::string kv_cache_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-group64";
    }
    return "unknown";
}

std::string proposal_head_name(ProposalHead head) {
    switch (head) {
    case ProposalHead::Full:
        return "full";
    case ProposalHead::Optimized:
        return "optimized";
    }
    return "unknown";
}

std::uint64_t file_size_or_zero(const std::string& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : static_cast<std::uint64_t>(size);
}

} // namespace ninfer::bench
