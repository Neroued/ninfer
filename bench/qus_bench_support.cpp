#include "qus_bench_support.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace qus::bench {
namespace {

int parse_int(std::string_view text, const char* label) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    int value         = 0;
    const auto* first = text.data();
    const auto* last  = text.data() + text.size();
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) {
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

std::size_t parse_positive_u64(std::string_view text, const char* label) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    unsigned long long value = 0;
    const auto* first        = text.data();
    const auto* last         = text.data() + text.size();
    const auto result        = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last || value == 0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return static_cast<std::size_t>(value);
}

int parse_mtp_draft_tokens(std::string_view text) {
    const int value = parse_nonnegative(text, "mtp-draft-tokens");
    if (value > model::kMaxMtpDraftTokens) {
        throw std::invalid_argument("--mtp-draft-tokens must be in [0,5]");
    }
    return value;
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
        const int p = parse_positive(piece.substr(0, comma), "prompt-gen P");
        const int g = parse_positive(piece.substr(comma + 1), "prompt-gen G");
        out.emplace_back(p, g);
        if (semi == std::string_view::npos) { break; }
        start = semi + 1;
    }
    return out;
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

std::string existing_read_path(const std::string& path) {
    if (std::filesystem::exists(path)) { return path; }
#ifdef QUS_SOURCE_DIR
    const std::filesystem::path source_relative =
        (std::filesystem::path(QUS_SOURCE_DIR) / std::filesystem::path(path)).lexically_normal();
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

std::string json_number(double value) {
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::uint32_t checked_context_u32(std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(std::string(label) + " exceeds uint32");
    }
    return static_cast<std::uint32_t>(value);
}

std::string format_bytes(std::size_t bytes) {
    if (bytes == 0) { return "-"; }
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    constexpr double mib = 1024.0 * 1024.0;
    std::ostringstream out;
    if (static_cast<double>(bytes) >= gib) {
        out << std::fixed << std::setprecision(2) << (static_cast<double>(bytes) / gib) << " GiB";
    } else {
        out << std::fixed << std::setprecision(1) << (static_cast<double>(bytes) / mib) << " MiB";
    }
    return out.str();
}

std::string rate_cell(const std::vector<double>& series) {
    if (series.empty()) { return "-"; }
    const Stats stats = compute_stats(series);
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << stats.mean;
    if (stats.count >= 2) {
        out << " \u00b1 " << std::fixed << std::setprecision(2) << stats.stddev;
    }
    return out.str();
}

void append_stat_fields(std::ostringstream& out, const std::string& prefix,
                        const std::vector<double>& series, const std::string& indent) {
    if (series.empty()) {
        out << indent << "\"" << prefix << "_mean\": null,\n";
        out << indent << "\"" << prefix << "_stddev\": null";
        return;
    }
    const Stats stats = compute_stats(series);
    out << indent << "\"" << prefix << "_mean\": " << json_number(stats.mean) << ",\n";
    out << indent << "\"" << prefix << "_stddev\": " << json_number(stats.stddev);
}

} // namespace

std::uint32_t BenchTest::required_context(int mtp_draft_tokens) const {
    const int k = std::max(0, mtp_draft_tokens);
    switch (kind) {
    case TestKind::Prefill:
        return checked_context_u32(static_cast<std::uint64_t>(n_prompt) +
                                       static_cast<std::uint64_t>(k > 0 ? k - 1 : 0),
                                   "prefill context requirement");
    case TestKind::Decode:
        return checked_context_u32(static_cast<std::uint64_t>(kDecodeSeedTokens) +
                                       static_cast<std::uint64_t>(n_gen) +
                                       static_cast<std::uint64_t>(k > 0 ? 2 * k : 0),
                                   "decode context requirement");
    case TestKind::PrefillDecode:
        return checked_context_u32(static_cast<std::uint64_t>(n_prompt) +
                                       static_cast<std::uint64_t>(n_gen) +
                                       static_cast<std::uint64_t>(k > 0 ? 2 * k : 0),
                                   "prefill-decode context requirement");
    }
    return 0;
}

std::string usage_text(std::string_view program) {
    if (program.empty()) { program = "qus_bench"; }
    std::ostringstream out;
    out << "Usage: " << program << " --weights <q5090-path> [options]\n"
        << "\n"
        << "llama-bench-style throughput benchmark. Prefill (pp) and decode (tg) rates are\n"
        << "measured separately using meaningful tokens sliced from a pre-baked .ids corpus.\n"
        << "\n"
        << "Options:\n"
        << "  --weights <path>            required q5090 weights path\n"
        << "  --corpus <path>             meaningful .ids corpus (default: " << kDefaultCorpusPath
        << ")\n"
        << "  -p, --n-prompt <list>       prefill lengths, comma list; adds pp{P} tests\n"
        << "  -n, --n-gen <list>          decode lengths, comma list; adds tg{G} tests\n"
        << "  -pg, --prompt-gen <P,G;..>  combined pp{P}+tg{G} tests\n"
        << "  -r, --repetitions <n>       measured repetitions (default: " << kDefaultRepetitions
        << ")\n"
        << "  --warmup <n>                warmup repetitions, discarded (default: "
        << kDefaultWarmup << ")\n"
        << "  --max-ctx <tokens>          override auto-sized max context\n"
        << "  --prefill-chunk <tokens>    prefill ubatch size, multiple of 128 (default: "
        << model::kDefaultPrefillChunk << ")\n"
        << "  --work-bytes <bytes>        explicit workspace arena override\n"
        << "  --device <id>               CUDA device ordinal (default: 0)\n"
        << "  --no-cuda-graph             disable CUDA graph decode (decode_path=eager or "
           "mtp_eager)\n"
        << "  --mtp-draft-tokens <0..5>   enable MTP draft rounds (default: 0)\n"
        << "  -o, --output <table|json|csv>  output format (default: table)\n"
        << "  --output-file <path>        write output to a file instead of stdout\n"
        << "  -h, --help                  show this help\n"
        << "\n"
        << "With no -p/-n/-pg, defaults to pp" << kDefaultNPrompt << " and tg" << kDefaultNGen
        << ".\n";
    return out.str();
}

BenchOptions parse_args(int argc, char** argv) {
    BenchOptions options;
    bool saw_weights = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(flag) + " requires value");
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            options.help_requested = true;
            return options;
        } else if (arg == "--weights") {
            options.weights_path = require_value("--weights");
            saw_weights          = true;
        } else if (arg == "--corpus") {
            options.corpus_path = require_value("--corpus");
        } else if (arg == "-p" || arg == "--n-prompt") {
            const std::vector<int> parsed = parse_int_list(require_value("--n-prompt"), "n-prompt");
            options.n_prompt.insert(options.n_prompt.end(), parsed.begin(), parsed.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            const std::vector<int> parsed = parse_int_list(require_value("--n-gen"), "n-gen");
            options.n_gen.insert(options.n_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-pg" || arg == "--prompt-gen") {
            const std::vector<std::pair<int, int>> parsed =
                parse_pair_list(require_value("--prompt-gen"), "prompt-gen");
            options.prompt_gen.insert(options.prompt_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-r" || arg == "--repetitions") {
            options.repetitions = parse_positive(require_value("--repetitions"), "repetitions");
        } else if (arg == "--warmup") {
            options.warmup = parse_nonnegative(require_value("--warmup"), "warmup");
        } else if (arg == "--max-ctx") {
            options.max_ctx =
                static_cast<std::uint32_t>(parse_positive(require_value("--max-ctx"), "max-ctx"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_positive(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--work-bytes") {
            options.work_bytes = parse_positive_u64(require_value("--work-bytes"), "work-bytes");
        } else if (arg == "--device") {
            options.device = parse_nonnegative(require_value("--device"), "device");
        } else if (arg == "--mtp-draft-tokens") {
            options.mtp_draft_tokens = parse_mtp_draft_tokens(require_value("--mtp-draft-tokens"));
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "-o" || arg == "--output") {
            const std::string value = require_value("--output");
            if (value == "table") {
                options.output = OutputFormat::Table;
            } else if (value == "json") {
                options.output = OutputFormat::Json;
            } else if (value == "csv") {
                options.output = OutputFormat::Csv;
            } else {
                throw std::invalid_argument("--output must be table, json, or csv");
            }
        } else if (arg == "--output-file") {
            options.output_file = require_value("--output-file");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!saw_weights) { throw std::invalid_argument("--weights is required"); }
    if (options.prefill_chunk % model::kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    return options;
}

std::vector<BenchTest> expand_tests(const BenchOptions& options) {
    std::vector<BenchTest> tests;
    std::vector<int> prompts                  = options.n_prompt;
    std::vector<int> gens                     = options.n_gen;
    std::vector<std::pair<int, int>> combined = options.prompt_gen;
    if (prompts.empty() && gens.empty() && combined.empty()) {
        prompts.push_back(kDefaultNPrompt);
        gens.push_back(kDefaultNGen);
    }
    for (const int p : prompts) {
        if (p <= 0) { throw std::invalid_argument("n-prompt must be positive"); }
        tests.push_back(BenchTest{TestKind::Prefill, p, 0, "pp" + std::to_string(p)});
    }
    for (const int g : gens) {
        if (g <= 0) { throw std::invalid_argument("n-gen must be positive"); }
        tests.push_back(BenchTest{TestKind::Decode, 0, g, "tg" + std::to_string(g)});
    }
    for (const auto& [p, g] : combined) {
        if (p <= 0 || g <= 0) { throw std::invalid_argument("prompt-gen P,G must be positive"); }
        tests.push_back(BenchTest{TestKind::PrefillDecode, p, g,
                                  "pp" + std::to_string(p) + "+tg" + std::to_string(g)});
    }
    return tests;
}

std::uint32_t resolve_max_ctx(const std::vector<BenchTest>& tests,
                              std::optional<std::uint32_t> override_max_ctx, int mtp_draft_tokens,
                              bool use_cuda_graph) {
    if (mtp_draft_tokens < 0 || mtp_draft_tokens > model::kMaxMtpDraftTokens) {
        throw std::invalid_argument("mtp-draft-tokens must be in [0,5]");
    }
    std::uint32_t required = 0;
    std::string driver;
    bool has_decode = false;
    for (const BenchTest& test : tests) {
        const std::uint32_t need = test.required_context(mtp_draft_tokens);
        if (need > required) {
            required = need;
            driver   = test.label;
        }
        has_decode = has_decode || test.has_decode();
    }
    if (use_cuda_graph && has_decode) {
        const std::uint32_t prime_need = decode_graph_prime_required_context(mtp_draft_tokens);
        if (prime_need > required) {
            required = prime_need;
            driver   = "decode graph prime";
        }
    }
    if (required == 0) { throw std::invalid_argument("no tests to size max_ctx for"); }
    if (override_max_ctx.has_value()) {
        if (*override_max_ctx < required) {
            throw std::invalid_argument("--max-ctx " + std::to_string(*override_max_ctx) +
                                        " is too small; test " +
                                        (driver.empty() ? std::string("?") : driver) + " needs " +
                                        std::to_string(required));
        }
        return *override_max_ctx;
    }
    return required;
}

void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens) {
    if (corpus_tokens < static_cast<std::size_t>(kDecodeSeedTokens)) {
        throw std::invalid_argument("corpus is too small to seed decode tests");
    }
    for (const BenchTest& test : tests) {
        if (test.has_prefill() && static_cast<std::size_t>(test.n_prompt) > corpus_tokens) {
            throw std::invalid_argument("test " + test.label + " prefill length " +
                                        std::to_string(test.n_prompt) + " exceeds corpus tokens " +
                                        std::to_string(corpus_tokens));
        }
    }
}

std::vector<int> load_corpus_ids(const std::string& path) {
    const std::string read_path = existing_read_path(path);
    std::ifstream in(read_path);
    if (!in) { throw std::runtime_error("failed to open corpus ids file: " + path); }
    std::vector<int> ids;
    std::string token;
    while (in >> token) {
        for (const char c : token) {
            if (c < '0' || c > '9') {
                throw std::invalid_argument("invalid token id in corpus: " + token);
            }
        }
        ids.push_back(parse_nonnegative(token, "corpus token id"));
    }
    if (ids.empty()) { throw std::invalid_argument("corpus ids file is empty: " + path); }
    return ids;
}

std::vector<int> prompt_slice(const std::vector<int>& corpus, int n_prompt) {
    if (n_prompt <= 0) { throw std::invalid_argument("prompt slice length must be positive"); }
    if (static_cast<std::size_t>(n_prompt) > corpus.size()) {
        throw std::invalid_argument("prompt slice length " + std::to_string(n_prompt) +
                                    " exceeds corpus tokens " + std::to_string(corpus.size()));
    }
    return std::vector<int>(corpus.begin(), corpus.begin() + n_prompt);
}

std::string decode_path_name(bool use_cuda_graph, int mtp_draft_tokens) {
    if (mtp_draft_tokens > 0) { return use_cuda_graph ? "mtp_cuda_graph" : "mtp_eager"; }
    return use_cuda_graph ? "cuda_graph" : "eager";
}

int decode_graph_prime_steps(int mtp_draft_tokens) {
    if (mtp_draft_tokens < 0 || mtp_draft_tokens > model::kMaxMtpDraftTokens) {
        throw std::invalid_argument("mtp-draft-tokens must be in [0,5]");
    }
    return mtp_draft_tokens > 0 ? mtp_draft_tokens + 2 : 2;
}

std::uint32_t decode_graph_prime_required_context(int mtp_draft_tokens) {
    if (mtp_draft_tokens < 0 || mtp_draft_tokens > model::kMaxMtpDraftTokens) {
        throw std::invalid_argument("mtp-draft-tokens must be in [0,5]");
    }
    if (mtp_draft_tokens <= 0) {
        return checked_context_u32(static_cast<std::uint64_t>(kDecodeSeedTokens) + 2ULL,
                                   "decode graph prime context requirement");
    }
    const auto k = static_cast<std::uint64_t>(mtp_draft_tokens);
    return checked_context_u32(static_cast<std::uint64_t>(kDecodeSeedTokens) + 3ULL * k + 1ULL,
                               "MTP decode graph prime context requirement");
}

Stats compute_stats(const std::vector<double>& values) {
    Stats stats;
    stats.count = static_cast<int>(values.size());
    if (values.empty()) { return stats; }
    double sum = 0.0;
    for (const double v : values) { sum += v; }
    stats.mean = sum / static_cast<double>(values.size());
    if (values.size() >= 2) {
        double acc = 0.0;
        for (const double v : values) { acc += (v - stats.mean) * (v - stats.mean); }
        stats.stddev = std::sqrt(acc / static_cast<double>(values.size() - 1));
    }
    return stats;
}

std::vector<double> prefill_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_prefill()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.prefill_time_s > 0.0) {
            out.push_back(static_cast<double>(result.test.n_prompt) / rep.prefill_time_s);
        }
    }
    return out;
}

std::int64_t decode_output_tokens(const TestResult& result) {
    return result.test.has_decode() ? static_cast<std::int64_t>(result.test.n_gen) : 0;
}

std::int64_t decode_engine_tokens(const TestResult& result, const RepTiming& rep) {
    if (!result.test.has_decode()) { return 0; }
    if (!rep.mtp.enabled || rep.mtp.k <= 0) { return static_cast<std::int64_t>(result.test.n_gen); }
    return rep.mtp.rounds + rep.mtp.accepted_tokens + rep.mtp.fallback_steps;
}

std::vector<double> decode_output_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    const std::int64_t tokens = decode_output_tokens(result);
    for (const RepTiming& rep : result.reps) {
        if (rep.decode_time_s > 0.0) {
            out.push_back(static_cast<double>(tokens) / rep.decode_time_s);
        }
    }
    return out;
}

std::vector<double> decode_engine_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        const std::int64_t tokens = decode_engine_tokens(result, rep);
        if (rep.decode_time_s > 0.0) {
            out.push_back(static_cast<double>(tokens) / rep.decode_time_s);
        }
    }
    return out;
}

std::vector<double> prefill_time_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_prefill()) { return out; }
    for (const RepTiming& rep : result.reps) { out.push_back(rep.prefill_time_s); }
    return out;
}

std::vector<double> decode_time_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) { out.push_back(rep.decode_time_s); }
    return out;
}

BenchMtpStats aggregate_mtp(const BenchEnvironment& env, const TestResult& result);

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "qus_bench throughput report\n";
    out << "  gpu:        " << env.gpu_name << " (device " << env.device_id << ")\n";
    out << "  cuda:       runtime " << env.cuda_runtime_version << ", driver "
        << env.cuda_driver_version << "\n";
    out << "  git:        " << env.git_commit << (env.worktree_dirty ? " (dirty)" : "") << "\n";
    out << "  weights:    " << env.weights_path << " (" << env.weights_file_size_bytes
        << " bytes)\n";
    out << "  corpus:     " << env.corpus_path << " (" << env.corpus_tokens << " tokens)\n";
    out << "  config:     max_ctx=" << env.max_ctx << " prefill_chunk=" << env.prefill_chunk
        << " mtp_k=" << env.mtp_draft_tokens << " work_bytes=" << env.work_bytes
        << " decode_path=" << env.decode_path << " repetitions=" << env.repetitions
        << " warmup=" << env.warmup;
    if (env.decode_graph_requested) {
        out << " graph_prime="
            << (env.decode_graph_primed ? std::to_string(env.decode_graph_prime_steps)
                                        : std::string("not_run"));
    } else {
        out << " graph_prime=n/a";
    }
    out << "\n\n";

    constexpr std::size_t kCols                  = 9;
    const std::array<std::string, kCols> headers = {
        "test",           "n_prompt", "n_gen",        "prefill t/s", "decode out t/s",
        "decode eng t/s", "mtp acc",  "mtp round/fb", "work peak"};
    std::vector<std::array<std::string, kCols>> rows;
    rows.reserve(results.size());
    for (const TestResult& result : results) {
        const BenchMtpStats mtp    = aggregate_mtp(env, result);
        const std::string mtp_rate = mtp.draft_tokens > 0
                                         ? json_number(static_cast<double>(mtp.accepted_tokens) /
                                                       static_cast<double>(mtp.draft_tokens))
                                         : std::string("n/a");
        const std::string mtp_rounds =
            mtp.enabled ? std::to_string(mtp.rounds) + "/" + std::to_string(mtp.fallback_steps)
                        : std::string("n/a");
        rows.push_back({result.test.label, std::to_string(result.test.n_prompt),
                        std::to_string(result.test.n_gen), rate_cell(prefill_tok_s_series(result)),
                        rate_cell(decode_output_tok_s_series(result)),
                        rate_cell(decode_engine_tok_s_series(result)), mtp_rate, mtp_rounds,
                        format_bytes(result.workspace_peak_bytes)});
    }

    std::array<std::size_t, kCols> width{};
    for (std::size_t c = 0; c < headers.size(); ++c) { width[c] = headers[c].size(); }
    for (const auto& row : rows) {
        for (std::size_t c = 0; c < row.size(); ++c) {
            width[c] = std::max(width[c], row[c].size());
        }
    }

    auto write_row = [&](const std::array<std::string, kCols>& row) {
        for (std::size_t c = 0; c < row.size(); ++c) {
            if (c != 0) { out << "  "; }
            if (c == 0) {
                out << std::left << std::setw(static_cast<int>(width[c])) << row[c];
            } else {
                out << std::right << std::setw(static_cast<int>(width[c])) << row[c];
            }
        }
        out << "\n";
    };

    write_row(headers);
    for (const auto& row : rows) { write_row(row); }
    return out.str();
}

void append_mtp_json(std::ostringstream& out, const BenchEnvironment& env, const BenchMtpStats& mtp,
                     const std::string& indent) {
    out << indent << "\"mtp\": {\n";
    out << indent << "  \"enabled\": " << (mtp.enabled ? "true" : "false") << ",\n";
    out << indent << "  \"k\": " << mtp.k << ",\n";
    out << indent << "  \"draft_tokens\": " << mtp.draft_tokens << ",\n";
    out << indent << "  \"accepted_tokens\": " << mtp.accepted_tokens << ",\n";
    out << indent << "  \"acceptance_rate\": ";
    if (mtp.draft_tokens > 0) {
        out << json_number(static_cast<double>(mtp.accepted_tokens) /
                           static_cast<double>(mtp.draft_tokens));
    } else {
        out << "null";
    }
    out << ",\n";
    out << indent << "  \"acceptance_length\": ";
    if (mtp.rounds > 0) {
        out << json_number(1.0 + static_cast<double>(mtp.accepted_tokens) /
                                     static_cast<double>(mtp.rounds));
    } else {
        out << "null";
    }
    out << ",\n";
    out << indent << "  \"rounds\": " << mtp.rounds << ",\n";
    out << indent << "  \"fallback_steps\": " << mtp.fallback_steps << ",\n";
    out << indent << "  \"accepted_per_pos\": [";
    for (int i = 0; i < mtp.k; ++i) {
        if (i != 0) { out << ", "; }
        out << mtp.accepted_per_pos[static_cast<std::size_t>(i)];
    }
    out << "]\n";
    out << indent << "}";
}

BenchMtpStats aggregate_mtp(const BenchEnvironment& env, const TestResult& result) {
    BenchMtpStats out;
    out.enabled = env.mtp_draft_tokens > 0;
    out.k       = env.mtp_draft_tokens;
    for (const RepTiming& rep : result.reps) {
        out.enabled = out.enabled || rep.mtp.enabled;
        out.k       = std::max(out.k, rep.mtp.k);
        out.draft_tokens += rep.mtp.draft_tokens;
        out.accepted_tokens += rep.mtp.accepted_tokens;
        out.rounds += rep.mtp.rounds;
        out.fallback_steps += rep.mtp.fallback_steps;
        for (std::size_t i = 0; i < out.accepted_per_pos.size(); ++i) {
            out.accepted_per_pos[i] += rep.mtp.accepted_per_pos[i];
        }
    }
    if (out.k < 0 || out.k > model::kMaxMtpDraftTokens) { out.k = 0; }
    return out;
}

std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": " << kSchemaVersion << ",\n"
        << "  \"artifact_type\": \"" << kArtifactType << "\",\n"
        << "  \"tool\": \"qus_bench\",\n"
        << "  \"command\": \"" << json_escape(command) << "\",\n"
        << "  \"git_commit\": \"" << json_escape(env.git_commit) << "\",\n"
        << "  \"worktree_dirty\": " << (env.worktree_dirty ? "true" : "false") << ",\n"
        << "  \"environment\": {\n"
        << "    \"gpu_name\": \"" << json_escape(env.gpu_name) << "\",\n"
        << "    \"cuda_runtime_version\": \"" << json_escape(env.cuda_runtime_version) << "\",\n"
        << "    \"cuda_driver_version\": \"" << json_escape(env.cuda_driver_version) << "\",\n"
        << "    \"device_id\": " << env.device_id << "\n"
        << "  },\n"
        << "  \"weights\": {\n"
        << "    \"path\": \"" << json_escape(env.weights_path) << "\",\n"
        << "    \"file_size_bytes\": " << env.weights_file_size_bytes << "\n"
        << "  },\n"
        << "  \"config\": {\n"
        << "    \"max_ctx\": " << env.max_ctx << ",\n"
        << "    \"prefill_chunk\": " << env.prefill_chunk << ",\n"
        << "    \"mtp_draft_tokens\": " << env.mtp_draft_tokens << ",\n"
        << "    \"work_bytes\": " << env.work_bytes << ",\n"
        << "    \"decode_path\": \"" << json_escape(env.decode_path) << "\",\n"
        << "    \"decode_graph_prime\": {\n"
        << "      \"requested\": " << (env.decode_graph_requested ? "true" : "false") << ",\n"
        << "      \"primed\": " << (env.decode_graph_primed ? "true" : "false") << ",\n"
        << "      \"decode_steps\": " << env.decode_graph_prime_steps << "\n"
        << "    },\n"
        << "    \"repetitions\": " << env.repetitions << ",\n"
        << "    \"warmup\": " << env.warmup << ",\n"
        << "    \"timing_boundary\": \"host_visible_phase_end\",\n"
        << "    \"corpus_path\": \"" << json_escape(env.corpus_path) << "\",\n"
        << "    \"corpus_tokens\": " << env.corpus_tokens << "\n"
        << "  },\n"
        << "  \"tests\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const TestResult& result                      = results[i];
        const std::vector<double> prefill_rates       = prefill_tok_s_series(result);
        const std::vector<double> decode_output_rates = decode_output_tok_s_series(result);
        const std::vector<double> decode_engine_rates = decode_engine_tok_s_series(result);
        const std::vector<double> prefill_times       = prefill_time_series(result);
        const std::vector<double> decode_times        = decode_time_series(result);
        out << "    {\n"
            << "      \"label\": \"" << json_escape(result.test.label) << "\",\n"
            << "      \"kind\": \"" << kind_string(result.test.kind) << "\",\n"
            << "      \"n_prompt\": " << result.test.n_prompt << ",\n"
            << "      \"n_gen\": " << result.test.n_gen << ",\n";
        append_stat_fields(out, "prefill_tok_s", prefill_rates, "      ");
        out << ",\n";
        append_stat_fields(out, "decode_output_tok_s", decode_output_rates, "      ");
        out << ",\n";
        append_stat_fields(out, "decode_engine_tok_s", decode_engine_rates, "      ");
        out << ",\n";
        append_stat_fields(out, "prefill_time_s", prefill_times, "      ");
        out << ",\n";
        append_stat_fields(out, "decode_time_s", decode_times, "      ");
        out << ",\n      \"workspace_peak_bytes\": " << result.workspace_peak_bytes << ",\n";
        append_mtp_json(out, env, aggregate_mtp(env, result), "      ");
        out << ",\n      \"reps\": [\n";
        for (std::size_t r = 0; r < result.reps.size(); ++r) {
            const RepTiming& rep = result.reps[r];
            out << "        {";
            out << "\"prefill_time_s\": "
                << (result.test.has_prefill() ? json_number(rep.prefill_time_s)
                                              : std::string("null"))
                << ", ";
            out << "\"prefill_tok_s\": "
                << (result.test.has_prefill() && rep.prefill_time_s > 0.0
                        ? json_number(static_cast<double>(result.test.n_prompt) /
                                      rep.prefill_time_s)
                        : std::string("null"))
                << ", ";
            out << "\"decode_time_s\": "
                << (result.test.has_decode() ? json_number(rep.decode_time_s) : std::string("null"))
                << ", ";
            out << "\"decode_output_tokens\": "
                << (result.test.has_decode() ? std::to_string(decode_output_tokens(result))
                                             : std::string("null"))
                << ", ";
            out << "\"decode_engine_tokens\": "
                << (result.test.has_decode() ? std::to_string(decode_engine_tokens(result, rep))
                                             : std::string("null"))
                << ", ";
            out << "\"decode_output_tok_s\": "
                << (result.test.has_decode() && rep.decode_time_s > 0.0
                        ? json_number(static_cast<double>(decode_output_tokens(result)) /
                                      rep.decode_time_s)
                        : std::string("null"))
                << ", ";
            out << "\"decode_engine_tok_s\": "
                << (result.test.has_decode() && rep.decode_time_s > 0.0
                        ? json_number(static_cast<double>(decode_engine_tokens(result, rep)) /
                                      rep.decode_time_s)
                        : std::string("null"));
            out << ", ";
            append_mtp_json(out, env, rep.mtp, "");
            out << "}" << (r + 1 < result.reps.size() ? "," : "") << "\n";
        }
        out << "      ]\n";
        out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n"
        << "}\n";
    return out.str();
}

std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "label,kind,n_prompt,n_gen,prefill_chunk,mtp_draft_tokens,decode_path,"
           "decode_graph_primed,decode_graph_prime_steps,mtp_rounds,mtp_fallback_steps,"
           "mtp_acceptance_rate,repetitions,prefill_tok_s_mean,prefill_tok_s_stddev,"
           "decode_output_tok_s_mean,decode_output_tok_s_stddev,decode_engine_tok_s_mean,"
           "decode_engine_tok_s_stddev,prefill_time_s_mean,decode_time_s_mean,"
           "workspace_peak_bytes\n";
    auto cell_mean = [](const std::vector<double>& series) -> std::string {
        return series.empty() ? std::string() : json_number(compute_stats(series).mean);
    };
    auto cell_stddev = [](const std::vector<double>& series) -> std::string {
        return series.empty() ? std::string() : json_number(compute_stats(series).stddev);
    };
    for (const TestResult& result : results) {
        const BenchMtpStats mtp    = aggregate_mtp(env, result);
        const std::string mtp_rate = mtp.draft_tokens > 0
                                         ? json_number(static_cast<double>(mtp.accepted_tokens) /
                                                       static_cast<double>(mtp.draft_tokens))
                                         : std::string();
        out << result.test.label << "," << kind_string(result.test.kind) << ","
            << result.test.n_prompt << "," << result.test.n_gen << "," << env.prefill_chunk << ","
            << env.mtp_draft_tokens << "," << env.decode_path << ","
            << (env.decode_graph_primed ? "true" : "false") << "," << env.decode_graph_prime_steps
            << "," << mtp.rounds << "," << mtp.fallback_steps << "," << mtp_rate << ","
            << result.reps.size() << "," << cell_mean(prefill_tok_s_series(result)) << ","
            << cell_stddev(prefill_tok_s_series(result)) << ","
            << cell_mean(decode_output_tok_s_series(result)) << ","
            << cell_stddev(decode_output_tok_s_series(result)) << ","
            << cell_mean(decode_engine_tok_s_series(result)) << ","
            << cell_stddev(decode_engine_tok_s_series(result)) << ","
            << cell_mean(prefill_time_series(result)) << ","
            << cell_mean(decode_time_series(result)) << "," << result.workspace_peak_bytes << "\n";
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

std::uint64_t file_size_or_zero(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<std::uint64_t>(size);
}

} // namespace qus::bench
