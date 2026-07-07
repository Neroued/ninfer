#include "qus/text/cli.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace qus::text {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

std::uint32_t parse_positive_u32(const char* text, const char* label) {
    const int value = parse_nonnegative_int(text, label);
    if (value <= 0) {
        throw std::invalid_argument(std::string("--") + label + " must be positive");
    }
    return static_cast<std::uint32_t>(value);
}

int parse_mtp_draft_tokens(const char* text) {
    const int value = parse_nonnegative_int(text, "mtp-draft-tokens");
    if (value > model::kMaxMtpDraftTokens) {
        throw std::invalid_argument("--mtp-draft-tokens must be in [0,5]");
    }
    return value;
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <weights.qus> --tokenizer <dir> (--prompt <text>|--messages <messages.json>) "
           "[--max-context N] [--prefill-chunk N] [--max-new N] [--device N] [--raw-output] "
           "[--print-token-ids] [--no-cuda-graph] [--lm-head-draft] [--no-thinking] "
           "[--mtp-draft-tokens N] [--stop-token-id N]... "
           "[--temperature F] [--top-p F] [--top-k N] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       streams decoded text to stdout; writes progress and timings to stderr\n"
           "       sampler defaults to Qwen3 thinking (temperature 0.6, top-p 0.95, "
           "top-k 20, presence-penalty 1.0); --greedy forces temperature 0 (exact argmax)\n";
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("weights path is required"); }
    options.weights_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--tokenizer") {
            options.tokenizer_path = require_value("--tokenizer");
        } else if (arg == "--prompt") {
            options.prompt = require_value("--prompt");
        } else if (arg == "--messages") {
            options.messages_path = require_value("--messages");
        } else if (arg == "--max-new") {
            options.max_new = parse_nonnegative_int(require_value("--max-new"), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk =
                parse_positive_u32(require_value("--prefill-chunk"), "prefill-chunk");
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--mtp-draft-tokens") {
            options.mtp_draft_tokens = parse_mtp_draft_tokens(require_value("--mtp-draft-tokens"));
        } else if (arg == "--raw-output") {
            options.output_mode = OutputMode::Raw;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--lm-head-draft") {
            options.use_lm_head_draft = true;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--stop-token-id") {
            options.stop_token_ids.push_back(
                parse_nonnegative_int(require_value("--stop-token-id"), "stop-token-id"));
        } else if (arg == "--temperature") {
            options.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.top_p = parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            options.top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
        } else if (arg == "--presence-penalty") {
            options.presence_penalty =
                parse_float_in(require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.tokenizer_path.empty()) { throw std::invalid_argument("--tokenizer is required"); }
    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    if (options.max_new <= 0) { throw std::invalid_argument("--max-new must be positive"); }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.prefill_chunk % model::kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    return options;
}

} // namespace qus::text
