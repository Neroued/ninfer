#include "qus/serve/serve_options.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::serve {
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

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end         = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    char* end                     = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

} // namespace

int derive_default_max_tokens(std::uint32_t max_context) {
    const std::uint32_t half = max_context / 2;
    const std::uint32_t capped =
        std::min(half, static_cast<std::uint32_t>(kDefaultMaxTokensCeiling));
    return std::max<int>(1, static_cast<int>(capped));
}

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <weights.qus> --tokenizer <dir> [--host H] [--port N] [--api-key KEY] "
           "[--model-id ID] [--max-context N] [--prefill-chunk N] [--device N] "
           "[--mtp-draft-tokens N] [--default-max-tokens N] [--no-cuda-graph] "
           "[--lm-head-draft] [--thinking] [--cors] "
           "[--temperature F] [--top-p F] [--top-k N] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       serves an OpenAI-compatible Chat Completions endpoint\n"
           "       --default-max-tokens defaults to min(max_context/2, "
           + std::to_string(kDefaultMaxTokensCeiling) +
           ") when omitted\n"
           "       sampler defaults to Qwen3 thinking (temperature 0.6, top-p 0.95, "
           "top-k 20, presence-penalty 1.0); a request may override any field.\n"
           "       --greedy forces temperature 0 (exact argmax) for determinism/parity.\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    bool default_max_tokens_explicit = false;
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
        } else if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id = require_value("--model-id");
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--mtp-draft-tokens") {
            options.mtp_draft_tokens =
                parse_nonnegative_int(require_value("--mtp-draft-tokens"), "mtp-draft-tokens");
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--lm-head-draft") {
            options.use_lm_head_draft = true;
        } else if (arg == "--thinking") {
            options.enable_thinking = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_top_p = parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            options.sampling_top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
        } else if (arg == "--presence-penalty") {
            options.sampling_presence_penalty =
                parse_float_in(require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.tokenizer_path.empty()) { throw std::invalid_argument("--tokenizer is required"); }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.prefill_chunk == 0 ||
        options.prefill_chunk % model::kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    } else {
        options.default_max_tokens = derive_default_max_tokens(options.max_context);
    }
    return options;
}

} // namespace qus::serve
