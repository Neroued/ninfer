#include "options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ninfer::cli {
namespace {

std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_u32(const char* text, std::string_view label, bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

int parse_device(const char* text) {
    const std::uint64_t value = parse_u64(text, "device");
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid device: ") + text);
    }
    return static_cast<int>(value);
}

float parse_float(const char* text, std::string_view label, float minimum, float maximum) {
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < static_cast<double>(minimum) || value > static_cast<double>(maximum)) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<float>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    return KvCapacityPolicy::explicit_capacity(parse_u32(text, "kv-capacity"));
}

ReasoningEffort parse_reasoning_effort(std::string_view text) {
    if (text == "low") { return ReasoningEffort::Low; }
    if (text == "medium") { return ReasoningEffort::Medium; }
    if (text == "xhigh") { return ReasoningEffort::XHigh; }
    throw std::invalid_argument("invalid reasoning-effort: " + std::string(text));
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> (--prompt <text>|--messages <messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "       [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--spec mtp|dflash --draft-tokens N]\n"
           "       [--lm-head-draft]\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--raw-output] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-effort low|medium|xhigh] [--vision]\n"
           "       [--no-cuda-graph]\n"
           "\n"
           "Streams answer content to stdout and reasoning plus diagnostics to stderr.\n"
           "Structured message content accepts text, image/image_url, and video/video_url parts;\n"
           "media sources may be local paths, HTTP(S) URLs, or base64 data URIs.\n"
           "--vision enables image/video input and loads the fixed Vision GPU allocations.\n"
           "--thinking-budget caps model-origin thinking tokens; inserted control tokens count "
           "toward --max-new.\n"
           "--kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom.\n"
           "Sampling defaults come from the loaded model and thinking mode; flags override "
           "individual fields.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (argc >= 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument(".ninfer model path is required"); }
    options.artifact_path     = argv[1];
    bool kv_capacity_explicit = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };

        if (arg == "--prompt") {
            options.prompt = value(arg);
        } else if (arg == "--messages") {
            options.messages_path = value(arg);
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value(arg), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = parse_u32(value(arg), "max-context");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(value(arg));
            kv_capacity_explicit = true;
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
        } else if (arg == "--device") {
            options.device = parse_device(value(arg));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value(arg));
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value(arg), "draft-tokens");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--thinking-budget") {
            options.thinking_budget = parse_u32(value(arg), "thinking-budget");
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value(arg), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg);
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_float(value(arg), "temperature", 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_float(value(arg), "top-p", 0.0F, 1.0F);
        } else if (arg == "--top-k") {
            const std::uint32_t top_k = parse_u32(value(arg), "top-k", true);
            if (top_k > 20) { throw std::invalid_argument("--top-k must be in [0,20]"); }
            options.sampling.top_k = static_cast<std::int32_t>(top_k);
        } else if (arg == "--min-p") {
            options.sampling.min_p = parse_float(value(arg), "min-p", 0.0F, 1.0F);
        } else if (arg == "--presence-penalty") {
            options.sampling.presence_penalty =
                parse_float(value(arg), "presence-penalty", -2.0F, 2.0F);
        } else if (arg == "--frequency-penalty") {
            options.sampling.frequency_penalty =
                parse_float(value(arg), "frequency-penalty", -2.0F, 2.0F);
        } else if (arg == "--seed") {
            options.sampling.seed = parse_u64(value(arg), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }

    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.speculative.backend == SpeculativeBackend::DFlash && options.enable_vision) {
        throw std::invalid_argument("--spec dflash cannot be combined with --vision");
    }
    if (!options.enable_thinking && options.reasoning_effort) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (!options.enable_thinking && options.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    if (options.greedy) { options.sampling.temperature = 0.0F; }
    return options;
}

} // namespace ninfer::cli
