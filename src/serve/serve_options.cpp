#include "serve/serve_options.h"
#include "core/byte_size.h"
#include "product/speculative_options.h"
#include "product/tensor_parallel_options.h"
#include "product/vision_options.h"

#include <cmath>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {
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
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

// --api-key-file (issue #25): the whole file is the key, minus one trailing newline.
std::string read_api_key_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::invalid_argument("--api-key-file: cannot read " + path); }
    std::string key((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (!key.empty() && key.back() == '\n') { key.pop_back(); }
    if (!key.empty() && key.back() == '\r') { key.pop_back(); }
    if (key.empty()) { throw std::invalid_argument("--api-key-file: " + path + " is empty"); }
    if (key.find_first_of("\r\n") != std::string::npos) {
        throw std::invalid_argument("--api-key-file: " + path + " must hold the key on one line");
    }
    return key;
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::uint32_t resolve_turn_anchors(std::optional<std::uint32_t> configured,
                                   std::string_view architecture) {
    if (configured) { return *configured; }
    return architecture == "Qwen4ExpForCausalLM" ? 0U : 1U;
}

namespace {

// Serving profiles: the configurations the release launchers start (two 16 GB boards; 27b-16gb:
// one). They are inserted before the command line's own flags, so an explicit flag overrides its
// profile value; the KV capacity is left to follow --max-context.
struct ServeProfile {
    std::string_view name;
    std::vector<std::string_view> flags;
};

// One flag or value per line; clang-format would pack the shorter lists into columns.
// clang-format off
const std::vector<ServeProfile>& serve_profiles() {
    static const std::vector<ServeProfile> profiles{
        {"flash-next",
         {"--tp",
          "2",
          "--devices",
          "0,1",
          "--kv-dtype",
          "int8",
          "--elastic",
          "--max-context",
          "262144",
          "--max-concurrency",
          "1",
          "--spec",
          "mtp",
          "--draft-tokens",
          "3",
          "--host-kv-mib",
          "4096",
          "--host-state-slots",
          "8",
          "--embedding-host"}},
        {"27b",
         {"--tp",
          "2",
          "--devices",
          "0,1",
          "--kv-dtype",
          "int8",
          "--max-context",
          "196608",
          "--device-state-slots",
          "4",
          "--max-concurrency",
          "1",
          "--spec",
          "mtp",
          "--draft-tokens",
          "3",
          "--vision",
          "--vision-device",
          "0",
          "--max-vision-tokens",
          "4096",
          "--host-state-slots",
          "8",
          "--host-kv-mib",
          "8192",
          "--prefill-chunk",
          "2048"}},
        // v0.7: Qwen3.8-Flash-Next on ONE 16 GB GPU (T242: GSQ-RCO 64 GB edition, 57 GiB of RAM,
        // LocalMaxxing 110 tok/s): the flash-next configuration at one device with small RAM
        // caches. v0.7.1: 8 Host StateImages (+0.66 GiB pinned): at one device only 1 Device
        // StateImage fits, and with 2 Host ones three interleaved agent sessions lost their turn
        // checkpoints (prompt reuse 34 % -> 83 %, replay 858 -> 276 s, T247).
        {"flash-next-1gpu",
         {"--device",
          "0",
          "--kv-dtype",
          "int8",
          "--elastic",
          "--max-context",
          "262144",
          "--max-concurrency",
          "1",
          "--spec",
          "mtp",
          "--draft-tokens",
          "3",
          "--embedding-host",
          "--host-kv-mib",
          "1024",
          "--host-state-slots",
          "8"}},
        // v0.7: Qwen3.8-27B with the 3-bit GSQ weights (q3_g128_fp16 artifact) on ONE 16 GB board
        // (docs/serving.md#profiles, profile 16gb-64k: int8 KV, 65 536 tokens, vision; VRAM 13.9
        // GiB).
        {"27b-16gb",
         {"--device",
          "0",
          "--kv-dtype",
          "int8",
          "--max-context",
          "65536",
          "--kv-capacity",
          "auto",
          "--vram-headroom-mib",
          "384",
          "--device-state-slots",
          "2",
          "--max-concurrency",
          "1",
          "--spec",
          "mtp",
          "--draft-tokens",
          "3",
          "--lm-head-draft",
          "--embedding-host",
          "--vision",
          "--vision-device",
          "0",
          "--max-vision-tokens",
          "4096",
          "--host-state-slots",
          "8",
          "--host-kv-mib",
          "4096",
          "--prefill-chunk",
          "2048"}},
    };
    return profiles;
}

// clang-format on

const ServeProfile& find_profile(std::string_view name) {
    for (const ServeProfile& profile : serve_profiles()) {
        if (profile.name == name) { return profile; }
    }
    throw std::invalid_argument("unknown --profile " + std::string(name) +
                                " (known: flash-next, flash-next-1gpu, 27b, 27b-16gb)");
}

std::string profile_text(std::string_view name) {
    std::string text;
    for (const std::string_view flag : find_profile(name).flags) {
        if (!text.empty()) { text += ' '; }
        text += flag;
    }
    return text;
}

// argv with every `--profile NAME` replaced by the profile's flags, placed right after the
// artifact.
std::vector<std::string> expand_profiles(int argc, char** argv) {
    std::vector<std::string> head, rest;
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i] == nullptr ? "" : argv[i];
        if (i >= 2 && arg == "--profile") {
            if (++i >= argc) { throw std::invalid_argument("--profile needs a value"); }
            for (const std::string_view flag : find_profile(argv[i]).flags) {
                head.emplace_back(flag);
            }
        } else {
            rest.push_back(arg);
        }
    }
    const auto at =
        rest.begin() + std::min<std::ptrdiff_t>(2, static_cast<std::ptrdiff_t>(rest.size()));
    rest.insert(at, head.begin(), head.end());
    return rest;
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [--profile flash-next|flash-next-1gpu|27b|27b-16gb] [--host H] "
           "[--port N] [--api-key KEY|--api-key-file FILE] "
           "[--model-id ID] [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] [--decode-share F] "
           "[--prefill-chunk N] [--kv-lease-window N] [--log-stats-interval-ms N] [--device N] "
           "[--tp 1|2 --devices A,B] "
           "[--context-cost-presets FILE] "
           "[--max-request-mib N] [--max-request-json-depth N] [--media-cache-mib N] "
           "[--media-live-mib N] "
           "[--constraint-max-kib N] [--constraint-max-depth N] [--constraint-max-items N] "
           "[--constraint-compile-timeout-ms N] [--constraint-compile-concurrency N] "
           "[--constraint-compile-wait-ms N] "
           "[--constraint-cache-mib N] [--constraint-round-budget-ms N] "
           "[--constraint-request-budget-ms N] [--constraint-round-allowance-ms N] "
           "[--constraint-matcher-mib N] [--constraint-json-strict on|off] "
           "[--media-preprocess-threads N] "
           "[--device-state-slots N] [--host-state-slots N] [--host-kv-mib N] "
           "[--disk-cache-dir DIR --disk-cache-gib N] [--disk-cache-min-free-gib N] "
           "[--disk-cache-idle-ms N] "
           "[--disk-cache-flush-seconds N] "
           "[--max-private-continuations N] [--max-shared-prefixes N] "
           "[--max-long-anchors-per-continuation N] [--turn-anchors N] "
           "[--elastic] [--kv-grow N] [--kv-budget-gib G] [--kv-ram-active] "
           "[--miss-split auto|calibrate|off] "
           "[--stream-min-tokens N] [--expert-cpu-threads N] "
           "[--request-log-jsonl FILE] [--log-speculation-detail] "
           "[--log-speculation-detail-max-steps N] "
           "[--prefill-miss-dump DIR [--prefill-miss-min-tokens N] [--prefill-miss-min-match F]] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--spec mtp|dflash2|off --draft-tokens N] "
           "[--rope-yarn-factor F] [--rope-original-max-position N] "
           "[--default-max-tokens N] [--default-thinking-budget N] "
           "[--vision|--no-vision] [--vision-device N] [--max-vision-tokens N] [--no-cuda-graph] "
           "[--no-tp-mailbox] [--embedding-host] [--host-pinned-max-gib G] [--no-prefix-reuse] "
           "[--reasoning-close-retry|--no-reasoning-close-retry] "
           "[--reasoning-eos-guard|--no-reasoning-eos-guard] "
           "[--chat-template FILE] [--lm-head-draft] [--no-thinking] [--preserve-thinking] "
           "[--cors] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       --profile applies the tested configuration of a model before the other "
           "flags, which override it (27b-16gb: one 16 GB GPU, 3-bit GSQ 27B artifact):\n"
           "         flash-next: " +
           profile_text("flash-next") +
           "\n"
           "         27b:        " +
           profile_text("27b") +
           "\n"
           "         flash-next-1gpu: " +
           profile_text("flash-next-1gpu") +
           "\n"
           "         27b-16gb:   " +
           profile_text("27b-16gb") +
           "\n"
           "       --default-max-tokens defaults to " +
           std::to_string(kDefaultMaxTokens) +
           " when omitted\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --max-request-json-depth defaults to " +
           std::to_string(kDefaultMaxRequestJsonDepth) + " (at most " +
           std::to_string(kMaximumMaxRequestJsonDepth) +
           ") nested objects/arrays in a body or in tool-call arguments, checked before "
           "parsing\n"
           "       --media-cache-mib defaults to 1024; 0 disables retained media reuse\n"
           "       structured output limits (docs/serving.md): --constraint-max-kib (256), "
           "--constraint-max-depth (256, at most 1000) and --constraint-max-items (4096) bound a "
           "schema, grammar, regex or choice list before compilation; a compilation runs at most "
           "--constraint-compile-timeout-ms (5000) once it holds one of "
           "--constraint-compile-concurrency (2) slots, waited for at most "
           "--constraint-compile-wait-ms (30000, then 429), into a --constraint-cache-mib (64) "
           "cache; a generating constrained request may "
           "spend --constraint-round-budget-ms (20) of matcher time per decode round and "
           "--constraint-request-budget-ms (1000) plus --constraint-round-allowance-ms (5) per "
           "round in total and keep --constraint-matcher-mib (512) of matcher history, or it "
           "alone fails (0 disables a time or memory limit); --constraint-json-strict on (default, "
           "as XGrammar/vLLM) closes JSON Schema objects without additionalProperties and arrays "
           "without items, off keeps literal JSON Schema\n"
           "       --media-live-mib defaults to 2048 and bounds all live BF16 patch payloads\n"
           "       --media-preprocess-threads defaults to 0 (auto, at most 16 workers)\n"
           "       --request-log-jsonl appends full-precision server/request records\n"
           "       --log-speculation-detail adds the per-step accepted draft lengths and the "
           "output/prompt 4-gram overlap to every request_done record (requires "
           "--request-log-jsonl; off by default);\n"
           "       --log-speculation-detail-max-steps caps them per request (default " +
           std::to_string(kDefaultSpeculationDetailMaxSteps) + ", at most " +
           std::to_string(kMaximumSpeculationDetailMaxSteps) +
           "; longer requests are marked truncated)\n"
           "       --prefill-miss-dump diagnoses prefix-reuse misses: when a request recomputes at "
           "least --prefill-miss-min-tokens prompt tokens (default 10000), shares at least "
           "--prefill-miss-min-match of its prompt (default 0.5) with one of the last 16 "
           "contexts, and either the server recomputed that many tokens identical to it or "
           "the client changed that many of its tokens, both texts, their token IDs and a "
           "summary.json with the divergence point go to a new directory under DIR (at most 64 "
           "per process; contains conversation text; off by default)\n"
           "       --model-id overrides the artifact metadata.name reported by the server\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
           "       --vision-device selects the CUDA device that holds the Vision tower and encodes "
           "(default: rank 0's); it must equal --device on one GPU and be one of --devices with "
           "--tp 2, where the other GPU only receives the encoded embeddings\n"
           "       --max-vision-tokens caps the merged Vision tokens of one image or video item "
           "(64-16384, default 16384): larger media are resized and the encode workspace is "
           "planned for N\n"
           "       --kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom; --vram-headroom-mib N changes that margin (requires auto)\n"
           "       --no-prefix-reuse disables compatible-prefix caching (enabled by default)\n"
           "       Qwen4Exp: --elastic backs the KV on demand and lends unused KV to the expert "
           "slot cache (grow step --kv-grow tokens, default 8192; --kv-budget-gib caps the "
           "KV of every request together); --kv-ram-active keeps the text attention KV in pinned "
           "host RAM, taken "
           "8192 tokens at a time as conversations grow (at most --kv-capacity tokens), and "
           "copies only the indexer's selected blocks to the GPU (results unchanged); "
           "--miss-split auto lets the GPU read part of each window's missing "
           "experts (best at one GPU), calibrate does the same with costs measured on this machine "
           "(GPU copy rate at startup, CPU time per expert over the first decode calls, logged), "
           "off keeps them all on the CPU (default); prompt chunks of "
           "at least --stream-min-tokens N tokens stream the non-resident experts to the GPU "
           "over PCIe, shorter ones compute them on the CPU (default " +
           std::to_string(RoutedExpertOptions{}.stream_min_tokens) +
           ", 0 = never stream); --expert-cpu-threads N: CPU threads of the missing experts "
           "(default auto from the machine, logged at startup: max(6, cores/2, 2 per L3 domain), "
           "one per physical core with ggml block expert banks)\n"
           "       context cache defaults: device-state=max-concurrency, private=2x concurrency, "
           "shared=max(max-concurrency,4), anchors=2; Host state=8 slots, Host KV=8192 MiB\n"
           "       --tp 2 --devices A,B splits the dense model across two GPUs (rank 0 on A) for "
           "ordinary, --spec mtp or --spec dflash2 --lm-head-draft decoding, with or without "
           "--vision, with bf16, int8 or nvfp4 KV; "
           "it defaults device-state to "
           "max(2x concurrency,8), private to max(2x concurrency,8) and the Host tiers to 0 "
           "unless set (--profile 27b: 8 slots, 8192 MiB)\n"
           "       --kv-lease-window N: a request reserves its prompt plus N output tokens of "
           "Device KV (default 4096) and grows as its answer grows; 0 reserves the whole "
           "max_tokens at admission\n"
           "       --no-tp-mailbox keeps the captured --tp 2 all-reduces on cross-device copies\n"
           "       --embedding-host keeps the token embedding table in one copy in pinned host "
           "memory that every GPU reads in place over PCIe, instead of one copy per GPU: frees "
           "the table's VRAM (1.18 GiB per GPU on Qwen3.8-27B) for the KV pool, identical "
           "results\n"
           "       --host-pinned-max-gib G keeps at most G GiB of the host-mapped weights "
           "(Flash-Next expert banks) page-locked; the banks of the least used layers beyond it "
           "stay in pageable memory (CPU reads them in place, GPU copies go through a pinned "
           "staging ring, no prefill streaming on them). Without it the whole page-locked "
           "allocation is tried first and the fallback starts only if the driver refuses it "
           "(Windows WDDM / WSL2 allow about half the RAM). Same results, slower where "
           "pageable; the Host KV planes of --kv-ram-active count in G "
           "(NINFER_HOST_PINNED_MAX_GIB also sets it)\n"
           "       --device-state-slots is extra checkpoint capacity beyond active lanes; "
           "--host-kv-mib uses MiB\n"
           "       --disk-cache-dir DIR --disk-cache-gib N (off by default): disk tier of the "
           "context cache, at most N GiB in DIR (least recently used out). Retained "
           "conversations are written after --disk-cache-idle-ms of inactivity (default 5000; 0 "
           "= only at shutdown) and on SIGTERM/SIGINT (at most --disk-cache-flush-seconds, "
           "default 60); after a restart a prompt that begins with a saved conversation loads "
           "it from disk instead of prefilling it. KV is stored in 256-token blocks addressed by "
           "the prefix, written once and shared between turns and conversations. Records carry a "
           "validity key (artifact, executable, KV dtype, tp, speculative mode, layouts): another "
           "key is ignored. One directory per server (locked). --disk-cache-min-free-gib N keeps N "
           "GiB free on the file system (default max(10 GiB, 5 %): the cache evicts, then skips "
           "saves)\n"
           "       --default-thinking-budget caps model-origin thinking for enabled requests; "
           "control tokens count toward the request output limit\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       --reasoning-close-retry: a reply that ends with the model's terminal token "
           "inside "
           "its thinking (or right after </think> with blank content and no tool call) is closed "
           "with \"\\n</think>\\n\\n\" and continued once (on by default; "
           "--no-reasoning-close-retry turns it off)\n"
           "       --reasoning-eos-guard: the first token of a reply that starts in thinking is "
           "never "
           "a terminal token (a terminal draw is redrawn without them; on by default; "
           "--no-reasoning-eos-guard turns it off)\n"
           "       sampler defaults come from the resolved thinking mode; server flags and request "
           "fields override individual values.\n"
           "       --greedy forces temperature 0 (exact argmax).\n"
           "       --api-key-file FILE reads the key from FILE (one trailing newline stripped); "
           "without --api-key or --api-key-file the key comes from NINFER_API_KEY when it is "
           "set.\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool api_key_explicit            = false;
    std::optional<std::string> api_key_file;
    bool kv_capacity_explicit      = false;
    bool context_capacity_explicit = false;
    bool device_explicit           = false;
    bool host_state_explicit       = false;
    bool host_kv_explicit          = false;
    bool detail_steps_explicit     = false;
    bool prefill_miss_tuned        = false;
    std::optional<std::size_t> vram_headroom_mib;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    std::vector<std::string> expanded = expand_profiles(argc, argv);
    std::vector<char*> expanded_argv;
    expanded_argv.reserve(expanded.size());
    for (std::string& arg : expanded) { expanded_argv.push_back(arg.data()); }
    argc                  = static_cast<int>(expanded_argv.size());
    argv                  = expanded_argv.data();
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key  = require_value("--api-key");
            api_key_explicit = true;
        } else if (arg == "--api-key-file") {
            api_key_file = require_value("--api-key-file");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--vram-headroom-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--vram-headroom-mib"), "vram-headroom-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--vram-headroom-mib is out of range");
            }
            vram_headroom_mib = static_cast<std::size_t>(mib);
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--decode-share") {
            const std::string value = require_value("--decode-share");
            double share            = -1.0;
            try {
                share = std::stod(value);
            } catch (const std::exception&) {
                throw std::invalid_argument("--decode-share must be a number");
            }
            if (!(share >= 0.0 && share <= 16.0)) {
                throw std::invalid_argument("--decode-share must be in [0,16]");
            }
            options.decode_share = static_cast<float>(share);
        } else if (arg == "--max-request-json-depth") {
            options.max_request_json_depth = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-request-json-depth"), "max-request-json-depth"));
        } else if (arg == "--constraint-max-kib") {
            const std::uint64_t kib =
                parse_u64(require_value("--constraint-max-kib"), "constraint-max-kib");
            if (kib == 0 || kib > (1ULL << 20)) {
                throw std::invalid_argument("--constraint-max-kib must be in [1,1048576]");
            }
            options.constraint_limits.max_bytes = static_cast<std::size_t>(kib << 10);
        } else if (arg == "--constraint-max-depth") {
            options.constraint_limits.max_depth = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--constraint-max-depth"), "constraint-max-depth"));
        } else if (arg == "--constraint-max-items") {
            options.constraint_limits.max_items = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--constraint-max-items"), "constraint-max-items"));
        } else if (arg == "--constraint-compile-timeout-ms") {
            options.constraint_limits.compile_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--constraint-compile-timeout-ms"),
                                      "constraint-compile-timeout-ms"));
        } else if (arg == "--constraint-compile-concurrency") {
            options.constraint_limits.compile_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--constraint-compile-concurrency"),
                                      "constraint-compile-concurrency"));
        } else if (arg == "--constraint-compile-wait-ms") {
            options.constraint_limits.compile_wait_ms =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--constraint-compile-wait-ms"), "constraint-compile-wait-ms"));
        } else if (arg == "--constraint-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--constraint-cache-mib"), "constraint-cache-mib");
            if (mib == 0 || mib > (1ULL << 20)) {
                throw std::invalid_argument("--constraint-cache-mib must be in [1,1048576]");
            }
            options.constraint_limits.cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--constraint-json-strict") {
            const std::string value = require_value("--constraint-json-strict");
            if (value != "on" && value != "off") {
                throw std::invalid_argument("--constraint-json-strict must be on or off");
            }
            options.constraint_limits.json_strict = value == "on";
        } else if (arg == "--constraint-matcher-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--constraint-matcher-mib"), "constraint-matcher-mib");
            if (mib > (1ULL << 20)) {
                throw std::invalid_argument("--constraint-matcher-mib must be at most 1048576");
            }
            options.constraint_limits.matcher_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--constraint-round-budget-ms") {
            options.constraint_limits.round_budget_ms =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--constraint-round-budget-ms"), "constraint-round-budget-ms"));
        } else if (arg == "--constraint-request-budget-ms") {
            options.constraint_limits.request_budget_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--constraint-request-budget-ms"),
                                      "constraint-request-budget-ms"));
        } else if (arg == "--constraint-round-allowance-ms") {
            options.constraint_limits.round_allowance_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--constraint-round-allowance-ms"),
                                      "constraint-round-allowance-ms"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--kv-lease-window") {
            options.kv_lease_window_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kv-lease-window"), "kv-lease-window"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-state-slots") {
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
            context_capacity_explicit = true;
            host_state_explicit       = true;
        } else if (arg == "--host-kv-mib") {
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                    = true;
            host_kv_explicit                             = true;
        } else if (arg == "--disk-cache-dir") {
            options.context_cache.disk_directory = require_value("--disk-cache-dir");
            if (options.context_cache.disk_directory.empty()) {
                throw std::invalid_argument("--disk-cache-dir must not be empty");
            }
        } else if (arg == "--disk-cache-gib") {
            const std::uint64_t gib =
                parse_u64(require_value("--disk-cache-gib"), "disk-cache-gib");
            if (gib == 0 || gib > (1ULL << 20)) {
                throw std::invalid_argument("--disk-cache-gib must be in [1,1048576]");
            }
            options.context_cache.disk_capacity_bytes = gib << 30;
        } else if (arg == "--disk-cache-min-free-gib") {
            const std::uint64_t gib =
                parse_u64(require_value("--disk-cache-min-free-gib"), "disk-cache-min-free-gib");
            if (gib > (1ULL << 20)) {
                throw std::invalid_argument("--disk-cache-min-free-gib must be in [0,1048576]");
            }
            options.context_cache.disk_min_free_bytes = gib << 30;
        } else if (arg == "--disk-cache-idle-ms") {
            const std::uint64_t ms =
                parse_u64(require_value("--disk-cache-idle-ms"), "disk-cache-idle-ms");
            if (ms > 86400000ULL) {
                throw std::invalid_argument("--disk-cache-idle-ms must be at most 86400000");
            }
            options.context_cache.disk_idle_ms = static_cast<std::uint32_t>(ms);
        } else if (arg == "--disk-cache-flush-seconds") {
            const std::uint64_t seconds =
                parse_u64(require_value("--disk-cache-flush-seconds"), "disk-cache-flush-seconds");
            if (seconds > 3600) {
                throw std::invalid_argument("--disk-cache-flush-seconds must be at most 3600");
            }
            options.context_cache.disk_flush_seconds = static_cast<std::uint32_t>(seconds);
        } else if (arg == "--max-private-continuations") {
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
            context_capacity_explicit = true;
        } else if (arg == "--max-shared-prefixes") {
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
            context_capacity_explicit = true;
        } else if (arg == "--max-long-anchors-per-continuation") {
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
            context_capacity_explicit = true;
        } else if (arg == "--elastic") {
            options.elastic_kv.enabled = true;
        } else if (arg == "--kv-ram-active") {
            options.kv_ram_active = true;
        } else if (arg == "--kv-grow") {
            options.elastic_kv.grow_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kv-grow"), "kv-grow"));
            if (options.elastic_kv.grow_tokens == 0) {
                throw std::invalid_argument("--kv-grow must be positive");
            }
        } else if (arg == "--kv-budget-gib") {
            const std::string value = require_value("--kv-budget-gib");
            double gib              = 0.0;
            try {
                gib = std::stod(value);
            } catch (const std::exception&) {
                throw std::invalid_argument("--kv-budget-gib must be a number");
            }
            if (!(gib > 0.0)) { throw std::invalid_argument("--kv-budget-gib must be positive"); }
            options.elastic_kv.kv_budget_bytes = static_cast<std::size_t>(gib * 1073741824.0);
        } else if (arg == "--miss-split") {
            const std::string value      = require_value("--miss-split");
            options.miss_split_calibrate = false;
            if (value == "auto") {
                options.miss_split = MissSplitMode::Auto;
            } else if (value == "calibrate") {
                options.miss_split           = MissSplitMode::Auto;
                options.miss_split_calibrate = true;
            } else if (value == "off") {
                options.miss_split = MissSplitMode::Off;
            } else {
                throw std::invalid_argument("--miss-split must be auto, calibrate or off");
            }
        } else if (arg == "--expert-cpu-threads") {
            const int n =
                parse_nonnegative_int(require_value("--expert-cpu-threads"), "expert-cpu-threads");
            if (n < 1 || n > 64) {
                throw std::invalid_argument("--expert-cpu-threads must be 1..64");
            }
            options.expert_cpu_threads = static_cast<std::uint32_t>(n);
        } else if (arg == "--stream-min-tokens") {
            options.stream_min_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--stream-min-tokens"), "stream-min-tokens"));
        } else if (arg == "--turn-anchors") {
            options.turn_anchors = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--turn-anchors"), "turn-anchors"));
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--prefill-miss-dump") {
            options.prefill_miss_dump_dir = require_value("--prefill-miss-dump");
            if (options.prefill_miss_dump_dir.empty()) {
                throw std::invalid_argument("--prefill-miss-dump must not be empty");
            }
        } else if (arg == "--prefill-miss-min-tokens") {
            const int tokens = parse_nonnegative_int(require_value("--prefill-miss-min-tokens"),
                                                     "prefill-miss-min-tokens");
            if (tokens == 0) {
                throw std::invalid_argument("--prefill-miss-min-tokens must be positive");
            }
            options.prefill_miss_min_tokens = static_cast<std::uint32_t>(tokens);
            prefill_miss_tuned              = true;
        } else if (arg == "--prefill-miss-min-match") {
            options.prefill_miss_min_match = parse_float_in(
                require_value("--prefill-miss-min-match"), "prefill-miss-min-match", 0.0f, 1.0f);
            prefill_miss_tuned = true;
        } else if (arg == "--log-speculation-detail") {
            options.log_speculation_detail = true;
        } else if (arg == "--log-speculation-detail-max-steps") {
            const std::uint64_t steps =
                parse_u64(require_value("--log-speculation-detail-max-steps"),
                          "log-speculation-detail-max-steps");
            if (steps == 0 || steps > kMaximumSpeculationDetailMaxSteps) {
                throw std::invalid_argument("--log-speculation-detail-max-steps must be in [1," +
                                            std::to_string(kMaximumSpeculationDetailMaxSteps) +
                                            "]");
            }
            options.speculation_detail_max_steps = static_cast<std::uint32_t>(steps);
            detail_steps_explicit                = true;
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device  = parse_nonnegative_int(require_value("--device"), "device");
            device_explicit = true;
        } else if (arg == "--tp") {
            options.tp = product::parse_tp(require_value("--tp"));
        } else if (arg == "--devices") {
            options.devices = product::parse_devices(require_value("--devices"));
        } else if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor = std::stof(require_value("--rope-yarn-factor"));
            // PORT NOTE (2026-10-24): the upstream patch allowed [1,8]; the follow-up commit
            // caps it at 4 because the attention visible-keys ceiling cannot serve more, so a
            // factor that cannot work is refused at the command line, not at engine start.
            if (!(options.rope_yarn_factor >= 1.0F) || options.rope_yarn_factor > 4.0F) {
                throw std::invalid_argument(
                    "--rope-yarn-factor must be in [1,4] (1 disables YaRN; 4 is the attention "
                    "visible-keys ceiling)");
            }
        } else if (arg == "--rope-original-max-position") {
            options.rope_original_max_position = static_cast<std::uint32_t>(
                std::stoul(require_value("--rope-original-max-position")));
            if (options.rope_original_max_position == 0) {
                throw std::invalid_argument("--rope-original-max-position must be positive");
            }
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
            if (options.speculative.backend == SpeculativeBackend::None) {
                // --spec off also drops a profile's draft window and proposal head (like
                // --no-vision)
                options.speculative.draft_tokens  = 0;
                options.speculative.proposal_head = ProposalHead::Full;
            }
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--no-vision") { // also drops a profile's Vision device and token cap
            options.enable_vision = false;
            options.vision_device.reset();
            options.max_vision_tokens.reset();
        } else if (arg == "--vision-device") {
            options.vision_device = product::parse_device_id(require_value("--vision-device"));
        } else if (arg == "--max-vision-tokens") {
            options.max_vision_tokens =
                product::parse_max_vision_tokens(require_value("--max-vision-tokens"));
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-tp-mailbox") {
            options.tp_mailbox = false;
        } else if (arg == "--embedding-host") {
            options.embedding_host = true;
        } else if (arg == "--host-pinned-max-gib") {
            const std::string value = require_value("--host-pinned-max-gib");
            const auto bytes        = parse_positive_gib(value.c_str());
            if (!bytes) {
                throw std::invalid_argument("--host-pinned-max-gib must be a number of GiB "
                                            "representing at least 1 byte and less than 16 EiB");
            }
            options.host_pinned_max_bytes = *bytes;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--reasoning-close-retry") {
            options.reasoning_close_retry = true;
        } else if (arg == "--no-reasoning-close-retry") {
            options.reasoning_close_retry = false;
        } else if (arg == "--reasoning-eos-guard") {
            options.reasoning_eos_guard = true;
        } else if (arg == "--no-reasoning-eos-guard") {
            options.reasoning_eos_guard = false;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (options.log_speculation_detail && options.request_log_jsonl.empty()) {
        throw std::invalid_argument("--log-speculation-detail requires --request-log-jsonl");
    }
    if (prefill_miss_tuned && options.prefill_miss_dump_dir.empty()) {
        throw std::invalid_argument(
            "--prefill-miss-min-tokens and --prefill-miss-min-match require --prefill-miss-dump");
    }
    if (detail_steps_explicit && !options.log_speculation_detail) {
        throw std::invalid_argument(
            "--log-speculation-detail-max-steps requires --log-speculation-detail");
    }
    if (vram_headroom_mib.has_value()) {
        if (options.kv_capacity.mode != KvCapacityMode::Automatic) {
            throw std::invalid_argument("--vram-headroom-mib requires --kv-capacity auto");
        }
        options.kv_capacity = KvCapacityPolicy::automatic(*vram_headroom_mib << 20);
    }
    if (options.context_cache.disk_directory.empty() !=
        (options.context_cache.disk_capacity_bytes == 0)) {
        throw std::invalid_argument("--disk-cache-dir and --disk-cache-gib go together");
    }
    if (!options.allow_prefix_reuse) {
        if (context_capacity_explicit || !options.context_cache.disk_directory.empty()) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with context-cache capacity options");
        }
        options.context_cache.enabled                = false;
        options.context_cache.host_state_slots       = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
    }
    product::resolve_tensor_parallel_devices(options.tp, options.devices, options.device,
                                             device_explicit);
    product::validate_vision_options(options.enable_vision, options.vision_device,
                                     options.max_vision_tokens, options.devices);
    if (options.tp == 2) {
        // The Host tier holds both ranks' halves of a checkpoint (one pinned record per page and
        // per StateImage), so --host-kv-mib and --host-state-slots work as at tp 1. Unset, they
        // stay 0 at tp 2: on Qwen3.8-Flash-Next the routed experts already pin most of the RAM,
        // and the PLE table is read through the page cache. --profile 27b turns them on;
        // --profile flash-next sets 4 GiB of Host KV for the idle conversations' KV (v0.6) and 8
        // Host StateImages (884 MiB pinned): without them a capture that finds the 8+1 Device
        // StateImages full releases a whole idle conversation (reclaim_private_owners_for_capture),
        // and agents with parallel subagents (Claude Code) re-prefilled the smaller conversation
        // from the root at almost every turn (banco B4 19/10: 71 of 114 subagent turns).
        if (!host_state_explicit) { options.context_cache.host_state_slots = 0; }
        if (!host_kv_explicit) { options.context_cache.host_kv_capacity_bytes = 0; }
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.max_request_json_depth == 0 ||
        options.max_request_json_depth > kMaximumMaxRequestJsonDepth) {
        throw std::invalid_argument("--max-request-json-depth must be in [1," +
                                    std::to_string(kMaximumMaxRequestJsonDepth) + "]");
    }
    if (options.constraint_limits.max_depth == 0 ||
        options.constraint_limits.max_depth > kMaximumConstraintDepth) {
        throw std::invalid_argument("--constraint-max-depth must be in [1," +
                                    std::to_string(kMaximumConstraintDepth) + "]");
    }
    if (options.constraint_limits.max_items == 0) {
        throw std::invalid_argument("--constraint-max-items must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    // The key without putting it on the command line (ps, /proc/PID/cmdline, systemctl status):
    // --api-key-file, else NINFER_API_KEY. --api-key KEY keeps working.
    if (api_key_file) {
        if (api_key_explicit) {
            throw std::invalid_argument("--api-key and --api-key-file cannot be used together");
        }
        options.api_key = read_api_key_file(*api_key_file);
    } else if (!api_key_explicit) {
        if (const char* key = std::getenv("NINFER_API_KEY"); key != nullptr) {
            options.api_key = key;
        }
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    return std::string(artifact_model_name);
}

} // namespace ninfer::serve
