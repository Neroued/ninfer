#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact identity.model_id
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context              = 8192;
    KvCapacityPolicy kv_capacity           = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency          = 1;
    std::uint32_t max_pending_requests     = 16;
    std::uint32_t pending_timeout_ms       = 30000;
    std::uint32_t prefill_chunk            = 1024;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision      = false;
    bool use_cuda_graph     = true;
    bool allow_prefix_reuse = true;
    bool enable_thinking =
        true; // default thinking mode for the generation prompt (--no-thinking opts out)
    bool preserve_thinking = false; // keep thinking blocks in cached prefix across user turns
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Default sampler applied when a request omits a field. Defaults match the
    // Qwen3 thinking recommendation so real chat clients get non-degenerate
    // decoding out of the box; a request may override any field, and --greedy
    // forces the exact-argmax path (temperature 0).
    float sampling_temperature       = 0.6f;
    float sampling_top_p             = 0.95f;
    int sampling_top_k               = 20;
    float sampling_presence_penalty  = 1.0f;
    float sampling_frequency_penalty = 0.0f;
    // Fixes the seed used when a request omits `seed`; when unset each such
    // request draws a fresh random seed so regenerations differ.
    std::optional<std::uint64_t> sampling_seed;
    bool greedy = false; // --greedy: force temperature 0 (exact argmax)

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
