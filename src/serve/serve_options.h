#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens               = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes = 384ULL << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                  // empty => no auth
    std::string model_id = "qwen3.6-27b"; // reported by /v1/models
    std::string request_log_jsonl;        // empty => structured request logging disabled
    std::uint32_t max_context     = 8192;
    std::uint32_t prefill_chunk   = 1024;
    std::size_t max_request_bytes = kDefaultMaxRequestBytes;
    int device                    = 0;
    int mtp_draft_tokens          = 0;
    KvCacheStorage kv_cache       = KvCacheStorage::BFloat16;
    bool enable_vision            = false;
    bool use_cuda_graph           = true;
    bool allow_prefix_reuse       = true;
    ProposalHead proposal_head    = ProposalHead::Full;
    bool enable_thinking =
        true; // default thinking mode for the generation prompt (--no-thinking opts out)
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
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
