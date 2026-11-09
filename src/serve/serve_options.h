#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens               = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes = 384ULL << 20;
// Nesting of objects and arrays in a request body (and in JSON carried as a string, such as tool
// call arguments), checked on the bytes before parsing: the prompt path reads client JSON with
// recursive code (rendering, copies, serialization), so a deep document must not reach it.
inline constexpr std::uint32_t kDefaultMaxRequestJsonDepth = 1000;
// The template renderer converts client JSON recursively: about 12 000 levels exhaust an 8 MiB
// thread stack (measured), so the maximum keeps a margin.
inline constexpr std::uint32_t kMaximumMaxRequestJsonDepth = 5000;
inline constexpr std::size_t kDefaultResponseStoreRecords  = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes    = 256ULL << 20;
// --log-speculation-detail records the accepted draft length of at most this many decode steps
// per request (one byte each in the JSONL record) unless --log-speculation-detail-max-steps
// changes it.
inline constexpr std::uint32_t kDefaultSpeculationDetailMaxSteps = 4096;
inline constexpr std::uint32_t kMaximumSpeculationDetailMaxSteps = 1U << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::filesystem::path chat_template_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context            = 8192;
    KvCapacityPolicy kv_capacity         = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency        = 1;
    std::uint32_t max_pending_requests   = 16;
    std::uint32_t pending_timeout_ms     = kDefaultPendingTimeoutMs;
    float decode_share                   = 0.0F; // --decode-share (EngineOptions::decode_share)
    std::uint32_t prefill_chunk          = 1024;
    std::uint32_t kv_lease_window_tokens = 4096; // --kv-lease-window; 0 = reserve max_tokens
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms  = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes        = kDefaultMaxRequestBytes;
    std::uint32_t max_request_json_depth = kDefaultMaxRequestJsonDepth; // --max-request-json-depth
    std::size_t media_cache_bytes        = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes         = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    // Structured output limits (--constraint-*): input size/depth/items checked in the HTTP layer
    // and again before compilation, compilation deadline and concurrency, grammar cache, matcher
    // time budgets of a generating request.
    OutputConstraintLimits constraint_limits;
    int device = 0;
    int tp     = 1;
    std::vector<int> devices; // One id per tensor-parallel rank; {device} at tp 1.
    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    // YaRN RoPE extension. factor <= 1 is DISABLED and must stay bit-identical to the
    // unmodified engine: the fixed-frequency path is bypassed only when factor > 1.
    float rope_yarn_factor                   = 1.0F;
    std::uint32_t rope_original_max_position = 262144;
    SpeculativeOptions speculative;
    ContextCacheOptions context_cache;
    bool enable_vision = false;
    std::optional<int> vision_device;               // CUDA device of the Vision tower
    std::optional<std::uint32_t> max_vision_tokens; // merged-token ceiling of one media item
    bool use_cuda_graph = true;
    bool tp_mailbox     = true;
    bool embedding_host = false; // token embedding table in mapped pinned host memory
    // Page-locked limit of the host-mapped weights (--host-pinned-max-gib); 0 = none.
    std::uint64_t host_pinned_max_bytes = 0;
    bool allow_prefix_reuse             = true;
    // --reasoning-close-retry (on by default since v0.7, every model; --no-reasoning-close-retry):
    // a reply that ends with the model's terminal token inside its thinking (or right after
    // "</think>" with blank content and no tool call) is closed with "\n</think>\n\n" and continued
    // once (serve/reasoning_retry.h).
    bool reasoning_close_retry = true;
    // --reasoning-eos-guard (on by default since v0.7, every model; --no-reasoning-eos-guard): the
    // first token of a reply that starts in thinking is never a terminal token; a terminal draw is
    // redrawn without the terminal ids (models/qwen3_5/execution/reasoning_eos_guard.h).
    // Token-identical when it does not fire.
    bool reasoning_eos_guard = true;
    // Private long-anchor opportunities at the message boundaries before the final message: the
    // last N of them (0 disables). They keep a checkpoint at the point where a cancelled or edited
    // turn branches from the conversation; see to_prompt_input.
    // Unset = the model's default (resolve_turn_anchors): 0 on Qwen4Exp, 1 elsewhere.
    // capy-flash (T152): on Qwen3.8-Flash-Next every prefill pass reads the non-resident experts
    // from host RAM (~0.8 s fixed), and the anchor splits one pass more per agentic turn (TTFT 2.67
    // -> 1.82 s without it); a cancelled or edited turn then re-prefills the conversation. T161: on
    // models whose weights are all on the GPU (Qwen3.8-27B) the extra pass costs nothing
    // measurable, so they keep the anchor.
    std::optional<std::uint32_t> turn_anchors;
    // T64/T66c Qwen4Exp product options (no effect on other models): elastic KV <-> expert slots
    // (--elastic, --kv-grow, --kv-budget-gib), the decode miss split (--miss-split
    // auto|calibrate|off) and the prompt-chunk width from which the missing experts stream over
    // PCIe (--stream-min-tokens, RoutedExpertOptions::stream_min_tokens; 0 = never).
    ElasticKvOptions elastic_kv;
    // --kv-ram-active (EngineOptions::kv_ram_active): the generating conversation's KV in RAM.
    bool kv_ram_active = false;
    std::optional<MissSplitMode> miss_split;
    bool miss_split_calibrate = false; // --miss-split calibrate (MissSplitOptions::calibrate)
    std::optional<std::uint32_t> stream_min_tokens;
    // RoutedExpertOptions::cpu_threads and gguf_cpu_threads (Qwen4Exp offload backend: the CPU
    // threads of the missing experts, pinned one per allowed physical core, spread over the L3
    // domains). Unset = resolved at load from the machine (ops::host::auto_expert_cpu_threads).
    std::optional<std::uint32_t> expert_cpu_threads;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    std::optional<std::uint32_t> default_thinking_budget;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    product::LogLevel log_level = product::LogLevel::Info;

    // Test measurement in request_done (per-step accepted draft lengths, prompt n-gram overlap);
    // requires the JSONL log.
    bool log_speculation_detail                = false;
    std::uint32_t speculation_detail_max_steps = kDefaultSpeculationDetailMaxSteps;

    // --prefill-miss-dump DIR (diagnostic, empty => off): writes the new prompt and the closest
    // recent context for a prefix-reuse miss; criteria in serve/prefill_miss_dump.h.
    std::string prefill_miss_dump_dir;
    std::uint32_t prefill_miss_min_tokens = 10000;
    double prefill_miss_min_match         = 0.5;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
// --turn-anchors when given, otherwise the model's default: 0 for Qwen4ExpForCausalLM (each prefill
// pass reads the non-resident experts from host RAM), 1 for every other architecture.
[[nodiscard]] std::uint32_t resolve_turn_anchors(std::optional<std::uint32_t> configured,
                                                 std::string_view architecture);

std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
