#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes  = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes   = 2ULL << 30;
// Pinned Host tier of the prefix cache when --host-cache-mib is not given.
inline constexpr std::size_t kDefaultHostCacheBytes = 8ULL << 30;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8KeyNvfp4Value,
};

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // Startup-fixed K: MTP 1..5; DFlash and DFlash2 1..15 (query width K+1).
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostCachePin,
    CudaGraphPrepare,
    PrefixCacheLoad,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

// Prefix reuse through the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache.md):
// content-addressed 64-token KV blocks in one radix tree and sparse recurrent-state snapshots on
// its nodes, kept on the Device and in one pinned Host tier. Engine construction derives the unset
// optionals from max_concurrency, prefill_chunk and whether a Host tier exists
// (host_cache_bytes != 0); Engine::options() reports the effective values.
struct ContextCacheOptions {
    // Disabled, every request prefills from the root and nothing is retained.
    bool enabled = true;
    // Pinned Host slab pool that KV blocks and state snapshot images share. 0 disables the Host
    // tier: Device blocks and snapshots are then the only cache.
    std::size_t host_cache_bytes = kDefaultHostCacheBytes;
    // Device StateImage slots holding inactive snapshots (and tap/endpoint staging). Total Device
    // StateImage capacity is max_concurrency + device_snapshot_slots. Default: one per request
    // lane plus one staging slot with a Host tier, plus two without one (Device slots are then
    // the only snapshot storage).
    std::optional<std::uint32_t> device_snapshot_slots;
    // New prefill state snapshots one request may create. Default 8 with a Host tier, 2 without.
    std::optional<std::uint32_t> max_new_taps;
    // Geometric ladder base G: ladder taps target prompt_tokens - G * 2^k. Default
    // max(4096, 2 * prefill_chunk): ladder taps land on prefill chunk boundaries.
    std::optional<std::uint32_t> tap_ladder_tokens;
    // Ladder taps closer than this to another snapshot on the same path are skipped. Default
    // max(1024, prefill_chunk).
    std::optional<std::uint32_t> tap_min_gap_tokens;
    // Opt-in persistence of the Host tier: saved to this file when the Engine shuts down cleanly
    // and restored from it at startup, but only when it was written for the same artifact, KV and
    // state formats and `persistent_identity` (the product binary's build). Empty disables it.
    std::filesystem::path persistent_file;
    std::string persistent_identity;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    bool use_cuda_graph                    = true;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    StartupObserver startup_observer;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Terminal interpretation of model-origin tool-call markup. Parameter schemas guide JSON
// normalization but do not validate the call; only a structure/identity failure can return a
// complete marker region to ordinary content.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen                            = false;
    std::uint32_t structured_call_count         = 0;
    std::uint32_t empty_arguments_omitted       = 0;
    std::uint32_t schema_mismatch_arguments     = 0;
    ToolCallParseFallbackReason fallback_reason = ToolCallParseFallbackReason::None;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
};

enum class CacheRetentionHint : std::uint8_t {
    Default,
    LiveSession,
    Disposable,
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
    EngineObserved     = 1U << 4U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

// Protocol cache hints. The prefix cache is content-addressed: markers become state-snapshot
// positions, and the session and retention hints are validated but select nothing.
struct ContextCacheHints {
    std::optional<std::string> session_key;
    CacheRetentionHint retention = CacheRetentionHint::Default;
    std::vector<PromptCacheMarker> markers;
    // Protocols with their own automatic/explicit write policy disable the Engine's own snapshot
    // at the end of the tool definitions.
    bool allow_engine_automatic_shared_prefixes = true;
    bool update_session_index                   = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live observations require a Streaming consumer;
// phase timings may also be retained for an Aggregate terminal response.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    // Request wall phases at committed model-state boundaries. Prompt begins when admission and
    // its exact reuse choice are published and ends at the first accepted output token. Generation
    // spans the first through last accepted output token and therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                   = 0.0;
    double engine_boundary_exposed_seconds      = 0.0;
    double program_submit_exposed_seconds       = 0.0;
    double program_post_exposed_seconds         = 0.0;
    double engine_commit_output_exposed_seconds = 0.0;
    double engine_maintenance_exposed_seconds   = 0.0;
    double device_wait_exposed_seconds          = 0.0;
    double decode_host_exposed_seconds          = 0.0;
    double decode_device_wait_exposed_seconds   = 0.0;
    std::uint64_t prefill_units                 = 0;
    std::uint64_t decode_rounds                 = 0;
    std::uint64_t control_units                 = 0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
};

// The prefix-cache snapshot an admitted request resumed from: none, a previous request's endpoint
// (its state after its last committed token), or a prefill snapshot inside an earlier prompt.
enum class PrefixReusePath : std::uint8_t {
    Root,
    Endpoint,
    Snapshot,
};

// Prefix-cache admission of one request. `cached_prefix_tokens` is the longest prompt prefix held
// as cached KV blocks whether or not it was reusable: reuse also needs a state snapshot inside it,
// so a gap to the reused token count is prefix lost to snapshot placement. `restored_host_bytes`
// is what the admission copied back from the Host tier; the copies overlap the request's first
// prefill pass, so their time is part of its prefill.
struct MaterializationDiagnostics {
    std::uint32_t cached_prefix_tokens = 0;
    std::uint64_t restored_host_bytes  = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationDiagnostics&,
               const MaterializationDiagnostics&) noexcept = default;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;
    MaterializationDiagnostics materialization;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::size_t host_cache_capacity_bytes         = 0;
    std::size_t host_cache_occupied_bytes         = 0;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns      = 0;
    std::uint64_t program_submit_ns       = 0;
    std::uint64_t program_post_ns         = 0;
    std::uint64_t engine_commit_output_ns = 0;
    std::uint64_t engine_maintenance_ns   = 0;
    std::uint64_t device_wait_ns          = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t admission_policy_ns           = 0;
    std::uint64_t context_progress_ns           = 0;
    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t admission_policy_invocations  = 0;
    std::uint64_t context_progress_invocations  = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    // Actual prompt tokens evaluated by prefill; reused checkpoint-prefix tokens are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t terminal_pending_requests = 0;

    // Admissions by the snapshot they resumed from, and the prompt tokens they reused.
    std::uint64_t root_selections               = 0;
    std::uint64_t endpoint_selections           = 0;
    std::uint64_t snapshot_selections           = 0;
    std::uint64_t reused_prompt_tokens          = 0;
    std::uint32_t last_selected_frontier_tokens = 0;

    // Physical occupancy gauges.
    std::uint32_t device_state_occupied_slots      = 0;
    std::uint32_t device_main_kv_occupied_pages    = 0;
    std::uint32_t device_backend_kv_occupied_pages = 0;

    // Prefix cache (docs/maintainer/hybrid-prefix-cache.md). Block, snapshot and Host-tier
    // gauges are absolute; the rest are cumulative event counters.
    std::uint32_t cached_blocks             = 0; // Device-resident tree blocks
    std::uint32_t evictable_blocks          = 0;
    std::uint32_t tree_blocks               = 0; // Device or Host
    std::uint32_t snapshots                 = 0;
    std::uint64_t host_cache_capacity_bytes = 0;
    std::uint64_t host_cache_used_bytes     = 0;
    std::uint64_t blocks_inserted           = 0;
    std::uint64_t blocks_reattached         = 0;
    std::uint64_t blocks_duplicate          = 0;
    std::uint64_t taps_created              = 0;
    std::uint64_t taps_skipped              = 0;
    std::uint64_t endpoints_created         = 0;
    std::uint64_t host_image_writes         = 0;
    std::uint64_t host_block_writes         = 0;
    std::uint64_t host_image_restores       = 0;
    std::uint64_t host_block_restores       = 0;
    std::uint64_t host_write_bytes          = 0;
    std::uint64_t host_restore_bytes        = 0;
    std::uint64_t evicted_blocks            = 0;
    std::uint64_t host_snapshot_evictions   = 0;
    std::uint64_t host_dead_reclaims        = 0;
    std::uint64_t unbacked_node_losses      = 0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    ContextCostSummary context_cost;

    // Hybrid prefix cache restored from its persistent file at startup.
    struct PrefixCacheRestore {
        bool attempted = false;
        bool restored  = false;
        // Why nothing was restored (no file yet, incompatible file, I/O error).
        std::string message;
        std::uint64_t blocks    = 0;
        std::uint64_t snapshots = 0;
        std::uint64_t bytes     = 0;
        double seconds          = 0.0;
        // What the file holds and the Host tier bytes all of it takes, against this Engine's
        // tier. When the tier is smaller, only the snapshots it values most were restored.
        std::uint64_t saved_blocks        = 0;
        std::uint64_t saved_snapshots     = 0;
        std::uint64_t required_host_bytes = 0;
        std::uint64_t host_bytes          = 0;
    } prefix_cache;
};

} // namespace ninfer
