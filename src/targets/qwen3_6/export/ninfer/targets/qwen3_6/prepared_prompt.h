#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6 {

namespace frontend_internal {
struct ToolCallOutputContract;
}

inline constexpr std::size_t kPreparedVisionPatchFeatures = 3ULL * 2ULL * 16ULL * 16ULL;
inline constexpr std::uint64_t kRawPatchesPerVisionToken  = 4;
// Aggregate prompt capacity and one-item execution capacity are intentionally distinct. Multiple
// media items are retained by one prepared prompt but pass through the Vision tower sequentially.
inline constexpr std::uint64_t kMaximumPromptVisionTokens = 32'768;
inline constexpr std::uint64_t kMaximumPromptVisionRawPatches =
    kMaximumPromptVisionTokens * kRawPatchesPerVisionToken;
inline constexpr std::uint64_t kMaximumVisionItemTokens = 16'384;
inline constexpr std::uint64_t kMaximumVisionItemRawPatches =
    kMaximumVisionItemTokens * kRawPatchesPerVisionToken;

struct PreparedMediaPayload {
    // Exact row-major BF16 input consumed by the Vision patch projection.
    std::unique_ptr<std::uint16_t[]> patches;
    std::size_t patch_elements = 0;

    [[nodiscard]] std::span<std::uint16_t> mutable_span() noexcept {
        return {patches.get(), patch_elements};
    }

    [[nodiscard]] std::span<const std::uint16_t> span() const noexcept {
        return {patches.get(), patch_elements};
    }
};

enum class PromptModality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    std::int32_t temporal = 0;
    std::int32_t height   = 0;
    std::int32_t width    = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    PromptModality modality = PromptModality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    // SHA-256 of the owned encoded media bytes. Grid/modality/span identity is carried
    // separately so this digest binds the content without retaining the request payload.
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

enum class RewriteCheckpointKind : std::uint8_t {
    TurnClosure,
    ResponseReplay,
};

struct RewriteCheckpointSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct PromptIdentity {
    bool reusable = true;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    // Exact token frontiers at which this serialization can agree with a typed rewrite captured
    // by an earlier turn. Prefill splits at these frontiers so resumed and root execution use the
    // same GDN decomposition; they are not capture requests by themselves.
    std::vector<std::uint32_t> rewrite_execution_frontiers;
};

inline constexpr std::size_t kPreparedSessionKeyCapacity = kMaximumContextCacheSessionKeyBytes;

struct PreparedSessionKey {
    std::uint16_t size = 0;
    std::array<char, kPreparedSessionKeyCapacity> bytes{};

    [[nodiscard]] std::string_view view() const noexcept { return {bytes.data(), size}; }

    [[nodiscard]] friend bool operator==(const PreparedSessionKey&,
                                         const PreparedSessionKey&) noexcept = default;
};

struct PreparedCacheOpportunity {
    PromptCacheMarkerKind kind       = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
    std::uint32_t frontier           = 0;
    std::uint32_t input_order        = 0;

    [[nodiscard]] friend bool operator==(PreparedCacheOpportunity,
                                         PreparedCacheOpportunity) noexcept = default;
};

struct PreparedContextCache {
    std::optional<PreparedSessionKey> session_key;
    runtime::RetentionClass retention = runtime::RetentionClass::RecentPrivate;
    std::vector<PreparedCacheOpportunity> opportunities;
    // Controls replacement of a named SessionIndex entry, not anonymous source ownership.
    bool update_session_index = true;
};

struct PrepareStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::uint64_t attention_pairs        = 0; // Informational; not enforced against any budget.
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    std::vector<std::uint8_t> token_types;
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    // One immutable payload per Vision item, in the same order as vision_items.
    std::vector<std::shared_ptr<const PreparedMediaPayload>> media_payloads;
    std::vector<VisionItem> vision_items;
    PromptIdentity identity;
    PreparedContextCache context_cache;
    std::shared_ptr<const frontend_internal::ToolCallOutputContract> tool_call_output;
    bool starts_in_reasoning = false;
    PrepareStats prepare;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;

    [[nodiscard]] bool has_media() const noexcept { return !vision_items.empty(); }

    // Payload slots remain indexed one-to-one with vision_items for the lifetime of the prompt.
    // Releasing host storage must not destroy that structural identity while a Vision prefill
    // session can still revisit the same item in a later Text chunk.
    void release_all_media_payloads() noexcept {
        for (auto& payload : media_payloads) { payload.reset(); }
    }
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::targets::qwen3_6
