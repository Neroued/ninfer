#pragma once

#include "ninfer/types.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace ninfer::models::frontend {

inline constexpr std::size_t kPreparedSessionKeyCapacity = kMaximumContextCacheSessionKeyBytes;

struct FrontendOptions {
    std::filesystem::path chat_template_path;
    std::size_t grammar_cache_bytes        = 256ULL * 1024 * 1024;
    bool vision_enabled                    = true;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
};

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
    std::vector<PreparedCacheOpportunity> opportunities;
    std::vector<std::uint32_t> semantic_boundaries;
    bool automatic_checkpoints = true;
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

struct PromptData {
    std::vector<TokenId> token_ids;
    PreparedContextCache context_cache;
    PrepareStats prepare;
};

} // namespace ninfer::models::frontend
