#pragma once

#include "models/common/qwen/frontend/frontend.h"
#include "models/common/context/prefix.h"
#include "models/common/frontend/media.h"
#include "models/common/frontend/prompt.h"
#include "models/common/qwen/frontend/output.h"
#include "models/common/qwen/frontend/resources.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen {

struct PromptIdentity {
    bool reusable = true;
    std::optional<models::frontend::RewriteCheckpointSpec> rewrite_checkpoint;
};

struct PreparedPromptData : models::frontend::PromptData {
    std::vector<std::uint8_t> token_types;
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    // One immutable payload per Vision item, in the same order as vision_items.
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> media_payloads;
    std::vector<models::frontend::VisionItem> vision_items;
    PromptIdentity identity;
    frontend::PreparedOutput output;
    SamplingPreset sampling;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;

    [[nodiscard]] bool has_media() const noexcept { return !vision_items.empty(); }
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static std::shared_ptr<const PreparedPromptData>
    shared(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

class PromptPrefixKeyer {
public:
    PromptPrefixKeyer() = default;
    PromptPrefixKeyer(const PreparedPrompt& prompt, std::uint32_t domain);

    [[nodiscard]] std::uint32_t max_frontier() const noexcept { return max_frontier_; }

    [[nodiscard]] runtime::PrefixKey key_at(std::uint32_t frontier) const;

private:
    std::shared_ptr<const PreparedPromptData> data_;
    std::uint32_t domain_       = 0;
    std::uint32_t max_frontier_ = 0;
    std::vector<context::MediaIdentityView> media_;
    context::PrefixFingerprints fingerprints_;
};

} // namespace ninfer::models::qwen
