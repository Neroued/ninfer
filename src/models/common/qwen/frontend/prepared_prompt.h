#pragma once

#include "models/common/qwen/frontend/frontend.h"
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
    // Exact token frontiers at which this serialization can agree with a typed rewrite captured
    // by an earlier turn. Prefill splits at these frontiers so resumed and root execution use the
    // same chunk decomposition; they are not capture requests by themselves.
    std::vector<std::uint32_t> rewrite_execution_frontiers;
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
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::models::qwen
