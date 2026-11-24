#pragma once

#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/common/qwen/vision/parameters.h"
#include "models/common/vision/window.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen {

struct VisionItemRange {
    std::uint32_t begin = 0, end = 0;
};

struct VisionExecutionInput {
    VisionConfig config;
    std::vector<models::frontend::VisionItem> vision_items;
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> media_payloads;
    std::vector<VisionItemRange> items;
    std::vector<vision::Unit> units;

    [[nodiscard]] bool can_checkpoint(std::uint32_t frontier) const noexcept;
};

[[nodiscard]] std::shared_ptr<const VisionExecutionInput>
prepare_vision_input(const PreparedPromptData&, const VisionConfig&, std::uint32_t capacity);

struct VisionEncoderBatch {
    std::vector<std::span<const std::uint16_t>> patches;
    std::vector<std::int32_t> positions, position_indices, cu_seqlens;
    std::vector<float> position_weights;
    std::uint32_t features = 0;
};

struct VisionWindowInput {
    std::vector<VisionEncoderBatch> batches;
    std::shared_ptr<const VisionExecutionInput> owner;
};

struct VisionWork {
    vision::WindowUse window;
    std::shared_ptr<const VisionWindowInput> input;
};

[[nodiscard]] VisionWork prepare_vision_work(vision::WindowUse,
                                             std::shared_ptr<const VisionExecutionInput>,
                                             std::uint32_t ranks);

[[nodiscard]] std::vector<vision::Unit> vision_units(std::span<const models::frontend::VisionItem>,
                                                     const VisionConfig&);
[[nodiscard]] VisionEncoderBatch encoder_batch(const VisionExecutionInput&,
                                               const vision::WindowPlan&, std::uint32_t rank);

} // namespace ninfer::models::qwen
