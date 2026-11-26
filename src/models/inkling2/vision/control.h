#pragma once

#include "models/common/vision/window.h"
#include "models/inkling2/config.h"

#include <array>

namespace ninfer::models::inkling2 {
struct PreparedPromptData;
}

namespace ninfer::models::inkling2::vision {

struct EncoderBatch {
    std::vector<std::span<const std::uint16_t>> patches;
    std::vector<std::int32_t> positions, cu_seqlens, patch_columns, pool_rows;
    std::vector<std::array<std::int32_t, 4>> position_indices;
    std::vector<std::array<float, 4>> position_weights;
    std::uint32_t features = 0;
};

struct ExecutionInput {
    VisionConfig config;
    std::vector<models::frontend::VisionItem> items;
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> payloads;
    std::vector<models::vision::Unit> units;
};

struct WindowInput {
    EncoderBatch batch;
    std::shared_ptr<const ExecutionInput> owner;
};

struct Work {
    models::vision::WindowUse window;
    std::shared_ptr<const WindowInput> input;
};

[[nodiscard]] std::uint64_t patches_per_feature(const VisionConfig&) noexcept;
[[nodiscard]] std::vector<models::vision::Unit> units(std::span<const models::frontend::VisionItem>,
                                                      const VisionConfig&);
[[nodiscard]] EncoderBatch prepare_batch(const ExecutionInput&, const models::vision::WindowPlan&);
[[nodiscard]] std::shared_ptr<const ExecutionInput>
prepare_input(const PreparedPromptData&, const VisionConfig&, std::uint32_t feature_capacity);
[[nodiscard]] Work prepare_work(models::vision::WindowUse, std::shared_ptr<const ExecutionInput>);

} // namespace ninfer::models::inkling2::vision
