#pragma once

#include "models/common/vision/window.h"
#include "models/glm5next/config.h"

namespace ninfer::models::glm5next {
struct PreparedPromptData;
}

namespace ninfer::models::glm5next::vision {

struct EncoderBatch {
    std::vector<std::span<const std::uint16_t>> patches;
    std::vector<std::int32_t> positions, cu_seqlens;
    std::uint32_t features = 0;
};

struct ExecutionInput {
    VisionConfig config;
    std::vector<models::frontend::VisionItem> items;
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> payloads;
    std::vector<models::vision::Unit> units;
};

struct WindowInput {
    std::vector<EncoderBatch> batches;
    std::shared_ptr<const ExecutionInput> owner;
};

struct Work {
    models::vision::WindowUse window;
    std::shared_ptr<const WindowInput> input;
};

[[nodiscard]] std::vector<models::vision::Unit> units(std::span<const models::frontend::VisionItem>,
                                                      const VisionConfig&);
[[nodiscard]] EncoderBatch
prepare_batch(std::span<const models::frontend::VisionItem>,
              std::span<const std::shared_ptr<const models::frontend::PreparedMediaPayload>>,
              const models::vision::WindowPlan&, std::uint32_t rank, const VisionConfig&);
[[nodiscard]] std::shared_ptr<const ExecutionInput>
prepare_input(const PreparedPromptData&, const VisionConfig&, std::uint32_t feature_capacity);
[[nodiscard]] Work prepare_work(models::vision::WindowUse, std::shared_ptr<const ExecutionInput>,
                                std::uint32_t ranks);

} // namespace ninfer::models::glm5next::vision
