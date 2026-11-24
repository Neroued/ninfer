#pragma once

#include "models/common/qwen/vision/parameters.h"

#include "models/qwen3_5/config.h"
#include "models/load_options.h"
#include "ninfer/ops/moe.h"

#include <array>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5 {
class ModelDefinition;

namespace execution {

struct DenseProfile {
    ops::LinearProfile gate_up, down;
};

struct MoeLayerProfile {
    ops::MoeProfile math;
    ExpertBankId bank;
};

using FfnProfile = std::variant<DenseProfile, MoeLayerProfile>;

struct AttentionProfile {
    ops::ProjectionProfile projection;
    std::array<std::int32_t, 4> projection_rows;
    ops::LinearProfile output;
};

struct GdnProfile {
    ops::ProjectionProfile projection, control;
    std::array<std::int32_t, 2> projection_rows;
    ops::LinearProfile output;
};

struct BlockProfile {
    std::variant<AttentionProfile, GdnProfile> mixer;
    FfnProfile ffn;
};

struct TextProfile {
    ops::LinearProfile output_head;
    std::vector<BlockProfile> layers;
};

struct MtpProjectionProfile {
    ops::LinearProfile packed;
    std::array<std::int32_t, 4> output_rows;
    std::optional<std::array<ops::LinearProfile, 4>> rows;
};

struct MtpProfile {
    ops::LinearProfile input_projection, output, output_head;
    MtpProjectionProfile projection;
    FfnProfile ffn;
};

struct DraftBlockProfile {
    ops::LinearProfile output;
    DenseProfile mlp;
};

struct SelectorProfile {
    ops::LinearProfile hidden_projection;
};

struct DraftProfile {
    ops::LinearProfile feature_projection, output_head;
    std::vector<DraftBlockProfile> layers;
    std::optional<SelectorProfile> selector;
};

struct ProposalProfile {
    ops::LinearProfile head;
    std::uint32_t rows;
};

// Immutable execution geometry. Contains no allocation or execution addresses.
struct ModelProfile {
    Config config;
    models::LoadOptions options;
    std::uint32_t public_tokens;
    std::uint32_t control_token_limit = 0;
    TextProfile text;
    std::optional<MtpProfile> mtp;
    std::optional<qwen::VisionProfile> vision;
    std::optional<DraftProfile> draft;
    std::optional<ProposalProfile> proposal;
};

[[nodiscard]] std::shared_ptr<const ModelProfile> describe_model(const ModelDefinition&);

} // namespace execution
} // namespace ninfer::models::qwen3_5
