#pragma once

#include "models/common/moe/weights.h"
#include "models/inkling2/definition.h"
#include "models/inkling2/parallel/plan.h"
#include "ninfer/ops/attention_geometry.h"
#include "ninfer/ops/block_attn_res.h"
#include "ninfer/ops/moe.h"
#include "ninfer/ops/projection_weights.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::inkling2::execution {

struct AttentionPassProfile {
    // SWA pass 0 fuses (key, value, query, output_gate, relative).
    // SWA pass > 0 and all Full passes use (query, output_gate, relative) + linear_pair(key, value).
    std::optional<ops::ProjectionProfile> qkv_split;
    std::array<std::int32_t, 5> qkv_rows{};
    ops::ProjectionProfile qrg_split;
    std::array<std::int32_t, 3> qrg_rows{};
    ops::LinearProfile key;
    ops::LinearProfile value;
    std::optional<ops::LinearProfile> k_conv_proj;
    std::optional<ops::LinearProfile> v_conv_proj;
    ops::LinearProfile output;
};

struct MoePassProfile {
    ops::SigmoidRouteProfile routing;
    ops::RoutedSwiGLUProfile routed;
    ExpertBankId bank;
    ops::ExpertSubsetKind subset = ops::ExpertSubsetKind::All;
};

struct PhysicalLayerProfile {
    bool is_swa  = false;
    bool in_core = false;
    ops::AttentionHeadGeometry attention_geometry;
    std::vector<AttentionPassProfile> attention_passes;
    std::vector<MoePassProfile> moe_passes;
};

struct TextProfile {
    ops::BlockAttnResParameters attn_res;
    ops::LinearProfile head;
    std::uint32_t max_blocks  = 0;
    std::uint32_t bank_blocks = 0;
    std::vector<PhysicalLayerProfile> layers;
};

struct VisionLayerProfile {
    ops::LinearProfile qkv, output, fc1, fc2;
};

struct VisionProfile {
    ops::LinearProfile patch, projector_in, projector_out;
    std::vector<VisionLayerProfile> layers;
};

struct LocalGeometry {
    std::uint32_t rank = 0;
    std::uint32_t size = 1;
    IndexRange vocabulary;
};

struct ModelProfile {
    Config config;
    LoadOptions options;
    LocalGeometry local;
    std::uint32_t public_tokens       = 0;
    std::uint32_t control_token_limit = 0;
    TextProfile text;
    std::optional<VisionProfile> vision;
};

[[nodiscard]] std::shared_ptr<const ModelProfile>
describe_model(const ModelDefinition&, std::span<const BoundWeight>,
               std::span<const moe::ExpertDescription>, const RankPlan&);

} // namespace ninfer::models::inkling2::execution
