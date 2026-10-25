#pragma once

#include "models/glm5next/definition.h"
#include "models/glm5next/parallel/plan.h"
#include "ninfer/ops/kpool_mla.h"
#include "ninfer/ops/mhc.h"
#include "ninfer/ops/moe.h"

#include <array>

namespace ninfer::models::glm5next::execution {

struct KdaProfile {
    ops::LinearProfile qkv, control, forget, gate, output;
    std::array<std::int32_t, 3> control_rows;
};

struct MlaProfile {
    std::array<ops::KpoolMlaPrepareProfile, 3> prepare;
    ops::LinearProfile query_index, query, output;
    std::array<std::int32_t, 2> query_rows;
    ops::KpoolIndexerProfile indexer;
    ops::KpoolSparseMlaProfile attention;
};

struct DenseProfile {
    ops::SwiGLUProfile math;
};

struct MoeProfile {
    ops::SigmoidMoeProfile math;
    ExpertBankId bank;
    ops::ExpertActivationDivisors activation_divisors;
};

struct BlockProfile {
    std::variant<KdaProfile, MlaProfile> mixer;
    std::variant<DenseProfile, MoeProfile> ffn;
};

struct TextProfile {
    ops::MhcGeometry mhc;
    ops::MhcPreParameters mhc_parameters;
    ops::LinearProfile head;
    std::vector<BlockProfile> layers;
};

struct MtpProfile {
    ops::LinearProfile stem, head;
    MlaProfile attention;
    MoeProfile moe;
};

struct VisionBlockProfile {
    ops::LinearProfile qkv, output, gate_up, down;
};

struct VisionProfile {
    ops::LinearProfile patch, downsample, merger_projection, merger_gate_up, merger_down;
    std::vector<VisionBlockProfile> layers;
};

// Describes selected rank-local weights without retaining their addresses or source Reader.
struct ModelProfile {
    Config config;
    LoadOptions options;

    struct LocalGeometry {
        std::uint32_t rank, size;
        IndexRange kda_heads, vocabulary, dense_intermediate, shared_intermediate;
    } local;

    std::uint32_t public_tokens;
    TextProfile text;
    std::optional<MtpProfile> mtp;
    std::optional<VisionProfile> vision;
};

[[nodiscard]] std::shared_ptr<const ModelProfile>
describe_model(const ModelDefinition&, std::span<const BoundWeight>, const RankPlan&);

} // namespace ninfer::models::glm5next::execution
