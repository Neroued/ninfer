#pragma once

#include "models/qwen4_exp/config.h"
#include "ninfer/ops/moe.h"
#include "ninfer/ops/gated_residual.h"
#include "core/weight_bank.h"

#include <memory>
#include <variant>

namespace ninfer::models::qwen4_exp {
class LoadPlan;
class Model;

namespace execution {
struct GdnProfile {
    ops::ProjectionProfile input, control;
    std::array<std::int32_t, 2> input_rows;
    ops::LinearProfile output;
};

struct MainQsaProjectionProfile {
    ops::LinearProfile query_gate, kv_index;
    std::array<std::int32_t, 2> query_gate_rows;
    std::array<std::int32_t, 4> kv_index_rows;
};

struct QsaProfile {
    std::variant<MainQsaProjectionProfile, ops::LinearProfile> input;
    std::array<std::int32_t, 6> output_rows;
    ops::LinearProfile output;
};

struct MoeProfile {
    ops::MoeProfile math;
    ExpertBankId bank;
    ops::ExpertActivationDivisors activation_divisors;
};

struct BlockProfile {
    ops::HcMixProfile attention_hc, moe_hc;
    std::variant<GdnProfile, QsaProfile> mixer;
    MoeProfile moe;
};

struct TextProfile {
    std::vector<BlockProfile> layers;
    ops::HcFinalMixProfile final_hc;
    ops::LinearProfile ple, head;
};

struct MtpProfile {
    ops::LinearProfile fc_embedding, fc_hidden, head;
    BlockProfile layer;
    ops::HcFinalMixProfile final_hc;
};

// Immutable, address-free descriptions of the selected actual representations.
struct BankLayout {
    WeightBankLayout gate_up, down;
};

struct ModelProfile {
    Config config;
    LoadOptions options;
    std::uint32_t public_tokens;
    TextProfile text;
    std::optional<MtpProfile> mtp;
    std::optional<qwen::VisionProfile> vision;
    std::vector<BankLayout> banks;
};

[[nodiscard]] std::shared_ptr<const ModelProfile> describe_model(const LoadPlan&);
[[nodiscard]] std::shared_ptr<const ModelProfile> describe_model(const Model&);
[[nodiscard]] std::size_t qsa_projection_workspace_bytes(const QsaProfile&, std::int32_t first,
                                                         std::int32_t last);
} // namespace execution
} // namespace ninfer::models::qwen4_exp
