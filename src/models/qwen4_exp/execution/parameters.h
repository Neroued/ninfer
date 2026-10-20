#pragma once

#include "models/qwen4_exp/execution/profile.h"
#include "models/qwen4_exp/model.h"
#include "models/common/moe/execution.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/ple.h"

namespace ninfer::models::qwen4_exp::execution {
using LinearParameters = ops::SingleProjectionWeight;

struct HcParameters {
    Tensor norm;
    ops::HcMixWeights mix;
};

struct FinalHcParameters {
    Tensor norm;
    ops::HcFinalMixWeights mix;
};

struct GdnParameters {
    ops::ProjectionWeights input, control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct MainQsaProjection {
    LinearParameters query_gate, kv_index;
};

struct QsaParameters {
    std::variant<MainQsaProjection, LinearParameters> input;
    ops::QsaPrepareParameters prepare;
    LinearParameters output;
};

struct QsaHistoryParameters {
    LinearParameters input;
    ops::QsaKvPrepareParameters prepare;
};

struct MtpHistoryParameters {
    ops::HcFinalMixWeights mix;
    QsaHistoryParameters attention;
};

struct BlockParameters {
    HcParameters attention_hc, moe_hc;
    std::variant<GdnParameters, QsaParameters> mixer;
    moe::Parameters moe;
};

struct PleParameters {
    LinearParameters projection;
    ops::PleGateParameters gate;
    Tensor convolution;
};

struct TextParameters {
    Weight embedding;
    LinearParameters head;
    FinalHcParameters final_hc;
    std::vector<BlockParameters> layers;
    PleParameters ple;
};

struct MtpParameters {
    Weight embedding;
    LinearParameters head, fc_embedding, fc_hidden;
    Tensor embedding_norm, hidden_norm;
    BlockParameters layer;
    FinalHcParameters final_hc;
    MtpHistoryParameters history;
};

class Parameters {
public:
    explicit Parameters(const ModelShard&, std::shared_ptr<const ModelProfile>);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    const ModelShard& model;
    const std::shared_ptr<const ModelProfile> profile;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<qwen::VisionParameters> vision;
};
} // namespace ninfer::models::qwen4_exp::execution
