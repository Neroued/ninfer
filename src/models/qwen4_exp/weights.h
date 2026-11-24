#pragma once

#include "models/common/weights.h"
#include "models/common/qwen/vision/parameters.h"
#include "core/weight_bank.h"

#include <optional>
#include <variant>

namespace ninfer::models::qwen4_exp {

struct HcWeights {
    WeightId norm, down, up;
    std::optional<WeightId> injection;
};

struct GdnWeights {
    WeightId query, key, value, z, a, b, a_log, dt_bias, convolution, norm, output;
};

struct QsaWeights {
    WeightId query, key, gate, value, index_query, index_key;
    WeightId query_norm, key_norm, index_query_norm, index_key_norm, output;
};

struct MoeWeights {
    WeightId router, shared_score, shared_gate, shared_up, shared_down;
    WeightId gate, up, down;
    ExpertBankId bank;
};

struct PleWeights {
    WeightId key, value, norm_key, norm_query, norm_conv, convolution;
};

struct BlockWeights {
    HcWeights attention_hc, moe_hc;
    std::variant<GdnWeights, QsaWeights> mixer;
    MoeWeights moe;
};

struct TextWeights {
    WeightId embedding, head;
    HcWeights final_hc;
    std::vector<BlockWeights> layers;
    PleWeights ple;
};

struct MtpWeights {
    WeightId embedding, head, embedding_norm, hidden_norm, fc_embedding, fc_hidden;
    BlockWeights layer;
    HcWeights final_hc;
};

struct ModelWeights {
    TextWeights text;
    std::optional<MtpWeights> mtp;
    std::optional<qwen::VisionWeights> vision;
};

} // namespace ninfer::models::qwen4_exp
