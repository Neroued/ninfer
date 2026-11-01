#pragma once

#include "artifact/materializer.h"
#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
};

struct WeightUseId {
    WeightId parameter;
    std::size_t use_index = std::numeric_limits<std::size_t>::max();
};

struct WeightUse {
    std::string input;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

struct BoundWeight {
    std::string name;
    std::vector<std::string> source_objects;
    WeightView view;
    std::vector<WeightUse> uses;
    // HostFile weights have no parent view; host code reads their mapped object segments.
    std::vector<artifact::MappedObjectSegment> mapped;
    WeightGeometry mapped_geometry;
};

struct AttentionWeights {
    WeightId query, key, gate, value;
    WeightId query_norm, key_norm, output;
};

struct GdnWeights {
    WeightId query, key, value, z;
    WeightId a_projection, b_projection, a_log, dt_bias;
    WeightId convolution, norm, output;
};

struct DenseWeights {
    WeightId gate, up, down;
};

struct MoeWeights {
    WeightId router, shared_score;
    std::vector<DenseWeights> experts;
    DenseWeights shared;
};

struct BlockWeights {
    WeightId input_norm, post_attention_norm;
    std::variant<AttentionWeights, GdnWeights> mixer;
    DenseWeights ffn;
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    WeightUseId output_head_use;
    std::vector<BlockWeights> layers;
};

struct MtpWeights {
    WeightId input_projection, embedding_norm, hidden_norm, final_norm;
    BlockWeights layer;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

struct NormWeights {
    WeightId weight, bias;
};

struct VisionBlockWeights {
    NormWeights norm1, norm2;
    WeightId query, key, value, query_bias, key_bias, value_bias;
    WeightId output, output_bias;
    WeightId fc1, fc1_bias, fc2, fc2_bias;
};

struct VisionWeights {
    WeightId patch_embedding, patch_embedding_bias, position_embedding;
    std::vector<VisionBlockWeights> layers;
    NormWeights merger_norm;
    WeightId merger_fc1, merger_fc1_bias, merger_fc2, merger_fc2_bias;
};

struct DraftAttentionWeights {
    WeightId query, key, value, context_key, context_value;
    WeightId query_norm, key_norm, output;
};

struct DynamicConvWeights {
    WeightId base_kernel, kernel_projection;
};

struct DraftBlockWeights {
    WeightId input_norm, post_attention_norm;
    DraftAttentionWeights attention;
    DenseWeights mlp;
    std::optional<DynamicConvWeights> attention_conv, mlp_conv;
};

struct SelectorWeights {
    WeightId hidden_projection, predecessor_codebook, successor_codebook;
};

struct DraftWeights {
    WeightId feature_projection, context_norm, final_norm;
    std::vector<DraftBlockWeights> layers;
    std::optional<SelectorWeights> selector;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

struct ProposalWeights {
    WeightId head;
    std::optional<WeightId> token_ids;
    std::uint32_t rows = 0;
    std::vector<std::int32_t> global_token_ids;
};

struct HyperConnectionWeights {
    WeightId norm, down, up;
    std::optional<WeightId> inject; // absent for the output mixers
};

struct QsaWeights {
    AttentionWeights attention;
    WeightId indexer_query, indexer_key, indexer_query_norm, indexer_key_norm;
};

struct PleWeights {
    WeightId table, key, value, key_norm, query_norm, conv_norm, convolution;
    // FP32 scalar of an nvfp4_row_e4m3_g16 table (FP4-Local); absent for row-scaled FP8 tables.
    WeightId table_global_scale;
};

// Qwen4Exp block: hyper-connection mixers around a QSA or GDN mixer and the FFN: the offloaded
// MoE (Qwen3.8-Flash-Next) or a dense SwiGLU FFN (mlp/gate, mlp/up, mlp/down).
struct Qwen4BlockWeights {
    HyperConnectionWeights attention_hc, ffn_hc;
    std::variant<QsaWeights, GdnWeights> mixer;
    std::variant<MoeWeights, DenseWeights> ffn;
    std::optional<PleWeights> ple;
};

struct Qwen4TextWeights {
    HyperConnectionWeights head;
    std::vector<Qwen4BlockWeights> layers;
};

struct Qwen4MtpWeights {
    WeightId embedding_norm, hidden_norm, embedding_projection, hidden_projection;
    HyperConnectionWeights head;
    Qwen4BlockWeights layer;
};

// Handles refer to the frozen model's weight array. No artifact ID lookup is needed in execution.
struct ModelWeights {
    TextWeights text;
    std::optional<VisionWeights> vision;
    std::optional<MtpWeights> mtp;
    std::optional<DraftWeights> draft;
    std::optional<ProposalWeights> proposal;
    std::optional<Qwen4TextWeights> qwen4;
    std::optional<Qwen4MtpWeights> qwen4_mtp;
};

} // namespace ninfer::models::qwen3_5
