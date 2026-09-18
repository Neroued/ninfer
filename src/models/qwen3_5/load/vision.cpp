#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

VisionWeights bind_vision(Bindings& b, const VisionConfig& config, const TextConfig& target) {
    const auto h            = config.hidden_size;
    const auto intermediate = config.intermediate_size;
    VisionWeights out;
    out.patch_embedding =
        b.parameter("vision/patch_embedding", {h, config.patch_width()}, {"vision/patch_input"});
    out.patch_embedding_bias = b.direct("vision/patch_embedding_bias", {h});
    out.position_embedding =
        b.direct("vision/position_embedding", {config.num_position_embeddings, h});
    out.layers.reserve(config.depth);
    for (std::uint32_t i = 0; i < config.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.norm1       = {b.direct(p + "norm1_weight", {h}), b.direct(p + "norm1_bias", {h})};
        layer.norm2       = {b.direct(p + "norm2_weight", {h}), b.direct(p + "norm2_bias", {h})};
        layer.query       = b.parameter(p + "attention/query", {h, h}, {p + "attention_input"});
        layer.key         = b.parameter(p + "attention/key", {h, h}, {p + "attention_input"});
        layer.value       = b.parameter(p + "attention/value", {h, h}, {p + "attention_input"});
        layer.query_bias  = b.direct(p + "attention/query_bias", {h});
        layer.key_bias    = b.direct(p + "attention/key_bias", {h});
        layer.value_bias  = b.direct(p + "attention/value_bias", {h});
        layer.output      = b.parameter(p + "attention/output", {h, h}, {p + "attention_output"});
        layer.output_bias = b.direct(p + "attention/output_bias", {h});
        layer.fc1         = b.parameter(p + "mlp/fc1", {intermediate, h}, {p + "mlp_input"});
        layer.fc1_bias    = b.direct(p + "mlp/fc1_bias", {intermediate});
        layer.fc2         = b.parameter(p + "mlp/fc2", {h, intermediate}, {p + "mlp_activation"});
        layer.fc2_bias    = b.direct(p + "mlp/fc2_bias", {h});
        out.layers.push_back(layer);
    }
    const auto merger = config.merger_width();
    out.merger_norm   = {b.direct("vision/merger/norm_weight", {h}),
                         b.direct("vision/merger/norm_bias", {h})};
    out.merger_fc1    = b.parameter("vision/merger/fc1", {merger, merger}, {"vision/merger/input"});
    out.merger_fc1_bias = b.direct("vision/merger/fc1_bias", {merger});
    out.merger_fc2      = b.parameter("vision/merger/fc2", {target.hidden_size, merger},
                                      {"vision/merger/activation"});
    out.merger_fc2_bias = b.direct("vision/merger/fc2_bias", {target.hidden_size});
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
