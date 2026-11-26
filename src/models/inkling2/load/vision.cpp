#include "models/inkling2/load/bindings.h"

#include <string>

namespace ninfer::models::inkling2::loading {

VisionWeights bind_vision(Bindings& b, const VisionConfig& c) {
    const auto h    = c.hidden;
    const auto grid = std::uint64_t(c.posemb_grid_size) * c.posemb_grid_size;
    const auto linear = [&](const std::string& name, std::uint64_t output, std::uint64_t input,
                            const std::string& use) {
        return b.parameter(name, {output, input}, {use});
    };

    VisionWeights out;
    out.patch_embedding =
        linear("vision/patch_embedding", h, c.patch_width(), "vision/patch_input");
    out.position_embedding     = b.direct("vision/position_embedding", {grid, h}, QType::BF16);
    out.class_embedding        = b.direct("vision/class_embedding", {h}, QType::BF16);
    out.cls_position_embedding = b.direct("vision/cls_position_embedding", {h}, QType::BF16);
    out.pre_norm_weight        = b.direct("vision/pre_norm_weight", {h}, QType::BF16);
    out.pre_norm_bias          = b.direct("vision/pre_norm_bias", {h}, QType::BF16);

    out.layers.reserve(c.depth);
    for (std::uint32_t i = 0; i < c.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.ln_1_weight = b.direct(p + "ln_1_weight", {h}, QType::BF16);
        layer.ln_1_bias   = b.direct(p + "ln_1_bias", {h}, QType::BF16);
        layer.ln_2_weight = b.direct(p + "ln_2_weight", {h}, QType::BF16);
        layer.ln_2_bias   = b.direct(p + "ln_2_bias", {h}, QType::BF16);
        layer.ls_1_gamma  = b.direct(p + "ls_1_gamma", {h}, QType::BF16);
        layer.ls_2_gamma  = b.direct(p + "ls_2_gamma", {h}, QType::BF16);
        layer.query       = linear(p + "attention/query", h, h, p + "attention_input");
        layer.key         = linear(p + "attention/key", h, h, p + "attention_input");
        layer.value       = linear(p + "attention/value", h, h, p + "attention_input");
        layer.query_bias  = b.direct(p + "attention/query_bias", {h}, QType::BF16);
        layer.key_bias    = b.direct(p + "attention/key_bias", {h}, QType::BF16);
        layer.value_bias  = b.direct(p + "attention/value_bias", {h}, QType::BF16);
        layer.output      = linear(p + "attention/output", h, h, p + "attention_output");
        layer.output_bias = b.direct(p + "attention/output_bias", {h}, QType::BF16);
        layer.fc1         = linear(p + "mlp/fc1", c.intermediate, h, p + "mlp_input");
        layer.fc1_bias    = b.direct(p + "mlp/fc1_bias", {c.intermediate}, QType::BF16);
        layer.fc2         = linear(p + "mlp/fc2", h, c.intermediate, p + "mlp_activation");
        layer.fc2_bias    = b.direct(p + "mlp/fc2_bias", {h}, QType::BF16);
        out.layers.push_back(layer);
    }

    out.projector_in = linear("vision/projector/in", c.projection_hidden_dim, c.merge_width(),
                              "vision/projector/input");
    out.projector_out = linear("vision/projector/out", c.text_hidden, c.projection_hidden_dim,
                               "vision/projector/activation");
    if (c.projector_bias) {
        out.projector_in_bias =
            b.direct("vision/projector/in_bias", {c.projection_hidden_dim}, QType::BF16);
        out.projector_out_bias =
            b.direct("vision/projector/out_bias", {c.text_hidden}, QType::BF16);
    }
    return out;
}

} // namespace ninfer::models::inkling2::loading
