#include "models/glm5next/load/bindings.h"

namespace ninfer::models::glm5next::loading {

VisionWeights bind_vision(Bindings& b, const VisionConfig& c) {
    const auto h      = c.hidden;
    const auto linear = [&](const std::string& name, std::uint64_t output, std::uint64_t input,
                            const std::string& use) {
        return b.parameter(name, {output, input}, {use}, QType::BF16);
    };

    VisionWeights out;
    out.patch_embedding =
        linear("vision/patch_embedding", h, c.patch_width(), "vision/patch_input");
    out.patch_embedding_bias = b.direct("vision/patch_embedding_bias", {h});
    out.post_norm            = b.direct("vision/post_norm", {h});
    out.layers.reserve(c.depth);
    for (std::uint32_t i = 0; i < c.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {h});
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        layer.query               = linear(p + "attention/query", h, h, p + "attention_input");
        layer.key                 = linear(p + "attention/key", h, h, p + "attention_input");
        layer.value               = linear(p + "attention/value", h, h, p + "attention_input");
        layer.query_bias          = b.direct(p + "attention/query_bias", {h});
        layer.key_bias            = b.direct(p + "attention/key_bias", {h});
        layer.value_bias          = b.direct(p + "attention/value_bias", {h});
        layer.query_norm          = b.direct(p + "attention/query_norm", {c.head_dim()});
        layer.key_norm            = b.direct(p + "attention/key_norm", {c.head_dim()});
        layer.output              = linear(p + "attention/output", h, h, p + "attention_output");
        layer.output_bias         = b.direct(p + "attention/output_bias", {h});
        layer.mlp                 = {linear(p + "mlp/gate", c.intermediate, h, p + "mlp_input"),
                                     linear(p + "mlp/up", c.intermediate, h, p + "mlp_input"),
                                     linear(p + "mlp/down", h, c.intermediate, p + "mlp/product")};
        layer.gate_bias           = b.direct(p + "mlp/gate_bias", {c.intermediate});
        layer.up_bias             = b.direct(p + "mlp/up_bias", {c.intermediate});
        layer.down_bias           = b.direct(p + "mlp/down_bias", {h});
        out.layers.push_back(layer);
    }
    const auto output = c.output_hidden;
    out.downsample =
        linear("vision/downsample", output, c.merge_width(), "vision/downsample_input");
    out.downsample_bias = b.direct("vision/downsample_bias", {output});
    out.merger_projection =
        linear("vision/merger/projection", output, output, "vision/merger/input");
    out.merger_norm_weight = b.direct("vision/merger/norm_weight", {output});
    out.merger_norm_bias   = b.direct("vision/merger/norm_bias", {output});
    out.merger             = {
        linear("vision/merger/gate", c.merger_intermediate, output, "vision/merger/activation"),
        linear("vision/merger/up", c.merger_intermediate, output, "vision/merger/activation"),
        linear("vision/merger/down", output, c.merger_intermediate, "vision/merger/product")};
    return out;
}

} // namespace ninfer::models::glm5next::loading
