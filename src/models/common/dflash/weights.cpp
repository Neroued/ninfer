#include "models/common/dflash/weights.h"
#include "models/common/bindings.h"

namespace ninfer::models::dflash {
namespace {

void bind_dflash2(Bindings& b, Weights& weights, const Config& config, TargetGeometry target) {
    const auto& extra = config.dflash2.value();
    const auto h      = target.hidden_size;
    const auto rows =
        artifact::checked_mul(2ULL * extra.conv_kernel_size, h / extra.conv_group_size,
                              "DFlash2 dynamic projection rows");
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p           = "dflash2/layers/" + std::to_string(i) + "/";
        const auto convolution = [&](const std::string& name) {
            return DynamicConvWeights{
                b.direct(p + name + "/base_kernel", {2, extra.conv_kernel_size, h}),
                b.parameter(p + name + "/kernel_projection", {rows, h}, {p + name + "_input"})};
        };
        weights.layers[i].attention_conv = convolution("attention_conv");
        weights.layers[i].mlp_conv       = convolution("mlp_conv");
    }
    weights.selector =
        SelectorWeights{b.parameter("dflash2/candidate_selector/hidden_projection",
                                    {extra.selector_rank, h}, {"dflash2/final_hidden"}),
                        b.direct("dflash2/candidate_selector/predecessor_codebook",
                                 {target.vocabulary, extra.selector_rank}),
                        b.direct("dflash2/candidate_selector/successor_codebook",
                                 {target.vocabulary, extra.selector_rank})};
}

} // namespace

Weights bind_weights(Bindings& b, const Config& config, TargetGeometry target,
                     const std::string& component) {
    const auto h = target.hidden_size;
    const auto q = config.attention.query_width();
    const auto k = config.attention.key_width();
    const auto d = config.attention.head_dim;
    Weights out;
    out.feature_projection = b.parameter(
        component + "/feature_projection",
        {h, artifact::checked_mul(config.target_layer_ids.size(), h, "target features")},
        {component + "/target_features"});
    out.context_norm = b.direct(component + "/context_norm", {h});
    out.final_norm   = b.direct(component + "/final_norm", {h});
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p = component + "/layers/" + std::to_string(i) + "/";
        BlockWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {h});
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        layer.attention.query =
            b.parameter(p + "attention/query", {q, h}, {p + "query_projection_input"});
        layer.attention.key =
            b.parameter(p + "attention/key", {k, h}, {p + "query_projection_input"});
        layer.attention.value =
            b.parameter(p + "attention/value", {k, h}, {p + "query_projection_input"});
        layer.attention.context_key =
            b.parameter(p + "attention/context_key", {k, h}, {component + "/context_input"});
        layer.attention.context_value =
            b.parameter(p + "attention/context_value", {k, h}, {component + "/context_input"});
        layer.attention.query_norm = b.direct(p + "attention/query_norm", {d});
        layer.attention.key_norm   = b.direct(p + "attention/key_norm", {d});
        layer.attention.output =
            b.parameter(p + "attention/output", {h, q}, {p + "attention_output"});
        const auto intermediate = config.intermediate_size;
        layer.mlp = {b.parameter(p + "mlp/gate", {intermediate, h}, {p + "mlp_input"}),
                     b.parameter(p + "mlp/up", {intermediate, h}, {p + "mlp_input"}),
                     b.parameter(p + "mlp/down", {h, intermediate}, {p + "mlp_product"})};
        out.layers.push_back(std::move(layer));
    }
    if (config.dflash2) { bind_dflash2(b, out, config, target); }
    return out;
}

} // namespace ninfer::models::dflash
