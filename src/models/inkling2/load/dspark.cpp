#include "models/inkling2/load/bindings.h"

#include <string>

namespace ninfer::models::inkling2::loading {

DSparkWeights bind_dspark(Bindings& b, const DSparkConfig& c, const TextWeights& main) {
    const auto h      = c.hidden;
    const auto q_dim  = c.query_width();
    const auto kv_dim = c.kv_width();
    const auto d      = c.head_dim;
    const auto inter  = c.intermediate;
    const auto rank   = std::uint64_t(c.markov_rank);

    DSparkWeights out;
    out.embedding = b.direct("dspark/token_embedding", {c.vocabulary, h}, QType::BF16);
    require_shared_binding(b.at(out.embedding).reference, b.at(main.embedding).reference);
    out.embed_norm = b.direct("dspark/embed_norm", {h}, QType::BF16);
    require_shared_binding(b.at(out.embed_norm).reference, b.at(main.embed_norm).reference);
    out.output_head =
        b.parameter("dspark/output_head", {c.vocabulary, h}, {"dspark/final_hidden"});
    require_shared_binding(b.at(out.output_head).reference, b.at(main.head).reference);

    out.fuse_alpha = b.direct("dspark/fuse_alpha",
                              {std::uint64_t(c.num_layers) + 1, c.target_layer_ids.size()},
                              QType::FP32);
    out.mask_embed = b.direct("dspark/mask_embed", {h}, QType::BF16);
    out.final_norm = b.direct("dspark/final_norm", {h}, QType::BF16);

    out.layers.reserve(c.num_layers);
    for (std::uint32_t i = 0; i < c.num_layers; ++i) {
        const auto p = "dspark/layers/" + std::to_string(i) + "/";
        DSparkLayerWeights layer;
        layer.dflare_norm         = b.direct(p + "dflare_norm", {h}, QType::BF16);
        layer.input_norm          = b.direct(p + "input_norm", {h}, QType::BF16);
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h}, QType::BF16);
        layer.query  = b.parameter(p + "attention/query", {q_dim, h}, {p + "attention_input"});
        layer.key    = b.parameter(p + "attention/key", {kv_dim, h}, {p + "attention_input"});
        layer.value  = b.parameter(p + "attention/value", {kv_dim, h}, {p + "attention_input"});
        layer.context_key =
            b.parameter(p + "attention/context_key", {kv_dim, h}, {p + "context_input"});
        layer.context_value =
            b.parameter(p + "attention/context_value", {kv_dim, h}, {p + "context_input"});
        layer.query_norm       = b.direct(p + "attention/query_norm", {d}, QType::BF16);
        layer.key_norm         = b.direct(p + "attention/key_norm", {d}, QType::BF16);
        layer.context_key_norm = b.direct(p + "attention/context_key_norm", {d}, QType::BF16);
        layer.output = b.parameter(p + "attention/output", {h, q_dim}, {p + "attention_output"});
        layer.mlp    = {b.parameter(p + "mlp/gate", {inter, h}, {p + "mlp_input"}),
                        b.parameter(p + "mlp/up", {inter, h}, {p + "mlp_input"}),
                        b.parameter(p + "mlp/down", {h, inter}, {p + "mlp_product"})};
        out.layers.push_back(layer);
    }

    const std::string mp = "dspark/markov/";
    out.markov.dflare_norm = b.direct(mp + "dflare_norm", {h}, QType::BF16);
    out.markov.w1_embed    = b.direct(mp + "w1_embed", {c.vocabulary, rank}, QType::BF16);
    out.markov.w_recurrent =
        b.parameter(mp + "w_recurrent", {3 * rank, 2 * rank}, {mp + "recurrent_input"});
    out.markov.w_hidden = b.parameter(mp + "w_hidden", {3 * rank, h}, {mp + "hidden_input"});
    out.markov.w_frnn   = b.parameter(mp + "w_frnn", {3 * rank, h}, {mp + "frnn_input"});
    out.markov.rnn_bias = b.direct(mp + "rnn_bias", {3 * rank}, QType::FP32);
    out.markov.fold_weight =
        b.parameter(mp + "fold_weight", {c.vocabulary, rank}, {mp + "latent"});
    out.markov.fold_bias = b.direct(mp + "fold_bias", {c.vocabulary}, QType::BF16);
    return out;
}

} // namespace ninfer::models::inkling2::loading
