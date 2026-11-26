#include "models/inkling2/load/bindings.h"

#include <string>
#include <vector>

namespace ninfer::models::inkling2::loading {
namespace {

std::vector<std::string> pass_inputs(const std::string& prefix, const char* stem,
                                     std::uint32_t passes) {
    std::vector<std::string> out;
    out.reserve(passes);
    for (std::uint32_t k = 0; k < passes; ++k)
        out.push_back(prefix + stem + "_p" + std::to_string(k));
    return out;
}

AttentionWeights bind_attention(Bindings& b, const TextConfig& c, const PhysicalLayerConfig& layer,
                                const std::string& p) {
    const auto& attn  = c.attention(layer.is_swa);
    const auto h      = c.hidden;
    const auto q_dim  = attn.query_width();
    const auto r_dim  = c.rel_width(layer.is_swa);
    const auto kv_dim = attn.kv_width();
    const auto taps   = c.sconv_kernel_size;

    const auto q_inputs = pass_inputs(p, "attention_q_input", layer.num_passes);
    const auto k_inputs = pass_inputs(p, "attention_k_input", layer.num_passes);
    const auto v_inputs = pass_inputs(p, "attention_v_input", layer.num_passes);
    const auto o_inputs = pass_inputs(p, "attention_output", layer.num_passes);

    AttentionWeights out;
    if (layer.physical_index > 0) {
        const artifact::Shape q_shape =
            layer.is_swa ? artifact::Shape{h} : artifact::Shape{3, h};
        out.attn_res_query = b.direct(p + "attention/attn_res_query", q_shape, QType::FP32);
    }
    const artifact::Shape norm_shape =
        layer.is_swa ? artifact::Shape{h} : artifact::Shape{3, h};
    out.input_norm  = b.direct(p + "attention/input_norm", norm_shape, QType::BF16);
    out.query       = b.parameter(p + "attention/query", {q_dim, h}, q_inputs);
    out.relative    = b.parameter(p + "attention/relative", {r_dim, h}, q_inputs);
    out.output_gate = b.parameter(p + "attention/output_gate", {q_dim, h}, q_inputs);
    out.key         = b.parameter(p + "attention/key", {kv_dim, h}, k_inputs);
    out.value       = b.parameter(p + "attention/value", {kv_dim, h}, v_inputs);
    if (!layer.is_swa) {
        const auto dyn_rows = std::uint64_t(taps) * attn.kv_heads;
        out.k_conv_proj = b.parameter(p + "attention/k_conv_proj", {dyn_rows, h}, k_inputs);
        out.v_conv_proj = b.parameter(p + "attention/v_conv_proj", {dyn_rows, h}, v_inputs);
    }
    out.sink            = b.direct(p + "attention/sink", {attn.heads, h}, QType::BF16);
    out.output          = b.parameter(p + "attention/output", {h, q_dim}, o_inputs);
    out.q_scale         = b.direct(p + "attention/q_scale", {attn.heads, attn.head_dim}, QType::BF16);
    out.r_scale         = b.direct(p + "attention/r_scale", {attn.heads, c.d_rel}, QType::BF16);
    out.rel_logits_proj = b.direct(p + "attention/rel_logits_proj", {c.rel_extent, c.d_rel}, QType::BF16);
    out.k_conv          = b.direct(p + "attention/k_conv", {taps, kv_dim}, QType::BF16);
    out.v_conv          = b.direct(p + "attention/v_conv", {taps, kv_dim}, QType::BF16);
    out.short_conv      = b.direct(p + "attention/short_conv", {taps, h}, QType::BF16);
    return out;
}

LoopWeights bind_loop(Bindings& b, const TextConfig& c, const std::string& p) {
    const auto h    = std::uint64_t(c.hidden);
    const auto rank = std::uint64_t(c.loop->gate_rank);
    return {b.direct(p + "loop/message_queries", {4, h}, QType::FP32),
            b.direct(p + "loop/attn_kv_down", {3, rank, 2 * h}, QType::BF16),
            b.direct(p + "loop/attn_kv_up", {3, 2 * h, rank}, QType::BF16),
            b.direct(p + "loop/mlp_down", {1, rank, 2 * h}, QType::BF16),
            b.direct(p + "loop/mlp_up", {1, 2 * h, rank}, QType::BF16)};
}

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const PhysicalLayerConfig& layer,
                    const std::string& p) {
    const auto& m     = c.moe;
    const auto h      = c.hidden;
    const auto in_use = pass_inputs(p, "mlp_in", layer.num_passes);
    const auto act_use = pass_inputs(p, "mlp_act", layer.num_passes);
    return {b.direct(p + "mlp/attn_res_query", {h}, QType::FP32),
            b.direct(p + "mlp/input_norm", {h}, QType::BF16),
            b.direct(p + "mlp/short_conv", {c.sconv_kernel_size, h}, QType::BF16),
            b.parameter(p + "moe/router", {m.experts, h}, in_use, QType::BF16),
            b.direct(p + "moe/correction_bias", {m.experts}, QType::FP32),
            b.parameter(p + "moe/experts/gate", {m.experts, m.intermediate, h}, in_use),
            b.parameter(p + "moe/experts/up", {m.experts, m.intermediate, h}, in_use),
            b.parameter(p + "moe/experts/down", {m.experts, h, m.intermediate}, act_use),
            {}};
}

} // namespace

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    TextWeights out;
    out.embedding    = b.direct("text/token_embedding", {c.vocabulary, c.hidden}, QType::BF16);
    out.embed_norm   = b.direct("text/embed_norm", {c.hidden}, QType::BF16);
    out.final_norm   = b.direct("text/final_norm", {c.hidden}, QType::BF16);
    out.output_query = b.direct("text/output_query", {c.hidden}, QType::FP32);
    out.head = b.parameter("text/output_head", {c.vocabulary, c.hidden}, {"text/final_hidden"});

    out.layers.reserve(c.physical_layers.size());
    for (const auto& phys : c.physical_layers) {
        const auto p = "text/layers/" + std::to_string(phys.physical_index) + "/";
        PhysicalLayerWeights layer;
        layer.attention = bind_attention(b, c, phys, p);
        if (phys.in_core) layer.loop = bind_loop(b, c, p);
        layer.moe = bind_moe(b, c, phys, p);
        out.layers.push_back(std::move(layer));
    }
    return out;
}

} // namespace ninfer::models::inkling2::loading
