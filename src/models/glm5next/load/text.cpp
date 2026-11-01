#include "models/glm5next/load/bindings.h"

#include <algorithm>

namespace ninfer::models::glm5next::loading {
namespace {

MhcWeights mhc(Bindings& b, const TextConfig& c, const std::string& p) {
    const auto rows = c.mhc.coefficient_width();
    return {b.direct(p + "projection", {rows, c.residual_width()}), b.direct(p + "base", {rows}),
            b.direct(p + "scale", {3})};
}

KdaWeights kda(Bindings& b, const TextConfig& c, const std::string& layer) {
    const auto& k         = c.kda;
    const auto p          = layer + "kda/";
    const auto projection = [&](const char* name, std::uint64_t rows) {
        return b.parameter(p + name, {rows, c.hidden}, {layer + "attention_input"});
    };
    return {projection("query", k.width()),
            projection("key", k.width()),
            projection("value", k.width()),
            b.parameter(p + "output", {c.hidden, k.width()}, {p + "gated_output"}),
            projection("beta", k.heads),
            projection("forget_a", k.gate_lowrank),
            b.parameter(p + "forget_b", {k.width(), k.gate_lowrank}, {p + "forget_lowrank"}),
            projection("gate_a", k.gate_lowrank),
            b.parameter(p + "gate_b", {k.width(), k.gate_lowrank}, {p + "gate_lowrank"}),
            b.direct(p + "convolution", {k.conv_width, k.conv_channels()}, QType::FP32),
            b.direct(p + "a_log", {k.heads}, QType::FP32),
            b.direct(p + "dt_bias", {k.width()}, QType::FP32),
            b.direct(p + "norm", {k.head_dim})};
}

MlaWeights mla(Bindings& b, const TextConfig& c, const std::string& layer) {
    const auto& m         = c.mla;
    const auto p          = layer + "attention/";
    const auto projection = [&](const char* name, std::uint64_t rows) {
        return b.parameter(p + name, {rows, c.hidden}, {layer + "attention_input"});
    };
    return {projection("latent", m.latent_dim),
            projection("index_key", m.index_dim),
            projection("pool_gate", m.index_dim),
            projection("query_a", m.query_lowrank),
            projection("index_weight", m.index_heads),
            b.parameter(p + "query_b", {m.query_width(), m.query_lowrank}, {p + "query_lowrank"}),
            b.parameter(p + "index_query", {m.index_query_width(), m.query_lowrank},
                        {p + "query_lowrank"}),
            b.direct(p + "key_up", {m.heads, m.query_dim, m.latent_dim}),
            b.direct(p + "value_up", {m.heads, m.value_dim, m.latent_dim}),
            b.parameter(p + "output", {c.hidden, m.output_width()}, {p + "output_heads"}),
            b.direct(p + "query_norm", {m.query_lowrank}),
            b.direct(p + "latent_norm", {m.latent_dim}),
            b.direct(p + "index_norm", {m.index_dim}),
            b.direct(p + "index_bias", {m.index_dim}),
            b.direct(p + "pool_ape", {m.pool_size, m.index_dim})};
}

DenseWeights dense(Bindings& b, const TextConfig& c, const std::string& layer) {
    const auto p = layer + "mlp/";
    return {b.parameter(p + "gate", {c.dense_intermediate, c.hidden}, {layer + "mlp_input"}),
            b.parameter(p + "up", {c.dense_intermediate, c.hidden}, {layer + "mlp_input"}),
            b.parameter(p + "down", {c.hidden, c.dense_intermediate}, {p + "product"})};
}

MoeWeights moe(Bindings& b, const TextConfig& c, const std::string& layer) {
    const auto& m = c.moe;
    const auto p = layer + "moe/", input = layer + "moe_input";
    return {b.parameter(p + "router", {m.experts, c.hidden}, {input}, QType::BF16),
            b.direct(p + "correction_bias", {m.experts}, QType::FP32),
            {b.parameter(p + "shared/gate", {m.shared_intermediate(), c.hidden}, {input}),
             b.parameter(p + "shared/up", {m.shared_intermediate(), c.hidden}, {input}),
             b.parameter(p + "shared/down", {c.hidden, m.shared_intermediate()},
                         {p + "shared/product"})},
            b.parameter(p + "experts/gate", {m.experts, m.intermediate, c.hidden}, {input}),
            b.parameter(p + "experts/up", {m.experts, m.intermediate, c.hidden}, {input}),
            b.parameter(p + "experts/down", {m.experts, c.hidden, m.intermediate},
                        {p + "experts/product"}),
            {}};
}

void require_shared(const artifact::ParameterReference& parameter,
                    const artifact::ParameterReference& main) {
    const auto& parts  = parameter.binding.parts;
    const auto& target = main.binding.parts;
    if (!std::equal(parts.begin(), parts.end(), target.begin(), target.end(),
                    [](const auto& a, const auto& b) {
                        return a.object == b.object && a.begin == b.begin && a.end == b.end;
                    }))
        throw artifact::ArtifactError(parameter.name + ": must share Main weight storage");
}

} // namespace

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    TextWeights out;
    out.embedding = b.direct("text/token_embedding", {c.vocabulary, c.hidden});
    out.head      = b.parameter("text/output_head", {c.vocabulary, c.hidden}, {"text/final_hidden"},
                                QType::BF16);
    out.final_norm = b.direct("text/final_norm", {c.hidden});
    out.layers.reserve(c.layers.size());
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
        const auto p = "text/layers/" + std::to_string(i) + "/";
        BlockWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {c.hidden});
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {c.hidden});
        layer.attention_hc        = mhc(b, c, p + "attention_hc/");
        layer.mlp_hc              = mhc(b, c, p + "mlp_hc/");
        if (c.layers[i].mixer == MixerKind::Kda)
            layer.mixer = kda(b, c, p);
        else
            layer.mixer = mla(b, c, p);
        if (c.layers[i].ffn == FfnKind::Dense)
            layer.ffn = dense(b, c, p);
        else
            layer.ffn = moe(b, c, p);
        out.layers.push_back(std::move(layer));
    }
    return out;
}

MtpWeights bind_mtp(Bindings& b, const TextConfig& c, const TextWeights& main,
                    ProposalHead proposal_head) {
    MtpWeights out;
    out.embedding = b.direct("mtp/token_embedding", {c.vocabulary, c.hidden});
    require_shared(b.at(out.embedding).reference, b.at(main.embedding).reference);
    require_shared(b.binder.parameter("mtp/output_head", {c.vocabulary, c.hidden}, QType::BF16),
                   b.at(main.head).reference);
    if (proposal_head == ProposalHead::Optimized) {
        const auto& proposal = b.binder.reader().directory().component("text").proposal;
        if (!proposal || proposal->indexed)
            throw artifact::ArtifactError("GLM5Next requires a full-vocabulary proposal head");
        out.head = b.parameter("proposal/head", {c.vocabulary, c.hidden}, {"mtp/final_hidden"},
                               QType::Q4_G64_FP16);
    } else if (proposal_head == ProposalHead::Full) {
        out.head = b.parameter("mtp/output_head", {c.vocabulary, c.hidden}, {"mtp/final_hidden"},
                               QType::BF16);
    } else {
        throw artifact::ArtifactError("invalid GLM5Next proposal head selection");
    }
    out.embedding_norm = b.direct("mtp/pre_fc_embedding_norm", {c.hidden});
    out.hidden_norm    = b.direct("mtp/pre_fc_hidden_norm", {c.hidden});
    out.fc =
        b.parameter("mtp/fc", {c.hidden, 2ULL * c.hidden}, {"mtp/normalized_input"}, QType::BF16);
    out.final_norm          = b.direct("mtp/final_norm", {c.hidden});
    out.input_norm          = b.direct("mtp/layers/0/input_norm", {c.hidden});
    out.post_attention_norm = b.direct("mtp/layers/0/post_attention_norm", {c.hidden});
    out.attention           = mla(b, c, "mtp/layers/0/");
    out.moe                 = moe(b, c, "mtp/layers/0/");
    return out;
}

} // namespace ninfer::models::glm5next::loading
