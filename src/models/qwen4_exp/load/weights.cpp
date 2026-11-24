#include "models/qwen4_exp/load/bindings.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::loading {
namespace {
HcWeights hc(Bindings& b, const TextConfig& c, const std::string& p, bool final = false) {
    HcWeights out{
        b.direct(p + "norm", {c.residual_width()}),
        b.parameter(p + "down", {c.hc.lowrank, c.residual_width()}, {p + "normalized_streams"}),
        b.parameter(p + "up", {c.residual_width(), c.hc.lowrank}, {p + "lowrank_activation"}),
        {}};
    if (!final)
        out.injection = b.parameter(p + "inject", {c.hc.streams, c.residual_width()},
                                    {p + "normalized_streams"});
    return out;
}

GdnWeights gdn(Bindings& b, const TextConfig& c, const std::string& p) {
    const auto& g     = c.gdn;
    const auto prefix = p + "gdn/", input = p + "attention_input";
    const auto projection = [&](const char* name, std::uint64_t width) {
        return b.parameter(prefix + name, {width, c.hidden}, {input});
    };
    return {projection("query", g.key_width()),
            projection("key", g.key_width()),
            projection("value", g.value_width()),
            projection("z", g.value_width()),
            projection("a_projection", g.value_heads),
            projection("b_projection", g.value_heads),
            b.direct(prefix + "a_log", {g.value_heads}, QType::FP32),
            b.direct(prefix + "dt_bias", {g.value_heads}, QType::FP32),
            b.direct(prefix + "convolution", {g.conv_width, g.conv_channels()}),
            b.direct(prefix + "norm", {g.value_dim}),
            b.parameter(prefix + "output", {c.hidden, g.value_width()}, {prefix + "gated_output"})};
}

QsaWeights qsa(Bindings& b, const TextConfig& c, const std::string& p) {
    const auto& q     = c.qsa;
    const auto prefix = p + "attention/", input = p + "attention_input";
    const auto projection = [&](const char* name, std::uint64_t width) {
        return b.parameter(prefix + name, {width, c.hidden}, {input});
    };
    return {projection("query", q.query_width()),
            projection("key", q.kv_width()),
            projection("output_gate", q.query_width()),
            projection("value", q.kv_width()),
            projection("index_query", q.index_query_width()),
            projection("index_key", q.index_key_width()),
            b.direct(prefix + "query_norm", {q.head_dim}),
            b.direct(prefix + "key_norm", {q.head_dim}),
            b.direct(prefix + "index_query_norm", {q.index_dim}),
            b.direct(prefix + "index_key_norm", {q.index_dim}),
            b.parameter(prefix + "output", {c.hidden, q.query_width()}, {prefix + "gated_output"})};
}

MoeWeights moe_weights(Bindings& b, const TextConfig& c, const std::string& p) {
    const auto& m     = c.moe;
    const auto prefix = p + "moe/", input = p + "moe_input";
    return {b.parameter(prefix + "router", {m.experts, c.hidden}, {input}),
            b.parameter(prefix + "shared_score", {1, c.hidden}, {input}),
            b.parameter(prefix + "shared/gate", {m.shared_intermediate, c.hidden}, {input}),
            b.parameter(prefix + "shared/up", {m.shared_intermediate, c.hidden}, {input}),
            b.parameter(prefix + "shared/down", {c.hidden, m.shared_intermediate},
                        {prefix + "shared/product"}),
            b.parameter(prefix + "experts/gate", {m.experts, m.intermediate, c.hidden}, {input}),
            b.parameter(prefix + "experts/up", {m.experts, m.intermediate, c.hidden}, {input}),
            b.parameter(prefix + "experts/down", {m.experts, c.hidden, m.intermediate},
                        {prefix + "experts/product"}),
            {}};
}

BlockWeights block(Bindings& b, const TextConfig& c, const std::string& p, MixerKind kind) {
    BlockWeights out;
    out.attention_hc = hc(b, c, p + "attention_hc/");
    out.moe_hc       = hc(b, c, p + "moe_hc/");
    if (kind == MixerKind::Gdn)
        out.mixer = gdn(b, c, p);
    else
        out.mixer = qsa(b, c, p);
    out.moe = moe_weights(b, c, p);
    return out;
}
} // namespace

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    TextWeights out;
    out.embedding = b.parameter("text/token_embedding", {c.vocabulary, c.hidden}, {}, QType::BF16);
    out.head     = b.parameter("text/output_head", {c.vocabulary, c.hidden}, {"text/final_hidden"});
    out.final_hc = hc(b, c, "text/final_hc/", true);
    for (std::size_t l = 0; l < c.layers.size(); ++l)
        out.layers.push_back(
            block(b, c, "text/layers/" + std::to_string(l) + "/", c.layers[l].mixer));
    const auto p = "text/layers/" + std::to_string(c.ple.layer) + "/ple/";
    out.ple = {b.parameter(p + "key", {c.residual_width(), c.ple.embedding}, {p + "embedding"}),
               b.parameter(p + "value", {c.hidden, c.ple.embedding}, {p + "embedding"}),
               b.direct(p + "norm_key", {c.residual_width()}),
               b.direct(p + "norm_query", {c.residual_width()}),
               b.direct(p + "norm_conv", {c.residual_width()}),
               b.direct(p + "convolution", {c.ple.conv_width, c.residual_width()})};
    return out;
}

MtpWeights bind_mtp(Bindings& b, const TextConfig& c, const TextWeights& main,
                    ProposalHead proposal_head) {
    MtpWeights out;
    out.embedding = b.parameter("mtp/token_embedding", {c.vocabulary, c.hidden}, {}, QType::BF16);
    const auto shared = [&](const artifact::ParameterReference& parameter, WeightId target_id) {
        const auto& a      = parameter.binding.parts;
        const auto& target = b.at(target_id).reference.binding.parts;
        if (!std::equal(a.begin(), a.end(), target.begin(), target.end(),
                        [](const auto& x, const auto& y) {
                            return x.object == y.object && x.begin == y.begin && x.end == y.end;
                        }))
            throw artifact::ArtifactError("Qwen4Exp MTP embedding/head must share Main storage");
    };
    shared(b.at(out.embedding).reference, main.embedding);
    shared(b.binder.parameter("mtp/output_head", {c.vocabulary, c.hidden}), main.head);
    switch (proposal_head) {
    case ProposalHead::Full:
        out.head = b.parameter("mtp/output_head", {c.vocabulary, c.hidden}, {"mtp/final_hidden"});
        break;
    case ProposalHead::Optimized: {
        const auto& proposal = b.binder.reader().directory().component("text").proposal;
        if (!proposal || proposal->indexed)
            throw artifact::ArtifactError("Qwen4Exp requires a full-vocabulary proposal head");
        out.head = b.parameter("proposal/head", {c.vocabulary, c.hidden}, {"mtp/final_hidden"});
        break;
    }
    default:
        throw artifact::ArtifactError("invalid Qwen4Exp proposal head selection");
    }
    out.embedding_norm = b.direct("mtp/pre_fc_embedding_norm", {c.hidden});
    out.hidden_norm    = b.direct("mtp/pre_fc_hidden_norm", {c.residual_width()});
    out.fc_embedding =
        b.parameter("mtp/fc_embedding", {c.hidden, c.hidden}, {"mtp/normalized_embedding"});
    out.fc_hidden =
        b.parameter("mtp/fc_hidden", {c.hidden, c.hidden}, {"mtp/normalized_hidden_streams"});
    out.layer    = block(b, c, "mtp/layers/0/", MixerKind::Qsa);
    out.final_hc = hc(b, c, "mtp/final_hc/", true);
    return out;
}

qwen::FrontendResources bind_resources(artifact::Binder& binder, const Config& c) {
    const auto resource = [&](const char* component, const char* name) {
        const auto bytes = binder.resource_bytes(binder.resource(component, name));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    qwen::FrontendResources out;
    out.text.tokenizer_json         = resource("text", "tokenizer.json");
    out.text.tokenizer_config_json  = resource("text", "tokenizer_config.json");
    out.text.generation_config_json = resource("text", "generation_config.json");
    out.text.chat_template_jinja    = resource("text", "chat_template.jinja");
    std::optional<qwen::MediaGeometry> media;
    if (c.vision) {
        out.preprocessor_config_json       = resource("vision", "preprocessor_config.json");
        out.video_preprocessor_config_json = resource("vision", "video_preprocessor_config.json");
        media = {c.vision->patch_size, c.vision->temporal_patch_size, c.vision->spatial_merge_size};
    }
    qwen::parse_resources(out, c.text.vocabulary, media);
    return out;
}
} // namespace ninfer::models::qwen4_exp::loading
