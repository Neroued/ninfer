#include "models/qwen3_5/load/bindings.h"

#include <limits>
#include <set>
#include <string>

namespace ninfer::models::qwen3_5::loading {

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& a = config.attention.value();
    const auto h  = config.hidden_size;
    const auto q  = a.query_width();
    const auto k  = a.key_width();
    AttentionWeights out;
    out.query      = b.parameter(p + "attention/query", {q, h}, {p + "mixer_input"});
    out.key        = b.parameter(p + "attention/key", {k, h}, {p + "mixer_input"});
    out.gate       = b.parameter(p + "attention/gate", {q, h}, {p + "mixer_input"});
    out.value      = b.parameter(p + "attention/value", {k, h}, {p + "mixer_input"});
    out.query_norm = b.direct(p + "attention/query_norm", {a.head_dim});
    out.key_norm   = b.direct(p + "attention/key_norm", {a.head_dim});
    out.output     = b.parameter(p + "attention/output", {h, q}, {p + "attention/gated_output"});
    return out;
}

DenseWeights bind_dense(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                        const std::string& p, bool draft) {
    const auto input = p + (draft ? "mlp_input" : "ffn_input");
    return {b.parameter(p + "mlp/gate", {intermediate, h}, {input}),
            b.parameter(p + "mlp/up", {intermediate, h}, {input}),
            b.parameter(p + "mlp/down", {h, intermediate},
                        {p + (draft ? "mlp_product" : "mlp/product")})};
}

namespace {

GdnWeights bind_gdn(Bindings& b, const TextConfig& text, const std::string& p) {
    const auto& g    = text.gdn.value();
    const auto h     = text.hidden_size;
    const auto k     = g.key_width();
    const auto v     = g.value_width();
    const auto heads = g.linear_num_value_heads;
    GdnWeights out;
    out.query        = b.parameter(p + "gdn/query", {k, h}, {p + "mixer_input"});
    out.key          = b.parameter(p + "gdn/key", {k, h}, {p + "mixer_input"});
    out.value        = b.parameter(p + "gdn/value", {v, h}, {p + "mixer_input"});
    out.z            = b.parameter(p + "gdn/z", {v, h}, {p + "mixer_input"});
    out.a_projection = b.parameter(p + "gdn/a_projection", {heads, h}, {p + "mixer_input"});
    out.b_projection = b.parameter(p + "gdn/b_projection", {heads, h}, {p + "mixer_input"});
    out.a_log        = b.direct(p + "gdn/a_log", {heads}, QType::FP32);
    out.dt_bias      = b.direct(p + "gdn/dt_bias", {heads}, QType::FP32);
    out.convolution =
        b.direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
    out.norm   = b.direct(p + "gdn/norm", {g.linear_value_head_dim});
    out.output = b.parameter(p + "gdn/output", {h, v}, {p + "gdn/gated_output"});
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& config, const std::string& prefix) {
    const auto& moe   = std::get<MoeConfig>(config.ffn);
    const auto p      = prefix + "moe/";
    const auto input  = prefix + "ffn_input";
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    MoeWeights out;
    out.router       = b.parameter(p + "router", {moe.num_experts, h}, {input});
    out.shared_score = b.parameter(p + "shared_score", {1, h}, {input});
    out.experts.reserve(moe.num_experts);
    for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
        const auto ep = p + "experts/" + std::to_string(e) + "/";
        out.experts.push_back({b.parameter(ep + "gate", {ir, h}, {input}),
                               b.parameter(ep + "up", {ir, h}, {input}),
                               b.parameter(ep + "down", {h, ir}, {ep + "product"})});
    }
    out.shared = {b.parameter(p + "shared/gate", {shared, h}, {input}),
                  b.parameter(p + "shared/up", {shared, h}, {input}),
                  b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    return out;
}

HyperConnectionWeights bind_hyper_connection(Bindings& b, const TextConfig& config,
                                             const std::string& p, bool inject) {
    const auto wide = config.residual_width();
    const auto& hc  = config.hyper_connection.value();
    HyperConnectionWeights out;
    out.norm = b.direct(p + "norm", {wide});
    out.down = b.parameter(p + "down", {hc.hc_lowrank, wide}, {p + "normed"});
    out.up   = b.parameter(p + "up", {wide, hc.hc_lowrank}, {p + "lowrank"});
    if (inject) { out.inject = b.parameter(p + "inject", {hc.hc_count, wide}, {p + "normed"}); }
    return out;
}

QsaWeights bind_qsa(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& qsa = config.sparse_attention.value();
    const auto h    = config.hidden_size;
    QsaWeights out;
    out.attention     = bind_attention(b, config, p);
    out.indexer_query = b.parameter(p + "attention/indexer_query",
                                    {std::uint64_t(qsa.indexer_heads) * qsa.indexer_head_dim, h},
                                    {p + "mixer_input"});
    out.indexer_key =
        b.parameter(p + "attention/indexer_key", {qsa.indexer_head_dim, h}, {p + "mixer_input"});
    out.indexer_query_norm = b.direct(p + "attention/indexer_query_norm", {qsa.indexer_head_dim});
    out.indexer_key_norm   = b.direct(p + "attention/indexer_key_norm", {qsa.indexer_head_dim});
    return out;
}

PleWeights bind_ple(Bindings& b, const TextConfig& config, const std::string& prefix) {
    const auto& ple = config.ple.value();
    const auto p    = prefix + "ple/";
    const auto wide = config.residual_width();
    PleWeights out;
    // A row-scaled FP8 table, or the FP4-Local row-scaled NVFP4 table, whose FP32 tensor
    // multiplier is the separate scalar table_global_scale (absent for FP8).
    out.table = b.parameter(p + "table", {ple.table_rows, ple.row_width()}, {}, {},
                            artifact::Residency::HostFile);
    const auto table_format =
        b.binder.reader().geometry(b.at(out.table).reference.binding.parts.at(0).object).format;
    const auto scale_name = p + "table_global_scale";
    if (table_format == QType::NVFP4_ROW_E4M3_G16) {
        if (!b.binder.contains(scale_name)) {
            throw artifact::ArtifactError(p + "table: row-scaled NVFP4 requires " + scale_name);
        }
        // One FP32 element, stored as a scalar or a one-element vector.
        const auto& binding = b.binder.reader().directory().bindings.at(scale_name);
        auto shape          = binding.whole_object
                                  ? b.binder.reader().directory().tensor(binding.parts.at(0).object).shape
                                  : artifact::Shape{binding.elements};
        if (binding.elements != 1) {
            throw artifact::ArtifactError(scale_name + ": must be one FP32 value");
        }
        out.table_global_scale = b.direct(scale_name, std::move(shape), QType::FP32);
    } else if (table_format != QType::FP8_E4M3FN_ROW_BF16 || b.binder.contains(scale_name)) {
        throw artifact::ArtifactError(
            p + "table: the n-gram table is row-scaled FP8, or row-scaled NVFP4 with " +
            scale_name);
    }
    out.key      = b.parameter(p + "key", {wide, ple.embed_dim}, {p + "embedding"});
    out.value    = b.parameter(p + "value", {config.hidden_size, ple.embed_dim}, {p + "embedding"});
    out.key_norm = b.direct(p + "key_norm", {wide});
    out.query_norm  = b.direct(p + "query_norm", {wide});
    out.conv_norm   = b.direct(p + "conv_norm", {wide});
    out.convolution = b.direct(p + "convolution", {ple.conv_kernel, wide});
    return out;
}

Qwen4BlockWeights bind_qwen4_block(Bindings& b, const TextConfig& config, const std::string& p,
                                   MixerKind mixer, bool ple) {
    Qwen4BlockWeights out;
    out.attention_hc = bind_hyper_connection(b, config, p + "attn_hc/", true);
    out.ffn_hc       = bind_hyper_connection(b, config, p + "ffn_hc/", true);
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_qsa(b, config, p);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    // Routed experts bind as device parameters; the Qwen4Exp shard rule places their banks
    // HostMapped (load/sharding.h), one page-locked copy that every device and the host read.
    // A dense FFN binds as the Qwen3.5 dense block's (row-split gate/up, column-split down at tp
    // 2).
    if (const auto* dense = std::get_if<DenseConfig>(&config.ffn)) {
        out.ffn = bind_dense(b, config.hidden_size, dense->intermediate_size, p);
    } else {
        out.ffn = bind_moe(b, config, p);
    }
    if (ple) { out.ple = bind_ple(b, config, p); }
    return out;
}

} // namespace

Qwen4TextWeights bind_qwen4_text(Bindings& b, const TextConfig& config) {
    Qwen4TextWeights out;
    out.head = bind_hyper_connection(b, config, "text/hc_head/", false);
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        out.layers.push_back(bind_qwen4_block(b, config, "text/layers/" + std::to_string(i) + "/",
                                              config.layer_types[i],
                                              config.ple && config.ple->layer == i));
    }
    return out;
}

Qwen4MtpWeights bind_qwen4_mtp(Bindings& b, const TextConfig& config) {
    const auto h = config.hidden_size;
    Qwen4MtpWeights out;
    out.embedding_norm = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm    = b.direct("mtp/hidden_norm", {config.residual_width()});
    out.embedding_projection =
        b.parameter("mtp/embedding_projection", {h, h}, {"mtp/embedding_input"});
    out.hidden_projection = b.parameter("mtp/hidden_projection", {h, h}, {"mtp/hidden_input"});
    out.head              = bind_hyper_connection(b, config, "mtp/hc_head/", false);
    out.layer = bind_qwen4_block(b, config, "mtp/layers/0/", MixerKind::FullAttention, false);
    return out;
}

BlockWeights bind_block(Bindings& b, const TextConfig& config, const std::string& p,
                        MixerKind mixer) {
    BlockWeights out;
    out.input_norm          = b.direct(p + "input_norm", {config.hidden_size});
    out.post_attention_norm = b.direct(p + "post_attention_norm", {config.hidden_size});
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    out.ffn =
        bind_dense(b, config.hidden_size, std::get<DenseConfig>(config.ffn).intermediate_size, p);
    return out;
}

TextWeights bind_text(Bindings& b, const TextConfig& config, const LoadOptions& options) {
    TextWeights out;
    if (config.qwen4()) {
        // Qwen4Exp keeps only the vocabulary here; its blocks bind through bind_qwen4_text.
        out.token_embedding =
            b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
        std::vector<std::string> inputs{"text/final_hidden"};
        if (options.mtp()) { inputs.push_back("mtp/final_hidden"); }
        out.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                      std::move(inputs));
        return out;
    }
    out.token_embedding =
        b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
    std::vector<std::string> head_inputs{"text/final_hidden"};
    if (options.speculative != SpeculativeBackend::None && !options.proposal_enabled()) {
        head_inputs.push_back(std::string(options.speculative_component()) + "/final_hidden");
    }
    out.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                  std::move(head_inputs));
    out.final_norm  = b.direct("text/final_norm", {config.hidden_size});
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        out.layers.push_back(
            bind_block(b, config, "text/layers/" + std::to_string(i) + "/", config.layer_types[i]));
    }
    return out;
}

ProposalWeights bind_proposal(Bindings& b, const artifact::Proposal& proposal,
                              const TextConfig& target, const LoadOptions& options,
                              std::uint32_t public_tokens) {
    const auto rows = proposal.indexed ? proposal.rows : target.vocab_size;
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        (proposal.indexed && rows > public_tokens)) {
        throw artifact::ArtifactError("proposal rows exceed the output domain");
    }
    ProposalWeights out;
    out.rows = static_cast<std::uint32_t>(rows);
    out.head = b.parameter("proposal/head", {rows, target.hidden_size},
                           {std::string(options.speculative_component()) + "/final_hidden"});
    if (proposal.indexed) {
        out.token_ids = b.direct("proposal/token_ids", {rows}, QType::INT32);
        out.global_token_ids =
            b.binder.values(b.at(*out.token_ids).reference.binding, QType::INT32).integers();
        std::set<std::int32_t> unique;
        for (const auto id : out.global_token_ids) {
            if (id < 0 || std::uint32_t(id) >= public_tokens || !unique.insert(id).second) {
                throw artifact::ArtifactError(
                    "proposal token IDs must be unique and in the public domain");
            }
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
