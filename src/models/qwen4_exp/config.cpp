#include "models/qwen4_exp/config.h"

#include "artifact/schema.h"

#include <cmath>

namespace ninfer::models::qwen4_exp {
namespace {
using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& value, std::string_view name) {
    const auto n = artifact::require_u64(value, name, true);
    if (n > UINT32_MAX) throw ArtifactError(std::string(name) + ": dimension exceeds uint32");
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& value, const char* name) {
    return integer(value.at(name), name);
}

float positive(const Json& value, const char* name) {
    if (!value.at(name).is_number()) throw ArtifactError(std::string(name) + ": expected number");
    const float result = value.at(name).get<float>();
    if (!(result > 0) || !std::isfinite(result))
        throw ArtifactError(std::string(name) + ": expected positive finite value");
    return result;
}

void expect(const Json& value, const char* key, const Json& expected) {
    if (value.at(key) != expected)
        throw ArtifactError(std::string("Qwen4Exp ") + key + ": unsupported value");
}

const Json& companion(const artifact::Directory& directory, const char* name) {
    const auto& component = directory.component(name);
    if (component.target != "text")
        throw ArtifactError(std::string(name) + ": target must be text");
    return component.config;
}

TextConfig text_config(const Json& c) {
    expect(c, "architectures", Json::array({"Qwen4ExpForCausalLM"}));
    expect(c, "model_type", "qwen4_exp_text");
    expect(c, "hidden_act", "silu");
    expect(c, "output_gate_type", "sigmoid");
    expect(c, "tie_word_embeddings", false);
    expect(c, "attention_bias", false);
    expect(c, "mamba_ssm_dtype", "float32");
    expect(c, "norm_topk_prob", true);
    TextConfig out{};
    out.hidden       = dimension(c, "hidden_size");
    out.vocabulary   = dimension(c, "vocab_size");
    out.max_context  = dimension(c, "max_position_embeddings");
    out.norm_epsilon = positive(c, "rms_norm_eps");
    out.hc           = {dimension(c, "hc_count"), dimension(c, "hc_lowrank")};
    out.gdn = {dimension(c, "linear_num_key_heads"), dimension(c, "linear_num_value_heads"),
               dimension(c, "linear_key_head_dim"), dimension(c, "linear_value_head_dim"),
               dimension(c, "linear_conv_kernel_dim")};
    const auto& rope = c.at("rope_parameters");
    expect(rope, "rope_type", "default");
    const auto compression = dimension(c, "indexer_compress_ratio");
    const auto budget      = dimension(c, "indexer_budget");
    if (budget % compression)
        throw ArtifactError("QSA token budget must contain whole compressed blocks");
    out.qsa = {dimension(c, "num_attention_heads"),
               dimension(c, "num_key_value_heads"),
               dimension(c, "head_dim"),
               dimension(c, "indexer_n_heads"),
               dimension(c, "indexer_head_dim"),
               compression,
               budget / compression,
               positive(rope, "rope_theta")};
    out.moe = {dimension(c, "num_experts"), dimension(c, "num_experts_per_tok"),
               dimension(c, "moe_intermediate_size"),
               dimension(c, "shared_expert_intermediate_size")};
    if (out.gdn.value_heads % out.gdn.key_heads || out.qsa.query_heads % out.qsa.kv_heads)
        throw ArtifactError("attention head counts must be divisible");
    if (out.moe.top_k > out.moe.experts)
        throw ArtifactError("selected experts exceed expert count");
    auto& q          = out.qsa;
    q.index_kv_heads = dimension(c, "indexer_kv_heads");
    if (q.index_heads % q.index_kv_heads)
        throw ArtifactError("index query heads must be divisible by key heads");
    const auto fraction = positive(rope, "partial_rotary_factor");
    if (fraction > 1) throw ArtifactError("partial_rotary_factor exceeds one");
    q.rotary_dim = static_cast<std::uint32_t>(double(q.head_dim) * fraction);
    if (!q.rotary_dim || q.rotary_dim % 2 || q.rotary_dim > q.index_dim)
        throw ArtifactError("rotary dimension must be even and fit both attention and index heads");
    if (!rope.at("mrope_interleaved").is_boolean())
        throw ArtifactError("mrope_interleaved must be boolean");
    q.rope_interleaved   = rope.at("mrope_interleaved").get<bool>();
    const auto& sections = rope.at("mrope_section");
    if (!sections.is_array() || sections.size() != q.rope_sections.size())
        throw ArtifactError("MRoPE requires three sections");
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < q.rope_sections.size(); ++i) {
        const auto n = artifact::require_u64(sections[i], "MRoPE section", false);
        if (n > UINT32_MAX) throw ArtifactError("MRoPE section exceeds uint32");
        q.rope_sections[i] = static_cast<std::uint32_t>(n);
        sum += n;
    }
    if (sum != q.rotary_dim / 2) throw ArtifactError("MRoPE sections differ from the rotary width");
    const auto layers = dimension(c, "num_hidden_layers");
    const auto& kinds = c.at("layer_types");
    if (!kinds.is_array() || kinds.size() != layers)
        throw ArtifactError("Qwen4Exp layer_types must cover every layer");
    for (const auto& kind : kinds) {
        if (kind == "linear_attention")
            out.layers.push_back({MixerKind::Gdn, out.gdn_layers++});
        else if (kind == "indexed_attention")
            out.layers.push_back({MixerKind::Qsa, out.qsa_layers++});
        else
            throw ArtifactError("unknown Qwen4Exp mixer");
    }
    const auto& positions = c.at("ple_layer_ids");
    if (!positions.is_array() || positions.size() != 1)
        throw ArtifactError("Qwen4Exp requires one PLE layer");
    const auto layer  = artifact::require_u64(positions[0], "PLE layer", false);
    const auto& shard = c.at("ple_shard_shape");
    if (layer >= layers || !shard.is_array() || shard.size() != 2)
        throw ArtifactError("invalid Qwen4Exp PLE geometry");
    const auto eos = artifact::require_u64(c.at("eos_token_id"), "EOS", false);
    if (eos >= out.vocabulary || eos > INT32_MAX)
        throw ArtifactError("PLE EOS is outside vocabulary");
    out.ple         = {static_cast<std::uint32_t>(layer),    dimension(c, "split_ngram_parts"),
                       integer(shard[0], "PLE shard rows"),  static_cast<std::int32_t>(eos),
                       dimension(c, "ple_embed_dim"),        dimension(c, "ngram_size"),
                       dimension(c, "heads_per_ngram"),      integer(shard[1], "PLE row width"),
                       dimension(c, "ple_conv_kernel_size"), dimension(c, "ple_conv_dilation")};
    const auto& ple = out.ple;
    if (ple.ngram_size < 2 ||
        (ple.embedding % ple.row_width ||
         std::uint64_t(ple.ngram_size - 1) * ple.heads_per_ngram != ple.embedding / ple.row_width))
        throw ArtifactError("PLE head geometry differs from its embedding width");
    return out;
}

qwen::VisionConfig vision_config(const Json& c, std::uint32_t hidden) {
    expect(c, "model_type", "qwen4_exp_vision");
    expect(c, "hidden_act", "gelu_pytorch_tanh");
    expect(c, "in_channels", 3);
    expect(c, "out_hidden_size", hidden);
    expect(c, "deepstack_visual_indexes", Json::array());
    qwen::VisionConfig out{dimension(c, "depth"),
                           dimension(c, "hidden_size"),
                           dimension(c, "intermediate_size"),
                           dimension(c, "num_heads"),
                           dimension(c, "patch_size"),
                           dimension(c, "temporal_patch_size"),
                           dimension(c, "spatial_merge_size"),
                           dimension(c, "num_position_embeddings")};
    out.position_grid_side = static_cast<std::uint32_t>(std::sqrt(out.num_position_embeddings));
    if (out.hidden_size % out.num_heads || (out.hidden_size / out.num_heads) % 4 ||
        std::uint64_t(out.position_grid_side) * out.position_grid_side !=
            out.num_position_embeddings)
        throw ArtifactError("invalid Vision head or square position geometry");
    return out;
}
} // namespace

std::uint64_t GdnConfig::conv_channels() const {
    return artifact::checked_add(artifact::checked_mul(2, key_width(), "GDN Q/K width"),
                                 value_width(), "GDN convolution channels");
}

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    if (options.masked_draft()) throw ArtifactError("Qwen4Exp supports native MTP");
    Config out{text_config(directory.component("text").config), {}, options.mtp()};
    if (options.vision) out.vision = vision_config(companion(directory, "vision"), out.text.hidden);
    if (out.mtp) {
        const auto& m = companion(directory, "mtp");
        expect(m, "architectures", Json::array({"Qwen4ExpMTP"}));
        expect(m, "model_type", "qwen4_exp_mtp");
        expect(m, "num_hidden_layers", 1);
        expect(m, "hybrid", true);
        expect(m, "layer_types", Json::array({"indexed_attention"}));
        expect(m, "mtp_use_hidden_state_from_layer", nullptr);
        if (positive(m, "rope_theta") != out.text.qsa.rope_theta)
            throw ArtifactError("MTP RoPE differs from Main");
    }
    return out;
}
} // namespace ninfer::models::qwen4_exp
