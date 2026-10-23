#include "models/glm5next/config.h"

#include "artifact/schema.h"

#include <cmath>

namespace ninfer::models::glm5next {
namespace {
using artifact::ArtifactError;
using artifact::Json;

std::uint32_t dimension(const Json& object, const char* name) {
    const auto value = artifact::require_u64(object.at(name), name, true);
    if (value > UINT32_MAX) throw ArtifactError(std::string(name) + ": dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

float number(const Json& object, const char* name) {
    if (!object.at(name).is_number())
        throw ArtifactError(std::string(name) + ": expected a finite number");
    const auto value = object.at(name).get<float>();
    if (!std::isfinite(value))
        throw ArtifactError(std::string(name) + ": expected a finite number");
    return value;
}

float positive(const Json& object, const char* name) {
    const auto value = number(object, name);
    if (value <= 0) throw ArtifactError(std::string(name) + ": expected a positive number");
    return value;
}

void expect(const Json& object, const char* name, const Json& value) {
    if (object.at(name) != value)
        throw ArtifactError(std::string("GLM5Next ") + name + ": unsupported model semantics");
}

const Json& companion(const artifact::Directory& directory, const char* name) {
    const auto& component = directory.component(name);
    if (component.target != "text")
        throw ArtifactError(std::string("GLM5Next ") + name + ": target must be text");
    return component.config;
}

TextConfig text_config(const Json& c) {
    expect(c, "architectures", Json::array({"Glm5NextForCausalLM"}));
    expect(c, "model_type", "glm5next_text");
    expect(c, "hidden_act", "silu");
    expect(c, "tie_word_embeddings", false);
    expect(c, "attention_bias", false);
    expect(c, "mhc", true);
    expect(c, "mla_use_nope", true);
    expect(c, "qk_rope_head_dim", 0);
    expect(c, "scoring_func", "sigmoid");
    expect(c, "topk_method", "noaux_tc");
    expect(c, "moe_router_dtype", "float32");
    expect(c, "norm_topk_prob", true);
    expect(c, "index_kpool_compress", true);
    expect(c, "index_kpool_always_select_tail", true);

    TextConfig out{};
    out.hidden             = dimension(c, "hidden_size");
    out.vocabulary         = dimension(c, "vocab_size");
    out.max_context        = dimension(c, "max_position_embeddings");
    out.dense_intermediate = dimension(c, "intermediate_size");
    out.norm_epsilon       = positive(c, "rms_norm_eps");
    out.swiglu_limit       = positive(c, "swiglu_limit");
    out.mhc = {dimension(c, "hc_mult"), dimension(c, "hc_sinkhorn_iters"), positive(c, "hc_eps")};
    out.kda = {dimension(c, "linear_num_heads"), dimension(c, "linear_head_dim"),
               dimension(c, "linear_conv_kernel_dim"), dimension(c, "linear_gate_lowrank"),
               number(c, "linear_lower_bound")};
    if (out.kda.gate_lower_bound >= 0)
        throw ArtifactError("GLM5Next KDA log-decay lower bound must be negative");
    const auto pool_size = dimension(c, "index_kpool");
    const auto budget    = dimension(c, "index_topk");
    if (budget % pool_size) throw ArtifactError("GLM5Next index budget must contain whole pools");
    if (!c.at("index_share_for_mtp_iteration").is_boolean())
        throw ArtifactError("GLM5Next index_share_for_mtp_iteration must be boolean");
    out.mla = {dimension(c, "num_attention_heads"),
               dimension(c, "qk_head_dim"),
               dimension(c, "v_head_dim"),
               dimension(c, "q_lora_rank"),
               dimension(c, "kv_lora_rank"),
               dimension(c, "index_n_heads"),
               dimension(c, "index_head_dim"),
               pool_size,
               budget / pool_size,
               c.at("index_share_for_mtp_iteration").get<bool>()};
    if (dimension(c, "qk_nope_head_dim") != out.mla.query_dim ||
        dimension(c, "num_key_value_heads") != out.mla.heads)
        throw ArtifactError("GLM5Next MLA dimensions disagree with the no-RoPE factorization");
    out.moe         = {dimension(c, "n_routed_experts"),
                       dimension(c, "num_experts_per_tok"),
                       dimension(c, "moe_intermediate_size"),
                       dimension(c, "n_shared_experts"),
                       dimension(c, "n_group"),
                       dimension(c, "topk_group"),
                       positive(c, "routed_scaling_factor")};
    const auto& moe = out.moe;
    if (moe.experts % moe.groups || moe.selected_groups > moe.groups ||
        moe.top_k > std::uint64_t(moe.experts / moe.groups) * moe.selected_groups)
        throw ArtifactError("GLM5Next routed expert and group counts are inconsistent");

    const auto count    = dimension(c, "num_hidden_layers");
    const auto& mixers  = c.at("layer_types");
    const auto& ffns    = c.at("mlp_layer_types");
    const auto& indices = c.at("indexer_types");
    if (!mixers.is_array() || mixers.size() != count || !ffns.is_array() || ffns.size() != count ||
        !indices.is_array() || indices.size() != count)
        throw ArtifactError("GLM5Next layer types must cover every Main layer");
    out.layers.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        LayerConfig layer{};
        if (mixers[i] == "linear_attention")
            layer = {MixerKind::Kda, {}, out.kda_layers++};
        else if (mixers[i] == "deepseek_sparse_attention")
            layer = {MixerKind::Mla, {}, out.mla_layers++};
        else
            throw ArtifactError("GLM5Next unknown attention layer type");
        if (indices[i] != "full")
            throw ArtifactError("GLM5Next requires independent Main layer indexers");
        if (ffns[i] == "dense")
            layer.ffn = FfnKind::Dense;
        else if (ffns[i] == "sparse")
            layer.ffn = FfnKind::Moe;
        else
            throw ArtifactError("GLM5Next unknown FFN layer type");
        out.layers.push_back(layer);
    }
    return out;
}

VisionConfig vision_config(const Json& c, std::uint32_t text_hidden) {
    expect(c, "model_type", "glm5next_vision");
    expect(c, "hidden_act", "silu");
    expect(c, "attention_bias", true);
    VisionConfig out{dimension(c, "depth"),
                     dimension(c, "hidden_size"),
                     dimension(c, "num_heads"),
                     dimension(c, "intermediate_size"),
                     dimension(c, "in_channels"),
                     dimension(c, "patch_size"),
                     dimension(c, "temporal_patch_size"),
                     dimension(c, "spatial_merge_size"),
                     dimension(c, "out_hidden_size"),
                     dimension(c, "projection_intermediate_size"),
                     positive(c, "rms_norm_eps"),
                     positive(c, "swiglu_limit")};
    if (out.hidden % out.heads || out.head_dim() % 4)
        throw ArtifactError("GLM5Next Vision head dimensions must support two-axis RoPE");
    if (out.output_hidden != text_hidden)
        throw ArtifactError("GLM5Next Vision output width differs from Text hidden width");
    return out;
}
} // namespace

std::uint64_t VisionConfig::patch_width() const {
    const auto spatial = artifact::checked_mul(patch_size, patch_size, "Vision patch area");
    const auto temporal =
        artifact::checked_mul(spatial, temporal_patch_size, "Vision patch volume");
    return artifact::checked_mul(temporal, channels, "Vision patch features");
}

std::uint64_t VisionConfig::merge_width() const {
    return artifact::checked_mul(artifact::checked_mul(merge_size, merge_size, "Vision merge area"),
                                 hidden, "Vision merge features");
}

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    if (options.masked_draft()) throw ArtifactError("GLM5Next uses its native MTP backend");
    try {
        Config out{text_config(directory.component("text").config), {}, options.mtp()};
        if (options.vision)
            out.vision = vision_config(companion(directory, "vision"), out.text.hidden);
        if (out.mtp) {
            const auto& c = companion(directory, "mtp");
            expect(c, "architectures", Json::array({"Glm5NextMTP"}));
            expect(c, "model_type", "glm5next_mtp");
            expect(c, "num_hidden_layers", 1);
            expect(c, "layer_types", Json::array({"deepseek_sparse_attention"}));
            expect(directory.component("text").config, "num_nextn_predict_layers", 1);
        }
        return out;
    } catch (const Json::exception& error) {
        throw ArtifactError(std::string("invalid GLM5Next config: ") + error.what());
    }
}

} // namespace ninfer::models::glm5next
