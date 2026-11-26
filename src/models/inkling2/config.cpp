#include "models/inkling2/config.h"

#include "artifact/schema.h"

#include <cmath>
#include <set>
#include <string>

namespace ninfer::models::inkling2 {
namespace {
using artifact::ArtifactError;
using artifact::Json;

std::uint32_t uint_field(const Json& object, const char* name, bool positive = true) {
    const auto value = artifact::require_u64(object.at(name), name, positive);
    if (value > UINT32_MAX) throw ArtifactError(std::string(name) + ": value exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

float float_field(const Json& object, const char* name) {
    if (!object.at(name).is_number())
        throw ArtifactError(std::string(name) + ": expected a finite number");
    const auto value = object.at(name).get<float>();
    if (!std::isfinite(value))
        throw ArtifactError(std::string(name) + ": expected a finite number");
    return value;
}

float positive_float(const Json& object, const char* name) {
    const auto value = float_field(object, name);
    if (value <= 0.0F) throw ArtifactError(std::string(name) + ": expected a positive number");
    return value;
}

float nonnegative_float(const Json& object, const char* name) {
    const auto value = float_field(object, name);
    if (value < 0.0F) throw ArtifactError(std::string(name) + ": expected a nonnegative number");
    return value;
}

void expect(const Json& object, const char* name, const Json& value) {
    if (object.at(name) != value)
        throw ArtifactError(std::string("Inkling2 ") + name + ": unsupported model semantics");
}

const Json& companion(const artifact::Directory& directory, const char* name) {
    const auto& component = directory.component(name);
    if (component.target != "text")
        throw ArtifactError(std::string("Inkling2 ") + name + ": target must be text");
    return component.config;
}

std::vector<bool> parse_layer_mask(const Json& array, std::uint32_t layer_count, const char* name) {
    if (!array.is_array()) throw ArtifactError(std::string(name) + ": expected array");
    std::vector<bool> mask(layer_count, false);
    for (const auto& entry : array) {
        const auto id = artifact::require_u64(entry, name, false);
        if (id >= layer_count || mask[id])
            throw ArtifactError(std::string(name) + ": invalid or duplicate layer index");
        mask[id] = true;
    }
    return mask;
}

TextConfig parse_text_config(const Json& c) {
    expect(c, "architectures", Json::array({"Inkling2ForCausalLM"}));
    expect(c, "model_type", "inkling2_text");
    expect(c, "q_bias", false);
    expect(c, "o_bias", false);
    expect(c, "use_value_norm_no_gamma", true);
    expect(c, "attn_res_normalize_values", true);
    expect(c, "n_shared_experts", 0);
    expect(c, "use_gate_bias", true);

    TextConfig out{};
    out.hidden              = uint_field(c, "hidden_size");
    out.num_layers          = uint_field(c, "num_hidden_layers");
    out.vocabulary          = uint_field(c, "vocab_size");
    out.unpadded_vocabulary = uint_field(c, "unpadded_vocab_size");
    if (out.unpadded_vocabulary > out.vocabulary)
        throw ArtifactError("Inkling2 unpadded_vocab_size exceeds vocab_size");
    out.max_context  = uint_field(c, "max_position_embeddings");
    out.eos_token_id = uint_field(c, "eos_token_id", false);
    if (out.eos_token_id >= out.unpadded_vocabulary)
        throw ArtifactError("Inkling2 eos_token_id is outside unpadded_vocab_size");

    out.attn_res_block_size = uint_field(c, "n_layers_per_attn_res_block");
    out.sconv_kernel_size   = uint_field(c, "sconv_kernel_size");
    out.d_rel               = uint_field(c, "d_rel");
    out.rel_extent          = uint_field(c, "rel_extent");
    out.sliding_window      = uint_field(c, "sliding_window_size");
    out.norm_epsilon        = positive_float(c, "rms_norm_eps");
    out.qk_l2_clip_threshold  = positive_float(c, "qk_l2_clip_threshold");
    out.rel_l2_clip_threshold = positive_float(c, "rel_l2_clip_threshold");
    if (const auto it = c.find("log_scaling_n_floor"); it != c.end() && !it->is_null()) {
        const auto floor = artifact::require_u64(*it, "log_scaling_n_floor", true);
        if (floor > UINT32_MAX)
            throw ArtifactError("Inkling2 log_scaling_n_floor exceeds uint32");
        out.log_scaling_n_floor = static_cast<std::uint32_t>(floor);
    }
    out.log_scaling_alpha       = nonnegative_float(c, "log_scaling_alpha");
    out.final_logit_softcapping = positive_float(c, "final_logit_softcapping");

    out.full_attention = {uint_field(c, "num_attention_heads"),
                          uint_field(c, "num_key_value_heads"), uint_field(c, "head_dim")};
    out.swa_attention  = {uint_field(c, "swa_num_attention_heads"),
                          uint_field(c, "swa_num_key_value_heads"), uint_field(c, "swa_head_dim")};
    if (out.full_attention.heads % out.full_attention.kv_heads != 0 ||
        out.swa_attention.heads % out.swa_attention.kv_heads != 0)
        throw ArtifactError("Inkling2 attention heads must be divisible by key-value heads");

    out.moe = {uint_field(c, "n_routed_experts"),
               uint_field(c, "num_experts_per_tok"),
               uint_field(c, "intermediate_size"),
               positive_float(c, "mlp_gate_softcap"),
               positive_float(c, "mlp_up_softcap"),
               positive_float(c, "route_scale")};
    if (out.moe.top_k > out.moe.experts)
        throw ArtifactError("Inkling2 num_experts_per_tok exceeds n_routed_experts");

    if (const auto it = c.find("loop_config"); it != c.end() && !it->is_null()) {
        if (!it->is_object()) throw ArtifactError("Inkling2 loop_config must be an object");
        expect(*it, "is_varloop", false);
        LoopConfig loop{uint_field(*it, "core_start"), uint_field(*it, "core_end"),
                        uint_field(*it, "num_passes"), uint_field(*it, "gate_rank")};
        if (loop.core_start >= loop.core_end || loop.core_end > out.num_layers ||
            loop.core_start % out.attn_res_block_size != 0 ||
            loop.core_end % out.attn_res_block_size != 0 || loop.num_passes < 2)
            throw ArtifactError("Inkling2 loop_config schedule is invalid");
        out.loop = loop;
    }

    const auto local_mask = parse_layer_mask(c.at("local_layer_ids"), out.num_layers,
                                             "Inkling2 local_layer_ids");
    out.physical_layers.reserve(out.num_layers);
    for (std::uint32_t layer = 0; layer < out.num_layers; ++layer) {
        const bool in_core =
            out.loop && layer >= out.loop->core_start && layer < out.loop->core_end;
        out.physical_layers.push_back(
            {layer, local_mask[layer], in_core, in_core ? out.loop->num_passes : 1U});
    }

    const auto append_logical = [&](std::uint32_t physical, std::uint32_t pass) {
        const auto& phys         = out.physical_layers[physical];
        const auto logical_index = static_cast<std::uint32_t>(out.logical_layers.size());
        const auto state_index   = phys.is_swa ? out.swa_layers++ : out.full_layers++;
        out.logical_layers.push_back(
            {logical_index, physical, pass, phys.is_swa, phys.in_core, state_index,
             logical_index / out.attn_res_block_size,
             logical_index > 0 && (logical_index % out.attn_res_block_size) == 0, std::nullopt});
    };
    if (!out.loop) {
        out.logical_layers.reserve(out.num_layers);
        for (std::uint32_t l = 0; l < out.num_layers; ++l) append_logical(l, 0);
    } else {
        const auto& loop = *out.loop;
        out.logical_layers.reserve(out.num_layers +
                                   loop.core_layers() * (loop.num_passes - 1U));
        for (std::uint32_t l = 0; l < loop.core_start; ++l) append_logical(l, 0);
        for (std::uint32_t pass = 0; pass < loop.num_passes; ++pass)
            for (std::uint32_t l = loop.core_start; l < loop.core_end; ++l)
                append_logical(l, pass);
        for (std::uint32_t l = loop.core_end; l < out.num_layers; ++l) append_logical(l, 0);
    }
    return out;
}

VisionConfig parse_vision_config(const Json& c, std::uint32_t text_hidden) {
    expect(c, "model_type", "inkling2_vision");
    if (!c.at("projector_bias").is_boolean())
        throw ArtifactError("Inkling2 vision projector_bias must be boolean");
    VisionConfig out{uint_field(c, "num_hidden_layers"),
                     uint_field(c, "hidden_size"),
                     uint_field(c, "num_attention_heads"),
                     uint_field(c, "head_dim"),
                     uint_field(c, "intermediate_size"),
                     uint_field(c, "native_patch_size"),
                     uint_field(c, "adapter_merge_size"),
                     uint_field(c, "temporal_patch_size"),
                     uint_field(c, "posemb_grid_size"),
                     uint_field(c, "projection_hidden_dim"),
                     uint_field(c, "text_hidden_size"),
                     c.at("projector_bias").get<bool>(),
                     positive_float(c, "layer_norm_eps")};
    if (out.hidden != out.heads * out.head_dim || (out.head_dim % 4) != 0)
        throw ArtifactError("Inkling2 Vision head geometry must support 2D RoPE");
    if (out.text_hidden != text_hidden)
        throw ArtifactError("Inkling2 Vision text_hidden_size differs from Text hidden_size");
    return out;
}

DSparkConfig parse_dspark_config(const Json& c, TextConfig& text) {
    expect(c, "architectures", Json::array({"Inkling2DSparkDraftModel"}));
    expect(c, "model_type", "inkling2_dspark");
    expect(c, "markov_head_type", "rnn_hidden");

    DSparkConfig out{};
    out.hidden              = uint_field(c, "hidden_size");
    out.num_layers          = uint_field(c, "num_hidden_layers");
    out.heads               = uint_field(c, "num_attention_heads");
    out.kv_heads            = uint_field(c, "num_key_value_heads");
    out.head_dim            = uint_field(c, "head_dim");
    out.intermediate        = uint_field(c, "intermediate_size");
    out.max_context         = uint_field(c, "max_position_embeddings");
    out.vocabulary          = uint_field(c, "vocab_size");
    out.unpadded_vocabulary = uint_field(c, "unpadded_vocab_size");
    out.sliding_window      = uint_field(c, "sliding_window_size");
    out.block_size          = uint_field(c, "block_size");
    out.markov_rank         = uint_field(c, "markov_rank");
    out.norm_epsilon        = positive_float(c, "rms_norm_eps");
    out.rope_theta          = positive_float(c, "rope_theta");
    out.final_logit_softcapping = positive_float(c, "final_logit_softcapping");

    if (out.hidden != text.hidden || out.vocabulary != text.vocabulary ||
        out.unpadded_vocabulary != text.unpadded_vocabulary)
        throw ArtifactError("Inkling2 DSpark hidden/vocabulary dimensions differ from Text");
    if (out.heads % out.kv_heads != 0 || (out.head_dim % 2) != 0)
        throw ArtifactError("Inkling2 DSpark attention geometry is invalid");
    if (out.block_size < 2)
        throw ArtifactError("Inkling2 DSpark block_size must be at least 2");

    const auto local_mask = parse_layer_mask(c.at("local_layer_ids"), out.num_layers,
                                             "Inkling2 dspark.local_layer_ids");
    out.layers.reserve(out.num_layers);
    for (std::uint32_t i = 0; i < out.num_layers; ++i) {
        const bool is_swa = local_mask[i];
        out.layers.push_back({is_swa, is_swa ? out.swa_layers++ : out.full_layers++});
    }

    const auto& taps = c.at("target_layer_ids");
    if (!taps.is_array() || taps.empty())
        throw ArtifactError("Inkling2 dspark.target_layer_ids must be a non-empty array");
    const auto logical_count = text.logical_layer_count();
    std::set<std::uint32_t> seen;
    out.target_layer_ids.reserve(taps.size());
    for (std::uint32_t slot = 0; slot < taps.size(); ++slot) {
        const auto id = artifact::require_u64(taps[slot], "dspark.target_layer_ids", false);
        if (id > logical_count || !seen.insert(static_cast<std::uint32_t>(id)).second)
            throw ArtifactError("Inkling2 dspark.target_layer_ids contains invalid or duplicate id");
        const auto tap = static_cast<std::uint32_t>(id);
        out.target_layer_ids.push_back(tap);
        if (tap == logical_count)
            out.final_aux_slot = slot;
        else
            text.logical_layers[tap].capture_aux_slot = slot;
    }
    return out;
}

} // namespace

std::uint64_t VisionConfig::patch_width() const {
    const auto area = artifact::checked_mul(native_patch_size, native_patch_size,
                                            "Inkling2 Vision patch area");
    return artifact::checked_mul(area, 3U, "Inkling2 Vision patch features");
}

std::uint64_t VisionConfig::merge_width() const {
    const auto area =
        artifact::checked_mul(merge_size, merge_size, "Inkling2 Vision merge area");
    return artifact::checked_mul(area, hidden, "Inkling2 Vision merge features");
}

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    if (options.mtp()) throw ArtifactError("Inkling2 has no MTP component; use DSpark");
    if (options.dflash() || options.dflash2())
        throw ArtifactError("Inkling2 has no DFlash component; use DSpark");
    if (options.dspark() && options.proposal_head != ProposalHead::Full)
        throw ArtifactError("Inkling2 DSpark requires the full proposal head");
    try {
        Config out{parse_text_config(directory.component("text").config), std::nullopt,
                   std::nullopt};
        if (options.vision)
            out.vision = parse_vision_config(companion(directory, "vision"), out.text.hidden);
        if (options.dspark())
            out.dspark = parse_dspark_config(companion(directory, "dspark"), out.text);
        return out;
    } catch (const Json::exception& error) {
        throw ArtifactError(std::string("invalid Inkling2 config: ") + error.what());
    }
}

} // namespace ninfer::models::inkling2
