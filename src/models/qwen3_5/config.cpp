#include "models/qwen3_5/config.h"

#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5 {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

bool is_prime(std::uint64_t value) {
    if (value < 2) { return false; }
    if (value % 2 == 0) { return value == 2; }
    for (std::uint64_t divisor = 3; divisor * divisor <= value; divisor += 2) {
        if (value % divisor == 0) { return false; }
    }
    return true;
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

std::string architecture(const Json& config) {
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

const artifact::Component& companion(const artifact::Directory& directory, std::string_view name) {
    const auto& component = directory.component(name);
    if (component.target != "text") {
        throw ArtifactError(std::string(name) + ": target must be text");
    }
    return component;
}

AttentionConfig attention(const Json& value) {
    AttentionConfig out{dimension(value, "num_attention_heads"),
                        dimension(value, "num_key_value_heads"), dimension(value, "head_dim")};
    if (out.num_attention_heads % out.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    return out;
}

RopeConfig rope(const Json& value, std::uint32_t head_dim) {
    require_members(value, {"rope_theta", "partial_rotary_factor", "mrope_section"}, {},
                    "text RoPE");
    RopeConfig out;
    out.rope_theta            = positive_float(value, "rope_theta");
    out.partial_rotary_factor = positive_float(value, "partial_rotary_factor");
    if (out.partial_rotary_factor > 1) { throw ArtifactError("partial_rotary_factor exceeds one"); }
    out.rotary_dim = static_cast<std::uint32_t>(double(head_dim) * out.partial_rotary_factor);
    if (!out.rotary_dim || out.rotary_dim % 2) {
        throw ArtifactError("rotary dimension must be positive and even");
    }
    const auto& sections = value.at("mrope_section");
    if (!sections.is_array() || sections.size() != 3) {
        throw ArtifactError("MRoPE requires three sections");
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        out.mrope_section[i] = integer(sections[i], "MRoPE section", false);
        sum += out.mrope_section[i];
    }
    if (sum != out.rotary_dim / 2) {
        throw ArtifactError("MRoPE sections differ from the rotary width");
    }
    std::array<std::uint32_t, 3> counts{};
    for (std::uint32_t pair = 0; pair < out.rotary_dim / 2; ++pair) {
        const std::uint8_t axis = pair % 3 == 1 && pair < 3ULL * out.mrope_section[1]   ? 1
                                  : pair % 3 == 2 && pair < 3ULL * out.mrope_section[2] ? 2
                                                                                        : 0;
        out.pair_axes.push_back(axis);
        ++counts[axis];
    }
    if (counts != out.mrope_section) {
        throw ArtifactError("MRoPE sections exceed interleaved axis ranges");
    }
    return out;
}

void qwen4_extensions(const Json& value, TextConfig& out) {
    if (!out.attention || !out.gdn || !out.rope_parameters) {
        throw ArtifactError("Qwen4Exp requires gated attention and GDN geometry");
    }
    const auto& gate = value.at("output_gate_type");
    if (gate == "sigmoid") {
        out.gdn_gate = GateActivation::Sigmoid;
    } else if (gate == "silu") {
        out.gdn_gate = GateActivation::Silu;
    } else {
        throw ArtifactError("unknown output_gate_type");
    }

    HyperConnectionConfig hc{dimension(value, "hc_count"), dimension(value, "hc_lowrank")};
    if (hc.hc_count < 2) { throw ArtifactError("hyper-connections need at least two streams"); }
    (void)artifact::checked_mul(hc.hc_count, out.hidden_size, "hyper-connection width");
    out.hyper_connection = hc;

    SparseAttentionConfig qsa{
        dimension(value, "indexer_n_heads"), dimension(value, "indexer_head_dim"),
        dimension(value, "indexer_budget"), dimension(value, "indexer_compress_ratio")};
    if (dimension(value, "indexer_kv_heads") != 1) {
        throw ArtifactError("QSA requires one indexer key head");
    }
    if (qsa.budget % qsa.compress_ratio) {
        throw ArtifactError("QSA budget must be a whole number of blocks");
    }
    if (out.rope_parameters->rotary_dim > qsa.indexer_head_dim) {
        throw ArtifactError("QSA indexer head is narrower than the rotary width");
    }
    out.sparse_attention = qsa;

    // PLE is optional: no (or an empty) ple_layer_ids means no n-gram embedding, and the n-gram
    // fields are then not read. One layer is implemented; more would need one table per layer.
    if (!value.contains("ple_layer_ids")) { return; }
    const auto& ids = value.at("ple_layer_ids");
    if (!ids.is_array() || ids.size() > 1) {
        throw ArtifactError("Qwen4Exp implements at most one one-indexed PLE layer");
    }
    if (ids.empty()) { return; }
    PleConfig ple;
    const auto one_indexed = integer(ids[0], "PLE layer");
    if (one_indexed > out.num_hidden_layers ||
        out.layer_types[one_indexed - 1] != MixerKind::LinearAttention) {
        throw ArtifactError("PLE layer must name a GDN block");
    }
    ple.layer           = one_indexed - 1;
    ple.ngram_size      = dimension(value, "ngram_size");
    ple.heads_per_ngram = dimension(value, "heads_per_ngram");
    ple.embed_dim       = dimension(value, "ple_embed_dim");
    ple.conv_kernel     = dimension(value, "ple_conv_kernel_size");
    ple.eos_token_id    = integer(value.at("eos_token_id"), "eos_token_id", false);
    if (ple.ngram_size < 2 || ple.embed_dim % ple.heads() || ple.eos_token_id >= out.vocab_size) {
        throw ArtifactError("invalid n-gram geometry");
    }
    const auto base = artifact::require_u64(value.at("ngram_vocab_size_base"), "vocab base");
    const auto divisor =
        artifact::require_u64(value.at("make_ngram_vocab_size_divisible_by"), "n-gram row divisor");
    const auto seed = artifact::require_u64(value.at("seed"), "seed", false);
    // Hash products stay in signed 64-bit range because each multiplier is below 2^63 / vocab.
    const std::uint64_t half_bound = std::max<std::uint64_t>(
        1, (std::uint64_t(std::numeric_limits<std::int64_t>::max()) / out.vocab_size) / 2);
    const std::uint64_t layer_seed = seed; // single PLE layer: ple_layer_index == 0
    for (std::uint32_t i = 0; i < ple.ngram_size; ++i) {
        ple.multipliers.push_back(
            2 * (splitmix64(layer_seed + 0x9E3779B97F4A7C15ULL * (i + 1)) % half_bound) + 1);
    }
    std::uint64_t prime = base - 1;
    std::uint64_t total = 0;
    for (std::uint32_t head = 0; head < ple.heads(); ++head) {
        do { ++prime; } while (!is_prime(prime));
        ple.head_moduli.push_back(prime);
        ple.head_offsets.push_back(total);
        total = artifact::checked_add(total, prime, "n-gram table rows");
    }
    ple.table_rows =
        artifact::checked_mul((total + divisor - 1) / divisor, divisor, "padded n-gram table rows");
    out.ple = std::move(ple);
}

TextConfig text(const Json& value, bool mtp) {
    require_members(value,
                    {"architectures", "model_type", "hidden_size", "vocab_size",
                     "num_hidden_layers", "max_position_embeddings", "tie_word_embeddings",
                     "rms_norm_eps", "layer_types"},
                    {"num_attention_heads",
                     "num_key_value_heads",
                     "head_dim",
                     "rope_parameters",
                     "linear_num_key_heads",
                     "linear_key_head_dim",
                     "linear_num_value_heads",
                     "linear_value_head_dim",
                     "linear_conv_kernel_dim",
                     "intermediate_size",
                     "num_experts",
                     "num_experts_per_tok",
                     "moe_intermediate_size",
                     "shared_expert_intermediate_size",
                     "output_gate_type",
                     "hc_count",
                     "hc_lowrank",
                     "indexer_n_heads",
                     "indexer_kv_heads",
                     "indexer_head_dim",
                     "indexer_budget",
                     "indexer_compress_ratio",
                     "ple_layer_ids",
                     "ple_embed_dim",
                     "ple_conv_kernel_size",
                     "ngram_size",
                     "heads_per_ngram",
                     "ngram_vocab_size_base",
                     "make_ngram_vocab_size_divisible_by",
                     "seed",
                     "eos_token_id"},
                    "text config");
    TextConfig out;
    out.architecture = resolve_architecture(
        architecture(value), artifact::require_id(value.at("model_type"), "model_type"));
    out.hidden_size             = dimension(value, "hidden_size");
    out.vocab_size              = dimension(value, "vocab_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    if (!value.at("tie_word_embeddings").is_boolean()) {
        throw ArtifactError("tie_word_embeddings must be boolean");
    }
    out.tie_word_embeddings = value.at("tie_word_embeddings").get<bool>();
    const auto& layers      = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must cover every text block");
    }
    for (const auto& layer : layers) {
        if (layer == "full_attention") {
            out.layer_types.push_back(MixerKind::FullAttention);
            out.compact_layer_indices.push_back(out.full_attention_layers++);
        } else if (layer == "linear_attention") {
            out.layer_types.push_back(MixerKind::LinearAttention);
            out.compact_layer_indices.push_back(out.linear_attention_layers++);
        } else {
            throw ArtifactError("unknown text layer_type");
        }
    }
    if (out.full_attention_layers || mtp || value.contains("head_dim") ||
        value.contains("rope_parameters") || value.contains("num_attention_heads") ||
        value.contains("num_key_value_heads")) {
        out.attention       = attention(value);
        out.rope_parameters = rope(value.at("rope_parameters"), out.attention->head_dim);
    }
    if (out.linear_attention_layers || value.contains("linear_num_key_heads") ||
        value.contains("linear_key_head_dim") || value.contains("linear_num_value_heads") ||
        value.contains("linear_value_head_dim") || value.contains("linear_conv_kernel_dim")) {
        GdnConfig gdn{
            dimension(value, "linear_num_key_heads"), dimension(value, "linear_key_head_dim"),
            dimension(value, "linear_num_value_heads"), dimension(value, "linear_value_head_dim"),
            dimension(value, "linear_conv_kernel_dim")};
        if (gdn.linear_num_value_heads % gdn.linear_num_key_heads) {
            throw ArtifactError("GDN value heads must be divisible by key heads");
        }
        (void)gdn.conv_channels();
        out.gdn = gdn;
    }
    const bool qwen4 = out.architecture == Architecture::Qwen4Exp;
    for (const auto* key :
         {"output_gate_type", "hc_count", "hc_lowrank", "indexer_n_heads", "indexer_kv_heads",
          "indexer_head_dim", "indexer_budget", "indexer_compress_ratio", "ple_layer_ids",
          "ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
          "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "seed", "eos_token_id"}) {
        if (!qwen4 && value.contains(key)) {
            throw ArtifactError(std::string("Qwen3.5 config carries Qwen4Exp field ") + key);
        }
    }
    if (qwen4) { qwen4_extensions(value, out); }
    // Qwen4Exp has either FFN: the routed-expert MoE (Qwen3.8-Flash-Next) or a dense SwiGLU FFN
    // (intermediate_size and no MoE field). Qwen3_5 is dense.
    const bool dense =
        out.architecture == Architecture::Qwen3_5 || (qwen4 && value.contains("intermediate_size"));
    if (!dense) {
        MoeConfig moe{dimension(value, "num_experts"), dimension(value, "num_experts_per_tok"),
                      dimension(value, "moe_intermediate_size"),
                      dimension(value, "shared_expert_intermediate_size")};
        if (moe.num_experts_per_tok > moe.num_experts) {
            throw ArtifactError("selected experts exceed expert count");
        }
        out.ffn = moe;
    } else {
        for (const auto* key : {"num_experts", "num_experts_per_tok", "moe_intermediate_size",
                                "shared_expert_intermediate_size"}) {
            if (value.contains(key)) { throw ArtifactError("Dense config carries MoE fields"); }
        }
        out.ffn = DenseConfig{dimension(value, "intermediate_size")};
    }
    return out;
}

VisionConfig vision(const Json& value) {
    require_members(value,
                    {"model_type", "depth", "hidden_size", "intermediate_size", "num_heads",
                     "patch_size", "temporal_patch_size", "spatial_merge_size",
                     "num_position_embeddings"},
                    {}, "vision config");
    if (value.at("model_type") != "qwen3_5_vision" &&
        value.at("model_type") != "qwen4_exp_vision") {
        throw ArtifactError("unknown Vision model_type");
    }
    VisionConfig out;
    out.depth                   = dimension(value, "depth");
    out.hidden_size             = dimension(value, "hidden_size");
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_heads               = dimension(value, "num_heads");
    out.patch_size              = dimension(value, "patch_size");
    out.temporal_patch_size     = dimension(value, "temporal_patch_size");
    out.spatial_merge_size      = dimension(value, "spatial_merge_size");
    out.num_position_embeddings = dimension(value, "num_position_embeddings");
    out.position_grid_side =
        static_cast<std::uint32_t>(std::sqrt(double(out.num_position_embeddings)));
    if (out.hidden_size % out.num_heads || (out.hidden_size / out.num_heads) % 4 ||
        std::uint64_t(out.position_grid_side) * out.position_grid_side !=
            out.num_position_embeddings) {
        throw ArtifactError("invalid Vision head or square position geometry");
    }
    (void)out.patch_width();
    (void)out.merger_width();
    return out;
}

DraftConfig draft(const Json& value, const TextConfig& target) {
    require_members(value,
                    {"architectures", "model_type", "intermediate_size", "num_attention_heads",
                     "num_key_value_heads", "head_dim", "num_hidden_layers",
                     "max_position_embeddings", "rms_norm_eps", "rope_parameters", "layer_types",
                     "sliding_window", "dflash_config"},
                    {}, "draft config");
    if (architecture(value) != "DFlash2DraftModel" || value.at("model_type") != "qwen3") {
        throw ArtifactError("draft architecture/model_type mismatch");
    }
    DraftConfig out;
    out.attention = attention(value);
    if (out.attention.head_dim % 2) { throw ArtifactError("draft head dimension must be even"); }
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "draft RoPE");
    out.rope_theta     = positive_float(value.at("rope_parameters"), "rope_theta");
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("draft layer_types length mismatch");
    }
    // Every DFlash2 drafter layer attends over its sliding window (its context KV is the local
    // ring the fused materialization writes).
    for (const auto& layer : layers) {
        if (layer != "sliding_attention") {
            throw ArtifactError("DFlash2 draft layers must be sliding_attention");
        }
    }
    out.sliding_window = dimension(value, "sliding_window");
    const auto& config = value.at("dflash_config");
    require_members(config,
                    {"target_layer_ids", "mask_token_id", "conv_kernel_size", "conv_group_size",
                     "selector_rank", "selector_top_k"},
                    {}, "DFlash2 config");
    const auto& taps = config.at("target_layer_ids");
    if (!taps.is_array() || taps.empty()) {
        throw ArtifactError("target_layer_ids must be nonempty");
    }
    std::set<std::uint32_t> unique;
    for (const auto& tap : taps) {
        const auto index = integer(tap, "target layer index", false);
        if (index >= target.num_hidden_layers || !unique.insert(index).second) {
            throw ArtifactError("invalid target layer index");
        }
        out.target_layer_ids.push_back(index);
    }
    out.mask_token_id = integer(config.at("mask_token_id"), "mask token", false);
    if (out.mask_token_id >= target.vocab_size) {
        throw ArtifactError("mask token exceeds target embedding rows");
    }
    out.dflash2 =
        DFlash2Config{dimension(config, "conv_kernel_size"), dimension(config, "conv_group_size"),
                      dimension(config, "selector_rank"), dimension(config, "selector_top_k")};
    if (target.hidden_size % out.dflash2.conv_group_size) {
        throw ArtifactError("DFlash2 conv group does not divide hidden width");
    }
    return out;
}

} // namespace

std::uint64_t GdnConfig::conv_channels() const {
    return artifact::checked_add(artifact::checked_mul(2, key_width(), "GDN key width"),
                                 value_width(), "GDN convolution channels");
}

std::uint64_t VisionConfig::patch_width() const {
    return artifact::checked_mul(artifact::checked_mul(3, temporal_patch_size, "patch channels"),
                                 std::uint64_t(patch_size) * patch_size, "patch width");
}

std::uint64_t VisionConfig::merger_width() const {
    return artifact::checked_mul(std::uint64_t(spatial_merge_size) * spatial_merge_size,
                                 hidden_size, "merger width");
}

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.purpose != EnginePurpose::Generation &&
            options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        if (options.purpose == EnginePurpose::CausalScoring &&
            (options.vision || options.speculative != SpeculativeBackend::None)) {
            throw ArtifactError("CausalScoring loads the Text backbone only");
        }
        if (options.speculative != SpeculativeBackend::None &&
            options.speculative_component().empty()) {
            throw ArtifactError("unknown speculative backend");
        }
        if (options.proposal_head != ProposalHead::Full &&
            options.proposal_head != ProposalHead::Optimized) {
            throw ArtifactError("unknown proposal head selection");
        }
        Config out;
        out.mtp  = options.speculative == SpeculativeBackend::Mtp;
        out.text = text(directory.component("text").config, out.mtp);
        if (options.vision) { out.vision = vision(companion(directory, "vision").config); }
        if (out.mtp) {
            const auto& config = companion(directory, "mtp").config;
            require_members(config, {"architectures"}, {}, "MTP config");
            const std::string_view expected =
                mtp_architecture(architecture(directory.component("text").config));
            if (expected.empty() || architecture(config) != expected) {
                throw ArtifactError("MTP architecture differs from target mathematics");
            }
        }
        if (options.speculative == SpeculativeBackend::DFlash2) {
            out.draft =
                draft(companion(directory, options.speculative_component()).config, out.text);
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Qwen3.5 config: ") + error.what());
    }
}

} // namespace ninfer::models::qwen3_5
