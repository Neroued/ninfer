#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized,
                               NumericFormat format) {
    DensePostMixerPayload out;
    out.gate_up = artifact::materialized_weight(materialized, plan.gate_up, format, 34816, 5120);
    out.down    = artifact::materialized_weight(
        materialized, plan.down,
        format == NumericFormat::W8G32_F16S ? NumericFormat::W8G32_F16S : NumericFormat::Q5G64_F16S,
        5120, 17408);
    return out;
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    out.token_embedding = artifact::bind_device_tensor(binder, "text/token_embedding",
                                                       NumericFormat::Q6G64_F16S, {248320, 5120});
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.query_key = artifact::bind_device_tensor(
                binder, prefix + "attention/query_key", NumericFormat::Q4G64_F16S, {7168, 5120});
            target.attention.gate_value = artifact::bind_device_tensor(
                binder, prefix + "attention/gate_value", NumericFormat::Q5G64_F16S, {7168, 5120});
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = artifact::bind_device_tensor(
                binder, prefix + "attention/output", NumericFormat::Q5G64_F16S, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.a_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/b_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.query_key = artifact::bind_device_tensor(
                binder, prefix + "gdn/query_key", NumericFormat::Q4G64_F16S, {4096, 5120});
            target.gdn.value_z = artifact::bind_device_tensor(
                binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S, {12288, 5120});
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {128});
            target.gdn.output = artifact::bind_device_tensor(
                binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up = artifact::bind_device_tensor(binder, prefix + "mlp/gate_up",
                                                          NumericFormat::Q4G64_F16S, {34816, 5120});
        target.mlp.down    = artifact::bind_device_tensor(binder, prefix + "mlp/down",
                                                          NumericFormat::Q5G64_F16S, {5120, 17408});
    }
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {5120});
    out.output_head = artifact::bind_device_tensor(binder, "text/output_head",
                                                   NumericFormat::Q6G64_F16S, {248320, 5120});
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal ? artifact::TensorPlacement::Device
                                    : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                           {131072, 5120}, proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {131072}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement =
        features.mtp ? artifact::TensorPlacement::Device : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {5120, 10240});
    out.mtp.embedding_norm       = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {5120});
    out.mtp.hidden_norm          = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {5120});
    out.mtp.input_norm           = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {5120});
    out.mtp.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                            NumericFormat::W8G32_F16S, {14336, 5120});
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output =
        bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {5120, 6144});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {5120});
    out.mtp.mlp.gate_up =
        bind_mtp("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S, {34816, 5120});
    out.mtp.mlp.down   = bind_mtp("mtp/layer/mlp/down", NumericFormat::W8G32_F16S, {5120, 17408});
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {5120});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {5120, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {5120}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding        = artifact::materialized_weight(backing, plan.token_embedding,
                                                           NumericFormat::Q6G64_F16S, 248320, 5120);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {5120});
            target.projection.query_key  = artifact::materialized_weight(
                backing, source.attention.query_key, NumericFormat::Q4G64_F16S, 7168, 5120);
            target.projection.gate_value = artifact::materialized_weight(
                backing, source.attention.gate_value, NumericFormat::Q5G64_F16S, 7168, 5120);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = artifact::materialized_weight(backing, source.attention.output,
                                                              NumericFormat::Q5G64_F16S, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing, NumericFormat::Q4G64_F16S);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {5120});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {48});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {48});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {10240, 4});
            target.projection.a_projection = artifact::materialized_weight(
                backing, source.gdn.a_projection, NumericFormat::BF16, 48, 5120);
            target.projection.b_projection = artifact::materialized_weight(
                backing, source.gdn.b_projection, NumericFormat::BF16, 48, 5120);
            target.projection.query_key = artifact::materialized_weight(
                backing, source.gdn.query_key, NumericFormat::Q4G64_F16S, 4096, 5120);
            const Weight value_z = artifact::materialized_weight(
                backing, source.gdn.value_z, NumericFormat::Q5G64_F16S, 12288, 5120);
            target.projection.value = row_view(value_z, 0, 6144);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.projection.z        = row_view(value_z, 6144, 6144);
            target.output              = artifact::materialized_weight(backing, source.gdn.output,
                                                                       NumericFormat::Q5G64_F16S, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing, NumericFormat::Q4G64_F16S);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {5120});
    output_head = artifact::materialized_weight(backing, plan.output_head,
                                                NumericFormat::Q6G64_F16S, 248320, 5120);
    if (plan.features.optimized_proposal) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_weight(backing, plan.draft_head,
                                                           NumericFormat::Q4G64_F16S, 131072, 5120);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, NumericFormat::W8G32_F16S, 5120, 10240);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, NumericFormat::W8G32_F16S, 14336, 5120);
        mtp.attention.query       = row_view(mtp.attention.packed, 0, 6144);
        mtp.attention.key         = row_view(mtp.attention.packed, 6144, 1024);
        mtp.attention.output_gate = row_view(mtp.attention.packed, 7168, 6144);
        mtp.attention.value       = row_view(mtp.attention.packed, 13312, 1024);
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {256});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {256});
        mtp.output              = artifact::materialized_weight(backing, plan.mtp.output,
                                                                NumericFormat::W8G32_F16S, 5120, 6144);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {5120});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing, NumericFormat::W8G32_F16S);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {5120});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 5120, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {5120});
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
