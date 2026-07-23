#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

NumericFormat routed_down_format(std::size_t layer) {
    return layer == 34 || layer == 38 || layer == 39 ? NumericFormat::Q6G64_F16S
                                                     : NumericFormat::Q5G64_F16S;
}

MoePlan bind_moe(artifact::Binder& binder, const std::string& prefix, NumericFormat routed_gate_up,
                 NumericFormat routed_down, artifact::TensorPlacement placement) {
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return MoePlan{
        .router_shared_gate = bind(prefix + "router_shared_gate", NumericFormat::BF16, {257, 2048}),
        .routed_gate_up     = bind(prefix + "routed_gate_up", routed_gate_up, {262144, 2048}),
        .routed_down        = bind(prefix + "routed_down", routed_down, {524288, 512}),
        .shared_gate_up = bind(prefix + "shared_gate_up", NumericFormat::W8G32_F16S, {1024, 2048}),
        .shared_down    = bind(prefix + "shared_down", NumericFormat::W8G32_F16S, {2048, 512}),
    };
}

SparseMoePayload load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& materialized,
                          NumericFormat routed_gate_up, NumericFormat routed_down) {
    return SparseMoePayload{
        .op = {
            .router_shared_gate = artifact::materialized_weight(
                materialized, plan.router_shared_gate, NumericFormat::BF16, 257, 2048),
            .routed_gate_up = artifact::materialized_weight(materialized, plan.routed_gate_up,
                                                            routed_gate_up, 262144, 2048),
            .routed_down    = artifact::materialized_weight(materialized, plan.routed_down,
                                                            routed_down, 524288, 512),
            .shared_gate_up = artifact::materialized_weight(materialized, plan.shared_gate_up,
                                                            NumericFormat::W8G32_F16S, 1024, 2048),
            .shared_down    = artifact::materialized_weight(materialized, plan.shared_down,
                                                            NumericFormat::W8G32_F16S, 2048, 512),
        }};
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
        if (id >= kTokenizerVocab || seen[id]) {
            throw artifact::ArtifactError("invalid optimized draft-head token ids");
        }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out    = load_plan.bindings;
    out.frontend        = qwen3_6::bind_frontend_resources(binder);
    out.features        = features;
    out.token_embedding = artifact::bind_device_tensor(binder, "text/token_embedding",
                                                       NumericFormat::W8G32_F16S, {248320, 2048});

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {2048});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.query_key_gate_value =
                artifact::bind_device_tensor(binder, prefix + "attention/query_key_gate_value",
                                             NumericFormat::W8G32_F16S, {9216, 2048});
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = artifact::bind_device_tensor(
                binder, prefix + "attention/output", NumericFormat::W8G32_F16S, {2048, 4096});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {32});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {32});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 8192});
            target.gdn.a_b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_b_projection", NumericFormat::BF16, {64, 2048});
            target.gdn.query_key_value_z = artifact::bind_device_tensor(
                binder, prefix + "gdn/query_key_value_z", NumericFormat::W8G32_F16S, {12288, 2048});
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {128});
            target.gdn.output = artifact::bind_device_tensor(
                binder, prefix + "gdn/output", NumericFormat::W8G32_F16S, {2048, 4096});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {2048});
        target.moe = bind_moe(binder, prefix + "moe/", NumericFormat::Q4G64_F16S,
                              routed_down_format(layer), artifact::TensorPlacement::Device);
    }

    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {2048});
    out.output_head = artifact::bind_device_tensor(binder, "text/output_head",
                                                   NumericFormat::Q6G64_F16S, {248320, 2048});
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal ? artifact::TensorPlacement::Device
                                    : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                           {131072, 2048}, proposal_placement);
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
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {2048, 4096});
    out.mtp.embedding_norm = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {2048});
    out.mtp.hidden_norm    = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {2048});
    out.mtp.input_norm     = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {2048});
    out.mtp.attention.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                                      NumericFormat::W8G32_F16S, {9216, 2048});
    out.mtp.attention.query_norm =
        bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.attention.key_norm =
        bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.attention.output =
        bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {2048, 4096});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {2048});
    out.mtp.moe        = bind_moe(binder, "mtp/layer/moe/", NumericFormat::W8G32_F16S,
                                  NumericFormat::W8G32_F16S, mtp_placement);
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {2048});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {2048, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {2048}, vision_placement);
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

    token_embedding = artifact::materialized_weight(backing, plan.token_embedding,
                                                    NumericFormat::W8G32_F16S, 248320, 2048);

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {2048});
            target.projection.query_key_gate_value =
                artifact::materialized_weight(backing, source.attention.query_key_gate_value,
                                              NumericFormat::W8G32_F16S, 9216, 2048);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = artifact::materialized_weight(backing, source.attention.output,
                                                              NumericFormat::W8G32_F16S, 2048, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            target.post_mixer =
                load_moe(source.moe, backing, NumericFormat::Q4G64_F16S, routed_down_format(layer));
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {2048});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {32});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {32});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {8192, 4});
            target.projection.a_b_projection = artifact::materialized_weight(
                backing, source.gdn.a_b_projection, NumericFormat::BF16, 64, 2048);
            target.projection.query_key_value_z = artifact::materialized_weight(
                backing, source.gdn.query_key_value_z, NumericFormat::W8G32_F16S, 12288, 2048);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output              = artifact::materialized_weight(backing, source.gdn.output,
                                                                       NumericFormat::W8G32_F16S, 2048, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            target.post_mixer =
                load_moe(source.moe, backing, NumericFormat::Q4G64_F16S, routed_down_format(layer));
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("35B Text topology binding is incomplete");
    }

    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {2048});
    output_head = artifact::materialized_weight(backing, plan.output_head,
                                                NumericFormat::Q6G64_F16S, 248320, 2048);
    if (plan.features.optimized_proposal) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_weight(backing, plan.draft_head,
                                                           NumericFormat::Q4G64_F16S, 131072, 2048);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(backing, plan.mtp.input_projection,
                                                             NumericFormat::W8G32_F16S, 2048, 4096);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.attention.query_key_gate_value =
            artifact::materialized_weight(backing, plan.mtp.attention.query_key_gate_value,
                                          NumericFormat::W8G32_F16S, 9216, 2048);
        mtp.query_norm = artifact::materialized_tensor(backing, plan.mtp.attention.query_norm,
                                                       NumericFormat::BF16, {256});
        mtp.key_norm   = artifact::materialized_tensor(backing, plan.mtp.attention.key_norm,
                                                       NumericFormat::BF16, {256});
        mtp.output     = artifact::materialized_weight(backing, plan.mtp.attention.output,
                                                       NumericFormat::W8G32_F16S, 2048, 4096);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {2048});
        mtp.post_mixer =
            load_moe(plan.mtp.moe, backing, NumericFormat::W8G32_F16S, NumericFormat::W8G32_F16S);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {2048});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 2048, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {2048});
    }
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
