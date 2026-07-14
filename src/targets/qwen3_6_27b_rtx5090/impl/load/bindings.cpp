#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {
namespace {

using artifact::NumericFormat;
using artifact::StorageLayout;

StorageLayout layout_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return StorageLayout::ContiguousLeV1;
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S:
    case NumericFormat::W8G32_F16S:
        return StorageLayout::RowSplitK128V1;
    }
    throw std::logic_error("unhandled numeric format");
}

artifact::ObjectHandle tensor(artifact::Binder& binder, std::string name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
    const artifact::ObjectHandle handle =
        binder.require_tensor(name, format, layout_for(format),
                              std::span<const std::uint64_t>(shape.begin(), shape.size()));
    binder.materialize_on_device(handle);
    return handle;
}

artifact::ObjectHandle resource(artifact::Binder& binder, std::string_view name) {
    const artifact::ObjectHandle handle =
        binder.require_resource(name, artifact::ResourceEncoding::RawBytesV1);
    binder.retain_on_host(handle);
    return handle;
}

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

QType qtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return QType::BF16_CTRL;
    case NumericFormat::FP32:
        return QType::FP32_CTRL;
    case NumericFormat::I32:
        return QType::I32_CTRL;
    case NumericFormat::Q4G64_F16S:
        return QType::Q4G64_F16S;
    case NumericFormat::Q5G64_F16S:
        return QType::Q5G64_F16S;
    case NumericFormat::Q6G64_F16S:
        return QType::Q6G64_F16S;
    case NumericFormat::W8G32_F16S:
        return QType::W8G32_F16S;
    }
    throw std::logic_error("unhandled numeric format");
}

DType dtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return DType::BF16;
    case NumericFormat::FP32:
        return DType::FP32;
    case NumericFormat::I32:
        return DType::I32;
    default:
        throw std::logic_error("quantized format has no direct dtype");
    }
}

Tensor direct_tensor(const artifact::MaterializedArtifact& materialized,
                     artifact::ObjectHandle handle, NumericFormat format,
                     std::initializer_list<std::int32_t> internal_shape) {
    return Tensor(materialized.device_data(handle), dtype_for(format), internal_shape);
}

Weight dense_weight(const artifact::MaterializedArtifact& materialized,
                    artifact::ObjectHandle handle, NumericFormat format, std::int32_t rows,
                    std::int32_t columns) {
    Weight out{};
    out.payload       = materialized.device_data(handle);
    out.qdata         = out.payload;
    out.payload_bytes = static_cast<std::uint64_t>(rows) * columns * dtype_size(dtype_for(format));
    out.qtype         = qtype_for(format);
    out.layout        = QuantLayout::Contiguous;
    out.n             = rows;
    out.k             = columns;
    out.ndim          = 2;
    out.shape[0]      = rows;
    out.shape[1]      = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

Weight quant_weight(const artifact::MaterializedArtifact& materialized,
                    artifact::ObjectHandle handle, NumericFormat format, std::int32_t rows,
                    std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape  = {static_cast<std::uint64_t>(rows),
                                                 static_cast<std::uint64_t>(columns)};
    const artifact::RowSplitGeometry geometry = artifact::row_split_geometry(format, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));
    Weight out{};
    out.payload          = bytes;
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = qtype_for(format);
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = bytes;
    out.qhigh       = geometry.high_plane_bytes == 0 ? nullptr : bytes + geometry.high_plane_offset;
    out.scales      = bytes + geometry.scale_plane_offset;
    out.n           = rows;
    out.k           = columns;
    out.group       = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype = DType::FP16;
    out.ndim        = 2;
    out.shape[0]    = rows;
    out.shape[1]    = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}

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

MlpWeights load_mlp(const MlpPlan& plan, const artifact::MaterializedArtifact& materialized,
                    NumericFormat format) {
    MlpWeights out;
    out.gate_up = quant_weight(materialized, plan.gate_up, format, 34816, 5120);
    out.gate    = row_view(out.gate_up, 0, 17408);
    out.up      = row_view(out.gate_up, 17408, 17408);
    out.down    = quant_weight(materialized, plan.down,
                            format == NumericFormat::W8G32_F16S ? NumericFormat::W8G32_F16S
                                                                   : NumericFormat::Q5G64_F16S,
                               5120, 17408);
    return out;
}

std::string take_resource_string(artifact::MaterializedArtifact& materialized,
                                 artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
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

ArtifactLoadPlan bind_artifact(artifact::Binder& binder) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out             = load_plan.bindings;
    out.tokenizer_json           = resource(binder, "frontend/tokenizer.json");
    out.tokenizer_config_json    = resource(binder, "frontend/tokenizer_config.json");
    out.chat_template_jinja      = resource(binder, "frontend/chat_template.jinja");
    out.generation_config_json   = resource(binder, "frontend/generation_config.json");
    out.preprocessor_config_json = resource(binder, "frontend/preprocessor_config.json");
    out.video_preprocessor_config_json =
        resource(binder, "frontend/video_preprocessor_config.json");

    out.token_embedding =
        tensor(binder, "text/token_embedding", NumericFormat::Q6G64_F16S, {248320, 5120});
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm = tensor(binder, prefix + "input_norm", NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.query_key  = tensor(binder, prefix + "attention/query_key",
                                                 NumericFormat::Q4G64_F16S, {7168, 5120});
            target.attention.gate_value = tensor(binder, prefix + "attention/gate_value",
                                                 NumericFormat::Q5G64_F16S, {7168, 5120});
            target.attention.query_norm =
                tensor(binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm =
                tensor(binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = tensor(binder, prefix + "attention/output",
                                             NumericFormat::Q5G64_F16S, {5120, 6144});
        } else {
            target.gdn.a_log   = tensor(binder, prefix + "gdn/a_log", NumericFormat::FP32, {48});
            target.gdn.dt_bias = tensor(binder, prefix + "gdn/dt_bias", NumericFormat::FP32, {48});
            target.gdn.convolution =
                tensor(binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.a_projection =
                tensor(binder, prefix + "gdn/a_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.b_projection =
                tensor(binder, prefix + "gdn/b_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.query_key =
                tensor(binder, prefix + "gdn/query_key", NumericFormat::Q4G64_F16S, {4096, 5120});
            target.gdn.value =
                tensor(binder, prefix + "gdn/value", NumericFormat::Q5G64_F16S, {6144, 5120});
            target.gdn.norm = tensor(binder, prefix + "gdn/norm", NumericFormat::BF16, {128});
            target.gdn.z =
                tensor(binder, prefix + "gdn/z", NumericFormat::Q5G64_F16S, {6144, 5120});
            target.gdn.output =
                tensor(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {5120, 6144});
        }
        target.post_attention_norm =
            tensor(binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            tensor(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {34816, 5120});
        target.mlp.down =
            tensor(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {5120, 17408});
    }
    out.final_norm  = tensor(binder, "text/final_norm", NumericFormat::BF16, {5120});
    out.output_head = tensor(binder, "text/output_head", NumericFormat::Q6G64_F16S, {248320, 5120});
    out.draft_head  = tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S, {131072, 5120});
    out.draft_head_token_ids =
        tensor(binder, "text/draft_head_token_ids", NumericFormat::I32, {131072});
    validate_draft_ids(binder, out.draft_head_token_ids);

    out.mtp.input_projection =
        tensor(binder, "mtp/input_projection", NumericFormat::W8G32_F16S, {5120, 10240});
    out.mtp.embedding_norm = tensor(binder, "mtp/embedding_norm", NumericFormat::BF16, {5120});
    out.mtp.hidden_norm    = tensor(binder, "mtp/hidden_norm", NumericFormat::BF16, {5120});
    out.mtp.input_norm     = tensor(binder, "mtp/layer/input_norm", NumericFormat::BF16, {5120});
    out.mtp.query_key_gate_value = tensor(binder, "mtp/layer/attention/query_key_gate_value",
                                          NumericFormat::W8G32_F16S, {14336, 5120});
    out.mtp.query_norm =
        tensor(binder, "mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm = tensor(binder, "mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output =
        tensor(binder, "mtp/layer/attention/output", NumericFormat::W8G32_F16S, {5120, 6144});
    out.mtp.post_attention_norm =
        tensor(binder, "mtp/layer/post_attention_norm", NumericFormat::BF16, {5120});
    out.mtp.mlp.gate_up =
        tensor(binder, "mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S, {34816, 5120});
    out.mtp.mlp.down =
        tensor(binder, "mtp/layer/mlp/down", NumericFormat::W8G32_F16S, {5120, 17408});
    out.mtp.final_norm = tensor(binder, "mtp/final_norm", NumericFormat::BF16, {5120});

    out.vision_patch_embedding =
        tensor(binder, "vision/patch_embedding", NumericFormat::Q6G64_F16S, {1152, 1536});
    out.vision_patch_embedding_bias =
        tensor(binder, "vision/patch_embedding_bias", NumericFormat::BF16, {1152});
    out.vision_position_embedding =
        tensor(binder, "vision/position_embedding", NumericFormat::BF16, {2304, 1152});
    for (std::size_t layer = 0; layer < kVisionLayers; ++layer) {
        VisionLayerPlan& target  = out.vision_layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv =
            tensor(binder, prefix + "attention/qkv", NumericFormat::Q4G64_F16S, {3456, 1152});
        target.qkv_bias =
            tensor(binder, prefix + "attention/qkv_bias", NumericFormat::BF16, {3456});
        target.output =
            tensor(binder, prefix + "attention/output", NumericFormat::Q5G64_F16S, {1152, 1152});
        target.output_bias =
            tensor(binder, prefix + "attention/output_bias", NumericFormat::BF16, {1152});
        target.fc1 = tensor(binder, prefix + "mlp/fc1", NumericFormat::Q4G64_F16S, {4304, 1152});
        target.fc1_bias = tensor(binder, prefix + "mlp/fc1_bias", NumericFormat::BF16, {4304});
        target.fc2 = tensor(binder, prefix + "mlp/fc2", NumericFormat::Q5G64_F16S, {1152, 4304});
        target.fc2_bias     = tensor(binder, prefix + "mlp/fc2_bias", NumericFormat::BF16, {1152});
        target.norm1_weight = tensor(binder, prefix + "norm1/weight", NumericFormat::BF16, {1152});
        target.norm1_bias   = tensor(binder, prefix + "norm1/bias", NumericFormat::BF16, {1152});
        target.norm2_weight = tensor(binder, prefix + "norm2/weight", NumericFormat::BF16, {1152});
        target.norm2_bias   = tensor(binder, prefix + "norm2/bias", NumericFormat::BF16, {1152});
    }
    out.vision_merger_fc1 =
        tensor(binder, "vision/merger/fc1", NumericFormat::W8G32_F16S, {4608, 4608});
    out.vision_merger_fc1_bias =
        tensor(binder, "vision/merger/fc1_bias", NumericFormat::BF16, {4608});
    out.vision_merger_fc2 =
        tensor(binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {5120, 4608});
    out.vision_merger_fc2_bias =
        tensor(binder, "vision/merger/fc2_bias", NumericFormat::BF16, {5120});
    out.vision_merger_norm_weight =
        tensor(binder, "vision/merger/norm/weight", NumericFormat::BF16, {1152});
    out.vision_merger_norm_bias =
        tensor(binder, "vision/merger/norm/bias", NumericFormat::BF16, {1152});

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend.tokenizer_json         = take_resource_string(backing, plan.tokenizer_json);
    frontend.tokenizer_config_json  = take_resource_string(backing, plan.tokenizer_config_json);
    frontend.chat_template_jinja    = take_resource_string(backing, plan.chat_template_jinja);
    frontend.generation_config_json = take_resource_string(backing, plan.generation_config_json);
    frontend.preprocessor_config_json =
        take_resource_string(backing, plan.preprocessor_config_json);
    frontend.video_preprocessor_config_json =
        take_resource_string(backing, plan.video_preprocessor_config_json);

    token_embedding =
        quant_weight(backing, plan.token_embedding, NumericFormat::Q6G64_F16S, 248320, 5120);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm =
                direct_tensor(backing, source.input_norm, NumericFormat::BF16, {5120});
            target.query_key   = quant_weight(backing, source.attention.query_key,
                                              NumericFormat::Q4G64_F16S, 7168, 5120);
            target.query       = row_view(target.query_key, 0, 6144);
            target.key         = row_view(target.query_key, 6144, 1024);
            target.gate_value  = quant_weight(backing, source.attention.gate_value,
                                              NumericFormat::Q5G64_F16S, 7168, 5120);
            target.output_gate = row_view(target.gate_value, 0, 6144);
            target.value       = row_view(target.gate_value, 6144, 1024);
            target.query_norm =
                direct_tensor(backing, source.attention.query_norm, NumericFormat::BF16, {256});
            target.key_norm =
                direct_tensor(backing, source.attention.key_norm, NumericFormat::BF16, {256});
            target.output = quant_weight(backing, source.attention.output,
                                         NumericFormat::Q5G64_F16S, 5120, 6144);
            target.post_attention_norm =
                direct_tensor(backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.mlp = load_mlp(source.mlp, backing, NumericFormat::Q4G64_F16S);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm =
                direct_tensor(backing, source.input_norm, NumericFormat::BF16, {5120});
            target.a_log   = direct_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {48});
            target.dt_bias = direct_tensor(backing, source.gdn.dt_bias, NumericFormat::FP32, {48});
            target.convolution_storage =
                direct_tensor(backing, source.gdn.convolution, NumericFormat::BF16, {10240, 4});
            target.convolution = target.convolution_storage;
            target.a_projection =
                dense_weight(backing, source.gdn.a_projection, NumericFormat::BF16, 48, 5120);
            target.b_projection =
                dense_weight(backing, source.gdn.b_projection, NumericFormat::BF16, 48, 5120);
            target.query_key =
                quant_weight(backing, source.gdn.query_key, NumericFormat::Q4G64_F16S, 4096, 5120);
            target.query = row_view(target.query_key, 0, 2048);
            target.key   = row_view(target.query_key, 2048, 2048);
            target.value =
                quant_weight(backing, source.gdn.value, NumericFormat::Q5G64_F16S, 6144, 5120);
            target.norm = direct_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.z = quant_weight(backing, source.gdn.z, NumericFormat::Q5G64_F16S, 6144, 5120);
            target.output =
                quant_weight(backing, source.gdn.output, NumericFormat::Q5G64_F16S, 5120, 6144);
            target.post_attention_norm =
                direct_tensor(backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.mlp = load_mlp(source.mlp, backing, NumericFormat::Q4G64_F16S);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm  = direct_tensor(backing, plan.final_norm, NumericFormat::BF16, {5120});
    output_head = quant_weight(backing, plan.output_head, NumericFormat::Q6G64_F16S, 248320, 5120);
    draft_head  = quant_weight(backing, plan.draft_head, NumericFormat::Q4G64_F16S, 131072, 5120);
    draft_head_token_ids =
        direct_tensor(backing, plan.draft_head_token_ids, NumericFormat::I32, {131072});

    mtp.input_projection =
        quant_weight(backing, plan.mtp.input_projection, NumericFormat::W8G32_F16S, 5120, 10240);
    mtp.embedding_norm =
        direct_tensor(backing, plan.mtp.embedding_norm, NumericFormat::BF16, {5120});
    mtp.hidden_norm = direct_tensor(backing, plan.mtp.hidden_norm, NumericFormat::BF16, {5120});
    mtp.input_norm  = direct_tensor(backing, plan.mtp.input_norm, NumericFormat::BF16, {5120});
    mtp.query_key_gate_value = quant_weight(backing, plan.mtp.query_key_gate_value,
                                            NumericFormat::W8G32_F16S, 14336, 5120);
    mtp.query                = row_view(mtp.query_key_gate_value, 0, 6144);
    mtp.key                  = row_view(mtp.query_key_gate_value, 6144, 1024);
    mtp.output_gate          = row_view(mtp.query_key_gate_value, 7168, 6144);
    mtp.value                = row_view(mtp.query_key_gate_value, 13312, 1024);
    mtp.query_norm = direct_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {256});
    mtp.key_norm   = direct_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {256});
    mtp.output     = quant_weight(backing, plan.mtp.output, NumericFormat::W8G32_F16S, 5120, 6144);
    mtp.post_attention_norm =
        direct_tensor(backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {5120});
    mtp.mlp        = load_mlp(plan.mtp.mlp, backing, NumericFormat::W8G32_F16S);
    mtp.final_norm = direct_tensor(backing, plan.mtp.final_norm, NumericFormat::BF16, {5120});

    vision.patch_embedding =
        quant_weight(backing, plan.vision_patch_embedding, NumericFormat::Q6G64_F16S, 1152, 1536);
    vision.patch_embedding_bias =
        direct_tensor(backing, plan.vision_patch_embedding_bias, NumericFormat::BF16, {1152});
    vision.position_embedding =
        direct_tensor(backing, plan.vision_position_embedding, NumericFormat::BF16, {1152, 2304});
    for (std::size_t layer = 0; layer < kVisionLayers; ++layer) {
        const VisionLayerPlan& source = plan.vision_layers[layer];
        VisionLayerWeights& target    = vision.layers[layer];
        target.qkv      = quant_weight(backing, source.qkv, NumericFormat::Q4G64_F16S, 3456, 1152);
        target.qkv_bias = direct_tensor(backing, source.qkv_bias, NumericFormat::BF16, {3456});
        target.output = quant_weight(backing, source.output, NumericFormat::Q5G64_F16S, 1152, 1152);
        target.output_bias =
            direct_tensor(backing, source.output_bias, NumericFormat::BF16, {1152});
        target.fc1      = quant_weight(backing, source.fc1, NumericFormat::Q4G64_F16S, 4304, 1152);
        target.fc1_bias = direct_tensor(backing, source.fc1_bias, NumericFormat::BF16, {4304});
        target.fc2      = quant_weight(backing, source.fc2, NumericFormat::Q5G64_F16S, 1152, 4304);
        target.fc2_bias = direct_tensor(backing, source.fc2_bias, NumericFormat::BF16, {1152});
        target.norm1_weight =
            direct_tensor(backing, source.norm1_weight, NumericFormat::BF16, {1152});
        target.norm1_bias = direct_tensor(backing, source.norm1_bias, NumericFormat::BF16, {1152});
        target.norm2_weight =
            direct_tensor(backing, source.norm2_weight, NumericFormat::BF16, {1152});
        target.norm2_bias = direct_tensor(backing, source.norm2_bias, NumericFormat::BF16, {1152});
    }
    vision.merger_fc1 =
        quant_weight(backing, plan.vision_merger_fc1, NumericFormat::W8G32_F16S, 4608, 4608);
    vision.merger_fc1_bias =
        direct_tensor(backing, plan.vision_merger_fc1_bias, NumericFormat::BF16, {4608});
    vision.merger_fc2 =
        quant_weight(backing, plan.vision_merger_fc2, NumericFormat::W8G32_F16S, 5120, 4608);
    vision.merger_fc2_bias =
        direct_tensor(backing, plan.vision_merger_fc2_bias, NumericFormat::BF16, {5120});
    vision.merger_norm_weight =
        direct_tensor(backing, plan.vision_merger_norm_weight, NumericFormat::BF16, {1152});
    vision.merger_norm_bias =
        direct_tensor(backing, plan.vision_merger_norm_bias, NumericFormat::BF16, {1152});
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
