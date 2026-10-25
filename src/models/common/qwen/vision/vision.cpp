#include "models/common/qwen/vision/vision.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/nvtx.h"
#include "core/transfer.h"
#include "models/common/qwen/vision/control.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen {
namespace {

std::int32_t dimension(std::uint64_t n) {
    if (!n || n > INT32_MAX) throw std::invalid_argument("Vision dimension exceeds int32");
    return static_cast<std::int32_t>(n);
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct VisionWorkspaceLayout {
    TensorRegion position_ids, cu_seqlens;
    LayoutRegion attention_scratch;
    TensorRegion pos_indices;
    TensorRegion pos_weights;
    TensorRegion x;
    TensorRegion patch_bf16;
    TensorRegion attended;
    TensorRegion qkv;
    TensorRegion attention_norm;
    TensorRegion projected;
    TensorRegion mlp_down;
    TensorRegion mlp_up;
    TensorRegion mlp_norm;
    TensorRegion normalized;
    TensorRegion merger_hidden;
    LayoutRegion patch_scratch, qkv_scratch, projection_scratch;
    LayoutRegion up_scratch, down_scratch, merger_first_scratch, merger_second_scratch;
    std::size_t bytes = 0;
};

TensorRegion alias_tensor(const TensorRegion& storage, DType dtype,
                          std::initializer_list<std::int32_t> shape, const char* label) {
    auto tensor = Tensor::metadata(dtype, shape);
    if (tensor.bytes() > storage.region.bytes) {
        throw std::logic_error(std::string("Vision ") + label +
                               " does not fit its aliased storage");
    }
    TensorRegion out;
    out.region = LayoutRegion{storage.region.offset, tensor.bytes(), storage.region.alignment};
    out.dtype  = dtype;
    std::copy(shape.begin(), shape.end(), out.shape.begin());
    return out;
}

template <class Profile>
VisionWorkspaceLayout build_workspace_layout(const VisionConfig& config, const Profile& parameters,
                                             std::size_t patches64, std::size_t tokens64) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 != checked_mul(tokens64,
                                 dimension(std::uint64_t(config.spatial_merge_size) *
                                           config.spatial_merge_size),
                                 "patch/token relation")) {
        throw std::invalid_argument(
            "Vision workspace requires the configured positive patch/token ratio");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }
    const auto patches = static_cast<std::int32_t>(patches64);
    const auto tokens  = static_cast<std::int32_t>(tokens64);

    const auto capacity = [](const auto& weight, std::int32_t first, std::int32_t last) {
        const auto p = ops::linear_profile(weight);
        return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last);
    };
    const auto minimum_patches =
        dimension(std::uint64_t(config.spatial_merge_size) * config.spatial_merge_size);
    std::size_t qkv_bytes = 0, projection_bytes = 0, up_bytes = 0, down_bytes = 0;
    for (const auto& layer : parameters.layers) {
        qkv_bytes = std::max(qkv_bytes, capacity(layer.qkv, minimum_patches, patches));
        projection_bytes =
            std::max(projection_bytes, capacity(layer.output, minimum_patches, patches));
        up_bytes   = std::max(up_bytes, capacity(layer.fc1, minimum_patches, patches));
        down_bytes = std::max(down_bytes, capacity(layer.fc2, minimum_patches, patches));
    }
    LayoutBuilder builder;
    const auto scratch = [&](std::size_t bytes, const char* label) {
        auto scope = builder.scope();
        // A borrowed MemoryArena needs backing even when this Op uses no scratch.
        return builder.add(std::max(std::size_t{1}, bytes), kWorkspaceAlignment, label);
    };
    VisionWorkspaceLayout out;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                         const char* label) {
        return builder.add_tensor(dtype, shape, kWorkspaceAlignment, label);
    };
    out.x = add(DType::BF16, {dimension(config.hidden_size), patches}, "vision residual");
    {
        auto position_lifetime = builder.scope();
        out.cu_seqlens         = add(DType::I32, {tokens + 1}, "vision segment offsets");
        out.position_ids       = add(DType::I32, {patches, 2}, "vision position ids");
        {
            auto patch_scope = builder.scope();
            out.patch_bf16 =
                add(DType::BF16, {dimension(config.patch_width()), patches}, "vision BF16 patches");
            out.patch_scratch =
                scratch(capacity(parameters.patch_embedding, minimum_patches, patches),
                        "patch projection scratch");
        }
        {
            auto position_scope = builder.scope();
            out.pos_indices     = add(DType::I32, {4, patches}, "vision position indices");
            out.pos_weights     = add(DType::FP32, {4, patches}, "vision position weights");
        }
        {
            auto attention_scope = builder.scope();
            out.qkv = add(DType::BF16, {3 * dimension(config.hidden_size), patches}, "vision QKV");
            out.attention_norm     = add(DType::BF16, {dimension(config.hidden_size), patches},
                                         "vision attention norm/attended");
            out.attended           = out.attention_norm;
            out.qkv_scratch        = scratch(qkv_bytes, "QKV scratch");
            out.projection_scratch = scratch(projection_bytes, "attention output scratch");
            out.projected =
                alias_tensor(out.qkv, DType::BF16, {dimension(config.hidden_size), patches},
                             "attention projection output");
            out.attention_scratch =
                scratch(ops::packed_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.hidden_size / config.num_heads),
                             dimension(config.num_heads), dimension(config.num_heads)},
                            1, patches, 1, tokens),
                        "attention descriptors");
        }
        {
            auto mlp_scope = builder.scope();
            out.mlp_up =
                add(DType::BF16, {dimension(config.intermediate_size), patches}, "vision MLP up");
            out.mlp_norm =
                add(DType::BF16, {dimension(config.hidden_size), patches}, "vision MLP norm/down");
            out.mlp_down     = out.mlp_norm;
            out.up_scratch   = scratch(up_bytes, "MLP up scratch");
            out.down_scratch = scratch(down_bytes, "MLP down scratch");
        }
    }
    {
        auto merger_scope = builder.scope();
        out.normalized =
            add(DType::BF16, {dimension(config.hidden_size), patches}, "vision merger norm");
        out.merger_first_scratch =
            scratch(capacity(parameters.merger_fc1, 1, tokens), "merger first scratch");
        out.merger_hidden = alias_tensor(
            out.x, DType::BF16, {dimension(config.merger_width()), tokens}, "merger hidden");
    }
    out.merger_second_scratch =
        scratch(capacity(parameters.merger_fc2, 1, tokens), "merger second scratch");
    out.bytes = builder.finish(kWorkspaceAlignment, "vision workspace");
    return out;
}

vision::Capacity plan_resources(const VisionConfig& config, const VisionProfile& parameters,
                                std::uint32_t maximum, std::size_t budget) {
    const auto ratio = std::uint64_t(config.spatial_merge_size) * config.spatial_merge_size;
    return vision::plan_capacity(
        budget, maximum, ops::linear_profile(parameters.merger_fc2).n, [&](std::uint32_t features) {
            const auto n      = checked_mul(features, ratio, "patch capacity");
            const auto layout = build_workspace_layout(config, parameters, n, features);
            LayoutBuilder host;
            (void)host.add_tensor(DType::BF16, {dimension(config.patch_width()), dimension(n)}, 256,
                                  "patch upload");
            (void)host.add_tensor(DType::I32, {dimension(n), 2}, 256, "positions");
            (void)host.add_tensor(DType::I32, {4, dimension(n)}, 256, "position indices");
            (void)host.add_tensor(DType::FP32, {4, dimension(n)}, 256, "position weights");
            (void)host.add_tensor(DType::I32, {dimension(features + 1ULL)}, 256, "segment offsets");
            return vision::EncoderMemory{layout.bytes, host.finish(256)};
        });
}

} // namespace

VisionContext::VisionContext(CudaExecutionView execution, const VisionConfig& config,
                             const VisionParameters& parameters)
    : execution_(execution), config_(config), parameters_(parameters) {}

vision::Capacity VisionContext::plan_resources(const VisionConfig& c, const VisionProfile& p,
                                               std::uint32_t maximum, std::size_t budget) {
    return qwen::plan_resources(c, p, maximum, budget);
}

std::size_t VisionContext::encode(const VisionEncoderBatch& input, Tensor& output,
                                  MemorySpan backing, MemorySpan staging) const {
    if (!input.features || input.cu_seqlens.size() < 2 || input.cu_seqlens.front() != 0)
        throw std::invalid_argument("Vision encoder batch has no segments");
    const auto patches = input.cu_seqlens.back();
    const auto tokens  = static_cast<std::int32_t>(input.features);
    if (patches <= 0 || input.positions.size() != 2ULL * patches ||
        input.position_indices.size() != 4ULL * patches ||
        input.position_weights.size() != 4ULL * patches)
        throw std::invalid_argument("Vision encoder positions do not cover its patches");
    for (std::size_t i = 1; i < input.cu_seqlens.size(); ++i)
        if (input.cu_seqlens[i] <= input.cu_seqlens[i - 1])
            throw std::invalid_argument("Vision encoder segments are not strictly ordered");
    if (output.dtype != DType::BF16 || output.ne[0] != parameters_.merger_fc2.weight.n ||
        output.ne[1] != tokens || output.ne[2] != 1 || output.ne[3] != 1 ||
        !output.is_contiguous() || !output.data || output.device != execution_.device())
        throw std::invalid_argument("Vision output must be contiguous device BF16 [H,F]");
    const auto segment_length = input.cu_seqlens[1];
    bool equal_segments       = true;
    for (std::size_t i = 2; i < input.cu_seqlens.size(); ++i)
        equal_segments &= input.cu_seqlens[i] - input.cu_seqlens[i - 1] == segment_length;
    const auto output_begin  = reinterpret_cast<std::uintptr_t>(output.data);
    const auto backing_begin = reinterpret_cast<std::uintptr_t>(backing.data);
    if (output_begin < backing_begin + backing.bytes &&
        backing_begin < output_begin + output.bytes())
        throw std::invalid_argument("Vision output overlaps encoder scratch");
    const auto layout = build_workspace_layout(config_, parameters_, patches, tokens);
    if (backing.bytes < layout.bytes || backing.device != execution_.device() ||
        staging.device != Device::cpu() || !staging.host_pinned)
        throw std::invalid_argument("Vision encoder backing differs from its planned resources");
    MemoryArena host(staging);
    auto upload = [&](const void* source, Tensor& destination) {
        auto buffer = host.alloc_bytes(destination.bytes());
        std::memcpy(buffer.data, source, destination.bytes());
        copy_bytes_async({destination.data, destination.bytes(), destination.device}, buffer,
                         execution_);
    };
    nvtx::ScopedRange encode_range(nvtx::Name::VisionEncode, nvtx::Category::Vision, patches);
    const auto project = [&](const Tensor& x, const ops::SingleProjectionWeight& p, Tensor& out,
                             const LayoutRegion& region) {
        MemoryArena scratch(region.bind(backing));
        ops::linear(x, p.weight, out, p.policy, scratch, execution_);
    };
    Tensor position_ids = layout.position_ids.bind(backing);
    Tensor x            = layout.x.bind(backing);
    Tensor patch_bf16   = layout.patch_bf16.bind(backing);
    Tensor pos_indices  = layout.pos_indices.bind(backing);
    Tensor pos_weights  = layout.pos_weights.bind(backing);
    {
        nvtx::ScopedRange patch_range(nvtx::Name::VisionPatchEmbedding, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(patches));
        auto patch_host    = host.alloc_bytes(patch_bf16.bytes());
        std::size_t copied = 0;
        for (auto part : input.patches) {
            if (part.size_bytes() > patch_host.bytes - copied)
                throw std::invalid_argument("Vision patch parts exceed the encoder extent");
            std::memmove(static_cast<std::byte*>(patch_host.data) + copied, part.data(),
                         part.size_bytes());
            copied += part.size_bytes();
        }
        if (copied != patch_host.bytes)
            throw std::invalid_argument("Vision patch parts do not fill the encoder extent");
        copy_bytes_async({patch_bf16.data, patch_bf16.bytes(), patch_bf16.device}, patch_host,
                         execution_);
        upload(input.positions.data(), position_ids);
        project(patch_bf16, parameters_.patch_embedding, x, layout.patch_scratch);
        ops::add_bias(parameters_.patch_embedding_bias, x, execution_);
        // The artifact records the source table shape [rows,hidden], while Tensor's
        // contiguous matrix convention is [inner,columns]. The payload is already
        // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
        upload(input.position_indices.data(), pos_indices);
        upload(input.position_weights.data(), pos_weights);
        auto segments = layout.cu_seqlens.bind(backing).slice(0, 0, input.cu_seqlens.size());
        upload(input.cu_seqlens.data(), segments);
        Tensor position_table = parameters_.position_embedding.reshape(
            {dimension(config_.hidden_size), dimension(config_.num_position_embeddings)});
        ops::embedding_interpolate_add(position_table, pos_indices, pos_weights, x, execution_);
    }
    for (std::size_t layer = 0; layer < parameters_.layers.size(); ++layer) {
        nvtx::ScopedRange layer_range(nvtx::Name::VisionLayer, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(layer));
        const auto& block = parameters_.layers[layer];
        {
            nvtx::ScopedRange attention_range(nvtx::Name::VisionAttention,
                                              nvtx::Category::Attention,
                                              static_cast<std::uint64_t>(layer));
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    ops::layer_norm(x, block.norm1.weight, block.norm1.bias, 1.0e-6F, h,
                                    execution_);
                    project(h, block.qkv, qkv, layout.qkv_scratch);
                }
                ops::add_bias(block.qkv_bias, qkv, execution_);
                const auto packed = qkv.view({dimension(config_.hidden_size / config_.num_heads),
                                              dimension(config_.num_heads), 3, patches});
                Tensor q          = packed.slice(2, 0, 1).permute({0, 1, 3, 2});
                Tensor k          = packed.slice(2, 1, 1).permute({0, 1, 3, 2});
                Tensor v          = packed.slice(2, 2, 1).permute({0, 1, 3, 2});
                ops::rope(position_ids, dimension(config_.hidden_size / config_.num_heads),
                          10'000.0F, q, k, execution_);
                Tensor attended_heads =
                    attended.view({dimension(config_.hidden_size / config_.num_heads),
                                   dimension(config_.num_heads), patches});
                const ops::AttentionHeadGeometry geometry{
                    dimension(config_.hidden_size / config_.num_heads),
                    dimension(config_.num_heads), dimension(config_.num_heads)};
                const auto scale = 1.0F / std::sqrt(float(geometry.head_dim));
                if (equal_segments)
                    ops::packed_softmax_attention(q, k, v, geometry, scale, segment_length,
                                                  attended_heads, execution_);
                else {
                    MemoryArena attention_scratch(layout.attention_scratch.bind(backing));
                    ops::packed_softmax_attention(
                        q, k, v, geometry, scale,
                        layout.cu_seqlens.bind(backing).slice(0, 0, input.cu_seqlens.size()),
                        attention_scratch, attended_heads, execution_);
                }
            }
            Tensor projected = layout.projected.bind(backing);
            project(attended, block.output, projected, layout.projection_scratch);
            ops::add_bias(block.output_bias, projected, execution_);
            ops::residual_add(projected, x, execution_);
        }
        {
            nvtx::ScopedRange mlp_range(nvtx::Name::VisionMlp, nvtx::Category::PostMixer,
                                        static_cast<std::uint64_t>(layer));
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                ops::layer_norm(x, block.norm2.weight, block.norm2.bias, 1.0e-6F, h, execution_);
                project(h, block.fc1, up, layout.up_scratch);
            }
            ops::add_bias(block.fc1_bias, up, execution_);
            ops::gelu(up, ops::GeluMode::Tanh, execution_);
            project(up, block.fc2, down, layout.down_scratch);
            ops::add_bias(block.fc2_bias, down, execution_);
            ops::residual_add(down, x, execution_);
        }
    }

    {
        nvtx::ScopedRange merge_range(nvtx::Name::VisionMerge, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(tokens));
        Tensor normalized = layout.normalized.bind(backing);
        ops::layer_norm(x, parameters_.merger_norm.weight, parameters_.merger_norm.bias, 1.0e-6F,
                        normalized, execution_);
        Tensor merged = normalized.view({dimension(config_.merger_width()), tokens});
        Tensor hidden = layout.merger_hidden.bind(backing);
        project(merged, parameters_.merger_fc1, hidden, layout.merger_first_scratch);
        ops::add_bias(parameters_.merger_fc1_bias, hidden, execution_);
        ops::gelu(hidden, ops::GeluMode::Exact, execution_);
        project(hidden, parameters_.merger_fc2, output, layout.merger_second_scratch);
        ops::add_bias(parameters_.merger_fc2_bias, output, execution_);
    }
    return layout.bytes;
}

} // namespace ninfer::models::qwen
