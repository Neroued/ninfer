#include "targets/qwen3_6_27b_rtx5090/impl/schedule/vision_context.h"

#include "core/device.h"
#include "core/layout.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "kernels/add_bias/add_bias.h"
#include "kernels/gelu/gelu.h"
#include "kernels/layer_norm/layer_norm.h"
#include "kernels/linear/linear.h"
#include "kernels/residual_add/residual_add.h"
#include "kernels/rope/rope.h"
#include "kernels/vision_attention/vision_attention.h"
#include "kernels/vision_pos_embed/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule {
namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct VisionWorkspaceLayout {
    TensorRegion output;
    TensorRegion position_ids;
    TensorRegion cu_seqlens;
    TensorRegion pos_indices;
    TensorRegion pos_weights;
    TensorRegion x;
    TensorRegion patch_bf16;
    TensorRegion patch_f32;
    TensorRegion attended;
    TensorRegion qkv;
    TensorRegion attention_norm;
    std::optional<TensorRegion> attention_tiles;
    TensorRegion projected;
    TensorRegion mlp_down;
    TensorRegion mlp_up;
    TensorRegion mlp_norm;
    TensorRegion normalized;
    TensorRegion merger_hidden;
    std::size_t bytes = 0;
};

VisionWorkspaceLayout build_workspace_layout(const ProcessedInput& input,
                                             const detail::VisionControl& control) {
    const auto patches64 = static_cast<std::size_t>(input.stats.raw_patches);
    const auto tokens64  = static_cast<std::size_t>(input.stats.vision_tokens);
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 != checked_mul(tokens64, VisionScheduleConfig::merge_unit,
                                 "patch/token relation")) {
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        control.cu_seqlens.size() >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }
    const auto patches = static_cast<std::int32_t>(patches64);
    const auto tokens  = static_cast<std::int32_t>(tokens64);

    LayoutBuilder builder;
    VisionWorkspaceLayout out;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                         const char* label) {
        return builder.add_tensor(dtype, shape, kWorkspaceAlignment, label);
    };
    out.output       = add(DType::BF16, {VisionScheduleConfig::out_hidden, tokens}, "vision output");
    out.position_ids = add(DType::I32, {patches, 2}, "vision position ids");
    out.cu_seqlens   = add(DType::I32,
                         {static_cast<std::int32_t>(control.cu_seqlens.size())},
                         "vision segment bounds");
    out.pos_indices  = add(DType::I32, {4, patches}, "vision position indices");
    out.pos_weights  = add(DType::FP32, {4, patches}, "vision position weights");
    out.x            = add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision residual");
    {
        auto scope    = builder.scope();
        out.patch_bf16 = add(DType::BF16, {VisionScheduleConfig::patch_dim, patches},
                             "vision BF16 patches");
        out.patch_f32 = add(DType::FP32, {VisionScheduleConfig::patch_dim, patches},
                            "vision FP32 patches");
    }
    {
        auto attention_scope = builder.scope();
        out.attended = add(DType::BF16, {VisionScheduleConfig::hidden, patches},
                           "vision attended");
        {
            auto qkv_scope = builder.scope();
            out.qkv = add(DType::BF16, {3 * VisionScheduleConfig::hidden, patches}, "vision QKV");
            {
                auto norm_scope = builder.scope();
                out.attention_norm = add(DType::BF16, {VisionScheduleConfig::hidden, patches},
                                         "vision attention norm");
            }
            const std::int32_t tile_count = kernels::vision_attention_scratch_tiles(
                patches, static_cast<std::int32_t>(control.cu_seqlens.size()) - 1);
            if (tile_count != 0) {
                out.attention_tiles = add(DType::I32, {4, tile_count}, "vision attention tiles");
            }
        }
        out.projected = add(DType::BF16, {VisionScheduleConfig::hidden, patches},
                            "vision projected");
    }
    {
        auto mlp_scope = builder.scope();
        out.mlp_down = add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision MLP down");
        out.mlp_up = add(DType::BF16, {VisionScheduleConfig::intermediate, patches}, "vision MLP up");
        {
            auto norm_scope = builder.scope();
            out.mlp_norm = add(DType::BF16, {VisionScheduleConfig::hidden, patches},
                               "vision MLP norm");
        }
    }
    out.normalized = add(DType::BF16, {VisionScheduleConfig::hidden, patches},
                         "vision merger norm");
    out.merger_hidden = add(DType::BF16, {VisionScheduleConfig::merger_hidden, tokens},
                            "vision merger hidden");
    out.bytes = builder.finish(kWorkspaceAlignment, "vision workspace");
    return out;
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

} // namespace

VisionContext::VisionContext(DeviceContext& ctx,
                             const targets::qwen3_6_27b_rtx5090::detail::LoadedModelData& weights)
    : ctx_(ctx) {
    patch_embed_      = &weights.vision.patch_embedding;
    patch_embed_bias_ = &weights.vision.patch_embedding_bias;
    position_embed_   = &weights.vision.position_embedding;
    for (std::uint32_t layer = 0; layer < blocks_.size(); ++layer) {
        const auto& source  = weights.vision.layers[layer];
        BlockW& out         = blocks_[layer];
        out.norm1_weight    = &source.norm1_weight;
        out.norm1_bias      = &source.norm1_bias;
        out.qkv             = &source.qkv;
        out.qkv_bias        = &source.qkv_bias;
        out.projection      = &source.output;
        out.projection_bias = &source.output_bias;
        out.norm2_weight    = &source.norm2_weight;
        out.norm2_bias      = &source.norm2_bias;
        out.fc1             = &source.fc1;
        out.fc1_bias        = &source.fc1_bias;
        out.fc2             = &source.fc2;
        out.fc2_bias        = &source.fc2_bias;
    }
    merger_.norm_weight = &weights.vision.merger_norm_weight;
    merger_.norm_bias   = &weights.vision.merger_norm_bias;
    merger_.fc1         = &weights.vision.merger_fc1;
    merger_.fc1_bias    = &weights.vision.merger_fc1_bias;
    merger_.fc2         = &weights.vision.merger_fc2;
    merger_.fc2_bias    = &weights.vision.merger_fc2_bias;
}

std::size_t VisionContext::workspace_bytes(const ProcessedInput& input) {
    const detail::VisionControl control = detail::build_vision_control(input);
    return build_workspace_layout(input, control).bytes;
}

Tensor VisionContext::encode(const ProcessedInput& input, WorkspaceArena& workspace, void* tap,
                             VisionTapCallback callback) const {
    if ((tap == nullptr) != (callback == nullptr)) {
        throw std::invalid_argument("Vision tap context and callback must be provided together");
    }
    const detail::VisionControl control = detail::build_vision_control(input);
    const auto patches64                = static_cast<std::size_t>(input.stats.raw_patches);
    const auto tokens64                 = static_cast<std::size_t>(input.stats.vision_tokens);
    if (input.patches.size() !=
        checked_mul(patches64, VisionScheduleConfig::patch_dim, "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(input, control);
    if (workspace.capacity() < layout.bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = ctx_.stream;
    workspace.reset();
    const DeviceSpan backing = workspace.alloc_bytes(layout.bytes, kWorkspaceAlignment);

    Tensor output       = layout.output.bind(backing);
    Tensor position_ids = layout.position_ids.bind(backing);
    Tensor cu_seqlens   = layout.cu_seqlens.bind(backing);
    Tensor pos_indices  = layout.pos_indices.bind(backing);
    Tensor pos_weights  = layout.pos_weights.bind(backing);
    copy_host(control.position_ids.data(), position_ids, stream);
    copy_host(control.cu_seqlens.data(), cu_seqlens, stream);
    copy_host(control.pos_indices.data(), pos_indices, stream);
    copy_host(control.pos_weights.data(), pos_weights, stream);

    Tensor x          = layout.x.bind(backing);
    Tensor patch_bf16 = layout.patch_bf16.bind(backing);
    Tensor patch_f32  = layout.patch_f32.bind(backing);
    copy_host(input.patches.data(), patch_f32, stream);
    detail::vision_f32_to_bf16(patch_f32, patch_bf16, stream);
    kernels::linear(patch_bf16, *patch_embed_, x, workspace, stream);
    kernels::add_bias(*patch_embed_bias_, x, stream);
    // The artifact records the source table shape [rows,hidden], while Tensor's
    // contiguous matrix convention is [inner,columns]. The payload is already
    // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
    Tensor position_table = position_embed_->reshape(
        {VisionScheduleConfig::hidden, VisionScheduleConfig::position_embeddings});
    kernels::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
    if (callback != nullptr) { callback(tap, VisionTapId::PatchEmbed, -1, x, stream); }

    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        const BlockW& block = blocks_[layer];
        {
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    kernels::layer_norm(x, *block.norm1_weight, *block.norm1_bias,
                                        VisionScheduleConfig::norm_eps, h, stream);
                    kernels::linear(h, *block.qkv, qkv, workspace, stream);
                }
                kernels::add_bias(*block.qkv_bias, qkv, stream);
                const std::int32_t plane      = VisionScheduleConfig::hidden;
                const std::size_t plane_bytes = static_cast<std::size_t>(plane) * 2;
                Tensor q(qkv.data, DType::BF16,
                         {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16,
                         {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16,
                         {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                q.nb[2] = qkv.nb[1];
                k.nb[2] = qkv.nb[1];
                v.nb[2] = qkv.nb[1];
                kernels::rope(position_ids, VisionScheduleConfig::rotary_dim,
                              VisionScheduleConfig::rope_theta, q, k, stream);
                Tensor attended_heads = attended.view(
                    {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                Tensor attention_tiles;
                Tensor* attention_tiles_ptr = nullptr;
                if (layout.attention_tiles) {
                    attention_tiles     = layout.attention_tiles->bind(backing);
                    attention_tiles_ptr = &attention_tiles;
                }
                kernels::vision_attention(q, k, v, cu_seqlens, attention_tiles_ptr,
                                          attended_heads, stream);
            }
            Tensor projected = layout.projected.bind(backing);
            kernels::linear(attended, *block.projection, projected, workspace, stream);
            kernels::add_bias(*block.projection_bias, projected, stream);
            kernels::residual_add(projected, x, stream);
        }
        {
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                kernels::layer_norm(x, *block.norm2_weight, *block.norm2_bias,
                                    VisionScheduleConfig::norm_eps, h, stream);
                kernels::linear(h, *block.fc1, up, workspace, stream);
            }
            kernels::add_bias(*block.fc1_bias, up, stream);
            kernels::gelu(up, kernels::GeluMode::Tanh, stream);
            kernels::linear(up, *block.fc2, down, workspace, stream);
            kernels::add_bias(*block.fc2_bias, down, stream);
            kernels::residual_add(down, x, stream);
        }
        if (callback != nullptr) {
            callback(tap, VisionTapId::Block, static_cast<int>(layer), x, stream);
        }
    }

    Tensor normalized = layout.normalized.bind(backing);
    kernels::layer_norm(x, *merger_.norm_weight, *merger_.norm_bias, VisionScheduleConfig::norm_eps,
                        normalized, stream);
    Tensor merged = normalized.view({VisionScheduleConfig::merger_hidden, tokens});
    Tensor hidden = layout.merger_hidden.bind(backing);
    kernels::linear(merged, *merger_.fc1, hidden, workspace, stream);
    kernels::add_bias(*merger_.fc1_bias, hidden, stream);
    kernels::gelu(hidden, kernels::GeluMode::Exact, stream);
    kernels::linear(hidden, *merger_.fc2, output, workspace, stream);
    kernels::add_bias(*merger_.fc2_bias, output, stream);
    if (callback != nullptr) { callback(tap, VisionTapId::Merger, -1, output, stream); }
    return output;
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule
