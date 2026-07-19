#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "core/device.h"
#include "core/layout.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/vision_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct VisionWorkspaceLayout {
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

VisionWorkspaceLayout build_workspace_layout(std::size_t patches64, std::size_t tokens64,
                                             std::size_t segment_count) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 !=
            checked_mul(tokens64, VisionScheduleConfig::merge_unit, "patch/token relation")) {
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        segment_count == 0 ||
        segment_count >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
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
    out.position_ids = add(DType::I32, {patches, 2}, "vision position ids");
    out.cu_seqlens =
        add(DType::I32, {static_cast<std::int32_t>(segment_count + 1)}, "vision segment bounds");
    out.pos_indices = add(DType::I32, {4, patches}, "vision position indices");
    out.pos_weights = add(DType::FP32, {4, patches}, "vision position weights");
    out.x           = add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision residual");
    {
        auto scope = builder.scope();
        out.patch_bf16 =
            add(DType::BF16, {VisionScheduleConfig::patch_dim, patches}, "vision BF16 patches");
        out.patch_f32 =
            add(DType::FP32, {VisionScheduleConfig::patch_dim, patches}, "vision FP32 patches");
    }
    {
        auto attention_scope = builder.scope();
        out.attended = add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision attended");
        {
            auto qkv_scope = builder.scope();
            out.qkv = add(DType::BF16, {3 * VisionScheduleConfig::hidden, patches}, "vision QKV");
            {
                auto norm_scope    = builder.scope();
                out.attention_norm = add(DType::BF16, {VisionScheduleConfig::hidden, patches},
                                         "vision attention norm");
            }
            const std::int32_t tile_count = ops::vision_attention_scratch_tiles(
                patches, static_cast<std::int32_t>(segment_count));
            if (tile_count != 0) {
                out.attention_tiles = add(DType::I32, {4, tile_count}, "vision attention tiles");
            }
        }
        out.projected =
            add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision projected");
    }
    {
        auto mlp_scope = builder.scope();
        out.mlp_down = add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision MLP down");
        out.mlp_up =
            add(DType::BF16, {VisionScheduleConfig::intermediate, patches}, "vision MLP up");
        {
            auto norm_scope = builder.scope();
            out.mlp_norm =
                add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision MLP norm");
        }
    }
    out.normalized =
        add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision merger norm");
    out.merger_hidden =
        add(DType::BF16, {VisionScheduleConfig::merger_hidden, tokens}, "vision merger hidden");
    out.bytes = builder.finish(kWorkspaceAlignment, "vision workspace");
    return out;
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

} // namespace

VisionContext::VisionContext(DeviceContext& ctx, const LoadedModelData& weights) : ctx_(ctx) {
    patch_embed_      = &weights.vision.common.patch_embedding;
    patch_embed_bias_ = &weights.vision.common.patch_embedding_bias;
    position_embed_   = &weights.vision.common.position_embedding;
    for (std::uint32_t layer = 0; layer < blocks_.size(); ++layer) {
        const auto& source  = weights.vision.common.layers[layer];
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
    merger_.norm_weight = &weights.vision.common.merger_norm_weight;
    merger_.norm_bias   = &weights.vision.common.merger_norm_bias;
    merger_.fc1         = &weights.vision.common.merger_fc1;
    merger_.fc1_bias    = &weights.vision.common.merger_fc1_bias;
    merger_.fc2         = &weights.vision.merger_fc2;
    merger_.fc2_bias    = &weights.vision.merger_fc2_bias;
}

std::size_t VisionContext::workspace_bytes(const qwen3_6::VisionItemControl& item) {
    return build_workspace_layout(item.patch_count, item.merged_count,
                                  static_cast<std::size_t>(item.segment_count))
        .bytes;
}

std::size_t VisionContext::maximum_workspace_bytes() {
    return build_workspace_layout(131072, 32768, 384).bytes;
}

void VisionContext::encode(std::uint32_t item_index, const VisionItemView& item, Tensor& output,
                           WorkspaceArena& workspace, void* tap, VisionTapCallback callback) const {
    if ((tap == nullptr) != (callback == nullptr)) {
        throw std::invalid_argument("Vision tap context and callback must be provided together");
    }
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const qwen3_6::VisionItemControl& control = *item.control;
    const auto patches64                      = control.patch_count;
    const auto tokens64                       = control.merged_count;
    if (item.patches.size() !=
        checked_mul(patches64, VisionScheduleConfig::patch_dim, "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != VisionScheduleConfig::out_hidden ||
        output.ne[1] != static_cast<std::int32_t>(tokens64) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("Vision output must be contiguous BF16 [H,V]");
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(
        patches64, tokens64, static_cast<std::size_t>(control.segment_count));
    if (workspace.capacity() < layout.bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = ctx_.stream;
    workspace.reset();
    const DeviceSpan backing = workspace.alloc_bytes(layout.bytes, kWorkspaceAlignment);

    Tensor position_ids = layout.position_ids.bind(backing);
    Tensor cu_seqlens   = layout.cu_seqlens.bind(backing);
    Tensor pos_indices  = layout.pos_indices.bind(backing);
    Tensor pos_weights  = layout.pos_weights.bind(backing);
    copy_host(control.position_ids.data(), position_ids, stream);
    copy_host(control.cu_seqlens.data(), cu_seqlens, stream);
    copy_host(control.position_table_indices.data(), pos_indices, stream);
    copy_host(control.position_table_weights.data(), pos_weights, stream);

    Tensor x          = layout.x.bind(backing);
    Tensor patch_bf16 = layout.patch_bf16.bind(backing);
    Tensor patch_f32  = layout.patch_f32.bind(backing);
    copy_host(item.patches.data(), patch_f32, stream);
    ops::cast_fp32_to_bf16(patch_f32, patch_bf16, stream);
    ops::linear(patch_bf16, *patch_embed_, x, workspace, stream);
    ops::add_bias(*patch_embed_bias_, x, stream);
    // The artifact records the source table shape [rows,hidden], while Tensor's
    // contiguous matrix convention is [inner,columns]. The payload is already
    // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
    Tensor position_table = position_embed_->reshape(
        {VisionScheduleConfig::hidden, VisionScheduleConfig::position_embeddings});
    ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
    if (callback != nullptr) { callback(tap, item_index, VisionTapId::PatchEmbed, -1, x, stream); }

    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        const BlockW& block = blocks_[layer];
        {
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    ops::layer_norm(x, *block.norm1_weight, *block.norm1_bias,
                                    VisionScheduleConfig::norm_eps, h, stream);
                    ops::linear(h, *block.qkv, qkv, workspace, stream);
                }
                ops::add_bias(*block.qkv_bias, qkv, stream);
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
                ops::rope(position_ids, VisionScheduleConfig::rotary_dim,
                          VisionScheduleConfig::rope_theta, q, k, stream);
                Tensor attended_heads = attended.view(
                    {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                Tensor attention_tiles;
                Tensor* attention_tiles_ptr = nullptr;
                if (layout.attention_tiles) {
                    attention_tiles     = layout.attention_tiles->bind(backing);
                    attention_tiles_ptr = &attention_tiles;
                }
                ops::vision_attention(q, k, v, cu_seqlens, attention_tiles_ptr, attended_heads,
                                      stream);
            }
            Tensor projected = layout.projected.bind(backing);
            ops::linear(attended, *block.projection, projected, workspace, stream);
            ops::add_bias(*block.projection_bias, projected, stream);
            ops::residual_add(projected, x, stream);
        }
        {
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                ops::layer_norm(x, *block.norm2_weight, *block.norm2_bias,
                                VisionScheduleConfig::norm_eps, h, stream);
                ops::linear(h, *block.fc1, up, workspace, stream);
            }
            ops::add_bias(*block.fc1_bias, up, stream);
            ops::gelu(up, ops::GeluMode::Tanh, stream);
            ops::linear(up, *block.fc2, down, workspace, stream);
            ops::add_bias(*block.fc2_bias, down, stream);
            ops::residual_add(down, x, stream);
        }
        if (callback != nullptr) {
            callback(tap, item_index, VisionTapId::Block, static_cast<int>(layer), x, stream);
        }
    }

    Tensor normalized = layout.normalized.bind(backing);
    ops::layer_norm(x, *merger_.norm_weight, *merger_.norm_bias, VisionScheduleConfig::norm_eps,
                    normalized, stream);
    Tensor merged = normalized.view({VisionScheduleConfig::merger_hidden, tokens});
    Tensor hidden = layout.merger_hidden.bind(backing);
    ops::linear(merged, *merger_.fc1, hidden, workspace, stream);
    ops::add_bias(*merger_.fc1_bias, hidden, stream);
    ops::gelu(hidden, ops::GeluMode::Exact, stream);
    ops::linear(hidden, *merger_.fc2, output, workspace, stream);
    ops::add_bias(*merger_.fc2_bias, output, stream);
    if (callback != nullptr) { callback(tap, item_index, VisionTapId::Merger, -1, output, stream); }
}

VisionPrefillSession::VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                                           WorkspaceArena& workspace,
                                           const qwen3_6::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           runtime::TransientRegion transient, void* tap,
                                           VisionTapCallback callback)
    : device_(device), workspace_(workspace), prompt_(prompt), plan_(plan), transient_(transient),
      context_(device, model), tap_(tap), callback_(callback) {
    if ((tap_ == nullptr) != (callback_ == nullptr)) {
        throw std::invalid_argument("Vision tap context and callback must be provided together");
    }
    if (plan_.control.items.empty() || plan_.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (transient_.data == nullptr || transient_.alignment < kWorkspaceAlignment) {
        throw std::invalid_argument("Vision item output transient is missing or misaligned");
    }
    timers_.reserve(plan_.uses.size());
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= prompt_.token_ids.size()) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    std::uint32_t end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(nominal_end64, prompt_.token_ids.size()));

    const VisionUseSpan* active = nullptr;
    for (const VisionUseSpan& use : plan_.uses) {
        if (use.end <= begin) { continue; }
        if (use.begin >= end) { break; }
        if (active == nullptr) {
            active = &use;
        } else {
            end = std::min(end, use.begin);
            break;
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    if (active->item_index >= plan_.control.items.size() ||
        active->item_index >= prompt_.vision_items.size()) {
        throw std::logic_error("Vision prefill item index is out of range");
    }
    const qwen3_6::VisionItemControl& control = plan_.control.items[active->item_index];
    const qwen3_6::VisionItem& source         = prompt_.vision_items[active->item_index];
    if (source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
        source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
        source.patch_begin != control.patch_begin || source.patch_count != control.patch_count) {
        throw std::invalid_argument("Vision prefill plan does not describe the prepared item");
    }
    if (control.merged_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision item output columns exceed int32");
    }
    const std::size_t output_bytes =
        checked_mul(checked_mul(static_cast<std::size_t>(VisionScheduleConfig::out_hidden),
                                control.merged_count, "item output elements"),
                    dtype_size(DType::BF16), "item output bytes");
    if (output_bytes > transient_.size) {
        throw std::invalid_argument("Vision item output transient is too small");
    }
    Tensor output(
        transient_.data, DType::BF16,
        {VisionScheduleConfig::out_hidden, static_cast<std::int32_t>(control.merged_count)});

    if (!active_item_ || *active_item_ != active->item_index) {
        if (active_item_ && active->item_index <= *active_item_) {
            throw std::logic_error("Vision items are not consumed in strictly increasing order");
        }
        const std::size_t patch_offset = checked_mul(
            control.patch_begin, static_cast<std::size_t>(VisionScheduleConfig::patch_dim),
            "item patch offset");
        const std::size_t patch_elements = checked_mul(
            control.patch_count, static_cast<std::size_t>(VisionScheduleConfig::patch_dim),
            "item patch elements");
        if (patch_offset > prompt_.patches.size() ||
            patch_elements > prompt_.patches.size() - patch_offset) {
            throw std::invalid_argument("Vision item patch range exceeds prepared payload");
        }
        timers_.emplace_back(device_);
        timers_.back().start();
        context_.encode(
            active->item_index,
            VisionItemView{
                std::span<const float>(prompt_.patches).subspan(patch_offset, patch_elements),
                &control},
            output, workspace_, tap_, callback_);
        timers_.back().record_stop();
        workspace_.reset();
        active_item_ = active->item_index;
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output};
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
