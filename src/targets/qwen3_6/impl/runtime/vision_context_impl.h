#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "core/device.h"
#include "core/layout.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
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

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a + b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("Vision ") + label +
                                    " alignment must be a power of two");
    }
    return checked_add(value, alignment - 1, label) & ~(alignment - 1);
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct VisionWorkspaceLayout {
    TensorRegion position_ids;
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
    std::size_t bytes = 0;
};

TensorRegion alias_tensor(const TensorRegion& storage, DType dtype,
                          std::initializer_list<std::int32_t> shape, const char* label) {
    Tensor tensor(nullptr, dtype, shape);
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

VisionWorkspaceLayout build_workspace_layout(std::size_t patches64, std::size_t tokens64) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 !=
            checked_mul(tokens64, VisionScheduleConfig::merge_unit, "patch/token relation")) {
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
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
    out.x = add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision residual");
    {
        auto position_lifetime = builder.scope();
        out.position_ids       = add(DType::I32, {patches, 2}, "vision position ids");
        {
            auto patch_scope = builder.scope();
            out.patch_bf16 =
                add(DType::BF16, {VisionScheduleConfig::patch_dim, patches}, "vision BF16 patches");
        }
        {
            auto position_scope = builder.scope();
            out.pos_indices     = add(DType::I32, {4, patches}, "vision position indices");
            out.pos_weights     = add(DType::FP32, {4, patches}, "vision position weights");
        }
        {
            auto attention_scope = builder.scope();
            out.qkv = add(DType::BF16, {3 * VisionScheduleConfig::hidden, patches}, "vision QKV");
            out.attention_norm = add(DType::BF16, {VisionScheduleConfig::hidden, patches},
                                     "vision attention norm/attended");
            out.attended       = out.attention_norm;
            out.projected =
                alias_tensor(out.qkv, DType::BF16, {VisionScheduleConfig::hidden, patches},
                             "attention projection output");
        }
        {
            auto mlp_scope = builder.scope();
            out.mlp_up =
                add(DType::BF16, {VisionScheduleConfig::intermediate, patches}, "vision MLP up");
            out.mlp_norm =
                add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision MLP norm/down");
            out.mlp_down = out.mlp_norm;
        }
    }
    {
        auto merger_scope = builder.scope();
        out.normalized =
            add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision merger norm");
        out.merger_hidden = alias_tensor(
            out.x, DType::BF16, {VisionScheduleConfig::merger_hidden, tokens}, "merger hidden");
    }
    out.bytes = builder.finish(1, "vision workspace");
    return out;
}

std::size_t output_handoff_bytes(std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision output handoff extent must fit positive int32");
    }
    LayoutBuilder layout;
    (void)layout.add_tensor(
        DType::BF16, {VisionScheduleConfig::out_hidden, static_cast<std::int32_t>(merged_tokens)},
        kWorkspaceAlignment, "Vision item output handoff");
    return layout.finish(kWorkspaceAlignment, "Vision item output handoff layout");
}

std::size_t merger_hidden_bytes(std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision merger hidden extent must fit positive int32");
    }
    Tensor tensor(nullptr, DType::BF16,
                  {VisionScheduleConfig::merger_hidden, static_cast<std::int32_t>(merged_tokens)});
    return tensor.bytes();
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

} // namespace

VisionContext::VisionContext(DeviceContext& ctx, const LoadedModelData& weights) : ctx_(ctx) {
    if (!weights.vision) {
        throw std::invalid_argument("Vision execution was requested without materialized weights");
    }
    const auto& vision = *weights.vision;
    patch_embed_       = &vision.common.patch_embedding;
    patch_embed_bias_  = &vision.common.patch_embedding_bias;
    position_embed_    = &vision.common.position_embedding;
    for (std::uint32_t layer = 0; layer < blocks_.size(); ++layer) {
        const auto& source  = vision.common.layers[layer];
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
    merger_.norm_weight = &vision.common.merger_norm_weight;
    merger_.norm_bias   = &vision.common.merger_norm_bias;
    merger_.fc1         = &vision.common.merger_fc1;
    merger_.fc1_bias    = &vision.common.merger_fc1_bias;
    merger_.fc2         = &vision.merger_fc2;
    merger_.fc2_bias    = &vision.merger_fc2_bias;
}

std::size_t VisionContext::workspace_bytes(const qwen3_6::VisionItemControl& item) {
    return workspace_bytes(item.patch_count, item.merged_count);
}

std::size_t VisionContext::workspace_bytes(std::size_t patches, std::size_t merged_tokens) {
    return build_workspace_layout(patches, merged_tokens).bytes;
}

VisionWorkspacePlan VisionContext::plan_workspace(std::uint32_t max_merged_tokens,
                                                  std::size_t general_capacity_bytes) {
    if (max_merged_tokens == 0) {
        throw std::invalid_argument("Vision workspace capacity bound must be positive");
    }
    if (general_capacity_bytes == 0) {
        throw std::invalid_argument("Vision general workspace capacity must be positive");
    }
    VisionWorkspacePlan out;
    out.max_merged_tokens      = max_merged_tokens;
    out.general_capacity_bytes = general_capacity_bytes;
    out.encode_peak_bytes =
        build_workspace_layout(checked_mul(max_merged_tokens, VisionScheduleConfig::merge_unit,
                                           "capacity patch count"),
                               max_merged_tokens)
            .bytes;
    out.handoff_offset_bytes =
        align_up(std::max(general_capacity_bytes, merger_hidden_bytes(max_merged_tokens)),
                 kWorkspaceAlignment, "handoff offset");
    out.handoff_capacity_bytes = output_handoff_bytes(max_merged_tokens);
    out.capacity_bytes         = std::max(
        out.encode_peak_bytes,
        checked_add(out.handoff_offset_bytes, out.handoff_capacity_bytes, "workspace capacity"));
    return out;
}

Tensor VisionContext::bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                  std::size_t merged_tokens) {
    if (backing.data == nullptr || backing.bytes < plan.capacity_bytes || merged_tokens == 0 ||
        merged_tokens > plan.max_merged_tokens) {
        throw std::invalid_argument("Vision output binding exceeds its workspace plan");
    }
    const std::size_t bytes = output_handoff_bytes(merged_tokens);
    if (bytes > plan.handoff_capacity_bytes) {
        throw std::logic_error("Vision output binding exceeds its handoff region");
    }
    TensorRegion region;
    region.region = LayoutRegion{plan.handoff_offset_bytes, bytes, kWorkspaceAlignment};
    region.dtype  = DType::BF16;
    region.shape  = {VisionScheduleConfig::out_hidden, static_cast<std::int32_t>(merged_tokens), 1,
                     1};
    return region.bind(backing);
}

void VisionContext::encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                           const VisionWorkspacePlan& plan) const {
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
    const Tensor planned_output = bind_output(backing, plan, tokens64);
    if (output.data != planned_output.data || output.bytes() != planned_output.bytes()) {
        throw std::invalid_argument("Vision output does not name the planned handoff region");
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(patches64, tokens64);
    if (layout.bytes > plan.encode_peak_bytes || backing.bytes < plan.capacity_bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = ctx_.stream;

    Tensor position_ids = layout.position_ids.bind(backing);
    copy_host(control.position_ids.data(), position_ids, stream);

    Tensor x          = layout.x.bind(backing);
    Tensor patch_bf16 = layout.patch_bf16.bind(backing);
    copy_host(item.patches.data(), patch_bf16, stream);
    ops::linear(patch_bf16, *patch_embed_, x, stream);
    ops::add_bias(*patch_embed_bias_, x, stream);
    // The artifact records the source table shape [rows,hidden], while Tensor's
    // contiguous matrix convention is [inner,columns]. The payload is already
    // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
    Tensor pos_indices = layout.pos_indices.bind(backing);
    Tensor pos_weights = layout.pos_weights.bind(backing);
    copy_host(control.position_table_indices.data(), pos_indices, stream);
    copy_host(control.position_table_weights.data(), pos_weights, stream);
    Tensor position_table = position_embed_->reshape(
        {VisionScheduleConfig::hidden, VisionScheduleConfig::position_embeddings});
    ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
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
                    ops::linear(h, *block.qkv, qkv, stream);
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
                ops::packed_softmax_attention(q, k, v,
                                              {VisionScheduleConfig::head_dim,
                                               VisionScheduleConfig::heads,
                                               VisionScheduleConfig::heads},
                                              VisionScheduleConfig::attention_scale,
                                              control.segment_length, attended_heads, stream);
            }
            Tensor projected = layout.projected.bind(backing);
            ops::linear(attended, *block.projection, projected, stream);
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
                ops::linear(h, *block.fc1, up, stream);
            }
            ops::add_bias(*block.fc1_bias, up, stream);
            ops::gelu(up, ops::GeluMode::Tanh, stream);
            ops::linear(up, *block.fc2, down, stream);
            ops::add_bias(*block.fc2_bias, down, stream);
            ops::residual_add(down, x, stream);
        }
    }

    Tensor normalized = layout.normalized.bind(backing);
    ops::layer_norm(x, *merger_.norm_weight, *merger_.norm_bias, VisionScheduleConfig::norm_eps,
                    normalized, stream);
    Tensor merged = normalized.view({VisionScheduleConfig::merger_hidden, tokens});
    Tensor hidden = layout.merger_hidden.bind(backing);
    ops::linear(merged, *merger_.fc1, hidden, stream);
    ops::add_bias(*merger_.fc1_bias, hidden, stream);
    ops::gelu(hidden, ops::GeluMode::Exact, stream);
    ops::linear(hidden, *merger_.fc2, output, stream);
    ops::add_bias(*merger_.fc2_bias, output, stream);
}

VisionPrefillSession::VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                                           DeviceSpan workspace,
                                           const VisionWorkspacePlan& workspace_plan,
                                           qwen3_6::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           std::size_t& handoff_peak_bytes)
    : device_(device), workspace_(workspace), workspace_plan_(workspace_plan), prompt_(prompt),
      plan_(plan), handoff_peak_bytes_(handoff_peak_bytes), context_(device, model) {
    if (plan_.control == nullptr || plan_.control->items.empty() || plan_.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (workspace_.data == nullptr || workspace_.bytes < workspace_plan_.capacity_bytes ||
        plan_.max_merged_count == 0 || plan_.max_merged_count > workspace_plan_.max_merged_tokens) {
        throw std::invalid_argument("Vision prefill workspace plan is invalid");
    }
    std::uint32_t previous_end = 0;
    std::optional<std::uint32_t> previous_item;
    for (const VisionUseSpan& use : plan_.uses) {
        if (use.begin >= use.end || use.begin < previous_end ||
            use.end > prompt_.token_ids.size()) {
            throw std::invalid_argument("Vision suffix item spans are invalid or unordered");
        }
        if (use.control_index >= plan_.control->items.size() ||
            use.prepared_item_index >= prompt_.vision_items.size() ||
            use.prepared_item_index >= prompt_.media_payloads.size() ||
            plan_.control->prepared_item_begin + use.control_index != use.prepared_item_index ||
            (previous_item && use.prepared_item_index <= *previous_item)) {
            throw std::invalid_argument("Vision suffix item indices are invalid or unordered");
        }
        const qwen3_6::VisionItemControl& control = plan_.control->items[use.control_index];
        const qwen3_6::VisionItem& source         = prompt_.vision_items[use.prepared_item_index];
        if (control.scatter_indices.empty() ||
            use.end != static_cast<std::uint32_t>(control.scatter_indices.back()) + 1U ||
            (use.begin != static_cast<std::uint32_t>(control.scatter_indices.front()) &&
             use.begin + 1U != static_cast<std::uint32_t>(control.scatter_indices.front())) ||
            source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
            source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
            source.patch_begin != control.patch_begin ||
            source.patch_count != control.patch_count) {
            throw std::invalid_argument("Vision suffix plan does not describe the prepared item");
        }
        if (control.merged_count > plan_.max_merged_count) {
            throw std::invalid_argument("Vision suffix item exceeds its request workspace extent");
        }
        const Tensor output =
            VisionContext::bind_output(workspace_, workspace_plan_, control.merged_count);
        const std::size_t patch_elements = checked_mul(
            control.patch_count, static_cast<std::size_t>(VisionScheduleConfig::patch_dim),
            "item patch elements");
        const auto& payload = prompt_.media_payloads[use.prepared_item_index];
        if (output.bytes() > workspace_plan_.handoff_capacity_bytes || !payload ||
            payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision suffix item storage has an invalid shape");
        }
        previous_end  = use.end;
        previous_item = use.prepared_item_index;
    }
    if (plan_.max_merged_count != 0 &&
        std::none_of(plan_.control->items.begin(), plan_.control->items.end(),
                     [&](const qwen3_6::VisionItemControl& item) {
                         return item.merged_count == plan_.max_merged_count;
                     })) {
        throw std::invalid_argument("Vision request workspace extent has no matching suffix item");
    }
    encoded_payloads_pending_release_.reserve(plan_.uses.size());
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

    while (next_use_ < plan_.uses.size() && plan_.uses[next_use_].end <= begin) { ++next_use_; }
    const VisionUseSpan* active = nullptr;
    if (next_use_ < plan_.uses.size() && plan_.uses[next_use_].begin < end) {
        active = &plan_.uses[next_use_];
        if (next_use_ + 1U < plan_.uses.size()) {
            end = std::min(end, plan_.uses[next_use_ + 1U].begin);
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    const qwen3_6::VisionItemControl& control = plan_.control->items[active->control_index];
    Tensor output = VisionContext::bind_output(workspace_, workspace_plan_, control.merged_count);

    if (!active_item_ || *active_item_ != active->prepared_item_index) {
        const auto& payload = prompt_.media_payloads[active->prepared_item_index];
        timers_.emplace_back(device_);
        timers_.back().start();
        context_.encode(VisionItemView{payload->span(), &control}, output, workspace_,
                        workspace_plan_);
        timers_.back().record_stop();
        active_item_          = active->prepared_item_index;
        active_handoff_bytes_ = output.bytes();
        handoff_peak_bytes_   = std::max(handoff_peak_bytes_, active_handoff_bytes_);
        encoded_payloads_pending_release_.push_back(active->prepared_item_index);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output};
}

void VisionPrefillSession::release_encoded_media_payloads() noexcept {
    for (const std::uint32_t item_index : encoded_payloads_pending_release_) {
        if (item_index >= prompt_.media_payloads.size()) { std::terminate(); }
        prompt_.media_payloads[item_index].reset();
    }
    encoded_payloads_pending_release_.clear();
}

void VisionPrefillSession::retire_handoff() noexcept {
    active_item_.reset();
    active_handoff_bytes_ = 0;
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
