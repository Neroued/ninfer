#include "ninfer/model/vision.h"

#include "model/vision_ops.h"
#include "ninfer/core/device.h"
#include "ninfer/core/weight_store_parser.h"
#include "ninfer/kernels/add_bias.h"
#include "ninfer/kernels/gelu.h"
#include "ninfer/kernels/layer_norm.h"
#include "ninfer/kernels/linear.h"
#include "ninfer/kernels/residual_add.h"
#include "ninfer/kernels/rope.h"
#include "ninfer/kernels/vision_attention.h"
#include "ninfer/kernels/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::model {
namespace {

constexpr ModuleKind kVision = ModuleKind::VisionEncoder;

std::uint32_t sk(SourceKind kind) { return static_cast<std::uint32_t>(kind); }

std::string source_label(SourceKind kind, std::uint32_t layer) {
    return "source_kind=" + std::to_string(sk(kind)) +
           " layer=" + (layer == kQ5090NoLayer ? std::string("NO_LAYER") : std::to_string(layer));
}

const Weight* require_weight(const WeightStore& store, SourceKind kind, std::uint32_t layer) {
    const Weight* value = store.qweight(kVision, sk(kind), layer);
    if (value == nullptr) {
        throw std::runtime_error("missing q5090 Vision weight: " + source_label(kind, layer));
    }
    return value;
}

const Tensor* require_tensor(const WeightStore& store, SourceKind kind, std::uint32_t layer) {
    const Tensor* value = store.tensor(kVision, sk(kind), layer);
    if (value == nullptr) {
        throw std::runtime_error("missing q5090 Vision tensor: " + source_label(kind, layer));
    }
    return value;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a + b;
}

std::size_t aligned_add(std::size_t offset, std::size_t bytes) {
    constexpr std::size_t align = 256;
    const std::size_t aligned   = (offset + align - 1) & ~(align - 1);
    return checked_add(aligned, bytes, "workspace size");
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

} // namespace

Qwen3_6_Vision::Qwen3_6_Vision(DeviceContext& ctx, const WeightStore& weights) : ctx_(ctx) {
    patch_embed_      = require_weight(weights, SourceKind::VisPatchEmbed, kQ5090NoLayer);
    patch_embed_bias_ = require_tensor(weights, SourceKind::VisPatchEmbedBias, kQ5090NoLayer);
    position_embed_   = require_tensor(weights, SourceKind::VisPosEmbed, kQ5090NoLayer);
    for (std::uint32_t layer = 0; layer < blocks_.size(); ++layer) {
        BlockW& out         = blocks_[layer];
        out.norm1_weight    = require_tensor(weights, SourceKind::VisBlockNorm1W, layer);
        out.norm1_bias      = require_tensor(weights, SourceKind::VisBlockNorm1B, layer);
        out.qkv             = require_weight(weights, SourceKind::VisBlockQkv, layer);
        out.qkv_bias        = require_tensor(weights, SourceKind::VisBlockQkvBias, layer);
        out.projection      = require_weight(weights, SourceKind::VisBlockProj, layer);
        out.projection_bias = require_tensor(weights, SourceKind::VisBlockProjBias, layer);
        out.norm2_weight    = require_tensor(weights, SourceKind::VisBlockNorm2W, layer);
        out.norm2_bias      = require_tensor(weights, SourceKind::VisBlockNorm2B, layer);
        out.fc1             = require_weight(weights, SourceKind::VisBlockFc1, layer);
        out.fc1_bias        = require_tensor(weights, SourceKind::VisBlockFc1Bias, layer);
        out.fc2             = require_weight(weights, SourceKind::VisBlockFc2, layer);
        out.fc2_bias        = require_tensor(weights, SourceKind::VisBlockFc2Bias, layer);
    }
    merger_.norm_weight = require_tensor(weights, SourceKind::VisMergerNormW, kQ5090NoLayer);
    merger_.norm_bias   = require_tensor(weights, SourceKind::VisMergerNormB, kQ5090NoLayer);
    merger_.fc1         = require_weight(weights, SourceKind::VisMergerFc1, kQ5090NoLayer);
    merger_.fc1_bias    = require_tensor(weights, SourceKind::VisMergerFc1Bias, kQ5090NoLayer);
    merger_.fc2         = require_weight(weights, SourceKind::VisMergerFc2, kQ5090NoLayer);
    merger_.fc2_bias    = require_tensor(weights, SourceKind::VisMergerFc2Bias, kQ5090NoLayer);
}

std::size_t Qwen3_6_Vision::workspace_bytes(const ProcessedInput& input) {
    const std::size_t patches = static_cast<std::size_t>(input.stats.raw_patches);
    const std::size_t tokens  = static_cast<std::size_t>(input.stats.vision_tokens);
    if (patches == 0 || tokens == 0 ||
        patches != checked_mul(tokens, VisionConfig::merge_unit, "patch/token relation")) {
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    }
    if (patches > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }

    std::size_t persistent = 0;
    persistent             = aligned_add(persistent,
                                         checked_mul(tokens, VisionConfig::out_hidden * 2ULL, "output bytes"));
    persistent = aligned_add(persistent, checked_mul(patches, 2ULL * 4ULL, "position bytes"));
    persistent = aligned_add(persistent, checked_mul(input.vision_items.size() + patches + 1ULL,
                                                     4ULL, "segment bound bytes"));
    persistent = aligned_add(persistent, checked_mul(patches, 4ULL * 4ULL, "position index bytes"));
    persistent =
        aligned_add(persistent, checked_mul(patches, 4ULL * 4ULL, "position weight bytes"));
    persistent = aligned_add(persistent,
                             checked_mul(patches, VisionConfig::hidden * 2ULL, "residual bytes"));

    std::size_t block_peak = persistent;
    block_peak             = aligned_add(block_peak,
                                         checked_mul(patches, VisionConfig::hidden * 2ULL, "block out bytes"));
    block_peak = aligned_add(block_peak, checked_mul(patches, VisionConfig::intermediate * 2ULL,
                                                     "block intermediate bytes"));
    block_peak = aligned_add(block_peak,
                             checked_mul(patches, VisionConfig::hidden * 2ULL, "block norm bytes"));

    std::size_t upload_peak = persistent;
    upload_peak             = aligned_add(
        upload_peak, checked_mul(patches, VisionConfig::patch_dim * 2ULL, "BF16 patch bytes"));
    upload_peak = aligned_add(
        upload_peak, checked_mul(patches, VisionConfig::patch_dim * 4ULL, "FP32 patch bytes"));

    // Account for alignment slack in nested operator scratch allocations.
    return checked_add(std::max(block_peak, upload_peak), 4096, "workspace slack");
}

Tensor Qwen3_6_Vision::encode(const ProcessedInput& input, WorkspaceArena& workspace, void* tap,
                              VisionTapCallback callback) const {
    if ((tap == nullptr) != (callback == nullptr)) {
        throw std::invalid_argument("Vision tap context and callback must be provided together");
    }
    const detail::VisionControl control = detail::build_vision_control(input);
    const auto patches64                = static_cast<std::size_t>(input.stats.raw_patches);
    const auto tokens64                 = static_cast<std::size_t>(input.stats.vision_tokens);
    if (input.patches.size() != checked_mul(patches64, VisionConfig::patch_dim, "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (workspace.capacity() < workspace_bytes(input)) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = ctx_.stream;
    workspace.reset();

    Tensor output       = workspace.alloc(DType::BF16, {VisionConfig::out_hidden, tokens});
    Tensor position_ids = workspace.alloc(DType::I32, {patches, 2});
    Tensor cu_seqlens =
        workspace.alloc(DType::I32, {static_cast<std::int32_t>(control.cu_seqlens.size())});
    Tensor pos_indices = workspace.alloc(DType::I32, {4, patches});
    Tensor pos_weights = workspace.alloc(DType::FP32, {4, patches});
    copy_host(control.position_ids.data(), position_ids, stream);
    copy_host(control.cu_seqlens.data(), cu_seqlens, stream);
    copy_host(control.pos_indices.data(), pos_indices, stream);
    copy_host(control.pos_weights.data(), pos_weights, stream);

    Tensor x = workspace.alloc(DType::BF16, {VisionConfig::hidden, patches});
    {
        auto input_scope  = workspace.scope();
        Tensor patch_bf16 = workspace.alloc(DType::BF16, {VisionConfig::patch_dim, patches});
        Tensor patch_f32  = workspace.alloc(DType::FP32, {VisionConfig::patch_dim, patches});
        copy_host(input.patches.data(), patch_f32, stream);
        detail::vision_f32_to_bf16(patch_f32, patch_bf16, stream);
        kernels::linear(patch_bf16, *patch_embed_, x, workspace, stream);
    }
    kernels::add_bias(*patch_embed_bias_, x, stream);
    // q5090 records the source table shape [rows,hidden], while Tensor's
    // contiguous matrix convention is [inner,columns]. The payload is already
    // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
    Tensor position_table =
        position_embed_->reshape({VisionConfig::hidden, VisionConfig::position_embeddings});
    kernels::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
    if (callback != nullptr) { callback(tap, VisionTapId::PatchEmbed, -1, x, stream); }

    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        const BlockW& block = blocks_[layer];
        {
            auto attention_scope = workspace.scope();
            Tensor attended      = workspace.alloc(DType::BF16, {VisionConfig::hidden, patches});
            {
                auto qkv_scope = workspace.scope();
                Tensor qkv     = workspace.alloc(DType::BF16, {3 * VisionConfig::hidden, patches});
                {
                    auto norm_scope = workspace.scope();
                    Tensor h        = workspace.alloc(DType::BF16, {VisionConfig::hidden, patches});
                    kernels::layer_norm(x, *block.norm1_weight, *block.norm1_bias,
                                        VisionConfig::norm_eps, h, stream);
                    kernels::linear(h, *block.qkv, qkv, workspace, stream);
                }
                kernels::add_bias(*block.qkv_bias, qkv, stream);
                const std::int32_t plane      = VisionConfig::hidden;
                const std::size_t plane_bytes = static_cast<std::size_t>(plane) * 2;
                Tensor q(qkv.data, DType::BF16,
                         {VisionConfig::head_dim, VisionConfig::heads, patches});
                Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16,
                         {VisionConfig::head_dim, VisionConfig::heads, patches});
                Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16,
                         {VisionConfig::head_dim, VisionConfig::heads, patches});
                q.nb[2] = qkv.nb[1];
                k.nb[2] = qkv.nb[1];
                v.nb[2] = qkv.nb[1];
                kernels::rope(position_ids, VisionConfig::rotary_dim, VisionConfig::rope_theta, q,
                              k, stream);
                Tensor attended_heads =
                    attended.view({VisionConfig::head_dim, VisionConfig::heads, patches});
                kernels::vision_attention(q, k, v, cu_seqlens, workspace, attended_heads, stream);
            }
            Tensor projected = workspace.alloc(DType::BF16, {VisionConfig::hidden, patches});
            kernels::linear(attended, *block.projection, projected, workspace, stream);
            kernels::add_bias(*block.projection_bias, projected, stream);
            kernels::residual_add(projected, x, stream);
        }
        {
            auto mlp_scope = workspace.scope();
            Tensor down    = workspace.alloc(DType::BF16, {VisionConfig::hidden, patches});
            Tensor up      = workspace.alloc(DType::BF16, {VisionConfig::intermediate, patches});
            {
                auto norm_scope = workspace.scope();
                Tensor h        = workspace.alloc(DType::BF16, {VisionConfig::hidden, patches});
                kernels::layer_norm(x, *block.norm2_weight, *block.norm2_bias,
                                    VisionConfig::norm_eps, h, stream);
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

    Tensor normalized = workspace.alloc(DType::BF16, {VisionConfig::hidden, patches});
    kernels::layer_norm(x, *merger_.norm_weight, *merger_.norm_bias, VisionConfig::norm_eps,
                        normalized, stream);
    Tensor merged = normalized.view({VisionConfig::merger_hidden, tokens});
    Tensor hidden = workspace.alloc(DType::BF16, {VisionConfig::merger_hidden, tokens});
    kernels::linear(merged, *merger_.fc1, hidden, workspace, stream);
    kernels::add_bias(*merger_.fc1_bias, hidden, stream);
    kernels::gelu(hidden, kernels::GeluMode::Exact, stream);
    kernels::linear(hidden, *merger_.fc2, output, workspace, stream);
    kernels::add_bias(*merger_.fc2_bias, output, stream);
    if (callback != nullptr) { callback(tap, VisionTapId::Merger, -1, output, stream); }
    return output;
}

} // namespace ninfer::model
