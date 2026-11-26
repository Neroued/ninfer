#include "models/inkling2/vision/execution.h"

#include "core/layout.h"
#include "core/transfer.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scaled_residual_add_layer_norm.h"
#include "ninfer/ops/softmax_attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ninfer::models::inkling2::vision {
namespace {

struct MetadataLayout {
    TensorRegion positions, segments, patch_columns, pool_rows, position_indices, position_weights;
};

struct Layout {
    MetadataLayout device_metadata, host_metadata;
    LayoutRegion x, y, transient, scratch;
    TensorRegion host_patches;
    std::size_t device_bytes, host_bytes;
};

MetadataLayout metadata_layout(LayoutBuilder& b, int patches, int tokens, int segments) {
    return {b.add_tensor(DType::I32, {tokens, 2}, 256, "Vision RoPE coordinates"),
            b.add_tensor(DType::I32, {segments + 1}, 256, "Vision segment offsets"),
            b.add_tensor(DType::I32, {patches}, 256, "Vision patch columns"),
            b.add_tensor(DType::I32, {patches}, 256, "Vision spatial merge rows"),
            b.add_tensor(DType::I32, {4, patches}, 256, "Vision interpolation indices"),
            b.add_tensor(DType::FP32, {4, patches}, 256, "Vision interpolation weights")};
}

ops::AttentionHeadGeometry attention_geometry(const VisionConfig& c) {
    return {dimension(c.head_dim), dimension(c.heads), dimension(c.heads)};
}

template <class Projections>
Layout layout(const VisionConfig& c, const Projections& p, int features, int segments) {
    const auto patches = dimension(std::uint64_t(features) * patches_per_feature(c));
    const auto tokens  = dimension(std::uint64_t(patches) + segments);
    const auto h       = dimension(c.hidden);
    const auto width   = dimension(c.patch_width());
    const auto bf16    = [](std::uint64_t rows, int columns) {
        return Tensor::metadata(DType::BF16, {dimension(rows), columns}).bytes();
    };
    const auto projection = [](const auto& weight, int columns, bool bias = false) {
        const auto w = ops::linear_profile(weight);
        return bias ? ops::linear_bias_workspace_capacity_bytes(w.qtype, w.n, w.k, w.policy, 1,
                                                                columns)
                    : ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, w.policy, 1, columns);
    };
    auto scratch = std::max(
        {projection(p.patch, patches), projection(p.projector_in, features, c.projector_bias),
         projection(p.projector_out, features, c.projector_bias),
         ops::packed_softmax_attention_workspace_capacity_bytes(attention_geometry(c), 1, tokens, 1,
                                                                segments)});
    const std::array split_rows{h, h, h};
    for (const auto& layer : p.layers) {
        const auto w = ops::linear_profile(layer.qkv);
        scratch =
            std::max({scratch, projection(layer.output, tokens, true),
                      projection(layer.fc1, tokens, true), projection(layer.fc2, tokens, true),
                      ops::linear_split_bias_workspace_capacity_bytes(w.qtype, w.n, w.k, split_rows,
                                                                      w.policy, 1, tokens)});
    }

    LayoutBuilder device, host;
    Layout out;
    out.device_metadata = metadata_layout(device, patches, tokens, segments);
    out.host_metadata   = metadata_layout(host, patches, tokens, segments);
    out.host_patches = host.add_tensor(DType::BF16, {width, patches}, 256, "Vision input patches");
    out.x = device.add(std::max(bf16(h, tokens), bf16(c.projection_hidden_dim, features)), 256,
                       "Vision assembled tokens, layer norm or projector hidden");
    out.y = device.add(bf16(h, tokens), 256, "Vision residual");
    out.transient =
        device.add(std::max({bf16(width, patches), bf16(4ULL * h, tokens),
                             bf16(std::uint64_t(c.intermediate) + h, tokens), bf16(h, patches)}),
                   256, "Vision patch input, attention, MLP or spatial merge");
    out.scratch      = device.add(std::max<std::size_t>(1, scratch), 256, "Vision op scratch");
    out.device_bytes = device.finish(256);
    out.host_bytes   = host.finish(256);
    return out;
}

MemorySpan bytes(const Tensor& t) { return {t.data, t.bytes(), t.device}; }

void upload(const void* source, const TensorRegion& device, const TensorRegion& host,
            MemorySpan backing, MemorySpan staging, CudaExecutionView cuda) {
    const auto pinned = host.region.bind(staging);
    std::memcpy(pinned.data, source, pinned.bytes);
    copy_bytes_async(device.region.bind(backing), pinned, cuda);
}

void project(const Tensor& x, const execution::LinearParameters& w, const std::optional<Tensor>& b,
             Tensor& out, MemoryArena& scratch, CudaExecutionView cuda) {
    if (b)
        ops::linear_bias(x, w.weight, *b, out, w.policy, scratch, cuda);
    else
        ops::linear(x, w.weight, out, w.policy, scratch, cuda);
}

} // namespace

models::vision::Capacity plan_resources(const VisionConfig& c, const execution::VisionProfile& p,
                                        std::uint32_t maximum, std::size_t budget) {
    return models::vision::plan_capacity(
        budget, maximum, dimension(c.text_hidden), [&](std::uint32_t features) {
            // Every image has at least one merged feature and exactly one extra CLS token.
            const auto l = layout(c, p, dimension(features), dimension(features));
            return models::vision::EncoderMemory{l.device_bytes, l.host_bytes};
        });
}

std::size_t encode(const VisionConfig& c, const execution::VisionParameters& p,
                   const EncoderBatch& input, Tensor& output, MemorySpan backing,
                   MemorySpan staging, CudaExecutionView cuda) {
    const auto f = dimension(input.features);
    const auto s = dimension(input.patches.size());
    const auto n = dimension(std::uint64_t(f) * patches_per_feature(c));
    const auto t = dimension(std::uint64_t(n) + s);
    const auto h = dimension(c.hidden);
    if (!f || !s || s > f || input.cu_seqlens.size() != std::size_t(s) + 1 ||
        input.cu_seqlens.front() != 0 || input.cu_seqlens.back() != t ||
        input.positions.size() != 2ULL * t || input.position_indices.size() != std::size_t(n) ||
        input.position_weights.size() != std::size_t(n) ||
        input.patch_columns.size() != std::size_t(n) || input.pool_rows.size() != std::size_t(n))
        throw std::invalid_argument("Inkling2 Vision metadata does not cover its packed tokens");
    if (output.dtype != DType::BF16 || output.ne[0] != dimension(c.text_hidden) ||
        output.ne[1] != f || output.ne[2] != 1 || output.ne[3] != 1 || !output.is_contiguous() ||
        output.device != cuda.device() || backing.device != cuda.device() ||
        staging.device != Device::cpu() || !staging.host_pinned || p.layers.size() != c.depth ||
        p.layers.empty())
        throw std::invalid_argument("Inkling2 Vision operands differ from the prepared geometry");
    const auto base = reinterpret_cast<std::uintptr_t>(backing.data);
    const auto dst  = reinterpret_cast<std::uintptr_t>(output.data);
    if (dst < base + backing.bytes && base < dst + output.bytes())
        throw std::invalid_argument("Inkling2 Vision output overlaps encoder workspace");

    const int first_length = input.cu_seqlens[1];
    bool equal             = true;
    for (int i = 0; i < s; ++i) {
        const auto length = input.cu_seqlens[i + 1] - input.cu_seqlens[i];
        if (length <= 1 || input.patches[i].size() != std::uint64_t(length - 1) * c.patch_width())
            throw std::invalid_argument("Inkling2 Vision image differs from its packed segment");
        equal &= length == first_length;
    }
    const auto l = layout(c, p, f, s);
    if (backing.bytes < l.device_bytes || staging.bytes < l.host_bytes)
        throw std::invalid_argument("Inkling2 Vision workspace or staging is too small");

    const auto transient = l.transient.bind(backing);
    const auto matrix    = [&](int rows, int columns, std::size_t begin = 0) {
        return Tensor(transient.subspan(begin, transient.bytes - begin), DType::BF16,
                         {rows, columns});
    };
    auto patches              = matrix(dimension(c.patch_width()), n);
    const auto pinned_patches = l.host_patches.region.bind(staging);
    auto* patch_dst           = static_cast<std::byte*>(pinned_patches.data);
    for (const auto part : input.patches) {
        std::memcpy(patch_dst, part.data(), part.size_bytes());
        patch_dst += part.size_bytes();
    }
    copy_bytes_async(bytes(patches), pinned_patches, cuda);
    const auto& d = l.device_metadata;
    const auto& a = l.host_metadata;
    upload(input.positions.data(), d.positions, a.positions, backing, staging, cuda);
    upload(input.pool_rows.data(), d.pool_rows, a.pool_rows, backing, staging, cuda);
    upload(input.position_indices.data(), d.position_indices, a.position_indices, backing, staging,
           cuda);
    upload(input.position_weights.data(), d.position_weights, a.position_weights, backing, staging,
           cuda);
    if (!equal) upload(input.cu_seqlens.data(), d.segments, a.segments, backing, staging, cuda);
    if (s > 1)
        upload(input.patch_columns.data(), d.patch_columns, a.patch_columns, backing, staging,
               cuda);

    auto x                = Tensor(l.x.bind(backing), DType::BF16, {h, t});
    auto residual         = Tensor(l.y.bind(backing), DType::BF16, {h, t});
    auto patch_embeddings = s == 1 ? x.slice(1, 1, n) : residual.slice(1, 0, n);
    MemoryArena scratch(l.scratch.bind(backing));
    ops::linear(patches, p.patch.weight, patch_embeddings, p.patch.policy, scratch, cuda);
    ops::embedding_interpolate_add(p.position_embedding, d.position_indices.bind(backing),
                                   d.position_weights.bind(backing), patch_embeddings, cuda);
    if (s > 1) ops::scatter(patch_embeddings, d.patch_columns.bind(backing), x, cuda);

    auto cls = matrix(h, 1);
    copy_bytes_async(bytes(cls), bytes(p.class_embedding), cuda);
    ops::residual_add(p.cls_position_embedding.view({h, 1}), cls, cuda);
    for (int i = 0; i < s; ++i)
        copy_bytes_async(bytes(x.slice(1, input.cu_seqlens[i], 1)), bytes(cls), cuda);
    ops::layer_norm(x, p.pre_norm_weight, p.pre_norm_bias, c.norm_epsilon, residual, cuda);
    const auto& first = p.layers.front();
    ops::layer_norm(residual, first.ln_1_weight, first.ln_1_bias, c.norm_epsilon, x, cuda);

    const auto plane = x.bytes();
    const std::array qkv{matrix(h, t), matrix(h, t, plane), matrix(h, t, 2 * plane)};
    const auto geometry = attention_geometry(c);
    const auto shape    = {dimension(c.head_dim), dimension(c.heads), t};
    auto q = qkv[0].view(shape), k = qkv[1].view(shape), v = qkv[2].view(shape);
    auto attended         = matrix(h, t, 3 * plane);
    auto attended_heads   = attended.view(shape);
    auto attention_update = qkv[0];
    auto mid              = matrix(dimension(c.intermediate), t);
    auto mlp_update       = matrix(h, t, mid.bytes());
    const auto scale      = 1.0F / std::sqrt(float(c.head_dim));
    for (std::size_t i = 0; i < p.layers.size(); ++i) {
        const auto& layer = p.layers[i];
        ops::linear_split(x, layer.qkv.weight, layer.qkv_bias, qkv, layer.qkv.policy, scratch,
                          cuda);
        ops::rope_2d_interleaved(d.positions.bind(backing), 10000.0F, q, k, cuda);
        if (equal)
            ops::packed_softmax_attention(q, k, v, geometry, scale, first_length, attended_heads,
                                          cuda);
        else
            ops::packed_softmax_attention(q, k, v, geometry, scale, d.segments.bind(backing),
                                          scratch, attended_heads, cuda);
        ops::linear_bias(attended, layer.output.weight, layer.output_bias, attention_update,
                         layer.output.policy, scratch, cuda);
        ops::scaled_residual_add_layer_norm(residual, attention_update, layer.ls_1_gamma,
                                            layer.ln_2_weight, layer.ln_2_bias, c.norm_epsilon,
                                            residual, x, cuda);
        ops::linear_bias(x, layer.fc1.weight, layer.fc1_bias, mid, layer.fc1.policy, scratch, cuda);
        ops::gelu(mid, ops::GeluMode::Exact, cuda);
        ops::linear_bias(mid, layer.fc2.weight, layer.fc2_bias, mlp_update, layer.fc2.policy,
                         scratch, cuda);
        if (i + 1 < p.layers.size()) {
            const auto& next = p.layers[i + 1];
            ops::scaled_residual_add_layer_norm(residual, mlp_update, layer.ls_2_gamma,
                                                next.ln_1_weight, next.ln_1_bias, c.norm_epsilon,
                                                residual, x, cuda);
        } else {
            ops::scaled_residual_add(residual, mlp_update, layer.ls_2_gamma, residual, cuda);
        }
    }

    auto pooled = matrix(h, n).view({h, 1, n});
    ops::gather_batch_rows(residual.view({h, 1, t}), d.pool_rows.bind(backing), pooled, cuda);
    auto projected =
        Tensor(l.x.bind(backing), DType::BF16, {dimension(c.projection_hidden_dim), f});
    project(pooled.view({dimension(c.merge_width()), f}), p.projector_in, p.projector_in_bias,
            projected, scratch, cuda);
    ops::gelu(projected, ops::GeluMode::Exact, cuda);
    project(projected, p.projector_out, p.projector_out_bias, output, scratch, cuda);
    return l.device_bytes;
}

} // namespace ninfer::models::inkling2::vision
