// ninfer::ops - GQA A1/A2/A3 validation and finite route dispatch.
#include "ninfer/ops/gqa_attention.h"

#include "core/layout.h"
#include "ops/launcher/gqa_attention.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr float kExpectedScale  = 0.0625f;

std::int32_t kv_heads_for_q_heads(std::int32_t q_heads, const char* op) {
    if (q_heads == 24) { return 4; }
    if (q_heads == 16) { return 2; }
    throw std::invalid_argument(std::string(op) + ": unsupported Q/KV head geometry");
}

void require_kv_heads(std::int32_t kv_heads, const char* op) {
    if (kv_heads != 4 && kv_heads != 2) {
        throw std::invalid_argument(std::string(op) + ": unsupported KV head geometry");
    }
}

std::int32_t checked_i32(std::uint32_t value, const char* op, const char* name) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": " + name + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void validate_cache(const KVCacheLayerView& cache, std::int32_t kv_heads, const char* op) {
    if ((cache.dtype != DType::BF16 && cache.dtype != DType::I8) ||
        cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (cache.max_context == 0 || cache.padded_context < cache.max_context) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }
    if (cache.dtype == DType::BF16 && cache.quant_group != 0) {
        throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have quant_group");
    }
    if (cache.dtype == DType::I8 && cache.quant_group != kKvQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KV cache must use quant_group 64");
    }

    const std::int32_t padded = checked_i32(cache.padded_context, op, "padded_context");
    const DType code_dtype    = cache.dtype == DType::I8 ? DType::I8 : DType::BF16;
    if (cache.k.dtype != code_dtype || cache.v.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k, kHeadDim, padded, kv_heads, 1, op, "cache k");
    require_shape(cache.v, kHeadDim, padded, kv_heads, 1, op, "cache v");
    require_contiguous_nonnull(cache.k, op, "cache k");
    require_contiguous_nonnull(cache.v, op, "cache v");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale.data != nullptr || cache.v_scale.data != nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
        }
        return;
    }

    constexpr std::int32_t groups = kHeadDim / kKvQuantGroup;
    if (cache.k_scale.dtype != DType::FP16 || cache.v_scale.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale, groups, padded, kv_heads, 1, op, "cache k scale");
    require_shape(cache.v_scale, groups, padded, kv_heads, 1, op, "cache v scale");
    require_contiguous_nonnull(cache.k_scale, op, "cache k scale");
    require_contiguous_nonnull(cache.v_scale, op, "cache v scale");
}

void validate_envelope(GqaExecutionEnvelope envelope, const KVCacheLayerView& cache,
                       std::int32_t tokens, const char* op) {
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > cache.max_context) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                const KVCacheLayerView& cache, GqaExecutionEnvelope envelope,
                                float scale, const char* op) {
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, op);
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    validate_cache(cache, kv_heads, op);
    validate_envelope(envelope, cache, tokens, op);
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

SmallTWorkspace allocate_small_t_workspace(WorkspaceArena& workspace, std::int32_t q_heads,
                                           std::int32_t tokens) {
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, "gqa_attention workspace");
    const std::int32_t splits   = detail::gqa_attention_decode_splits(q_heads, kv_heads);
    return {
        workspace.alloc(DType::BF16, {kHeadDim, q_heads, tokens, splits}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits}),
    };
}

} // namespace

std::size_t gqa_attention_workspace_bytes(std::int32_t q_heads, std::int32_t tokens) {
    if (tokens <= 0 || !detail::gqa_attention_uses_small_t(tokens)) { return 0; }
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, "gqa_attention_workspace_bytes");
    const std::int32_t splits   = detail::gqa_attention_decode_splits(q_heads, kv_heads);
    const Tensor acc(nullptr, DType::BF16, {kHeadDim, q_heads, tokens, splits});
    const Tensor stat(nullptr, DType::FP32, {q_heads, tokens, splits});
    LayoutBuilder layout;
    (void)layout.add(acc.bytes(), 256, "GQA partial accumulator");
    (void)layout.add(stat.bytes(), 256, "GQA partial max");
    (void)layout.add(stat.bytes(), 256, "GQA partial sum");
    return layout.finish(256, "GQA workspace");
}

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCacheLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "gqa_attention";
    validate_attention_tensors(q, positions, out, cache, envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention: k/v must be BF16");
    }
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q.ne[1], op);
    require_shape(k, kHeadDim, kv_heads, tokens, 1, op, "k");
    require_shape(v, kHeadDim, kv_heads, tokens, 1, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");

    auto scope = workspace.scope();
    if (detail::gqa_attention_uses_small_t(tokens)) {
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], tokens);
        detail::gqa_attention_launch(q, k, v, positions, scale, cache, envelope, &partial.acc,
                                     &partial.m, &partial.l, out, stream);
        return;
    }
    detail::gqa_attention_launch(q, k, v, positions, scale, cache, envelope, nullptr, nullptr,
                                 nullptr, out, stream);
}

void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   KVCacheLayerView cache, cudaStream_t stream) {
    constexpr const char* op = "gqa_kv_append";
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_kv_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_kv_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    require_kv_heads(kv_heads, op);
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_kv_append: T must be positive"); }
    require_shape(k, kHeadDim, kv_heads, tokens, 1, op, "k");
    require_shape(v, kHeadDim, kv_heads, tokens, 1, op, "v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    validate_cache(cache, kv_heads, op);
    if (static_cast<std::uint32_t>(tokens) > cache.max_context) {
        throw std::invalid_argument("gqa_kv_append: T exceeds KV cache capacity");
    }
    detail::gqa_kv_append_launch(k, v, positions, cache, stream);
}

void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const KVCacheLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "gqa_attention_cached";
    validate_attention_tensors(q, positions, out, cache, envelope, scale, op);

    auto scope = workspace.scope();
    if (detail::gqa_attention_uses_small_t(q.ne[2])) {
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], q.ne[2]);
        detail::gqa_attention_cached_small_t_launch(q, positions, scale, cache, envelope,
                                                    partial.acc, partial.m, partial.l, out, stream);
        return;
    }
    detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops
