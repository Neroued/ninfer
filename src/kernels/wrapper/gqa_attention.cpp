// qus::kernels - gqa_attention wrapper: public api validation and phase dispatch.
#include "qus/kernels/gqa_attention.h"

#include "kernels/launcher/gqa_attention.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads  = 24;
constexpr std::int32_t kKVHeads = 4;
constexpr float kExpectedScale  = 0.0625f;

struct ArenaScope {
    WorkspaceArena& ws;
    std::size_t mark;

    explicit ArenaScope(WorkspaceArena& arena) : ws(arena), mark(arena.mark()) {}

    ~ArenaScope() { ws.rewind(mark); }

    ArenaScope(const ArenaScope&)            = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;
};

std::int32_t checked_i32(std::uint32_t value, const char* op, const char* name) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": " + name + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void validate_cache(KVCache& kv, int layer, const char* op) {
    if (layer < 0 || static_cast<std::uint32_t>(layer) >= kv.layer_count()) {
        throw std::invalid_argument(std::string(op) + ": layer out of range");
    }
    if (kv.dtype != DType::BF16 || kv.num_kv_heads != kKVHeads || kv.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) +
                                    ": KVCache must be BF16 [256,padded_context,4]");
    }
    if (kv.padded_context < kv.max_context) {
        throw std::invalid_argument(std::string(op) + ": KVCache padded_context is too small");
    }
    if (kv.k.size() != kv.v.size() || kv.k.size() != kv.layer_count()) {
        throw std::invalid_argument(std::string(op) + ": invalid KVCache layer vectors");
    }

    const Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    const Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];
    if (cache_k.dtype != DType::BF16 || cache_v.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": KVCache tensors must be BF16");
    }
    const std::int32_t padded_context = checked_i32(kv.padded_context, op, "padded_context");
    if (cache_k.ne[0] != kHeadDim || cache_k.ne[1] != padded_context || cache_k.ne[2] != kKVHeads ||
        cache_k.ne[3] != 1 || cache_v.ne[0] != kHeadDim || cache_v.ne[1] != padded_context ||
        cache_v.ne[2] != kKVHeads || cache_v.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid KVCache tensor shape");
    }
    require_contiguous_nonnull(cache_k, op, "cache k");
    require_contiguous_nonnull(cache_v, op, "cache v");
}

} // namespace

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCache& kv, int layer, WorkspaceArena& ws, Tensor& out,
                   cudaStream_t stream) {
    constexpr const char* op = "gqa_attention";
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention: q/k/v/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_attention: positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument("gqa_attention: scale must be 1/sqrt(256)");
    }

    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_attention: T must be positive"); }
    require_shape(q, kHeadDim, kQHeads, tokens, 1, op, "q");
    require_shape(k, kHeadDim, kKVHeads, tokens, 1, op, "k");
    require_shape(v, kHeadDim, kKVHeads, tokens, 1, op, "v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, kQHeads, tokens, 1, op, "out");

    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    validate_cache(kv, layer, op);

    const std::uint32_t token_count = static_cast<std::uint32_t>(tokens);
    if (token_count > kv.max_context) {
        throw std::invalid_argument("gqa_attention: T exceeds KVCache max_context");
    }
    if (token_count > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("gqa_attention: T exceeds int32");
    }

    ArenaScope arena_scope(ws);
    Tensor partial_acc;
    Tensor partial_m;
    Tensor partial_l;
    Tensor* partial_acc_ptr = nullptr;
    Tensor* partial_m_ptr   = nullptr;
    Tensor* partial_l_ptr   = nullptr;
    if (detail::gqa_attention_uses_small_t(tokens)) {
        partial_acc     = ws.alloc(DType::BF16, {kHeadDim, kQHeads, tokens, kGqaDecodeSplits});
        partial_m       = ws.alloc(DType::FP32, {kQHeads, tokens, kGqaDecodeSplits});
        partial_l       = ws.alloc(DType::FP32, {kQHeads, tokens, kGqaDecodeSplits});
        partial_acc_ptr = &partial_acc;
        partial_m_ptr   = &partial_m;
        partial_l_ptr   = &partial_l;
    }

    detail::gqa_attention_launch(q, k, v, positions, scale, kv, layer, partial_acc_ptr,
                                 partial_m_ptr, partial_l_ptr, out, stream);
}

} // namespace qus::kernels
