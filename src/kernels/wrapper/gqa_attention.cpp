// qus::kernels - gqa_attention wrapper: public api validation and phase dispatch.
#include "qus/kernels/gqa_attention.h"

#include "kernels/launcher/gqa_attention.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads  = 24;
constexpr std::int32_t kKVHeads = 4;
constexpr float kExpectedScale  = 0.0625f;

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string("gqa_attention_decode: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("gqa_attention_decode: ") + name +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("gqa_attention_decode: ") + name +
                                    " data must be non-null");
    }
}

void validate_cache(KVCache& kv, int layer) {
    if (layer < 0 || static_cast<std::uint32_t>(layer) >= kv.layer_count()) {
        throw std::invalid_argument("gqa_attention_decode: layer out of range");
    }
    if (kv.dtype != DType::BF16 || kv.num_kv_heads != kKVHeads || kv.head_dim != kHeadDim) {
        throw std::invalid_argument("gqa_attention_decode: KVCache must be BF16 [4,256,max_ctx]");
    }
    if (kv.k.size() != kv.v.size() || kv.k.size() != kv.layer_count()) {
        throw std::invalid_argument("gqa_attention_decode: invalid KVCache layer vectors");
    }

    const Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    const Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];
    if (cache_k.dtype != DType::BF16 || cache_v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention_decode: KVCache tensors must be BF16");
    }
    if (cache_k.ne[0] != kKVHeads || cache_k.ne[1] != kHeadDim ||
        cache_k.ne[2] != static_cast<std::int32_t>(kv.max_context) || cache_k.ne[3] != 1 ||
        cache_v.ne[0] != kKVHeads || cache_v.ne[1] != kHeadDim ||
        cache_v.ne[2] != static_cast<std::int32_t>(kv.max_context) || cache_v.ne[3] != 1) {
        throw std::invalid_argument("gqa_attention_decode: invalid KVCache tensor shape");
    }
    require_contiguous_nonnull(cache_k, "cache k");
    require_contiguous_nonnull(cache_v, "cache v");
}

} // namespace

void gqa_attention_prefill(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
                           KVCache& kv, int layer, Tensor& out, cudaStream_t stream) {
    (void)q;
    (void)k;
    (void)v;
    (void)scale;
    (void)kv;
    (void)layer;
    (void)out;
    (void)stream;
    throw std::invalid_argument("gqa_attention_prefill: unsupported until A-2 implements prefill");
}

void gqa_attention_decode(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                          float scale, KVCache& kv, int layer, Tensor& out, cudaStream_t stream) {
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention_decode: q/k/v/out must be BF16");
    }
    if (pos.dtype != DType::I32) {
        throw std::invalid_argument("gqa_attention_decode: pos must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument("gqa_attention_decode: scale must be 1/sqrt(256)");
    }

    require_shape(q, kHeadDim, kQHeads, 1, 1, "q");
    require_shape(k, kHeadDim, kKVHeads, 1, 1, "k");
    require_shape(v, kHeadDim, kKVHeads, 1, 1, "v");
    require_shape(out, kHeadDim, kQHeads, 1, 1, "out");
    require_shape(pos, 1, 1, 1, 1, "pos");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(pos, "pos");
    require_contiguous_nonnull(out, "out");
    validate_cache(kv, layer);

    detail::gqa_attention_decode_launch(q, k, v, pos, scale, kv, layer, out, stream);
}

} // namespace qus::kernels
