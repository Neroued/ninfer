// qus::kernels - int8 KV append quantization launcher.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_kv_quant.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <stdexcept>
#include <string>

namespace qus::kernels::detail {
namespace {

void require_quant_append_shape(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1,
                                std::int32_t n2, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("gqa_attention_kv_quantize_append: invalid ") +
                                    name);
    }
}

} // namespace

void gqa_attention_kv_quantize_append_launch(const Tensor& k, const Tensor& v,
                                             const Tensor& positions, KVCache& kv, int layer,
                                             bool positions_are_base, cudaStream_t stream) {
    if (layer < 0 || static_cast<std::uint32_t>(layer) >= kv.layer_count()) {
        throw std::invalid_argument("gqa_attention_kv_quantize_append: layer out of range");
    }
    if (kv.dtype != DType::I8 || kv.quant_group != kGqaKvQuantGroup ||
        kv.num_kv_heads != kGqaKvQuantKVHeads || kv.head_dim != kGqaKvQuantHeadDim) {
        throw std::invalid_argument(
            "gqa_attention_kv_quantize_append: KVCache must be I8 group-64");
    }

    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    require_quant_append_shape(k, DType::BF16, kGqaKvQuantHeadDim, kGqaKvQuantKVHeads, tokens, "k");
    require_quant_append_shape(v, DType::BF16, kGqaKvQuantHeadDim, kGqaKvQuantKVHeads, tokens, "v");
    require_quant_append_shape(positions, DType::I32, tokens, 1, 1, "positions");
    if (tokens <= 0) {
        throw std::invalid_argument("gqa_attention_kv_quantize_append: tokens must be positive");
    }

    Tensor& cache_k                   = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v                   = kv.v[static_cast<std::uint32_t>(layer)];
    Tensor& cache_k_scale             = kv.k_scale[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v_scale             = kv.v_scale[static_cast<std::uint32_t>(layer)];
    const std::int32_t padded_context = static_cast<std::int32_t>(kv.padded_context);
    const std::int32_t max_context    = static_cast<std::int32_t>(kv.max_context);

    const dim3 grid(kGqaKvQuantGroups, kGqaKvQuantKVHeads, tokens);
    gqa_attention_kv_quantize_append_kernel<<<grid, kGqaKvQuantBlockSize, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<std::int8_t*>(cache_k.data),
        static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
        static_cast<__half*>(cache_v_scale.data), tokens, padded_context, max_context,
        positions_are_base ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
