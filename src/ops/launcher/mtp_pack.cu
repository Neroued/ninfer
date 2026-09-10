#include "ops/launcher/mtp_pack.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/mtp_pack.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void mtp_pack_fc_input_launch(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                              cudaStream_t stream) {
    constexpr int kBlock = 256;
    for_each_token_slice(embedding_norm.ne[1], 1, [&](int token_offset, int token_count) {
        const Tensor embedding_slice = embedding_norm.slice(1, token_offset, token_count);
        const Tensor hidden_slice    = hidden_norm.slice(1, token_offset, token_count);
        Tensor out_slice             = out.slice(1, token_offset, token_count);
        const dim3 grid(static_cast<unsigned int>(div_up(embedding_norm.ne[0], kBlock)),
                        static_cast<unsigned int>(token_count));
        mtp_pack_fc_input_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(embedding_slice.data),
            static_cast<const __nv_bfloat16*>(hidden_slice.data),
            static_cast<__nv_bfloat16*>(out_slice.data), embedding_norm.ne[0]);
        CUDA_CHECK(cudaGetLastError());
    });
}

namespace {

// A fused row kernel is bit-exact with ops::rmsnorm only if it lands on the SAME
// instantiation the standalone launcher would pick for that width: a different block width or a
// different pairs-per-thread splits the sum of squares differently, and FP32 addition is not
// associative. This mirrors the ladder in src/ops/launcher/rmsnorm.cu for the Offset epilogue,
// including its first branch, which intercepts d == 5120 before everything else. Widths the
// standalone launcher sends to its warp or generic kernels are declined here.
enum class MtpRowRoute { None, Cta256x6, Cta256x10 };

// Only the two registered MTP stem widths are admitted: 2048 for Qwen3.6-27B and 5120 for
// 35B-A3B. Everything else stays on the composed three-Op path, which serves it correctly --
// admitting a width no target asks for would be a route kept alive for a hypothetical one. Each
// admitted width lands on the instantiation src/ops/launcher/rmsnorm.cu would have picked for it,
// which is what makes the fused row bit-exact with ops::rmsnorm.
MtpRowRoute mtp_row_route(std::int32_t d) {
    if (d == 5120) { return MtpRowRoute::Cta256x10; }
    if (d == 2048) { return MtpRowRoute::Cta256x6; }
    return MtpRowRoute::None;
}

// The route table and the kernel's compile-time bound have to agree; if a future width is added to
// one and not the other the kernel would silently drop the trailing pairs from the sum and from
// the store. Checked rather than assumed.
void require_row_route(std::int32_t d, int block, int max_pairs) {
    const int pairs = d / 2;
    if (pairs % block != 0 || pairs / block < 1 || pairs / block > max_pairs) {
        throw std::invalid_argument("mtp fused row kernel: width outside the instantiated route");
    }
}

bool mtp_row_route_admits(std::int32_t d, std::uintptr_t pointer_bits) {
    if ((pointer_bits & (alignof(__nv_bfloat162) - 1)) != 0) { return false; }
    return mtp_row_route(d) != MtpRowRoute::None;
}

} // namespace

bool mtp_norm_pack_fc_input_admits(std::int32_t d, const Tensor& embedding,
                                   const Tensor& embedding_weight, const Tensor& hidden,
                                   const Tensor& hidden_weight, const Tensor& out) {
    return mtp_row_route_admits(d, reinterpret_cast<std::uintptr_t>(embedding.data) |
                                       reinterpret_cast<std::uintptr_t>(embedding_weight.data) |
                                       reinterpret_cast<std::uintptr_t>(hidden.data) |
                                       reinterpret_cast<std::uintptr_t>(hidden_weight.data) |
                                       reinterpret_cast<std::uintptr_t>(out.data));
}

void mtp_norm_pack_fc_input_launch(const Tensor& embedding, const Tensor& embedding_weight,
                                   const Tensor& hidden, const Tensor& hidden_weight, Tensor& out,
                                   float eps, cudaStream_t stream) {
    const std::int32_t d    = embedding.ne[0];
    const std::int64_t rows = embedding.ne[1];
    const dim3 grid(static_cast<unsigned int>(rows), 2);
    const auto* e  = reinterpret_cast<const __nv_bfloat162*>(embedding.data);
    const auto* ew = reinterpret_cast<const __nv_bfloat162*>(embedding_weight.data);
    const auto* h  = reinterpret_cast<const __nv_bfloat162*>(hidden.data);
    const auto* hw = reinterpret_cast<const __nv_bfloat162*>(hidden_weight.data);
    auto* o        = reinterpret_cast<__nv_bfloat162*>(out.data);
    switch (mtp_row_route(d)) {
    case MtpRowRoute::Cta256x6:
        require_row_route(d, 256, 6);
        mtp_norm_pack_fc_input_kernel<RmsEpilogue::Offset, 256, 6, true>
            <<<grid, 256, 0, stream>>>(e, ew, h, hw, o, d, rows, eps);
        break;
    case MtpRowRoute::Cta256x10:
        require_row_route(d, 256, 10);
        mtp_norm_pack_fc_input_kernel<RmsEpilogue::Offset, 256, 10, true>
            <<<grid, 256, 0, stream>>>(e, ew, h, hw, o, d, rows, eps);
        break;
    case MtpRowRoute::None:
        throw std::invalid_argument("mtp_norm_pack_fc_input: unsupported width");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool mtp_residual_norm_admits(std::int32_t d, const Tensor& delta, const Tensor& residual,
                              const Tensor& weight, const Tensor& out) {
    return mtp_row_route_admits(d, reinterpret_cast<std::uintptr_t>(delta.data) |
                                       reinterpret_cast<std::uintptr_t>(residual.data) |
                                       reinterpret_cast<std::uintptr_t>(weight.data) |
                                       reinterpret_cast<std::uintptr_t>(out.data));
}

void mtp_residual_norm_launch(const Tensor& delta, Tensor& residual, const Tensor& weight,
                              Tensor& out, float eps, cudaStream_t stream) {
    const std::int32_t d    = residual.ne[0];
    const std::int64_t rows = residual.ne[1];
    const auto* dv          = reinterpret_cast<const __nv_bfloat162*>(delta.data);
    auto* rv                = reinterpret_cast<__nv_bfloat162*>(residual.data);
    const auto* wv          = reinterpret_cast<const __nv_bfloat162*>(weight.data);
    auto* ov                = reinterpret_cast<__nv_bfloat162*>(out.data);
    const auto grid         = static_cast<unsigned int>(rows);
    switch (mtp_row_route(d)) {
    case MtpRowRoute::Cta256x6:
        require_row_route(d, 256, 6);
        mtp_residual_norm_kernel<RmsEpilogue::Offset, 256, 6, true>
            <<<grid, 256, 0, stream>>>(dv, rv, wv, ov, d, rows, eps);
        break;
    case MtpRowRoute::Cta256x10:
        require_row_route(d, 256, 10);
        mtp_residual_norm_kernel<RmsEpilogue::Offset, 256, 10, true>
            <<<grid, 256, 0, stream>>>(dv, rv, wv, ov, d, rows, eps);
        break;
    case MtpRowRoute::None:
        throw std::invalid_argument("mtp_residual_norm: unsupported width");
    }
    CUDA_CHECK(cudaGetLastError());
}

void mtp_split_attn_in_launch(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                              cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int64_t n = static_cast<std::int64_t>(attn_in.ne[0]) * attn_in.ne[1];
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
    mtp_split_attn_in_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(attn_in.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data), attn_in.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
