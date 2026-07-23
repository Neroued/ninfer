// ninfer::ops - rope launcher: private token-count tuning and generic fallback.
#include "ops/launcher/rope.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/rope.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kLargeBlock               = 256;
constexpr int kFullChunkBlock           = 192;
constexpr int kSmallBlock               = 128;
constexpr int kDefaultChunkTargetTokens = 1024;
// RTX 5090 has 170 SMs and admits six of these 256-thread CTAs per SM.
constexpr int kLargeBlockWaveCapacity = 1020;

template <RopeKernelMode Mode>
inline constexpr bool kTextMode =
    Mode == RopeKernelMode::Text1D || Mode == RopeKernelMode::TextMrope ||
    Mode == RopeKernelMode::DflashText1D;

std::int64_t token_stride(const Tensor* tensor) {
    return tensor == nullptr ? 0 : tensor->nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

bool bf16x2_aligned(const Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data) & (alignof(__nv_bfloat162) - 1)) == 0 &&
           tensor.nb[2] % static_cast<std::int64_t>(alignof(__nv_bfloat162)) == 0;
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed_block(const Tensor& positions, Tensor* q, Tensor* k, int block,
                        cudaStream_t stream) {
    const int tokens = positions.ne[0];
    rope_fixed_kernel<Mode, QHeads, KHeads><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
        token_stride(k));
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream) {
    const int tokens = positions.ne[0];
    int block        = kSmallBlock;
    if constexpr (kTextMode<Mode>) {
        if (tokens <= 6) {
            block = (QHeads + KHeads) * 32;
        } else if (tokens <= kLargeBlockWaveCapacity) {
            block = kLargeBlock;
        } else if (tokens <= kDefaultChunkTargetTokens) {
            block = kFullChunkBlock;
        }
        const int head_warps = (QHeads + KHeads) * 32;
        if (block > head_warps) { block = head_warps; }
        if (block > 1024) { block = 1024; }
    }
    launch_fixed_block<Mode, QHeads, KHeads>(positions, q, k, block, stream);
}

template <int HeadsPerBlock, int QHeads, int KHeads>
void launch_dflash_split(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream) {
    constexpr int kGroups = (QHeads + KHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int kBlock  = HeadsPerBlock <= 2 ? 64 : HeadsPerBlock * 32;
    const int tokens      = positions.ne[0];
    rope_fixed_split_kernel<RopeKernelMode::DflashText1D, QHeads, KHeads, HeadsPerBlock>
        <<<tokens * kGroups, kBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data),
            q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
            k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
            token_stride(k));
}

bool launch_fixed_pair(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                       cudaStream_t stream) {
    if (!bf16x2_aligned(q) || !bf16x2_aligned(k)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && q.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F && q.ne[1] == 32 &&
        k.ne[1] == 8) {
        const int tokens = positions.ne[0];
        if (tokens <= 16) {
            launch_dflash_split<5, 32, 8>(positions, &q, &k, stream);
        } else if (tokens <= 400) {
            launch_dflash_split<8, 32, 8>(positions, &q, &k, stream);
        } else {
            launch_fixed_block<RopeKernelMode::DflashText1D, 32, 8>(positions, &q, &k, 160, stream);
        }
        return true;
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (q.ne[1] == 24 && k.ne[1] == 4) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 24, 4>(positions, &q, &k, stream);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 24, 4>(positions, &q, &k, stream);
                return true;
            }
        }
        if (q.ne[1] == 16 && k.ne[1] == 2) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 16, 2>(positions, &q, &k, stream);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 16, 2>(positions, &q, &k, stream);
                return true;
            }
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && q.ne[1] == 16 && k.ne[1] == 16) {
        launch_fixed<RopeKernelMode::Vision2D, 16, 16>(positions, &q, &k, stream);
        return true;
    }
    return false;
}

template <RopeKernelMode Mode, int Heads>
void launch_fixed_single(const Tensor& positions, Tensor& x, cudaStream_t stream) {
    launch_fixed<Mode, Heads, 0>(positions, &x, nullptr, stream);
}

template <int Heads>
bool launch_text_single(const Tensor& positions, int axes, Tensor& x, cudaStream_t stream) {
    if (x.ne[1] != Heads) { return false; }
    if (axes == 1) {
        launch_fixed_single<RopeKernelMode::Text1D, Heads>(positions, x, stream);
        return true;
    }
    if (axes == 3) {
        launch_fixed_single<RopeKernelMode::TextMrope, Heads>(positions, x, stream);
        return true;
    }
    return false;
}

bool launch_fixed_single_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                                  cudaStream_t stream) {
    if (!bf16x2_aligned(x)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && x.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F) {
        if (x.ne[1] == 32) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 32>(positions, x, stream);
            return true;
        }
        if (x.ne[1] == 8) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 8>(positions, x, stream);
            return true;
        }
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (launch_text_single<24>(positions, axes, x, stream) ||
            launch_text_single<4>(positions, axes, x, stream) ||
            launch_text_single<16>(positions, axes, x, stream) ||
            launch_text_single<2>(positions, axes, x, stream)) {
            return true;
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && x.ne[1] == 16) {
        launch_fixed_single<RopeKernelMode::Vision2D, 16>(positions, x, stream);
        return true;
    }
    return false;
}

void launch_generic(const Tensor& positions, int rotary_dim, float theta, Tensor* q, Tensor* k,
                    cudaStream_t stream) {
    constexpr int block = 128;
    Tensor& sample      = q != nullptr ? *q : *k;
    const int tokens    = sample.ne[2];
    rope_generic_kernel<<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), positions.ne[1],
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), sample.ne[0], rotary_dim,
        theta, q == nullptr ? 0 : q->ne[1], k == nullptr ? 0 : k->ne[1], tokens, token_stride(q),
        token_stride(k));
}

} // namespace

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream) {
    if (!launch_fixed_pair(positions, rotary_dim, theta, q, k, stream)) {
        launch_generic(positions, rotary_dim, theta, &q, &k, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        cudaStream_t stream) {
    if (!launch_fixed_single_dispatch(positions, rotary_dim, theta, x, stream)) {
        launch_generic(positions, rotary_dim, theta, &x, nullptr, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
