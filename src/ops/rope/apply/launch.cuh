#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ops/rope/kernels/rotate.cuh"
#include "ops/rope/kernels/axial64.cuh"

namespace ninfer::ops::detail {

struct RopeArguments {
    const Tensor& positions;
    Tensor& first;
    Tensor* second;
    int rotary_dim;
    float theta;

    [[nodiscard]] __nv_bfloat16* second_data() const {
        return second ? static_cast<__nv_bfloat16*>(second->data) : nullptr;
    }

    [[nodiscard]] std::int64_t second_stride() const { return second ? second->nb[2] / 2 : 0; }
};

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_rope_fixed(const RopeArguments& a, int threads, cudaStream_t stream) {
    rope_fixed_kernel<Mode, QHeads, KHeads><<<a.positions.ne[0], threads, 0, stream>>>(
        static_cast<const std::int32_t*>(a.positions.data),
        static_cast<__nv_bfloat16*>(a.first.data), a.second_data(), a.positions.ne[0],
        a.first.nb[2] / 2, a.second_stride());
}

template <int HeadsPerBlock, int QHeads, int KHeads>
void launch_rope_split(const RopeArguments& a, int threads, cudaStream_t stream) {
    constexpr int Groups = (QHeads + KHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    rope_fixed_split_kernel<RopeKernelMode::D128R128, QHeads, KHeads, HeadsPerBlock>
        <<<a.positions.ne[0] * Groups, threads, 0, stream>>>(
            static_cast<const std::int32_t*>(a.positions.data),
            static_cast<__nv_bfloat16*>(a.first.data), a.second_data(), a.positions.ne[0],
            a.first.nb[2] / 2, a.second_stride());
}

template <int Block>
void launch_rope_axial_vector(const RopeArguments& a, int, cudaStream_t stream) {
    rope_axial_vector_kernel<Block><<<a.positions.ne[0], Block, 0, stream>>>(
        static_cast<const std::int32_t*>(a.positions.data),
        static_cast<__nv_bfloat16*>(a.first.data), a.second_data(), a.positions.ne[0],
        a.first.nb[2] / 2, a.second_stride());
}

inline void launch_rope_generic(const RopeArguments& a, int threads, cudaStream_t stream) {
    rope_generic_kernel<<<a.positions.ne[0], threads, 0, stream>>>(
        static_cast<const std::int32_t*>(a.positions.data), a.positions.ne[1],
        static_cast<__nv_bfloat16*>(a.first.data), a.second_data(), a.first.ne[0], a.rotary_dim,
        a.theta, a.first.ne[1], a.second ? a.second->ne[1] : 0, a.positions.ne[0],
        a.first.nb[2] / 2, a.second_stride());
}

template <int HeadsPerBlock>
void launch_rope_axial64(const RopeArguments& a, int, cudaStream_t stream) {
    launch_axial_rope64<false, HeadsPerBlock>(
        {static_cast<const int*>(a.positions.data), nullptr, nullptr,
         static_cast<__nv_bfloat16*>(a.first.data), a.second_data(), a.first.nb[2] / 2,
         a.second_stride(), a.positions.ne[0], a.first.ne[1], a.second ? a.second->ne[1] : 0,
         -std::log2(a.theta) / 16.F, 0},
        stream);
}

} // namespace ninfer::ops::detail
