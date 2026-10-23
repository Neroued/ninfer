#include "ops/rope/apply/dispatch.h"
#include "ops/common/host/operands.h"
#include "ops/rope/apply/launch.cuh"
#include "core/platform.h"

#include <algorithm>

namespace ninfer::ops::detail {
namespace {

struct Selection {
    void (*launch)(const RopeArguments&, int, cudaStream_t);
    int threads;
};

bool pair_aligned(const Tensor& x) {
    return common::operand::aligned(x.data, 4) && x.nb[2] % 4 == 0;
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
Selection select_fixed(int tokens, int multiprocessors) {
    int threads = 128;
    if constexpr (Mode != RopeKernelMode::AxialD72R72) {
        if (tokens <= 6)
            threads = (QHeads + KHeads) * 32;
        else if (tokens <= static_cast<std::int64_t>(multiprocessors) * 6)
            threads = 256;
        else if (tokens <= 1024)
            threads = 192;
        threads = std::min(threads, std::min((QHeads + KHeads) * 32, 1024));
    }
    return {launch_rope_fixed<Mode, QHeads, KHeads>, threads};
}

template <int QHeads, int KHeads>
Selection select_r64(int axes, int tokens, int multiprocessors) {
    if (axes == 1)
        return select_fixed<RopeKernelMode::D256R64, QHeads, KHeads>(tokens, multiprocessors);
    return select_fixed<RopeKernelMode::MropeD256R64, QHeads, KHeads>(tokens, multiprocessors);
}

Selection select_rope(const RopeArguments& a, int multiprocessors) {
    const Selection generic{launch_rope_generic, 128};
    if (!pair_aligned(a.first) || (a.second && !pair_aligned(*a.second))) return generic;
    const int axes         = a.positions.ne[1];
    const int tokens       = a.positions.ne[0];
    const int first_heads  = a.first.ne[1];
    const int second_heads = a.second ? a.second->ne[1] : 0;

    if (axes == 2 && a.rotary_dim == 64) {
        if (tokens <= 2) return {launch_rope_axial64<2>, 32};
        if ((tokens >= 32 && tokens <= 64) || tokens >= 8192) return {launch_rope_axial64<16>, 256};
        return {launch_rope_axial64<8>, 128};
    }

    if (axes == 1 && a.first.ne[0] == 128 && a.rotary_dim == 128 && a.theta == 1.0e7F) {
        if (first_heads == 32 && second_heads == 8) {
            if (tokens <= 16) return {launch_rope_split<5, 32, 8>, 160};
            if (tokens <= 400) return {launch_rope_split<8, 32, 8>, 256};
            return {launch_rope_fixed<RopeKernelMode::D128R128, 32, 8>, 160};
        }
        if (!a.second) {
            if (first_heads == 32)
                return select_fixed<RopeKernelMode::D128R128, 32, 0>(tokens, multiprocessors);
            if (first_heads == 8)
                return select_fixed<RopeKernelMode::D128R128, 8, 0>(tokens, multiprocessors);
        }
    }
    if (a.rotary_dim == 64 && a.theta == 1.0e7F && (axes == 1 || axes == 3)) {
        if (a.second) {
            if (first_heads == 24 && second_heads == 4)
                return select_r64<24, 4>(axes, tokens, multiprocessors);
            if (first_heads == 16 && second_heads == 2)
                return select_r64<16, 2>(axes, tokens, multiprocessors);
        } else {
            switch (first_heads) {
            case 24:
                return select_r64<24, 0>(axes, tokens, multiprocessors);
            case 4:
                return select_r64<4, 0>(axes, tokens, multiprocessors);
            case 16:
                return select_r64<16, 0>(axes, tokens, multiprocessors);
            case 2:
                return select_r64<2, 0>(axes, tokens, multiprocessors);
            }
        }
    }
    if (axes == 2 && a.rotary_dim == 72 && a.theta == 10'000.0F && first_heads == 16) {
        if (a.second && second_heads == 16) {
#if NINFER_TARGET_SPARK
            if (tokens < 32) return {launch_rope_fixed<RopeKernelMode::AxialD72R72, 16, 16>, 512};
            if (common::operand::aligned(a.first.data, 8) &&
                common::operand::aligned(a.second->data, 8) && a.first.nb[2] % 8 == 0 &&
                a.second->nb[2] % 8 == 0)
                return {launch_rope_axial_vector<256>, 256};
            return {launch_rope_fixed<RopeKernelMode::AxialD72R72, 16, 16>, 256};
#else
            return select_fixed<RopeKernelMode::AxialD72R72, 16, 16>(tokens, multiprocessors);
#endif
        }
        if (!a.second)
            return select_fixed<RopeKernelMode::AxialD72R72, 16, 0>(tokens, multiprocessors);
    }
    return generic;
}

void apply(const RopeArguments& arguments, CudaExecutionView execution) {
    const auto selected = select_rope(arguments, execution.multiprocessor_count());
    selected.launch(arguments, selected.threads, execution.stream());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                   CudaExecutionView execution) {
    apply({positions, q, &k, rotary_dim, theta}, execution);
}

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                   CudaExecutionView execution) {
    apply({positions, x, nullptr, rotary_dim, theta}, execution);
}

} // namespace ninfer::ops::detail
