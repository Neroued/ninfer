#include "ops/linear_attention/gated_delta_net/control/kernels/mma_launch.cuh"
#include "ops/linear_attention/gated_delta_net/control/kernels/gemv.cuh"

#include <array>

namespace ninfer::ops::detail::gdn_control {
namespace {
constexpr int kLast = std::numeric_limits<std::int32_t>::max();

constexpr Plan mma(int tile, int split, RowMode rows = RowMode::Paired) {
    return {Kernel::Mma, tile, split, rows};
}

constexpr std::array kN96K2560{
    Route{1, 1, {Kernel::Gemv}},
    Route{2, 8, {Kernel::TiledGemv, 4}},
    Route{9, 192, mma(64, 8, RowMode::Merged)},
    Route{193, 448, mma(64, 8)},
    Route{449, 896, mma(128, 8)},
    Route{897, 1280, mma(64, 8)},
    Route{1281, 2688, mma(64, 4)},
    Route{2689, 5120, mma(64, 2)},
    Route{5121, 7168, mma(128, 1)},
    Route{7169, 8192, mma(64, 1)},
    Route{8193, kLast, mma(128, 1)},
};
static_assert(closed(kN96K2560));

template <int Tile, bool Paired>
void mma_n96_k2560(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                   CudaExecutionView e) {
    using Geometry = MmaGeometry<48, 2560, Tile, Paired>;
    bool launched  = false;
    switch (plan.split_k) {
    case 8:
        launched = launch_mma<Geometry, 8>(in, out, scratch, e);
        break;
    case 4:
        if constexpr (Paired && Tile == 64) {
            launched = launch_mma<Geometry, 4>(in, out, scratch, e);
            break;
        } else
            throw std::logic_error("gdn_control: invalid split4");
    case 2:
        if constexpr (Paired && Tile == 64) {
            launched = launch_mma<Geometry, 2>(in, out, scratch, e);
            break;
        } else
            throw std::logic_error("gdn_control: invalid split2");
    case 1:
        launched = launch_mma<Geometry, 1>(in, out, {}, e);
        break;
    default:
        throw std::logic_error("gdn_control: invalid 96x2560 plan");
    }
    if (!launched) (void)launch_mma<Geometry, 1>(in, out, {}, e);
}

} // namespace

std::span<const Route> routes_n96_k2560() { return kN96K2560; }

void launch_n96_k2560(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      CudaExecutionView e) {
    if (plan.kernel == Kernel::Gemv) {
        gemv_kernel<48, 2560><<<96, 256, 0, e.stream()>>>(
            static_cast<const __nv_bfloat16*>(in.x.data), static_cast<const __nv_bfloat16*>(in.a),
            static_cast<const __nv_bfloat16*>(in.b), static_cast<const float*>(in.a_log.data),
            static_cast<const float*>(in.dt_bias.data), static_cast<float*>(out.g.data),
            static_cast<float*>(out.beta.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (plan.kernel == Kernel::TiledGemv) {
        tiled_gemv_kernel<4, 256><<<dim3(96, (in.x.ne[1] + 3) / 4), 256, 0, e.stream()>>>(
            static_cast<const __nv_bfloat16*>(in.x.data), static_cast<const __nv_bfloat16*>(in.a),
            static_cast<const __nv_bfloat16*>(in.b), static_cast<const float*>(in.a_log.data),
            static_cast<const float*>(in.dt_bias.data), static_cast<float*>(out.g.data),
            static_cast<float*>(out.beta.data), in.x.ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (plan.rows == RowMode::Merged)
        mma_n96_k2560<64, false>(in, out, plan, scratch, e);
    else if (plan.tile_cols == 64)
        mma_n96_k2560<64, true>(in, out, plan, scratch, e);
    else
        mma_n96_k2560<128, true>(in, out, plan, scratch, e);
}
} // namespace ninfer::ops::detail::gdn_control
