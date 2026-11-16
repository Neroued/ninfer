#include "ops/dynamic_grouped_conv/prepare/launch.h"
#include "ops/dynamic_grouped_conv/prepare/tma.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail::dynamic_grouped_conv::prepare {
namespace {
constexpr int kPrepareHidden = 4096;
constexpr PrepareGeometry kGeometry{kPrepareHidden};
constexpr int kPrepareGroups = kGeometry.groups();

template <class S>
struct GroupedCoefficients : Bf16ScheduleInstance<S, kPrepareHidden> {
    // Four adjacent coefficient rows belong to one group. Native P remains [side,tap,group].
    __device__ static int weight_row(int row) { return (row % 4) * kPrepareGroups + row / 4; }
};

struct InputConvolution {
    enum { kSlicedTile = true };

    template <class S>
    __device__ void finish_sliced_tile(FusedOutputs out, const float* coefficients, int row0,
                                       int token0, int, int tokens) const {
        constexpr int groups_per_tile   = S::kBlockRows / 4;
        constexpr int channels_per_tile = groups_per_tile * kGroupWidth;
        const int columns               = min(S::kBlockTokens, tokens - token0);
        for (int item = threadIdx.x; item < columns * channels_per_tile; item += S::kThreads) {
            const int column = item / channels_per_tile;
            const int token  = token0 + column;
            const int local  = item % channels_per_tile;
            const int g = local / kGroupWidth, h = row0 / 4 * kGroupWidth + local;
            const auto offset = std::int64_t(token) * kPrepareHidden + h;
            const auto c      = column * S::kBlockRows + 4 * g;
            float value       = (__bfloat162float(out.base[h]) + coefficients[c]) *
                          out.template normalized<kPrepareHidden>(token, h);
            if (token % out.width)
                value = fmaf(__bfloat162float(out.base[kPrepareHidden + h]) + coefficients[c + 1],
                             out.template normalized<kPrepareHidden>(token - 1, h), value);
            out.prepared[offset] = __float2bfloat16_rn(value);
        }
        for (int item = threadIdx.x; item < columns * groups_per_tile * 2; item += S::kThreads) {
            const int group   = item % groups_per_tile;
            const int tap     = (item / groups_per_tile) % 2;
            const int column  = item / (groups_per_tile * 2);
            const int token   = token0 + column;
            const auto offset = (std::int64_t(token) * 2 + tap) * kPrepareGroups + row0 / 4 + group;
            out.finish[offset] =
                __float2bfloat16_rn(coefficients[column * S::kBlockRows + group * 4 + 2 + tap]);
        }
    }
};

template <int Columns, int Warps, int WarpK>
void run(const NormSource& source, const Tensor& normalized, const Tensor& base,
         const Weight& projection, Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    using S = GroupedCoefficients<
        Bf16A16SlicedKMmaSchedule<16, Columns, Warps, WarpK, 2, Cache::cg, Cache::cg, 1, 7>>;
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(normalized.data),
                            static_cast<const __nv_bfloat16*>(projection.qdata),
                            kGeometry.coefficient_rows(), kPrepareHidden,
                            normalized.ne[1] * normalized.ne[2]};
    const FusedOutputs out{static_cast<const __nv_bfloat16*>(source.residual.data),
                           static_cast<const __nv_bfloat16*>(source.weight.data),
                           static_cast<const __nv_bfloat16*>(base.data),
                           static_cast<const float*>(source.inverse.data),
                           static_cast<__nv_bfloat16*>(prepared.data),
                           static_cast<__nv_bfloat16*>(finish.data),
                           normalized.ne[1]};
    launch_bf16_a16_sliced_k_mma<S>(p, out, InputConvolution{}, stream);
}
} // namespace

void launch_fused(const NormSource& source, const Tensor& normalized, const Tensor& base,
                  const Weight& projection, Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    const int columns = normalized.ne[1] * normalized.ne[2];
    if (columns <= 8)
        return run<8, 16, 32>(source, normalized, base, projection, prepared, finish, stream);
    if (columns <= 16)
        return run<16, 16, 32>(source, normalized, base, projection, prepared, finish, stream);
    if (columns <= 32)
        return run<32, 16, 32>(source, normalized, base, projection, prepared, finish, stream);
    if (columns <= 64)
        return run<64, 8, 32>(source, normalized, base, projection, prepared, finish, stream);
    if (columns <= 128)
        return launch_tma_fused<4096, 32, 32, 128, 16, 16, 2, true>(source, normalized, projection,
                                                                    base, prepared, finish, stream);
    if (columns <= 256)
        return launch_tma_fused<4096, 32, 64, 64, 16, 32, 3>(source, normalized, projection, base,
                                                             prepared, finish, stream);
    if (columns < 512)
        return launch_tma_fused<4096, 64, 64, 64, 32, 32, 2, true>(source, normalized, projection,
                                                                   base, prepared, finish, stream);
    if (columns >= 1024)
        return launch_tma_fused<4096, 64, 128, 64, 32, 32, 2, true, Bf16MmaRaster::RowFast, true>(
            source, normalized, projection, base, prepared, finish, stream);
    launch_tma_fused<4096, 64, 128, 64, 32, 32, 2, true>(source, normalized, projection, base,
                                                         prepared, finish, stream);
}
} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare
