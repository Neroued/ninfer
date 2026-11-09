#include "ops/dynamic_grouped_conv/prepare/launch.h"
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

struct Outputs {
    const __nv_bfloat16* normalized;
    const __nv_bfloat16* base;
    __nv_bfloat16* prepared;
    __nv_bfloat16* finish;
    int width;
};

struct InputConvolution {
    enum { kSlicedTile = true };

    template <class S>
    __device__ void finish_sliced_tile(Outputs out, const float* coefficients, int row0, int token0,
                                       int, int tokens) const {
        constexpr int groups_per_tile   = S::kBlockRows / 4;
        constexpr int channels_per_tile = groups_per_tile * kGroupWidth;
        for (int item = threadIdx.x; item < tokens * channels_per_tile; item += S::kThreads) {
            const int token = item / channels_per_tile;
            const int local = item % channels_per_tile;
            const int g = local / kGroupWidth, h = row0 / 4 * kGroupWidth + local;
            const auto offset = std::int64_t(token) * kPrepareHidden + h;
            const auto c      = (token - token0) * S::kBlockRows + 4 * g;
            float value       = (__bfloat162float(out.base[h]) + coefficients[c]) *
                          __bfloat162float(out.normalized[offset]);
            if (token % out.width)
                value = fmaf(__bfloat162float(out.base[kPrepareHidden + h]) + coefficients[c + 1],
                             __bfloat162float(out.normalized[offset - kPrepareHidden]), value);
            out.prepared[offset] = __float2bfloat16_rn(value);
        }
        for (int item = threadIdx.x; item < tokens * groups_per_tile * 2; item += S::kThreads) {
            const int group    = item % groups_per_tile;
            const int tap      = (item / groups_per_tile) % 2;
            const int token    = item / (groups_per_tile * 2);
            const auto offset  = (token * 2 + tap) * kPrepareGroups + row0 / 4 + group;
            out.finish[offset] = __float2bfloat16_rn(
                coefficients[(token - token0) * S::kBlockRows + group * 4 + 2 + tap]);
        }
    }
};

template <int Columns, int Warps, int WarpK>
void run(const Tensor& normalized, const Tensor& base, const Weight& projection, Tensor& prepared,
         Tensor& finish, cudaStream_t stream) {
    using S = GroupedCoefficients<
        Bf16A16SlicedKMmaSchedule<16, Columns, Warps, WarpK, 2, Cache::cg, Cache::cg, 1, 7>>;
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(normalized.data),
                            static_cast<const __nv_bfloat16*>(projection.qdata),
                            kGeometry.coefficient_rows(), kPrepareHidden,
                            normalized.ne[1] * normalized.ne[2]};
    const Outputs out{p.x, static_cast<const __nv_bfloat16*>(base.data),
                      static_cast<__nv_bfloat16*>(prepared.data),
                      static_cast<__nv_bfloat16*>(finish.data), normalized.ne[1]};
    launch_bf16_a16_sliced_k_mma<S>(p, out, InputConvolution{}, stream);
}
} // namespace

void launch_fused(const Tensor& normalized, const Tensor& base, const Weight& projection,
                  Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    const int columns = normalized.ne[1] * normalized.ne[2];
    if (columns <= 8) return run<8, 16, 32>(normalized, base, projection, prepared, finish, stream);
    if (columns <= 16)
        return run<16, 16, 32>(normalized, base, projection, prepared, finish, stream);
    if (columns <= 32)
        return run<32, 16, 32>(normalized, base, projection, prepared, finish, stream);
    if (columns <= 64)
        return run<64, 8, 32>(normalized, base, projection, prepared, finish, stream);
    return run<128, 8, 16>(normalized, base, projection, prepared, finish, stream);
}
} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare
