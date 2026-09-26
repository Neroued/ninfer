#include "ops/linear_add/q4/q4_linear_add_dispatch.h"

#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using GemvR1W8 =
    Q4A16GemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedByte2,
                      Q4GemvDecodeMode::ScalarInteger, Q4GemvCodeTransfer::SyncVector16,
                      Q4GemvScaleAccess::Scalar16Shuffle, Cache::ca, 6144 / 64, 1>;
using MmaR32T32  = Q4A16MmaSchedule<32, 32, 64, 16, 16, 3, 2, Q4MmaFragmentPipeline::Serial,
                                    Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32T64  = Q4A16MmaSchedule<32, 64, 64, 16, 32, 3, 2, Q4MmaFragmentPipeline::Serial,
                                    Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64T128 = Q4A16MmaSchedule<64, 128, 64, 64, 32, 2, 1, Q4MmaFragmentPipeline::Serial,
                                    Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

template <int Capacity>
void launch_sliced(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, 6144, Capacity>;
    auto* data     = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = static_cast<std::int64_t>(residual.nb[1] / sizeof(__nv_bfloat16));
    launch_q4_a16_sliced_k_mma<Schedule>(q4_linear_operands(x, w),
                                         LinearBf16StridedOutput{data, stride, 0},
                                         LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}

template <class Schedule>
void launch_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(out.data);
    const auto stride = static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16));
    launch_q4_a16_gemv<Schedule>(q4_linear_operands(x, w), LinearBf16StridedOutput{data, stride, 0},
                                 LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(out.data);
    const auto stride = static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16));
    launch_q4_a16_mma<Schedule>(q4_linear_operands(x, w), LinearBf16StridedOutput{data, stride, 0},
                                LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}
} // namespace

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (rows != 5120 || k != 6144 || tokens <= 0) {
        throw std::invalid_argument("q4 linear_add: unsupported shape or token extent");
    }
    if (tokens == 1) return launch_gemv<GemvR1W8>;
    if (tokens <= 4) return launch_sliced<4>;
    if (tokens <= 8) return launch_sliced<8>;
    if (tokens <= 16) return launch_sliced<16>;
    if (tokens <= 24) return launch_sliced<24>;
    if (tokens <= 32) return launch_sliced<32>;
    if (tokens <= 96) return launch_mma<MmaR32T32>;
    if (tokens <= 192) return launch_mma<MmaR32T64>;
    return launch_mma<MmaR64T128>;
}

} // namespace ninfer::ops::detail
