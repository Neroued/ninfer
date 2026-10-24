#include "ops/linear_attention/kimi_delta_attention/input_conv/kernels.cuh"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/common/numeric/math.cuh"
#include <algorithm>

namespace ninfer::ops::detail::kda_input {
namespace {
template <int Width>
void window(const Q8LinearOperands& operands, const Arguments& a, cudaStream_t stream) {
    const LinearBf16Output unused{a.q, hidden};
    const WindowEpilogue<Width> epilogue{a};
    if constexpr (Width == 1) {
        if (a.batch == 1) {
            launch_q8_a16_gemv<Q8A16GemvSchedule<4, 1, 2, 4096, 8>>(operands, unused, epilogue,
                                                                    stream);
            return;
        }
    }
    constexpr int columns = (16 / Width) * Width;
    using Schedule =
        Q8A16SlicedKMmaSchedule<16, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                Q8ActivationStage::PaddedZero, 4096, columns>;
    launch_q8_a16_sliced_k_mma<Schedule>(operands, unused, epilogue, stream);
}
} // namespace

void launch(const Tensor& x, const Weight& w, const Arguments& a, float* projected,
            cudaStream_t stream) {
    const auto operands = q8_linear_operands(x.view({hidden, a.width * a.batch}), w);
    using Window        = void (*)(const Q8LinearOperands&, const Arguments&, cudaStream_t);
    static constexpr Window windows[] = {window<1>, window<2>, window<3>, window<4>,
                                         window<5>, window<6>, window<7>, window<8>};
    if (!projected) {
        windows[a.width - 1](operands, a, stream);
        return;
    }
    if (a.width <= 8) {
        using Schedule = Q8A16MmaSchedule<32, 32, 64, 16, 16, 2, 2>;
        launch_q8_a16_mma<Schedule>(operands, FloatOutput{projected}, LinearIdentityEpilogue{},
                                    stream);
    } else if (operands.tokens <= 64) {
        using Schedule = Q8A16SlicedKMmaSchedule<32, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca,
                                                 Cache::cg, Q8ActivationStage::PaddedZero, 4096>;
        launch_q8_a16_sliced_k_mma<Schedule>(operands, FloatOutput{projected},
                                             LinearIdentityEpilogue{}, stream);
    } else if (operands.tokens <= 256) {
        using Schedule = Q8A16MmaSchedule<64, 64, 64, 16, 16, 2, 2>;
        launch_q8_a16_mma<Schedule>(operands, FloatOutput{projected}, LinearIdentityEpilogue{},
                                    stream);
    } else {
        using Schedule = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2>;
        launch_q8_a16_mma<Schedule>(operands, FloatOutput{projected}, LinearIdentityEpilogue{},
                                    stream);
    }
    conv_kernel<<<dim3(channels / 128, std::min<std::int64_t>((a.width + 3LL) / 4, 65535), a.batch),
                  128, 0, stream>>>(a, projected);
    CUDA_CHECK(cudaGetLastError());
    if (a.destination) {
        history_kernel<<<dim3(channels / 128, a.batch), 128, 0, stream>>>(a, projected);
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops::detail::kda_input
