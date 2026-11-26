#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n3072_k1024.cuh"

#include <type_traits>

namespace ninfer::ops::detail::linear_split {
namespace {

template <class Output>
struct Bf16SplitBiasRoutes {
    Bf16A16Operands operands;
    Output output;
    LinearBiasEpilogue epilogue;
    cudaStream_t stream;

    template <class S>
    void gemv() const {
        launch_bf16_a16_gemv<S>(operands, output, epilogue, stream);
    }

    template <class S>
    void sliced() const {
        launch_bf16_a16_sliced_k_mma<S>(operands, output, epilogue, stream);
    }

    template <class S>
    void mma() const {
        launch_bf16_a16_mma<S>(operands, output, epilogue, stream);
    }

    template <class S, class Epilogue = LinearIdentityEpilogue>
    void tma() const {
        if constexpr (std::is_same_v<Epilogue, LinearIdentityEpilogue>) {
            launch_bf16_a16_tma_mma<S>(operands, output, epilogue, stream);
        } else {
            static_assert(std::is_same_v<Epilogue, Bf16MmaWarpStoreEpilogue> ||
                          std::is_same_v<Epilogue, Bf16TmaVectorEpilogue>);
            launch_bf16_a16_tma_mma<S>(operands, output,
                                       Bf16BiasWarpStoreEpilogue{{}, epilogue.bias}, stream);
        }
    }
};

void launch_split_bias(const Tensor& x, const Weight& weight, const Tensor& bias,
                       std::span<const Tensor> outputs, LinearPolicy, MemoryArena*,
                       cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<1024, 1024, 1024>;
    const Output output{{static_cast<__nv_bfloat16*>(outputs[0].data),
                         static_cast<__nv_bfloat16*>(outputs[1].data),
                         static_cast<__nv_bfloat16*>(outputs[2].data)}};
    const auto operands = bf16_a16_operands(x, weight);
    bf16_n3072_k1024::select(
        operands.tokens,
        Bf16SplitBiasRoutes<Output>{
            operands, output, {static_cast<const __nv_bfloat16*>(bias.data)}, stream});
}

} // namespace

const SplitBiasShape kBf16SplitBiasN3072K1024{3072, 1024, kSplitRows<1024, 1024, 1024>,
                                              launch_split_bias};

} // namespace ninfer::ops::detail::linear_split
