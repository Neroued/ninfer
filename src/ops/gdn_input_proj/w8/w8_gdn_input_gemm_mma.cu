#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr int kRows   = 12288;
constexpr int kHidden = 2048;
using Output          = W8SplitOutput2<8192, 4096>;
using Schedule        = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;

template <bool Full>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    static_assert((8192 % Schedule::BM) == 0 && (4096 % Schedule::BM) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    const dim3 grid(kRows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    constexpr std::size_t kDynamicBytes = w8_rowsplit_gemm_mma_dynamic_bytes<Schedule>();
    if constexpr (kDynamicBytes > kW8SmallTMmaStaticSharedBytes) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::Store, Output>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
        CUDA_CHECK(attribute);
    }
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, kDynamicBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, kRows, kHidden, x.ne[1],
            kHidden);
}

} // namespace

void w8_gdn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream) {
    if ((x.ne[1] % Schedule::BN) == 0) {
        launch_variant<true>(x, weight, qkv, z, stream);
    } else {
        launch_variant<false>(x, weight, qkv, z, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
