#include "core/weight.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_rowsplit_gemm_mma.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr int kRows   = 12288;
constexpr int kHidden = 2048;
using Output          = Q8SplitOutput2<8192, 4096>;
using Schedule        = Q8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;

template <bool Full>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    static_assert((8192 % Schedule::BM) == 0 && (4096 % Schedule::BM) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    const dim3 grid(kRows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(weight.qdata),
                                                 static_cast<const std::uint8_t*>(weight.scales),
                                                 output, kRows, kHidden, x.ne[1], kHidden);
}

} // namespace

void q8_gdn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream) {
    if ((x.ne[1] % Schedule::BN) == 0) {
        launch_variant<true>(x, weight, qkv, z, stream);
    } else {
        launch_variant<false>(x, weight, qkv, z, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
