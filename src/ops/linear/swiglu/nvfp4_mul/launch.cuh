#pragma once

#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/swiglu/product_epilogue.cuh"

namespace ninfer::ops::detail {

template <class Schedule, class Product>
void nvfp4_mul_swiglu_gemv(const Nvfp4A16Operands& operands, LinearBf16Output output,
                           Product product, cudaStream_t stream) {
    launch_nvfp4_a16_gemv<Schedule>(operands, output, BasicSwiGluPairEpilogue<Product>{product},
                                    stream, SwiGluBranchRows<1>{});
}

template <class Schedule, class Product>
void nvfp4_mul_swiglu_simt(const Nvfp4A16Operands& operands, LinearBf16Output output,
                           Product product, cudaStream_t stream) {
    launch_nvfp4_a16_simt<Schedule>(operands, output, BasicSwiGluPairEpilogue<Product>{product},
                                    stream, SwiGluBranchRows<1>{});
}

template <class Schedule, class Product>
void nvfp4_mul_swiglu_sliced(const Nvfp4A16Operands& operands, LinearBf16Output output,
                             Product product, cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Schedule>(
        operands, output, BasicSwiGluPairEpilogue<Product>{product}, stream, SwiGluBranchRows<8>{});
}

template <class Schedule, class Product>
void nvfp4_mul_swiglu_mma(const Nvfp4A16Operands& operands, LinearBf16Output output,
                          Product product, cudaStream_t stream) {
    launch_nvfp4_a16_mma<Schedule>(operands, output,
                                   BasicSwiGluRowMajorMmaEpilogue<Product>{product}, stream,
                                   SwiGluBranchRows<Schedule::kBlockRows / 2>{});
}

} // namespace ninfer::ops::detail
