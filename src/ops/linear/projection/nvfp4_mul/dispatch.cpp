#include "ops/linear/projection/nvfp4_mul/dispatch.h"

#include "core/platform.h"
#include "ops/common/host/operands.h"

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {

Nvfp4MulRoute select_nvfp4_mul_route(int rows, int k, int tokens, LinearPolicy policy) {
    if (tokens <= 0 || !valid_linear_policy(policy))
        throw std::invalid_argument("NVFP4_MUL linear: invalid tokens or policy");
#if NINFER_TARGET_SPARK
    struct Shape {
        int rows, k;
        Nvfp4MulRoute (*select)(int);
    };

    constexpr std::array shapes{
        Shape{8192, 1024, select_nvfp4_mul_n8192_k1024},
        Shape{1024, 4096, select_nvfp4_mul_n1024_k4096},
        Shape{4096, 4096, select_nvfp4_mul_n4096_k4096},
        Shape{20480, 4096, select_nvfp4_mul_n20480_k4096},
        Shape{4096, 10240, select_nvfp4_mul_n4096_k10240},
    };
    for (const auto& shape : shapes)
        if (shape.rows == rows && shape.k == k) return shape.select(tokens);
#endif
    throw std::invalid_argument("NVFP4_MUL linear: unsupported weight geometry");
}

std::size_t nvfp4_mul_linear_workspace_capacity_bytes(int rows, int k, LinearPolicy policy,
                                                      int first, int last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("NVFP4_MUL linear workspace: invalid token interval");
    (void)select_nvfp4_mul_route(rows, k, first, policy);
    (void)select_nvfp4_mul_route(rows, k, last, policy);
    return 0;
}

void nvfp4_mul_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                        cudaStream_t stream) {
    (void)common::validate_nvfp4_mul_weight(weight, "NVFP4_MUL linear");
    select_nvfp4_mul_route(weight.n, weight.k, x.ne[1], policy).launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
