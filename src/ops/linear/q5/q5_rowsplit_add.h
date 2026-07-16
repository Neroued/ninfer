#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q5AddRoute {
    GemvResidual,
    Materialized,
    MmaR64C64Residual,
    MmaR64C128Residual,
};

Q5AddRoute q5_rowsplit_add_route(std::int32_t columns);
std::size_t q5_rowsplit_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                            std::int32_t columns);

void q5_rowsplit_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
