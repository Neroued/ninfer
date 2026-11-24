#pragma once
#include "ops/linear/add/dispatch.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"

namespace ninfer::ops::detail {

struct Fp8LinearAddPlan {
    LinearAddLaunch a16             = nullptr;
    void (*quantized)(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace,
                      cudaStream_t) = nullptr;
    std::size_t partial_bytes       = 0;
};

Fp8LinearAddPlan select_fp8_add_n5120_k6144(LinearPolicy policy, std::int32_t tokens);
Fp8LinearAddPlan select_fp8_add_n5120_k17408(LinearPolicy policy, std::int32_t tokens);
std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t rows, std::int32_t k,
                                                    LinearPolicy policy, std::int32_t first,
                                                    std::int32_t last);
void fp8_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, MemoryArena&,
                             cudaStream_t);
} // namespace ninfer::ops::detail
