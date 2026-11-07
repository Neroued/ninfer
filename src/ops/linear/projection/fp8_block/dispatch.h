#pragma once
#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail {
using Fp8BlockLinearLaunch          = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
using Fp8BlockLinearWorkspaceLaunch = void (*)(const Tensor&, const Weight&, Tensor&, MemorySpan,
                                               cudaStream_t);

struct Fp8BlockLinearRoute {
    Fp8BlockLinearLaunch launch;
    bool tensor_core;
    Fp8BlockLinearWorkspaceLaunch workspace_launch = nullptr;
    std::size_t workspace_bytes                    = 0;
    std::size_t scratch_traffic_bytes              = 0;
};

struct Fp8BlockLinearShape {
    int n, k;
    Fp8BlockLinearRoute (*select)(int, LinearPolicy);
    std::size_t (*workspace_capacity)(int, int) = nullptr;
};

Fp8BlockLinearRoute select_fp8_block_linear_route(int n, int k, int tokens, LinearPolicy policy);
void fp8_block_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, MemoryArena*,
                        cudaStream_t);
std::size_t fp8_block_linear_workspace_capacity_bytes(int, int, LinearPolicy, int, int);
extern const Fp8BlockLinearShape kFp8BlockN2048K4096;
extern const Fp8BlockLinearShape kFp8BlockN512K4096;
extern const Fp8BlockLinearShape kFp8BlockN16384K1536;
extern const Fp8BlockLinearShape kFp8BlockN4096K16384;
extern const Fp8BlockLinearShape kFp8BlockN4096K1024;
} // namespace ninfer::ops::detail
