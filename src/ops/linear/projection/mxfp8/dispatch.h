#pragma once
#include "ninfer/ops/linear.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"
#include <span>

namespace ninfer::ops::detail {
using Mxfp8LinearLaunch = void (*)(const Tensor&, const Weight&, Tensor&, Mxfp8A8Workspace,
                                   cudaStream_t);
enum class Mxfp8LinearCompute { Simt, A16, A8 };

struct Mxfp8LinearRoute {
    int last_tokens;
    Mxfp8LinearLaunch launch;
    Mxfp8LinearCompute compute;
    int code_token_alignment = 1;

    int code_tokens(int tokens) const {
        return mxfp8_a8_padded_tokens(tokens, code_token_alignment);
    }
};

struct Mxfp8LinearShape {
    int n, k;
    std::span<const Mxfp8LinearRoute> a16, a8;
};

const Mxfp8LinearRoute& select_mxfp8_linear_route(int n, int k, int tokens, LinearPolicy policy);
void mxfp8_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, MemoryArena*,
                    cudaStream_t);
std::size_t mxfp8_linear_workspace_capacity_bytes(int, int, LinearPolicy, int, int);
extern const Mxfp8LinearShape kMxfp8N12288K4096;
extern const Mxfp8LinearShape kMxfp8N4096K4096;
extern const Mxfp8LinearShape kMxfp8N4096K6144;
extern const Mxfp8LinearShape kMxfp8N77440K4096;
extern const Mxfp8LinearShape kMxfp8N2048K4096;
extern const Mxfp8LinearShape kMxfp8N4096K1024;
extern const Mxfp8LinearShape kMxfp8N2208K4096;
extern const Mxfp8LinearShape kMxfp8N2176K4096;
extern const Mxfp8LinearShape kMxfp8N640K4096;
extern const Mxfp8LinearShape kMxfp8N20480K1536;
extern const Mxfp8LinearShape kMxfp8N16384K1536;
extern const Mxfp8LinearShape kMxfp8N4096K16384;
extern const Mxfp8LinearShape kMxfp8N3072K1024;
extern const Mxfp8LinearShape kMxfp8N1024K1024;
} // namespace ninfer::ops::detail
