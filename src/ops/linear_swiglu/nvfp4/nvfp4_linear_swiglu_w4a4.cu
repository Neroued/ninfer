#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4MlpGateUpGeometry;
// Column tiles amortize gate/up decode over the complete speculative block.
using M64N128 = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 4, 2, 1>;
using M128N128 = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 4, 2, 1>;
using M96N128 = Nvfp4W4a4MmaSchedule<96, 128, 256, 3, 4, 2, 1>;

constexpr int kIntermediate = Geometry::kOutputRows / 2;

struct Nvfp4SwiGluRows {
    static constexpr bool kContiguous   = false;
    static constexpr int kRowsPerBranch = M64N128::kBlockN / 2;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row & (kRowsPerBranch - 1)) +
               (local_row >= kRowsPerBranch ? kIntermediate : 0);
    }
};

union Nvfp4SwiGluBf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

struct Nvfp4SwiGluOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ unsigned combine(unsigned gate_bits, unsigned up_bits) const {
        Nvfp4SwiGluBf16Pair gate{gate_bits};
        Nvfp4SwiGluBf16Pair up{up_bits};
        const float2 gate_values = __bfloat1622float2(gate.values);
        const float2 up_values   = __bfloat1622float2(up.values);
        Nvfp4SwiGluBf16Pair result;
        result.values = __floats2bfloat162_rn(silu(gate_values.x) * up_values.x,
                                              silu(gate_values.y) * up_values.y);
        return result.bits;
    }

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t token,
                                                      uint4 gate, uint4 up) const {
        const uint4 values = make_uint4(combine(gate.x, up.x), combine(gate.y, up.y),
                                        combine(gate.z, up.z), combine(gate.w, up.w));
        store_vec(data + static_cast<std::int64_t>(token) * kIntermediate + row, values);
    }
};

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    constexpr int kPairRows = Schedule::kBlockN / 2;
    static_assert(kPairRows == Nvfp4SwiGluRows::kRowsPerBranch);
    const dim3 grid(kIntermediate / kPairRows,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4SwiGluRows row_policy{};
    const Nvfp4SwiGluOutput output{static_cast<__nv_bfloat16*>(out.data)};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule, Nvfp4IdentityEpilogue, Nvfp4SwiGluOutput,
                          Nvfp4SwiGluRows, true><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output, row_policy);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
            cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4W4a4Workspace scratch =
        allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_nvfp4_w4a4_quantize(x, weight, scratch, stream);
    launch_gemm<Schedule>(weight, out, scratch, x.ne[1], stream);
}

} // namespace

void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.ne[1] <= M64N128::kBlockM) {
        launch<M64N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M96N128::kBlockM) {
        launch<M96N128>(x, weight, out, workspace, stream);
    } else {
        launch<M128N128>(x, weight, out, workspace, stream);
    }
}

} // namespace ninfer::ops::detail
