#include "ops/linear/fp8/fp8_shapes.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_ksplit_mma.cuh"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"

namespace ninfer::ops::detail {
namespace {
template <int ActiveTokens>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Fp8N248320K5120;
    using Schedule = Fp8A16KSplitSchedule<(ActiveTokens <= 8 ? 16 : (ActiveTokens <= 24 ? 8 : 4)),
                                          ActiveTokens, ActiveTokens <= 8 ? 1 : 2>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_ksplit_mma_kernel<Geometry, ActiveTokens, Schedule, Fp8ContiguousOutput, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 8) return launch_tile<8>(x, weight, out, stream);
    if (tokens <= 16) return launch_tile<16>(x, weight, out, stream);
    if (tokens <= 24) return launch_tile<24>(x, weight, out, stream);
    if (tokens <= 32) return launch_tile<32>(x, weight, out, stream);
    if (tokens <= 40) return launch_tile<40>(x, weight, out, stream);
    if (tokens <= 48) return launch_tile<48>(x, weight, out, stream);
    throw std::logic_error("fp8 K-split exceeds shape capacity");
}

// Measured schedules for [248320,5120]. The 128-token
// schedule is the large-T computation core. The 64- and 96-token schedules avoid executing a
// mostly empty final token tile; dispatch emits at most one such tail launch.
using Main128 = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
using Tail64  = Fp8A16GemmSchedule<128, 64, 64, 64, 16, 2, 2>;
using Tail96  = Fp8A16GemmSchedule<64, 96, 64, 64, 16, 2, 2>;

template <class Schedule, bool FullTokens>
void launch_slice(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Fp8N248320K5120;
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    constexpr int row_tiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = div_up(x.ne[1], Schedule::kBlockTokens);
    const dim3 grid(static_cast<unsigned>(row_tiles), static_cast<unsigned>(token_tiles), 1U);
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
        <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_schedule(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::kBlockTokens,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor input = x.slice(1, offset, count);
                             Tensor output      = out.slice(1, offset, count);
                             if ((count % Schedule::kBlockTokens) == 0) {
                                 launch_slice<Schedule, true>(input, weight, output, stream);
                             } else {
                                 launch_slice<Schedule, false>(input, weight, output, stream);
                             }
                         });
}

void launch_tail(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] < 42) {
        launch_ksplit(x, weight, out, stream);
    } else if (x.ne[1] <= Tail64::kBlockTokens) {
        launch_schedule<Tail64>(x, weight, out, stream);
    } else {
        launch_schedule<Tail96>(x, weight, out, stream);
    }
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] < 42) return launch_ksplit(x, weight, out, stream);

    const std::int32_t tokens = x.ne[1];
    if (tokens <= Tail64::kBlockTokens) {
        launch_schedule<Tail64>(x, weight, out, stream);
        return;
    }
    if (tokens <= Tail96::kBlockTokens) {
        launch_schedule<Tail96>(x, weight, out, stream);
        return;
    }
    if (tokens <= Main128::kBlockTokens) {
        launch_schedule<Main128>(x, weight, out, stream);
        return;
    }

    // Two early packing intervals are faster as whole 96-token GEMMs than as a 128-token prefix
    // plus a tail launch. Beyond 384 tokens the main schedule's higher full-tile throughput has
    // amortized this packing effect.
    if ((tokens >= 161 && tokens <= 192) || (tokens >= 257 && tokens <= 288)) {
        launch_schedule<Tail96>(x, weight, out, stream);
        return;
    }

    const std::int32_t tail = tokens % Main128::kBlockTokens;
    if (tail == 0 || tail > Tail96::kBlockTokens) {
        launch_schedule<Main128>(x, weight, out, stream);
        return;
    }

    const std::int32_t prefix = tokens - tail;
    const Tensor input_prefix = x.slice(1, 0, prefix);
    Tensor output_prefix      = out.slice(1, 0, prefix);
    launch_schedule<Main128>(input_prefix, weight, output_prefix, stream);

    const Tensor input_tail = x.slice(1, prefix, tail);
    Tensor output_tail      = out.slice(1, prefix, tail);
    launch_tail(input_tail, weight, output_tail, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N248320K5120{248320, 5120, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
