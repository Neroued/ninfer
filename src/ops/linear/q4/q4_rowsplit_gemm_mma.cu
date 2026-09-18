#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using Q4MmaR64C32Schedule =
    Q4RowSplitMmaGemmSchedule<64, 32, 64, 16, 8, 2, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C48Schedule =
    Q4RowSplitMmaGemmSchedule<64, 48, 64, 16, 16, 2, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C56Schedule =
    Q4RowSplitMmaGemmSchedule<64, 56, 64, 32, 8, 2, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C64Schedule =
    Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 3, Q4FragmentPipeline::PingPong, Cache::ca,
                              Cache::ca, Q4ScaleLoad::Scalar16>;


using Q4MmaR64C72Schedule =
    Q4RowSplitMmaGemmSchedule<64, 72, 64, 32, 24, 2, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C80Schedule =
    Q4RowSplitMmaGemmSchedule<64, 80, 64, 16, 40, 2, 1, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C96Schedule =
    Q4RowSplitMmaGemmSchedule<64, 96, 64, 32, 16, 2, 1, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;


using Q4MmaR64C112PartialSchedule =
    Q4RowSplitMmaGemmSchedule<64, 112, 64, 32, 16, 2, 1, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C112Schedule =
    Q4RowSplitMmaGemmSchedule<64, 112, 64, 16, 112, 2, 1, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C120PartialSchedule =
    Q4RowSplitMmaGemmSchedule<64, 120, 64, 32, 24, 2, 1, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C120Schedule =
    Q4RowSplitMmaGemmSchedule<64, 120, 64, 16, 120, 2, 1, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

using Q4MmaR64C128Schedule =
    Q4RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;


} // namespace

void launch_q4_mma_r64_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_mma<Q4MmaR64C32Schedule>(x, w, out, stream);
}

void launch_q4_mma_r64_c48(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_mma<Q4MmaR64C48Schedule>(x, w, out, stream);
}

void launch_q4_mma_r64_c56(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_mma<Q4MmaR64C56Schedule>(x, w, out, stream);
}

void launch_q4_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_mma<Q4MmaR64C64Schedule>(x, w, out, stream);
}

void launch_q4_mma_r64_c72(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_mma<Q4MmaR64C72Schedule>(x, w, out, stream);
}

void launch_q4_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_mma<Q4MmaR64C80Schedule>(x, w, out, stream);
}

void launch_q4_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_mma<Q4MmaR64C96Schedule>(x, w, out, stream);
}

void launch_q4_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // Complete and partial column tiles favor different fragment mappings.
    if (x.ne[1] % 112 == 0) {
        launch_q4_mma<Q4MmaR64C112Schedule>(x, w, out, stream);
    } else {
        launch_q4_mma<Q4MmaR64C112PartialSchedule>(x, w, out, stream);
    }
}

void launch_q4_mma_r64_c120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // Complete and partial column tiles favor different fragment mappings.
    if (x.ne[1] % 120 == 0) {
        launch_q4_mma<Q4MmaR64C120Schedule>(x, w, out, stream);
    } else {
        launch_q4_mma<Q4MmaR64C120PartialSchedule>(x, w, out, stream);
    }
}

void launch_q4_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_mma<Q4MmaR64C128Schedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
