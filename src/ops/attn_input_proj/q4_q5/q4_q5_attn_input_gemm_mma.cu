#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, Tensor& out) {
    return RowSplitGroupedMmaJob{
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales),
        static_cast<__nv_bfloat16*>(out.data),
        weight.n,
        out.ne[0],
        0,
        weight.qtype == QType::Q5G64_F16S,
    };
}

template <class Cfg, RowSplitGroupedMmaCodec Codec>
void launch_pair(Q4KernelVariant variant, const Tensor& x, RowSplitGroupedMmaJob first,
                 RowSplitGroupedMmaJob second, cudaStream_t stream) {
    const int tiles = div_up(first.n, Cfg::BM) + div_up(second.n, Cfg::BM);
    const int cols  = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(div_up(cols, Cfg::BN)));
    RowSplitGroupedMmaJob empty{};

    if (variant == Q4KernelVariant::Full) {
        rowsplit_grouped_mma_kernel<Cfg, true, Codec, 2>
            <<<grid, Cfg::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), first,
                                                second, empty, empty, x.ne[0], cols, x.ne[0]);
    } else if (variant == Q4KernelVariant::Predicated) {
        rowsplit_grouped_mma_kernel<Cfg, false, Codec, 2>
            <<<grid, Cfg::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), first,
                                                second, empty, empty, x.ne[0], cols, x.ne[0]);
    } else {
        throw std::invalid_argument(
            "Q4/Q5 attention input grouped MMA requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_attn_input_grouped_mma_launch(Q4KernelVariant variant, const Tensor& x,
                                         const Weight& q_weight, const Weight& gate_weight,
                                         const Weight& k_weight, const Weight& v_weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true, true>;
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q4>(variant, x, make_job(q_weight, q),
                                                       make_job(k_weight, k), stream);
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q5>(variant, x, make_job(gate_weight, gate),
                                                       make_job(v_weight, v), stream);
}

} // namespace ninfer::ops::detail
