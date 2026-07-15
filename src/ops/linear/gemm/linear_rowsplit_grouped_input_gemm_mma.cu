#include "ops/linear/gemm/linear_rowsplit_grouped_input_gemm_mma.cuh"
#include "ops/launcher/input_proj.h"

#include "ops/common/math.h"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

RowsplitGroupedJob make_job(const Weight& weight, Tensor& out, std::int32_t out_ld,
                            std::int32_t out_row_offset, bool q5) {
    return RowsplitGroupedJob{
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales),
        static_cast<__nv_bfloat16*>(out.data),
        weight.n,
        out_ld,
        out_row_offset,
        q5,
    };
}

template <class Cfg, GroupedInputCodec Codec = GroupedInputCodec::Mixed, int Jobs = 4>
void launch_grouped_cfg(const Tensor& x, RowsplitGroupedJob job0, RowsplitGroupedJob job1,
                        RowsplitGroupedJob job2, RowsplitGroupedJob job3, cudaStream_t stream) {
    const int tiles = div_up(job0.n, Cfg::BM) + div_up(job1.n, Cfg::BM) + div_up(job2.n, Cfg::BM) +
                      div_up(job3.n, Cfg::BM);
    const int t = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(div_up(t, Cfg::BN)));
    const bool full_tiles = (t % Cfg::BN) == 0;
    if (full_tiles) {
        linear_rowsplit_grouped_input_gemm_mma_kernel<Cfg, true, Codec, Jobs>
            <<<grid, Cfg::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), job0,
                                                job1, job2, job3, x.ne[0], t, x.ne[0]);
    } else {
        linear_rowsplit_grouped_input_gemm_mma_kernel<Cfg, false, Codec, Jobs>
            <<<grid, Cfg::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), job0,
                                                job1, job2, job3, x.ne[0], t, x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_grouped(const Tensor& x, RowsplitGroupedJob job0, RowsplitGroupedJob job1,
                    RowsplitGroupedJob job2, RowsplitGroupedJob job3, cudaStream_t stream) {
    launch_grouped_cfg<GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>>(x, job0, job1, job2,
                                                                              job3, stream);
}

} // namespace

void linear_rowsplit_attn_input_grouped_mma_launch(const Tensor& x, const Weight& q_weight,
                                                   const Weight& gate_weight,
                                                   const Weight& k_weight, const Weight& v_weight,
                                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                                   cudaStream_t stream) {
    using Cfg = GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true, true>;
    RowsplitGroupedJob empty{};
    launch_grouped_cfg<Cfg, GroupedInputCodec::Q4, 2>(x, make_job(q_weight, q, q.ne[0], 0, false),
                                                      make_job(k_weight, k, k.ne[0], 0, false),
                                                      empty, empty, stream);
    launch_grouped_cfg<Cfg, GroupedInputCodec::Q5, 2>(
        x, make_job(gate_weight, gate, gate.ne[0], 0, true),
        make_job(v_weight, v, v.ne[0], 0, true), empty, empty, stream);
}

void linear_rowsplit_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                                  const Weight& v_weight, Tensor& qkv,
                                                  cudaStream_t stream) {
    RowsplitGroupedJob empty{};
    launch_grouped(x, make_job(qk_weight, qkv, qkv.ne[0], 0, false),
                   make_job(v_weight, qkv, qkv.ne[0], qk_weight.n, true), empty, empty, stream);
}

} // namespace ninfer::ops::detail
