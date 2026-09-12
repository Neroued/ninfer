#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                               Tensor& out) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > weight.n ||
        out.ne[0] != row_count) {
        throw std::invalid_argument("Q4/Q5 attention input grouped MMA row view is invalid");
    }
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(row_begin) * groups * 32;
    const auto* high   = weight.qtype == QType::Q5G64_F16S
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(row_begin) * groups * 8
                             : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(row_begin) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,     high,      scales, static_cast<__nv_bfloat16*>(out.data),
        row_count, out.ne[0], 0,      weight.qtype == QType::Q5G64_F16S,
    };
}

template <class Schedule, RowSplitGroupedMmaCodec Codec>
void launch_pair(bool full, const Tensor& x, RowSplitGroupedMmaJob first,
                 RowSplitGroupedMmaJob second, cudaStream_t stream) {
    const int tiles = div_up(first.n, Schedule::BM) + div_up(second.n, Schedule::BM);
    const int cols  = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));
    RowSplitGroupedMmaJob empty{};

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first, second, empty, empty, x.ne[0], cols,
                                                     x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first, second, empty, empty, x.ne[0], cols,
                                                     x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_slice(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
                  Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const bool full = (x.ne[1] % Schedule::BN) == 0;
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q4>(
        full, x, make_job(query_key_weight, 0, 6144, q), make_job(query_key_weight, 6144, 1024, k),
        stream);
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q5>(
        full, x, make_job(gate_value_weight, 0, 6144, gate),
        make_job(gate_value_weight, 6144, 1024, v), stream);
}

template <class Schedule>
void launch(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
            Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kSliceCols = Schedule::BN;
    for_each_token_slice(x.ne[1], kSliceCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice       = q.slice(1, offset, count);
        Tensor gate_slice    = gate.slice(1, offset, count);
        Tensor k_slice       = k.slice(1, offset, count);
        Tensor v_slice       = v.slice(1, offset, count);
        launch_slice<Schedule>(x_slice, query_key_weight, gate_value_weight, q_slice, gate_slice,
                               k_slice, v_slice, stream);
    });
}

using MmaR32C64S4 = GemmCfg<32, 64, 64, 16, 16, 4, 1, false, true, true>;

template <class S, bool Full>
void mixed_slice(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& q, Tensor& g,
                 Tensor& k, Tensor& v, cudaStream_t stream) {
    const dim3 grid(14336 / S::BM, (x.ne[1] + S::BN - 1) / S::BN);
    rowsplit_grouped_mma_kernel<S, Full, RowSplitGroupedMmaCodec::Mixed, 4>
        <<<grid, S::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                          make_job(w0, 0, 6144, q), make_job(w0, 6144, 1024, k),
                                          make_job(w1, 0, 6144, g), make_job(w1, 6144, 1024, v),
                                          5120, x.ne[1], 5120);
    CUDA_CHECK(cudaGetLastError());
}

template <class S>
void launch_mixed(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& q, Tensor& g,
                  Tensor& k, Tensor& v, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], S::BN, [&](int begin, int count) {
        const Tensor xs = x.slice(1, begin, count);
        Tensor qs = q.slice(1, begin, count), gs = g.slice(1, begin, count),
               ks = k.slice(1, begin, count), vs = v.slice(1, begin, count);
        if (count % S::BN == 0)
            mixed_slice<S, true>(xs, w0, w1, qs, gs, ks, vs, stream);
        else
            mixed_slice<S, false>(xs, w0, w1, qs, gs, ks, vs, stream);
    });
}
} // namespace

void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream) {
    launch<MmaR32C64S4>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
}

void q4_q5_attn_input_mixed_r32_c64_s3_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream) {
    launch_mixed<GemmCfg<32, 64, 64, 16, 16, 3, 3, false, true, true>>(x, w0, w1, q, g, k, v,
                                                                       stream);
}

void q4_q5_attn_input_pair_r32_c64_s3_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                             Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                             cudaStream_t stream) {
    launch<GemmCfg<32, 64, 64, 32, 16, 3, 2, false, true, true>>(x, w0, w1, q, g, k, v, stream);
}

void q4_q5_attn_input_mixed_r64_c128_s2_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                               Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                               cudaStream_t stream) {
    launch_mixed<GemmCfg<64, 128, 64, 64, 16, 2, 2, false, true, true>>(x, w0, w1, q, g, k, v,
                                                                        stream);
}
} // namespace ninfer::ops::detail
