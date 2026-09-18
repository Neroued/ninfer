#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t weight_row_offset,
                               std::int32_t rows, Tensor& out, std::int32_t output_row_offset) {
    const std::int64_t groups = weight.padded_shape[1] / weight.group;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(weight_row_offset) * groups * 32;
    const auto* high   = weight.qtype == QType::Q5_G64_FP16
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(weight_row_offset) * groups * 8
                             : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(weight_row_offset) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,
        high,
        scales,
        static_cast<__nv_bfloat16*>(out.data),
        rows,
        out.ne[0],
        output_row_offset,
        weight.qtype == QType::Q5_G64_FP16,
    };
}

void launch_slice(bool full, const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr std::int32_t kValueRows = 6144;
    using Schedule                    = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;
    const RowSplitGroupedMmaJob qk    = make_job(qk_weight, 0, qk_weight.n, qkv, 0);
    const RowSplitGroupedMmaJob value = make_job(value_z_weight, 0, kValueRows, qkv, qk_weight.n);
    const RowSplitGroupedMmaJob output_gate =
        make_job(value_z_weight, kValueRows, kValueRows, z, 0);
    RowSplitGroupedMmaJob empty{};
    const int tiles = div_up(qk.n, Schedule::BM) + div_up(value.n, Schedule::BM) +
                      div_up(output_gate.n, Schedule::BM);
    const int cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, RowSplitGroupedMmaCodec::Mixed, 4>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     value, output_gate, empty, x.ne[0], cols,
                                                     x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, RowSplitGroupedMmaCodec::Mixed, 4>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     value, output_gate, empty, x.ne[0], cols,
                                                     x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        cudaStream_t stream) {
    constexpr std::int32_t kTileCols = 128;
    const bool full                  = (x.ne[1] % kTileCols) == 0;
    for_each_token_slice(x.ne[1], kTileCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor qkv_slice     = qkv.slice(1, offset, count);
        Tensor z_slice       = z.slice(1, offset, count);
        launch_slice(full, x_slice, qk_weight, value_z_weight, qkv_slice, z_slice, stream);
    });
}

} // namespace ninfer::ops::detail
