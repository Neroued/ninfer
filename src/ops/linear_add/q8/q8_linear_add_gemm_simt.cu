#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_simt_launch.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kStages       = 2;

template <int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8A16GemvSchedule<RowsPerCta, 1, 2, 6144, 24>;
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual_out.data),
                                  residual_out.ne[0]};
    launch_q8_a16_gemv<Schedule>(q8_linear_operands(x, w), output,
                                 LinearResidualAddEpilogue{{output.data, output.rows, 0}}, stream);
}

template <int ColsPerTile>
void launch_variant(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8A16SimtSchedule<kRowsPerBlock, ColsPerTile, 1, 32, kStages, Cache::cg, 1>;
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual_out.data),
                                  residual_out.ne[0]};
    launch_q8_a16_simt<Schedule>(q8_linear_operands(x, w), output,
                                 LinearResidualAddEpilogue{{output.data, output.rows, 0}}, stream);
}

} // namespace

void q8_linear_add_decode_r4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream) {
    launch_decode<4>(x, w, residual_out, stream);
}

void q8_linear_add_decode_r8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream) {
    launch_decode<8>(x, w, residual_out, stream);
}

void q8_linear_add_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    launch_decode<16>(x, w, residual_out, stream);
}

void q8_linear_add_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    launch_variant<4>(x, w, residual_out, stream);
}

void q8_linear_add_simt_r8_c8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    launch_variant<8>(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
