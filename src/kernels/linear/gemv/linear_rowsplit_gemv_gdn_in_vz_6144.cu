#include "kernels/linear/gemv/linear_rowsplit_gemv_gdn_in_vz_6144.cuh"

#include "kernels/linear/gemv/linear_rowsplit_gemv_q5_core.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::kernels::detail {
namespace {

constexpr int kN            = 6144;
constexpr int kK            = 5120;
constexpr int kRowsPerBlock = 12;
constexpr int kStages       = 2;

void require_shape(const Weight& w, const char* name) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_in_vz_decode: ") + name +
                                    " requires 6144x5120 Q5");
    }
}

} // namespace

void linear_rowsplit_gemv_gdn_in_vz_6144_q5_launch(const Tensor& x, const Weight& v_weight,
                                                   const Weight& z_weight, Tensor& v_out,
                                                   Tensor& z_out, cudaStream_t stream) {
    require_shape(v_weight, "v_weight");
    require_shape(z_weight, "z_weight");

    launch_q5_rowsplit_gemv_dual<kN, kK, kRowsPerBlock, kStages, /*kStageX=*/true>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(v_weight.qdata),
        static_cast<const std::uint8_t*>(v_weight.qhigh),
        static_cast<const std::uint8_t*>(v_weight.scales),
        static_cast<const std::uint8_t*>(z_weight.qdata),
        static_cast<const std::uint8_t*>(z_weight.qhigh),
        static_cast<const std::uint8_t*>(z_weight.scales), static_cast<__nv_bfloat16*>(v_out.data),
        static_cast<__nv_bfloat16*>(z_out.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
