#include "qus/kernels/gdn_in_vz.h"

#include "kernels/linear/gemv/linear_rowsplit_gemv_gdn_in_vz_6144.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

void require_q5_rowsplit(const Weight& w, const char* name) {
    if (w.qtype != QType::Q5G64_F16S || w.group_size != 64 || w.qdata == nullptr ||
        w.qhigh == nullptr || w.scales == nullptr) {
        throw std::invalid_argument(std::string("gdn_in_vz_decode: ") + name +
                                    " must be Q5G64_F16S row-split");
    }
}

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("gdn_in_vz_decode: invalid ") + name);
    }
}

} // namespace

void gdn_in_vz_decode(const Tensor& x, const Weight& v_weight, const Weight& z_weight,
                      Tensor& v_out, Tensor& z_out, cudaStream_t stream) {
    require_tensor(x, DType::BF16, 5120, "x");
    require_tensor(v_out, DType::BF16, 6144, "v_out");
    require_tensor(z_out, DType::BF16, 6144, "z_out");
    require_q5_rowsplit(v_weight, "v_weight");
    require_q5_rowsplit(z_weight, "z_weight");
    detail::linear_rowsplit_gemv_gdn_in_vz_6144_q5_launch(x, v_weight, z_weight, v_out, z_out,
                                                          stream);
}

} // namespace qus::kernels
