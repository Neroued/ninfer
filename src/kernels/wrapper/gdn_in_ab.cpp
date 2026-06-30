#include "qus/kernels/gdn_in_ab.h"

#include "kernels/linear/gemv/linear_dense_gdn_in_ab_48.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

void require_dense_bf16(const Weight& w, const char* name) {
    if (w.qtype != QType::BF16_CTRL || w.layout != QuantLayout::Contiguous ||
        w.qdata == nullptr) {
        throw std::invalid_argument(std::string("gdn_in_ab_decode: ") + name +
                                    " must be dense BF16");
    }
}

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("gdn_in_ab_decode: invalid ") + name);
    }
}

} // namespace

void gdn_in_ab_decode(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                      Tensor& a_out, Tensor& b_out, cudaStream_t stream) {
    require_tensor(x, DType::BF16, 5120, "x");
    require_tensor(a_out, DType::BF16, 48, "a_out");
    require_tensor(b_out, DType::BF16, 48, "b_out");
    require_dense_bf16(a_weight, "a_weight");
    require_dense_bf16(b_weight, "b_weight");
    detail::linear_dense_gdn_in_ab_48_launch(x, a_weight, b_weight, a_out, b_out, stream);
}

} // namespace qus::kernels
