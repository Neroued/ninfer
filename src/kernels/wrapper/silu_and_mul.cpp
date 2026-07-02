// qus::kernels — silu_and_mul wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/l1-kernel-layering.md §5.
#include "qus/kernels/silu_and_mul.h"

#include "kernels/launcher/silu_and_mul.h"  // detail::silu_and_mul_launch

#include <stdexcept>

namespace qus::kernels {

void silu_and_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("silu_and_mul: gate/up/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("silu_and_mul: gate/up/out shapes must match");
        }
    }
    if (!out.is_contiguous()) {
        throw std::invalid_argument("silu_and_mul: out must be contiguous");
    }
    if (out.numel() == 0) { return; }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("silu_and_mul: gate/up/out data must be non-null");
    }

    detail::silu_and_mul_launch(gate, up, out, stream);  // single variant -> direct dispatch
}

} // namespace qus::kernels
