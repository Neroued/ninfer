// ninfer::kernels — silu_mul wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/kernel-development.md §2.
#include "kernels/silu_mul/silu_mul.h"

#include "kernels/silu_mul/launch.h" // detail::silu_and_mul_launch

#include <stdexcept>

namespace ninfer::kernels {

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("silu_mul: gate/up/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("silu_mul: gate/up/out shapes must match");
        }
    }
    if (!out.is_contiguous()) { throw std::invalid_argument("silu_mul: out must be contiguous"); }
    if (out.numel() == 0) { return; }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("silu_mul: gate/up/out data must be non-null");
    }

    detail::silu_and_mul_launch(gate, up, out, stream); // single variant -> direct dispatch
}

} // namespace ninfer::kernels
