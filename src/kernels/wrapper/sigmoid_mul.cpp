// ninfer::kernels - sigmoid_mul wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/kernel-development.md §2.
#include "ninfer/kernels/sigmoid_mul.h"

#include "kernels/launcher/sigmoid_gate_mul.h" // detail::sigmoid_gate_mul_launch

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::kernels {
namespace {

std::int64_t numel_allow_zero(const Tensor& t) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument("sigmoid_mul: gate/x dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("sigmoid_mul: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("sigmoid_mul: gate/x must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != x.ne[d]) {
            throw std::invalid_argument("sigmoid_mul: gate/x shapes must match");
        }
    }
    if (numel_allow_zero(x) == 0) { return; }
    if (!gate.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("sigmoid_mul: gate/x must be contiguous");
    }
    if (gate.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("sigmoid_mul: gate/x data must be non-null");
    }

    detail::sigmoid_gate_mul_launch(gate, x, stream); // single variant -> direct dispatch
}

} // namespace ninfer::kernels
