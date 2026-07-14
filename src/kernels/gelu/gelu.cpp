#include "kernels/gelu/gelu.h"

#include "kernels/gelu/launch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::kernels {

void gelu(Tensor& x, GeluMode mode, cudaStream_t stream) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("gelu: x must be BF16"); }
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (x.ne[i] < 0) { throw std::invalid_argument("gelu: negative dimension"); }
        if (x.ne[i] == 0) { return; }
        if (n > std::numeric_limits<std::int64_t>::max() / x.ne[i]) {
            throw std::overflow_error("gelu: tensor size overflows int64");
        }
        n *= x.ne[i];
    }
    if (!x.is_contiguous()) { throw std::invalid_argument("gelu: x must be contiguous"); }
    if (x.data == nullptr) { throw std::invalid_argument("gelu: x data must be non-null"); }
    detail::gelu_launch(x, mode, stream);
}

} // namespace ninfer::kernels
