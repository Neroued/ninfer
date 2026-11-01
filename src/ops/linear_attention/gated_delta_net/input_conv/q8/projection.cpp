#include "core/platform.h"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {
using Launcher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

Launcher select_n12288_k2048(int columns) {
    if (columns == 1) return launch_q8_gdn_input_decode;
    if (columns <= 96) return launch_q8_gdn_input_splitk_mma;
    return launch_q8_gdn_input_mma_r64_c128;
}
} // namespace

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream) {
#if NINFER_TARGET_SPARK
    if (weight.n == 8192 && weight.k == 2560) {
        return q8_gdn_projection_n8192_k2560(x, weight, qkv, z, stream);
    }
#endif
    if (weight.n == 16384 && weight.k == 2560)
        return q8_gdn_projection_n16384_k2560(x, weight, qkv, z, stream);
    if (weight.n == 12288 && weight.k == 2048)
        return select_n12288_k2048(x.ne[1])(x, weight, qkv, z, stream);
    throw std::invalid_argument("GDN Q8 projection: unsupported full geometry");
}
} // namespace ninfer::ops::detail
