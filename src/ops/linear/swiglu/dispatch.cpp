#include "core/platform.h"
#include "ops/linear/swiglu/dispatch.h"
#include <stdexcept>

namespace ninfer::ops::detail {
void validate_swiglu_geometry(QType type, int rows, int k, bool clamped) {
    if (clamped) {
#if NINFER_TARGET_SPARK
        if (type == QType::NVFP4 && rows == 12288 && k == 4096) return;
        if ((type == QType::BF16 || type == QType::Q8_G32_FP16) && rows == 2048 && k == 4096)
            return;
#endif
        throw std::invalid_argument("linear_swiglu: unsupported clamped geometry");
    }
#if NINFER_TARGET_SPARK
    if (type == QType::NVFP4 && rows == 12288 && k == 4096) return;
    if ((type == QType::BF16 || type == QType::Q8_G32_FP16) && rows == 2048 && k == 4096) return;
#endif
    if (type == QType::Q8_G32_FP16 && rows == 12288 && k == 2048) return;
    if (rows == 34816 && k == 5120 &&
        (type == QType::Q4_G64_FP16 || type == QType::Q8_G32_FP16 ||
         type == QType::FP8_E4M3FN_ROW_BF16 || type == QType::NVFP4))
        return;
    throw std::invalid_argument("linear_swiglu: unsupported weight geometry");
}

LinearSwiGluLaunch select_q8_linear_swiglu(int rows, int k, int tokens) {
    if (tokens > 0) {
        if (rows == 12288 && k == 2048) return select_q8_swiglu_n12288_k2048(tokens);
        if (rows == 34816 && k == 5120) return select_q8_swiglu_n34816_k5120(tokens);
    }
    throw std::invalid_argument("q8 linear_swiglu: unsupported shape or token extent");
}

void q8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, float limit,
                               bool clamped, cudaStream_t stream) {
#if NINFER_TARGET_SPARK
    if (w.n == 2048 && w.k == 4096)
        return launch_q8_swiglu_n2048_k4096(x, w, out, limit, clamped, stream);
#endif
    if (clamped) throw std::invalid_argument("q8 linear_swiglu: unsupported clamped geometry");
    select_q8_linear_swiglu(w.n, w.k, x.ne[1])(x, w, out, stream);
}
} // namespace ninfer::ops::detail
