#pragma once

#include "core/dtype.h"
#include "core/record_view.h"
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::kimi_delta_attention {
inline constexpr int kStateDim      = 128;
inline constexpr float kQkL2NormEps = 1.0e-6F;

[[nodiscard]] constexpr bool is_supported_state_dtype(DType dtype) noexcept {
    return dtype == DType::FP32 || dtype == DType::BF16;
}

struct Arguments {
    const __nv_bfloat16 *q, *k, *v, *g, *beta;
    const float *a_log, *dt_bias;
    const void* state_in;
    void* state_out;
    DType state_dtype;
    __nv_bfloat16* out;
    int qk_heads, value_heads, tokens;
    float lower_bound, scale;
    const std::int64_t* source_offset      = nullptr;
    const std::int64_t* destination_offset = nullptr;
};

struct RecurrenceRecords {
    __nv_bfloat16* key = nullptr;
    float* delta = nullptr;
    float* alpha = nullptr;
};

struct Geometry {
    int qk_heads, value_heads, tokens, batch;
};

Geometry validate_inputs(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                         float lower_bound, float scale, const Tensor& out, Device device);
DType validate_state(ConstStateFieldView state, const Geometry&, Device device);

inline bool valid_heads(int qk, int value) { return qk > 0 && value >= qk && value % qk == 0; }

} // namespace ninfer::ops::detail::kimi_delta_attention
