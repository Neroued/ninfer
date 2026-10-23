#pragma once

#include "core/record_view.h"
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::kimi_delta_attention {
inline constexpr int kStateDim      = 128;
inline constexpr float kQkL2NormEps = 1.0e-6F;

struct Arguments {
    const __nv_bfloat16 *q, *k, *v, *g, *beta;
    const float *a_log, *dt_bias, *state_in;
    float* state_out;
    __nv_bfloat16* out;
    int qk_heads, value_heads, tokens;
    float lower_bound, scale;
};

struct RecurrenceRecords {
    __nv_bfloat16 *key = nullptr, *value = nullptr;
    float *log_decay = nullptr, *beta = nullptr;
};

struct Geometry {
    int qk_heads, value_heads, tokens, batch;
};

Geometry validate_inputs(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                         float lower_bound, float scale, const Tensor& out, Device device);
void validate_state(ConstStateFieldView state, const Geometry&, Device device);

inline bool valid_heads(int qk, int value) { return qk > 0 && value >= qk && value % qk == 0; }

} // namespace ninfer::ops::detail::kimi_delta_attention
