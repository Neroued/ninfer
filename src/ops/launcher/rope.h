#pragma once

// ninfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        cudaStream_t stream);

// Overwrite the text RoPE inverse-frequency table with a YaRN-scaled one. Call ONCE at
// startup and only when YaRN is enabled; not calling it leaves the compiled-in constant
// table, which is why factor<=1 is bit-identical to the pre-YaRN engine. `count` must be
// 32 (rotary_dim 64 / 2). Throws on any other size rather than truncating silently.
void rope_install_text_inv_frequency(const float* host_inv_freq, int count);

// YaRN attention scaling. Call ONCE at startup only when YaRN is enabled; the default
// constant is 1.0F, so not calling it leaves the maths untouched.
void rope_install_text_attention_scale(float scale);

} // namespace ninfer::ops::detail
