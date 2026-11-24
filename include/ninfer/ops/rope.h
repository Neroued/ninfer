#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   y[i]     = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   y[i+R/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi), R = rotary_dim.
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=D in {64,72}; the first D/4
 *   pairs use axis 0 and the remaining pairs use axis 1, with local frequency
 *   theta^(-2*(i%(D/4))/(D/2)). Positions are stored as two contiguous coordinate axes.
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. q and k must not overlap one another or positions. The Op mutates only
 * dimensions [0,rotary_dim); unrotated dimensions and token padding remain bit-exact. Empty token
 * extents perform no device work. The Op uses no workspace or persistent state. `execution`
 * supplies the stream and the selected device's positive physical SM count.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          CudaExecutionView execution);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          CudaExecutionView execution);

} // namespace ninfer::ops
