#pragma once

#include "core/tensor.h"
#include "core/record_view.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail::gated_delta_net {
void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream);

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, ConstStateFieldView source_state,
                                   StateFieldView destination_state, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
