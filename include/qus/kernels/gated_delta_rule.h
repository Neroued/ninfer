#pragma once

// qus::kernels - gated_delta_rule phase-split GDN recurrence.

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void gated_delta_rule_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                const Tensor& beta, float scale, Tensor& ssm_state, Tensor& out,
                                cudaStream_t stream);

void gated_delta_rule_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, int chunk_size, WorkspaceArena& ws,
                              Tensor& ssm_state, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels
