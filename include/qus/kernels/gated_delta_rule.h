#pragma once

// qus::kernels - gated_delta_rule phase-split GDN recurrence.

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void gated_delta_rule_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                const Tensor& beta, float scale, WorkspaceArena& ws,
                                Tensor& ssm_state, Tensor& out, cudaStream_t stream);

void gated_delta_rule_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         WorkspaceArena& ws, Tensor& ssm_states,
                                         const Tensor& initial_slot, Tensor& out,
                                         cudaStream_t stream);

// Chunked GDN prefill with distinct read/write SSM state. The initial recurrent state is read
// from `ssm_state_in` and the running state is written to `ssm_state_out`. Passing the same tensor
// for both is the in-place form used by reset prefill and continuation chunks; distinct views let
// prefix-append prefill read a committed GDN snapshot slot and publish the running state to slot 0.
void gated_delta_rule_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, int chunk_size, WorkspaceArena& ws,
                              const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                              cudaStream_t stream);
void gated_delta_rule_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, int chunk_size, WorkspaceArena& ws,
                              Tensor& ssm_state, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels
