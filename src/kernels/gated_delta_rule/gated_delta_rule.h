#pragma once

// ninfer::kernels - GDN gated delta rule. Token-count dispatch is internal.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::kernels {

[[nodiscard]] std::size_t gated_delta_rule_workspace_bytes(std::int32_t tokens);

void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, WorkspaceArena& ws, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream);

// Distinct read/write state form used by prefix-append prefill.
void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, WorkspaceArena& ws,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream);

// Snapshot-state semantics used by speculative verification.
void gated_delta_rule_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, WorkspaceArena& ws,
                               Tensor& ssm_states, const Tensor& initial_slot, Tensor& out,
                               cudaStream_t stream);

} // namespace ninfer::kernels
