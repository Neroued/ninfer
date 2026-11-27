#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops {

// Compact expert-major assignments. Device I32 counts are {unique_rows, assignments}.
// offsets[local_slots+1] spans [0,M), logical_ids[local_slots] names weight/calibration
// IDs, rows[M_cap] indexes the BF16 hidden prefix [H,U), and coefficients are original FP32
// router coefficients. Only valid rows of FP32 partial are written, including zero sums.
// The producer guarantees valid device indices, monotonic offsets and disjoint storage.
// All views and caller scratch remain live through the last consumer on execution.
struct CompactExpertAssignments {
    Tensor counts, offsets, logical_ids, rows, coefficients;
};

[[nodiscard]] std::size_t compact_routed_swiglu_workspace_bytes(const RoutedSwiGLUProfile&,
                                                                std::int32_t assignment_capacity);
void compact_routed_swiglu(const Tensor& hidden, const RoutedSwiGLUProfile&,
                           const CudaSwiGLUExperts&, const CompactExpertAssignments&,
                           Tensor& partial, MemoryArena&, CudaExecutionView);

} // namespace ninfer::ops
