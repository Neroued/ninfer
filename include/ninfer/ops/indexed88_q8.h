#pragma once

// ThinkingCap 135K Indexed88 proposal path (88,064 selected rows).
// This operator borrows the ONE physical Q8 output head and a device-resident,
// validated ascending list of selected vocabulary IDs. No second weight matrix.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kThinkingCapIndexed88Rows = 88064;

// The caller owns selected_ids_device and all tensors throughout asynchronous execution.
// selected_ids_device contains exactly kThinkingCapIndexed88Rows distinct ascending I32
// IDs in [0,135000), validated and uploaded before any CUDA Graph capture.
// logits must be contiguous BF16 [88064, T]; proposal_ids contiguous I32 [T].
// T=1..8. No allocations, synchronization, or persistent state in this operation.
void indexed88_q8_propose(const Tensor& hidden, const Weight& physical_q8_head,
                          const std::int32_t* selected_ids_device, Tensor& logits,
                          Tensor& proposal_ids, CudaExecutionView execution);

} // namespace ninfer::ops
