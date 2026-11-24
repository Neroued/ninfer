#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "distributed/communication.h"

namespace ninfer::ops {

// Collectives use contiguous, positive-length tensors on the group's local device. Participants
// agree on operation, count, dtype, reduction and root. SUM/MAX reduce represented values;
// floating reductions may round in backend reduction order. Integer MAX respects signedness.
// BF16/FP16/FP32/I8/U8/I32/U32/I64/U64 are supported. Transport-only operations preserve bits.
// Calls enqueue on execution.stream(). Buffers and groups outlive all consumers and Graphs.
// Outputs are caller-owned; no caller workspace is needed. Size-one groups copy or do nothing.

// output[i] = reduce_r input_r[i]. Equal extents; exact in-place is allowed.
void all_reduce(const Tensor& input, Tensor& output, distributed::Reduction,
                const distributed::Group&, CudaExecutionView);

// All ranks receive root's value. root is a group-local rank; value is input/output on root.
void broadcast(Tensor& value, std::uint32_t root, const distributed::Group&, CudaExecutionView);

// output[r*N+i] = input_r[i], N=input.numel(). In-place input is exactly its rank's output
// segment. Other partial overlap is invalid.
void all_gather(const Tensor& input, Tensor& output, const distributed::Group&, CudaExecutionView);

// output_rank[i] = reduce_r input_r[rank*N+i], N=output.numel(). In-place output is exactly its
// rank's input segment. Other partial overlap is invalid.
void reduce_scatter(const Tensor& input, Tensor& output, distributed::Reduction,
                    const distributed::Group&, CudaExecutionView);

} // namespace ninfer::ops
