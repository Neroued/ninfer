#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/kernels/noncausal/operands.h"
#include "ops/softmax_attention/dense/plain/launch.h"

namespace ninfer::ops {
void softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                       AttentionHeadGeometry geometry, float scale, MemoryArena& workspace,
                       Tensor& out, CudaExecutionView execution) {
    (void)detail::noncausal::validate_qkv(q, k, v, out, geometry, scale, "softmax_attention",
                                          execution.device());
    common::operand::device(workspace.device(), execution.device(), "attention", "workspace");
    detail::launch_plain_attention(q, k, v, out, execution.stream());
}
} // namespace ninfer::ops
