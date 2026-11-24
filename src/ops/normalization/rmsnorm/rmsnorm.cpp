#include "ninfer/ops/rmsnorm.h"
#include "ops/normalization/rmsnorm/dispatch.h"

namespace ninfer::ops {
namespace norm = detail::rmsnorm;

namespace {
void execute(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps,
             norm::Epilogue kind, Tensor& out, CudaExecutionView execution, const char* op) {
    const auto operands = norm::validate(x, weight, gate, eps, out, kind, execution, op);
    if (operands.rows) norm::dispatch(x, weight, gate, eps, kind, out, operands, execution);
}
} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             CudaExecutionView execution) {
    execute(x, weight, nullptr, eps, unit_offset ? norm::Epilogue::Offset : norm::Epilogue::Plain,
            out, execution, "rmsnorm");
}

void silu_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                        Tensor& out, CudaExecutionView execution) {
    execute(x, weight, &gate, eps, norm::Epilogue::SiluGated, out, execution, "silu_gated_rmsnorm");
}

void sigmoid_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                           Tensor& out, CudaExecutionView execution) {
    execute(x, weight, &gate, eps, norm::Epilogue::SigmoidGated, out, execution,
            "sigmoid_gated_rmsnorm");
}
} // namespace ninfer::ops
