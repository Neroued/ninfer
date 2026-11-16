#include "ninfer/ops/rmsnorm.h"
#include "ops/normalization/rmsnorm/dispatch.h"

namespace ninfer::ops {
namespace norm = detail::rmsnorm;

namespace {
void execute(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps,
             norm::Epilogue kind, Tensor& out, Tensor* inverse, CudaExecutionView execution,
             const char* op) {
    const auto operands = norm::validate(x, weight, gate, eps, out, inverse, kind, execution, op);
    if (operands.rows)
        norm::dispatch(x, weight, gate, eps, kind, out, inverse, operands, execution);
}
} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             CudaExecutionView execution) {
    execute(x, weight, nullptr, eps, unit_offset ? norm::Epilogue::Offset : norm::Epilogue::Plain,
            out, nullptr, execution, "rmsnorm");
}

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             Tensor& inverse, CudaExecutionView execution) {
    execute(x, weight, nullptr, eps, unit_offset ? norm::Epilogue::Offset : norm::Epilogue::Plain,
            out, &inverse, execution, "rmsnorm");
}

void silu_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                        Tensor& out, CudaExecutionView execution) {
    execute(x, weight, &gate, eps, norm::Epilogue::SiluGated, out, nullptr, execution,
            "silu_gated_rmsnorm");
}

void sigmoid_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                           Tensor& out, CudaExecutionView execution) {
    execute(x, weight, &gate, eps, norm::Epilogue::SigmoidGated, out, nullptr, execution,
            "sigmoid_gated_rmsnorm");
}
} // namespace ninfer::ops
