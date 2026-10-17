#include "ninfer/ops/gated_residual.h"
#include "ops/gated_residual/operands.h"

namespace ninfer::ops::detail::hc {
namespace {
void execute(Form form, const Tensor& input, const Tensor* block, const Tensor* injection,
             const Tensor* weight, float epsilon, Tensor* residual, Tensor* normalized,
             CudaExecutionView execution, const char* name) {
    const auto operands = validate(form, input, block, injection, weight, epsilon, residual,
                                   normalized, execution, name);
    launch_norm_combine(form, operands, execution);
}
} // namespace
} // namespace ninfer::ops::detail::hc

namespace ninfer::ops {
namespace hc = detail::hc;

void hc_expand_norm(const Tensor& embedding, const Tensor& weight, float eps, Tensor& residual,
                    Tensor& normalized, CudaExecutionView execution) {
    hc::execute(hc::Form::ExpandNorm, embedding, nullptr, nullptr, &weight, eps, &residual,
                &normalized, execution, "hc_expand_norm");
}

void hc_grouped_rmsnorm(const Tensor& residual, const Tensor& weight, float eps, Tensor& normalized,
                        CudaExecutionView execution) {
    hc::execute(hc::Form::GroupedNorm, residual, nullptr, nullptr, &weight, eps, nullptr,
                &normalized, execution, "hc_grouped_rmsnorm");
}

void hc_combine(const Tensor& residual, const Tensor& block, const Tensor& injection,
                Tensor& updated, CudaExecutionView execution) {
    hc::execute(hc::Form::Combine, residual, &block, &injection, nullptr, 0, &updated, nullptr,
                execution, "hc_combine");
}

void hc_combine_norm(const Tensor& residual, const Tensor& block, const Tensor& injection,
                     const Tensor& weight, float eps, Tensor& updated, Tensor& normalized,
                     CudaExecutionView execution) {
    hc::execute(hc::Form::CombineNorm, residual, &block, &injection, &weight, eps, &updated,
                &normalized, execution, "hc_combine_norm");
}

void hc_combine_norm(const Tensor& residual, const Tensor& block, const Tensor& weight, float eps,
                     Tensor& updated, Tensor& normalized, CudaExecutionView execution) {
    hc::execute(hc::Form::AddNorm, residual, &block, nullptr, &weight, eps, &updated, &normalized,
                execution, "hc_combine_norm");
}
} // namespace ninfer::ops
