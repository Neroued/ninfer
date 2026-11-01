#include "ops/gated_residual/norm_combine/launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail::hc {
namespace {
template <Form Kind>
Launch select_fused_norm(int columns) {
    return columns >= 1024 ? launch_norm_rows<Kind, NormSchedule::Vectors320>
                           : launch_norm_rows<Kind>;
}

Launch select_implementation(Form form, int columns) {
    switch (form) {
    case Form::ExpandNorm:
        return columns >= 2048 ? launch_norm_rows<Form::ExpandNorm, NormSchedule::Vectors320>
                               : launch_norm_rows<Form::ExpandNorm>;
    case Form::GroupedNorm:
        return columns >= 2816
                   ? launch_norm_rows<Form::GroupedNorm, NormSchedule::StreamingVectors320>
                   : launch_norm_rows<Form::GroupedNorm>;
    case Form::Combine:
        return launch_combine<Form::Combine>;
    case Form::CombineNorm:
        return select_fused_norm<Form::CombineNorm>(columns);
    case Form::AddNorm:
        return select_fused_norm<Form::AddNorm>(columns);
    }
    throw std::logic_error("gated residual: invalid form");
}
} // namespace

void launch_norm_combine(Form form, Operands operands, CudaExecutionView execution) {
    select_implementation(form, operands.columns)(operands, execution);
}
} // namespace ninfer::ops::detail::hc
