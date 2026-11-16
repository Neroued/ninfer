#include "ops/gated_residual/norm_combine/launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail::hc {
namespace {
template <Form Kind>
Launch select_fused_norm(int columns) {
    return columns >= 1024 ? launch_norm_rows<2560, Kind, 320, 8> : launch_norm_rows<2560, Kind>;
}

Launch select_implementation(Form form, int columns) {
    switch (form) {
    case Form::ExpandNorm:
        return columns >= 2048 ? launch_norm_rows<2560, Form::ExpandNorm, 320, 8>
                               : launch_norm_rows<2560, Form::ExpandNorm>;
    case Form::GroupedNorm:
        return columns >= 2816 ? launch_norm_rows<2560, Form::GroupedNorm, 320, 8, true>
                               : launch_norm_rows<2560, Form::GroupedNorm>;
    case Form::Combine:
        return launch_combine<2560, Form::Combine>;
    case Form::CombineNorm:
        return select_fused_norm<Form::CombineNorm>(columns);
    case Form::AddNorm:
        return select_fused_norm<Form::AddNorm>(columns);
    }
    throw std::logic_error("gated residual: invalid form");
}
} // namespace

void validate_norm_combine_width(int width) {
    if (width != 2560) throw std::invalid_argument("gated residual: unsupported residual width");
}

void launch_norm_combine(Form form, Operands operands, CudaExecutionView execution) {
    select_implementation(form, operands.columns)(operands, execution);
}
} // namespace ninfer::ops::detail::hc
