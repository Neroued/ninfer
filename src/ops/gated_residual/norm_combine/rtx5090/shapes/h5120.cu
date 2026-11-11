#include "ops/gated_residual/norm_combine/launch.cuh"
#include "ops/gated_residual/norm_combine/rtx5090/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::hc {
namespace {
template <Form Kind>
Launch select_combine_norm(Operands o) {
    if (o.input == o.residual) {
        // The two-pass route wins near the L2 capacity knee for in-place writes.
        // The output residual is also the norm input; no scratch.
        if (o.columns >= 832) return launch_materialized_norm<5120, Kind, 320, 8, true>;
        return o.columns <= 256 ? launch_norm_rows<5120, Kind, 512> : launch_norm_rows<5120, Kind>;
    }
    return o.columns <= 640 ? launch_norm_rows<5120, Kind, 512> : launch_norm_rows<5120, Kind>;
}

Launch select_implementation(Form form, Operands o) {
    switch (form) {
    case Form::ExpandNorm:
        return launch_norm_rows<5120, Form::ExpandNorm, 512>;
    case Form::GroupedNorm:
        return launch_norm_rows<5120, Form::GroupedNorm, 512>;
    case Form::Combine:
        return launch_combine<5120, Form::Combine>;
    case Form::CombineNorm:
        return select_combine_norm<Form::CombineNorm>(o);
    case Form::AddNorm:
        return select_combine_norm<Form::AddNorm>(o);
    }
    throw std::logic_error("gated residual: invalid form");
}
} // namespace

void launch_h5120(Form form, Operands operands, CudaExecutionView execution) {
    select_implementation(form, operands)(operands, execution);
}
} // namespace ninfer::ops::detail::hc
