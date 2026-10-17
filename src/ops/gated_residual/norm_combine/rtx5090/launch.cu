#include "ops/gated_residual/norm_combine/launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail::hc {
namespace {
constexpr int kExpandVectorStart        = 2048;
constexpr int kGroupedStreamingStart    = 2816;
constexpr int kDisjointStreamingStart   = 1536;
constexpr int kDisjointMaterializeStart = 1841;
constexpr int kDisjointMaterializeEnd   = 3072;
constexpr int kInPlaceMaterializeStart  = 1824;
constexpr int kInPlaceMaterializeEnd    = 3680;

template <Form Kind>
Launch select_combine_norm(Operands o) {
    const bool in_place = o.input == o.residual;
    if (!in_place && o.columns >= kDisjointStreamingStart && o.columns < kDisjointMaterializeStart)
        return launch_norm_rows<Kind, NormSchedule::StreamingVectors320>;
    const bool materialize =
        in_place ? o.columns >= kInPlaceMaterializeStart && o.columns <= kInPlaceMaterializeEnd
                 : o.columns >= kDisjointMaterializeStart && o.columns <= kDisjointMaterializeEnd;
    return materialize ? launch_materialized_norm<Kind> : launch_norm_rows<Kind>;
}

Launch select_implementation(Form form, Operands o) {
    switch (form) {
    case Form::ExpandNorm:
        return o.columns >= kExpandVectorStart
                   ? launch_norm_rows<Form::ExpandNorm, NormSchedule::Vectors320>
                   : launch_norm_rows<Form::ExpandNorm>;
    case Form::GroupedNorm:
        return o.columns >= kGroupedStreamingStart
                   ? launch_norm_rows<Form::GroupedNorm, NormSchedule::StreamingVectors320>
                   : launch_norm_rows<Form::GroupedNorm>;
    case Form::Combine:
        return launch_combine<Form::Combine>;
    case Form::CombineNorm:
        return select_combine_norm<Form::CombineNorm>(o);
    case Form::AddNorm:
        return select_combine_norm<Form::AddNorm>(o);
    }
    throw std::logic_error("gated residual: invalid form");
}

} // namespace

void launch_norm_combine(Form form, Operands operands, CudaExecutionView execution) {
    select_implementation(form, operands)(operands, execution);
}
} // namespace ninfer::ops::detail::hc
