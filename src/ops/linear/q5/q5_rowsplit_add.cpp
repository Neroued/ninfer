#include "ops/linear/q5/q5_rowsplit_add.h"

#include "ops/linear/q5/q5_rowsplit_kernels.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

Q5KernelVariant resolve_mma_variant(Q5ScheduleId schedule, const Q5Problem& problem) {
    if (q5_candidate_is_legal(schedule, Q5KernelVariant::Full, problem)) {
        return Q5KernelVariant::Full;
    }
    if (q5_candidate_is_legal(schedule, Q5KernelVariant::Predicated, problem)) {
        return Q5KernelVariant::Predicated;
    }
    throw std::logic_error("q5 linear_add: residual MMA route is not physically legal");
}

} // namespace

Q5AddRoute q5_rowsplit_add_route(std::int32_t columns) {
    if (columns <= 0) { throw std::invalid_argument("q5 linear_add: columns must be positive"); }
    if (columns == 1) { return Q5AddRoute::GemvResidual; }
    if (columns <= 24) { return Q5AddRoute::Materialized; }
    if (columns <= 128) { return Q5AddRoute::MmaR64C64Residual; }
    return Q5AddRoute::MmaR64C128Residual;
}

std::size_t q5_rowsplit_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                            std::int32_t columns) {
    if (output_rows <= 0 || input_rows <= 0) {
        throw std::invalid_argument("q5 linear_add workspace: dimensions must be positive");
    }
    if (q5_rowsplit_add_route(columns) != Q5AddRoute::Materialized) { return 0; }
    return Tensor(nullptr, DType::BF16, {output_rows, columns}).bytes();
}

void q5_rowsplit_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                              cudaStream_t stream) {
    const Q5Problem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const bool supported_shape =
        problem.rows == 5120 && ((problem.k == 6144 && problem.padded_k == 6144) ||
                                 (problem.k == 17408 && problem.padded_k == 17408));
    if (!supported_shape) {
        throw std::invalid_argument("q5 linear_add: unsupported exact Q5 problem");
    }

    const Q5AddRoute route = q5_rowsplit_add_route(problem.cols);
    if (route == Q5AddRoute::GemvResidual) {
        q5_rowsplit_gemv_residual_launch(x, w, residual_out, stream);
        return;
    }
    if (route == Q5AddRoute::Materialized) {
        throw std::invalid_argument(
            "q5 linear_add: columns 2..24 require the materialized semantic fallback");
    }

    const Q5ScheduleId schedule =
        route == Q5AddRoute::MmaR64C64Residual ? Q5ScheduleId::MmaR64C64 : Q5ScheduleId::MmaR64C128;
    const Q5KernelVariant variant = resolve_mma_variant(schedule, problem);
    if (schedule == Q5ScheduleId::MmaR64C64) {
        q5_rowsplit_mma_residual_r64_c64_launch(variant, x, w, residual_out, stream);
    } else {
        q5_rowsplit_mma_residual_r64_c128_launch(variant, x, w, residual_out, stream);
    }
}

} // namespace ninfer::ops::detail
