#include "ops/gated_residual/norm_combine/rtx5090/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::hc {
namespace {
struct Geometry {
    int width;
    void (*launch)(Form, Operands, CudaExecutionView);
};

constexpr Geometry kGeometries[]{{2560, launch_h2560}, {5120, launch_h5120}};

const Geometry& geometry(int width) {
    for (const auto& entry : kGeometries)
        if (entry.width == width) return entry;
    throw std::invalid_argument("gated residual: unsupported residual width");
}
} // namespace

void validate_norm_combine_width(int width) { (void)geometry(width); }

void launch_norm_combine(Form form, Operands operands, CudaExecutionView execution) {
    geometry(operands.width).launch(form, operands, execution);
}
} // namespace ninfer::ops::detail::hc
