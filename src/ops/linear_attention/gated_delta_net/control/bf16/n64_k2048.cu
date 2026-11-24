#include "ops/linear_attention/gated_delta_net/control/kernels/mma_launch.cuh"

namespace ninfer::ops::detail::gdn_control {
void launch_n64_k2048(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      CudaExecutionView e) {
    using Geometry = MmaGeometry<32, 2048, 64>;
    bool launched  = false;
    switch (plan.split_k) {
    case 16:
        launched = launch_mma<Geometry, 16>(in, out, scratch, e);
        break;
    case 8:
        launched = launch_mma<Geometry, 8>(in, out, scratch, e);
        break;
    case 4:
        launched = launch_mma<Geometry, 4>(in, out, scratch, e);
        break;
    case 2:
        launched = launch_mma<Geometry, 2>(in, out, scratch, e);
        break;
    case 1:
        launched = launch_mma<Geometry, 1>(in, out, {}, e);
        break;
    default:
        throw std::logic_error("gdn_control: invalid 64x2048 plan");
    }
    if (!launched) (void)launch_mma<Geometry, 1>(in, out, {}, e);
}

bool launch_norm_n64_k2048(const Inputs& in, const Norm& norm, GdnControlOutputs out, NormPlan plan,
                           Scratch scratch, CudaExecutionView e) {
    using Geometry = MmaGeometry<32, 2048, 64>;
    switch (plan.capacity) {
    case 6:
        return launch_mma<Geometry, 32, 8, true, 6>(in, out, scratch, e, &norm);
    case 8:
        return launch_mma<Geometry, 32, 8, true, 8>(in, out, scratch, e, &norm);
    case 12:
        return launch_mma<Geometry, 32, 8, true, 12>(in, out, scratch, e, &norm);
    case 16:
        return launch_mma<Geometry, 32, 8, true, 16>(in, out, scratch, e, &norm);
    }
    throw std::logic_error("gdn_control: invalid normalized column capacity");
}
} // namespace ninfer::ops::detail::gdn_control
