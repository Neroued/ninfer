#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<8, 1, 16, ActiveTokens, 1, Fp8SimtActivationAccess::TokenPacked,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    auto* output = static_cast<__nv_bfloat16*>(residual.data);
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        fp8_a16_operands(x, weight), LinearBf16Output{output, weight.n},
        LinearResidualAddEpilogue{{output, weight.n}}, stream);
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, 2 + static_cast<int>(Offsets)>...};
}

template <class Geometry>
const auto& launchers() {
    static constexpr auto kLaunchers =
        make_launchers<Geometry>(std::make_index_sequence<kFp8LinearAddLastSimtTokens - 2 + 1>{});
    return kLaunchers;
}

} // namespace

void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > kFp8LinearAddLastSimtTokens) {
        throw std::invalid_argument("fp8 linear_add small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - 2);
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        launchers<Fp8N5120K6144>()[index](x, weight, residual, stream);
        return;
    case Fp8GeometryId::N5120K17408:
        launchers<Fp8N5120K17408>()[index](x, weight, residual, stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add small-T: unsupported problem");
}

template <int K>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    const auto operands = fp8_a16_operands(x, weight);
    auto* data          = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output output{data, weight.n};
    const LinearResidualAddEpilogue epilogue{{data, weight.n}};
    const auto sliced = [&]<int T, int W, int Stages>() {
        launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<T, W, Stages>, K>>(
            operands, output, epilogue, stream);
    };
    if (x.ne[1] <= 8) return sliced.template operator()<8, 8, 2>();
    if (x.ne[1] <= 16) return sliced.template operator()<16, 8, 2>();
    if constexpr (K == 6144) {
        if (x.ne[1] <= 32) return sliced.template operator()<16, 4, 2>();
        if (x.ne[1] <= 64) return sliced.template operator()<32, 4, 1>();
    } else {
        if (x.ne[1] <= 32) return sliced.template operator()<32, 8, 1>();
        if (x.ne[1] <= 64) return sliced.template operator()<64, 2, 2>();
    }
    if (x.ne[1] <= 128)
        return launch_fp8_a16_mma<
            Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>, K>>(operands, output,
                                                                                  epilogue, stream);
    launch_fp8_a16_mma<Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
        operands, output, epilogue, stream);
}

void fp8_linear_add_matrix_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream) {
    if (weight.k == 6144)
        launch_matrix<6144>(x, weight, residual, stream);
    else
        launch_matrix<17408>(x, weight, residual, stream);
}
} // namespace ninfer::ops::detail
