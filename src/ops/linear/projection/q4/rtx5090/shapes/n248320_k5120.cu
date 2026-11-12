#include "ops/linear/projection/q4/q4_shapes.h"
#include "ops/linear/kernels/q4/q4_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv =
    Q4A16GemvSchedule<4, 1, 16, 1, Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedWord8,
                      Q4GemvDecodeMode::Fp16Mantissa, Q4GemvCodeTransfer::AsyncVector16,
                      Q4GemvScaleAccess::SharedPair32, Cache::ca, 80, 1>;
template <int Capacity>
using Small = Q4A16SlicedKMmaSchedule<32, 8, 4, 2, Cache::cg, Cache::ca, 2, 5120, Capacity>;

template <int Capacity>
void launch_small(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] == 5120)
        launch_q4_a16_sliced_instance<Small<Capacity>>(x, w, out, stream);
    else
        launch_q4_a16_sliced_r32_t8_w4_s2(x, w, out, stream);
}

template <class Schedule>
struct HeadSchedule : Schedule {
    static constexpr bool kIndependentTokenTail = true;
};
} // namespace

Q4Launch select_q4_n248320_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_instance<Gemv>;
    if (tokens <= 4) return launch_small<4>;
    if (tokens <= 8) return launch_small<8>;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 80)
        return launch_q4_a16_mma_instance<HeadSchedule<q4_instances::MmaR64T80K64Wr16Wt40S2A2B1>>;
    if (tokens <= 96)
        return launch_q4_a16_mma_instance<HeadSchedule<q4_instances::MmaR64T96K64Wr32Wt16S2A2B1>>;
    return launch_q4_a16_mma_instance<HeadSchedule<q4_instances::MmaR64T128K64Wr64Wt32S2A2B1>>;
}
} // namespace ninfer::ops::detail
