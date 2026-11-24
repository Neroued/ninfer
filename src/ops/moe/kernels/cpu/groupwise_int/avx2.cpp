#include "ops/common/cpu/avx2.h"
#include "ops/moe/kernels/cpu/groupwise_int/compute.h"

namespace ninfer::ops::detail::moe_cpu::groupwise_int {
namespace {
struct Avx2 : common::cpu::avx2::F32 {
    static constexpr int rows = 2, max_columns = 2;
    static constexpr bool bf16_gate   = false;
    static constexpr auto first_chunk = FirstChunkPreparation::Worker;

    static Float scale(std::uint16_t bits) noexcept { return _mm256_set1_ps(_cvtsh_ss(bits)); }

    template <int Bits>
    static Float decode(const std::uint8_t* code, const std::uint8_t* high) noexcept {
        return common::cpu::avx2::decode_signed_q<Bits>(code, high);
    }
};

} // namespace

const Kernel avx2 = make_kernel<Avx2>();

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
