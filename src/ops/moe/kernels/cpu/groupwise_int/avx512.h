#pragma once

#include "ops/common/cpu/avx512.h"
#include "ops/moe/kernels/cpu/groupwise_int/work.h"

namespace ninfer::ops::detail::moe_cpu::groupwise_int {
struct Avx512 : common::cpu::avx512::F32 {
    static constexpr int rows = 4, max_columns = 4;
    static constexpr bool bf16_gate   = false;
    static constexpr auto first_chunk = FirstChunkPreparation::Caller;

    static Float scale(std::uint16_t bits) noexcept { return _mm512_set1_ps(_cvtsh_ss(bits)); }

    template <int Bits>
    static Float decode(const std::uint8_t* code, const std::uint8_t* high) noexcept {
        return common::cpu::avx512::decode_signed_q<Bits>(code, high);
    }
};
} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
