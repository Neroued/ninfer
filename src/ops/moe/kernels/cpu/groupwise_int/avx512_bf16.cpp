#include "ops/common/cpu/avx512.h"
#include "ops/moe/kernels/cpu/groupwise_int/avx512.h"
#include "ops/moe/kernels/cpu/groupwise_int/compute.h"

namespace ninfer::ops::detail::moe_cpu::groupwise_int {
namespace {
struct Avx512Bf16 : Avx512 {
    static constexpr bool bf16_gate = true;

    static unsigned safe_columns(const std::uint16_t* input, int columns) noexcept {
        unsigned safe = 0;
        for (int t = 0; t < columns; ++t) {
            bool usable = true;
            for (int k = 0; k < hidden; k += 32) {
                const auto words    = _mm512_loadu_si512(input + t * hidden + k);
                const auto absolute = _mm512_and_si512(words, _mm512_set1_epi16(0x7fff));
                const auto nonzero  = _mm512_test_epi16_mask(absolute, absolute);
                // VDPBF16PS flushes denormal arithmetic. If each nonzero BF16 input is
                // at least 2^-119, its quantum is >=2^-126: an integer-code dot cannot
                // create a nonzero FP32 subnormal through cancellation. Smaller inputs
                // use the ordinary FMA implementation with gradual underflow.
                if (nonzero & _mm512_cmplt_epu16_mask(absolute, _mm512_set1_epi16(0x0400))) {
                    usable = false;
                    break;
                }
            }
            if (usable) { safe |= 1U << t; }
        }
        return safe;
    }

    template <int Rows, int Columns>
    static void gate_dot(const Matrix& matrix, const int (&rows)[Rows],
                         const std::uint16_t* const (&input)[Columns],
                         float (&out)[Rows][Columns]) noexcept {
        Float total[Rows][Columns];
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) { total[r][c] = zero(); }
        }
        for (int group = 0; group < hidden / 64; ++group) {
            Float partial[Rows][Columns];
            for (int r = 0; r < Rows; ++r) {
                for (int c = 0; c < Columns; ++c) { partial[r][c] = zero(); }
            }
            for (int k = 0; k < 64; k += 32) {
                __m512bh x[Columns];
                for (int c = 0; c < Columns; ++c) {
                    x[c] = (__m512bh)_mm512_loadu_si512(input[c] + group * 64 + k);
                }
                for (int r = 0; r < Rows; ++r) {
                    const auto* source = matrix.codes + rows[r] * (hidden / 2) + group * 32 + k / 2;
                    const auto q = (__m512bh)common::cpu::avx512::decode_signed_q4_bf16(source);
                    for (int c = 0; c < Columns; ++c) {
                        partial[r][c] = _mm512_dpbf16_ps(partial[r][c], q, x[c]);
                    }
                }
            }
            for (int r = 0; r < Rows; ++r) {
                const auto s = scale(matrix.scales[rows[r] * (hidden / 64) + group]);
                for (int c = 0; c < Columns; ++c) {
                    total[r][c] = fma(partial[r][c], s, total[r][c]);
                }
            }
        }
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) { out[r][c] = sum(total[r][c]); }
        }
    }
};
} // namespace

const Kernel avx512_bf16 = make_kernel<Avx512Bf16>();

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
