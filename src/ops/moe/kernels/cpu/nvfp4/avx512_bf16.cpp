#include "ops/common/cpu/avx512.h"
#include "ops/moe/kernels/cpu/nvfp4/compute.h"

namespace ninfer::ops::detail::moe_cpu::nvfp4 {
namespace {
struct Simd : common::cpu::avx512::F32 {
    static constexpr int rows = 4, max_columns = 8;
    static constexpr bool bf16_gate = true;

    static Float decode(const std::uint8_t* codes, std::uint8_t scale) noexcept {
        return common::cpu::avx512::decode_nvfp4(codes, scale);
    }

    static bool safe(const std::uint16_t* input) noexcept {
        for (int k = 0; k < hidden; k += 32) {
            const auto words    = _mm512_loadu_si512(input + k);
            const auto absolute = _mm512_and_si512(words, _mm512_set1_epi16(0x7fff));
            const auto nonzero  = _mm512_test_epi16_mask(absolute, absolute);
            // Scaled E2M1 values are multiples of 2^-10. BF16 inputs >=2^-109 have
            // quantum >=2^-116, so dot products cannot produce FP32 subnormals.
            if (nonzero & _mm512_cmplt_epu16_mask(absolute, _mm512_set1_epi16(0x0900))) {
                return false;
            }
        }
        return true;
    }

    template <int Rows, int Columns>
    static void gate_dot(const Matrix& matrix, const int (&rows)[Rows],
                         const std::uint16_t* const (&x)[Columns],
                         float (&out)[Rows][Columns]) noexcept {
        Float sums[Rows][Columns];
        const std::uint8_t* codes[Rows];
        const std::uint8_t* scales[Rows];
        for (int r = 0; r < Rows; ++r) {
            codes[r]  = matrix.codes + rows[r] * (hidden / 2);
            scales[r] = matrix.scales + common::nvfp4_scale_byte_offset(rows[r], 0, hidden);
            for (int c = 0; c < Columns; ++c) { sums[r][c] = zero(); }
        }
        for (int k = 0; k < hidden; k += 32) {
            __m512bh input[Columns];
            for (int c = 0; c < Columns; ++c) { input[c] = (__m512bh)_mm512_loadu_si512(x[c] + k); }
            const int s = (k / 64) * 512 + ((k / 16) & 3);
            for (int r = 0; r < Rows; ++r) {
                const auto w = (__m512bh)common::cpu::avx512::decode_nvfp4_bf16(
                    codes[r] + k / 2, scales[r][s], scales[r][s + 1]);
                for (int c = 0; c < Columns; ++c) {
                    sums[r][c] = _mm512_dpbf16_ps(sums[r][c], w, input[c]);
                }
            }
        }
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) { out[r][c] = sum(sums[r][c]) / matrix.divisor; }
        }
    }
};
} // namespace

void run_avx512_bf16(void* argument, const CpuWorkerContext& worker) noexcept {
    compute<Simd>(argument, worker);
}
} // namespace ninfer::ops::detail::moe_cpu::nvfp4
