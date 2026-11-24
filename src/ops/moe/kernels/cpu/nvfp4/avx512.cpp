#include "ops/common/cpu/avx512.h"
#include "ops/moe/kernels/cpu/nvfp4/compute.h"

namespace ninfer::ops::detail::moe_cpu::nvfp4 {
namespace {
struct Simd : common::cpu::avx512::F32 {
    static constexpr int rows = 4, max_columns = 8;
    static constexpr bool bf16_gate = false;

    static Float decode(const std::uint8_t* codes, std::uint8_t scale) noexcept {
        return common::cpu::avx512::decode_nvfp4(codes, scale);
    }
};
} // namespace

void run_avx512(void* argument, const CpuWorkerContext& worker) noexcept {
    compute<Simd>(argument, worker);
}
} // namespace ninfer::ops::detail::moe_cpu::nvfp4
