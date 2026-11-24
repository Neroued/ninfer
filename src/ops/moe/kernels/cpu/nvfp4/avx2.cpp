#include "ops/common/cpu/avx2.h"
#include "ops/moe/kernels/cpu/nvfp4/compute.h"

namespace ninfer::ops::detail::moe_cpu::nvfp4 {
namespace {
struct Simd : common::cpu::avx2::F32 {
    static constexpr int rows = 2, max_columns = 2;
    static constexpr bool bf16_gate = false;

    static Float decode(const std::uint8_t* codes, std::uint8_t scale) noexcept {
        return common::cpu::avx2::decode_nvfp4(codes, scale);
    }
};
} // namespace

void run_avx2(void* argument, const CpuWorkerContext& worker) noexcept {
    compute<Simd>(argument, worker);
}
} // namespace ninfer::ops::detail::moe_cpu::nvfp4
