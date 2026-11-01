#include "core/cpu/system.h"

#include <cpuid.h>

#include <cstring>

namespace ninfer {

CpuFeatures cpu_detail::usable_features(const FeatureRegisters& r) noexcept {
    CpuFeatures result;
    constexpr auto avx_state = (1U << 26) | (1U << 27) | (1U << 28);
    if ((r.leaf1_ecx & avx_state) != avx_state || (r.xcr0 & 0x6U) != 0x6U) return result;
    const auto add = [&](bool available, CpuFeature feature) {
        if (available) result.bits |= static_cast<std::uint32_t>(feature);
    };
    add(r.leaf1_ecx & (1U << 12), CpuFeature::Fma);
    add(r.leaf1_ecx & (1U << 29), CpuFeature::F16c);
    add(r.leaf7_0_ebx & (1U << 5), CpuFeature::Avx2);
    add(r.leaf7_1_eax & (1U << 4), CpuFeature::AvxVnni);
    if ((r.xcr0 & 0xe6U) != 0xe6U || !(r.leaf7_0_ebx & (1U << 16))) return result;
    add(true, CpuFeature::Avx512F);
    add(r.leaf7_0_ebx & (1U << 17), CpuFeature::Avx512Dq);
    add(r.leaf7_0_ebx & (1U << 30), CpuFeature::Avx512Bw);
    add(r.leaf7_0_ebx & (1U << 31), CpuFeature::Avx512Vl);
    add(r.leaf7_0_ecx & (1U << 11), CpuFeature::Avx512Vnni);
    add(r.leaf7_1_eax & (1U << 5), CpuFeature::Avx512Bf16);
    return result;
}

CpuFeatures current_cpu_features() noexcept {
    cpu_detail::FeatureRegisters state;
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return {};
    state.leaf1_ecx = ecx;
    if ((ecx & ((1U << 26) | (1U << 27))) == ((1U << 26) | (1U << 27))) {
        unsigned low, high;
        asm volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        state.xcr0 = (static_cast<std::uint64_t>(high) << 32) | low;
    }
    if (__get_cpuid_max(0, nullptr) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        const auto max_subleaf = eax;
        state.leaf7_0_ebx      = ebx;
        state.leaf7_0_ecx      = ecx;
        if (max_subleaf >= 1) {
            __cpuid_count(7, 1, eax, ebx, ecx, edx);
            state.leaf7_1_eax = eax;
        }
    }
    return cpu_detail::usable_features(state);
}

CpuIdentity current_cpu_identity() noexcept {
    CpuIdentity result;
    unsigned eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
    std::memcpy(result.vendor.data(), &ebx, 4);
    std::memcpy(result.vendor.data() + 4, &edx, 4);
    std::memcpy(result.vendor.data() + 8, &ecx, 4);
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return result;
    const auto family = (eax >> 8) & 15U;
    result.family     = family + (family == 15 ? (eax >> 20) & 255U : 0);
    result.model    = ((eax >> 4) & 15U) | ((family == 6 || family == 15) ? (eax >> 12) & 240U : 0);
    result.stepping = eax & 15U;
    result.virtualized = (ecx & (1U << 31)) != 0;
    return result;
}

} // namespace ninfer
