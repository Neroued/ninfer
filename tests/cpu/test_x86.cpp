#include "core/cpu/system.h"

#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void features() {
    cpu_detail::FeatureRegisters r{0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xe6};
    auto f = cpu_detail::usable_features(r);
    require(f.has(CpuFeature::Avx2) && f.has(CpuFeature::Avx512Bf16), "missing SIMD features");
    r.xcr0 = 6;
    f      = cpu_detail::usable_features(r);
    require(f.has(CpuFeature::AvxVnni) && !f.has(CpuFeature::Avx512F) &&
                !f.has(CpuFeature::Avx512Bf16),
            "OS-disabled AVX-512 accepted");
    for (auto state : {0U, 2U, 4U}) {
        r.xcr0 = state;
        require(cpu_detail::usable_features(r).bits == 0, "incomplete AVX state accepted");
    }
    r.xcr0 = 0xe6;
    r.leaf1_ecx &= ~(1U << 27);
    require(cpu_detail::usable_features(r).bits == 0, "OSXSAVE not required");
    r.leaf1_ecx = 0xffffffffU;
    r.leaf7_1_eax &= ~(1U << 5);
    f = cpu_detail::usable_features(r);
    require(f.has(CpuFeature::Avx512F) && !f.has(CpuFeature::Avx512Bf16),
            "AVX-512 extensions conflated");
    require(current_cpu_identity().vendor[0] != 0, "missing CPU identity");
}

} // namespace

int main() {
    try {
        features();
        std::cout << "OK CPU x86 features\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
