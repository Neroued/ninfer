#include "qus/core/dtype.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {

int expect_size(qus::DType dtype, std::size_t expected, const char* name) {
    const std::size_t actual = qus::dtype_size(dtype);
    if (actual != expected) {
        std::cerr << name << " expected " << expected << " bytes, got " << actual << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    static_assert(std::is_same_v<std::underlying_type_t<qus::DType>, std::uint8_t>);
    static_assert(sizeof(qus::DType) == 1);

    int failures = 0;
    failures += expect_size(qus::DType::BF16, 2, "BF16");
    failures += expect_size(qus::DType::FP32, 4, "FP32");
    failures += expect_size(qus::DType::I32, 4, "I32");
    failures += expect_size(qus::DType::U8, 1, "U8");

    bool threw = false;
    try {
        (void)qus::dtype_size(static_cast<qus::DType>(255));
    } catch (const std::invalid_argument&) { threw = true; }
    if (!threw) {
        std::cerr << "invalid DType did not throw std::invalid_argument\n";
        ++failures;
    }

    return failures == 0 ? 0 : 1;
}
