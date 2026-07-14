#include "ninfer/core/tensor.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

template <typename Fn>
int expect_invalid(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << " did not throw std::invalid_argument\n";
    return 1;
}

template <typename Fn>
int expect_overflow(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::overflow_error&) { return 0; }
    std::cerr << label << " did not throw std::overflow_error\n";
    return 1;
}

int expect_i64(std::int64_t actual, std::int64_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& t, const std::int32_t (&expected)[4], const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got " << t.ne[i]
                      << '\n';
        }
    }
    return failures;
}

int check_strides(const ninfer::Tensor& t, const std::int64_t (&expected)[4], const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (t.nb[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".nb[" << i << "] expected " << expected[i] << ", got " << t.nb[i]
                      << '\n';
        }
    }
    return failures;
}

} // namespace

int main() {
    alignas(16) unsigned char storage[512] = {};
    auto* base                             = storage;
    int failures                           = 0;

    ninfer::Tensor t(base, ninfer::DType::BF16, {2, 3, 4});
    failures += check_shape(t, {2, 3, 4, 1}, "t");
    failures += check_strides(t, {2, 4, 12, 48}, "t");
    failures += expect_i64(t.numel(), 24, "t.numel");
    failures += expect_size(t.bytes(), 48, "t.bytes");
    if (!t.is_contiguous()) {
        ++failures;
        std::cerr << "t expected contiguous\n";
    }

    ninfer::Tensor viewed = t.view({4, 6});
    if (viewed.data != base) {
        ++failures;
        std::cerr << "view changed data pointer\n";
    }
    failures += check_shape(viewed, {4, 6, 1, 1}, "viewed");
    failures += check_strides(viewed, {2, 8, 48, 48}, "viewed");
    failures += expect_i64(viewed.numel(), 24, "viewed.numel");
    if (!viewed.is_contiguous()) {
        ++failures;
        std::cerr << "viewed expected contiguous\n";
    }
    failures += expect_invalid([&] { (void)t.view({5, 5}); }, "mismatched view");

    ninfer::Tensor sliced = t.slice(1, 1, 2);
    if (sliced.data != base + 4) {
        ++failures;
        std::cerr << "slice did not advance by dim-1 stride\n";
    }
    failures += check_shape(sliced, {2, 2, 4, 1}, "sliced");
    failures += check_strides(sliced, {2, 4, 12, 48}, "sliced");
    if (sliced.is_contiguous()) {
        ++failures;
        std::cerr << "sliced expected non-contiguous\n";
    }

    ninfer::Tensor permuted = t.permute({2, 1, 0, 3});
    if (permuted.data != base) {
        ++failures;
        std::cerr << "permute changed data pointer\n";
    }
    failures += check_shape(permuted, {4, 3, 2, 1}, "permuted");
    failures += check_strides(permuted, {12, 4, 2, 48}, "permuted");
    if (permuted.is_contiguous()) {
        ++failures;
        std::cerr << "permuted expected non-contiguous\n";
    }

    ninfer::Tensor reshaped = t.reshape({6, 4});
    if (reshaped.data != base) {
        ++failures;
        std::cerr << "reshape changed data pointer\n";
    }
    failures += check_shape(reshaped, {6, 4, 1, 1}, "reshaped");
    failures += check_strides(reshaped, {2, 12, 48, 48}, "reshaped");
    failures += expect_i64(reshaped.numel(), 24, "reshaped.numel");

    failures += expect_invalid([&] { (void)sliced.reshape({4, 4}); }, "non-contiguous reshape");
    failures += expect_invalid([&] { (void)ninfer::Tensor(base, ninfer::DType::BF16, {2, 0}); },
                               "zero dimension");
    failures += expect_invalid([&] { (void)t.slice(4, 0, 1); }, "invalid slice dim");
    failures += expect_invalid([&] { (void)t.slice(1, 2, 2); }, "out-of-range slice");
    failures += expect_invalid([&] { (void)t.permute({0, 0, 1, 2}); }, "duplicate permutation");
    failures += expect_overflow(
        [&] {
            (void)ninfer::Tensor(nullptr, ninfer::DType::FP32,
                              {std::numeric_limits<std::int32_t>::max(),
                               std::numeric_limits<std::int32_t>::max(), 2});
        },
        "oversized tensor");
    failures += expect_overflow(
        [&] {
            ninfer::Tensor huge_offset(base, ninfer::DType::U8,
                                    {std::numeric_limits<std::int32_t>::max(),
                                     std::numeric_limits<std::int32_t>::max(), 1, 4});
            (void)huge_offset.slice(3, 3, 1);
        },
        "oversized slice offset");
    failures += expect_overflow(
        [&] {
            ninfer::Tensor null_huge_offset(nullptr, ninfer::DType::U8,
                                         {std::numeric_limits<std::int32_t>::max(),
                                          std::numeric_limits<std::int32_t>::max(), 1, 4});
            (void)null_huge_offset.slice(3, 3, 1);
        },
        "oversized null slice offset");

    return failures == 0 ? 0 : fail("tensor test failed");
}
