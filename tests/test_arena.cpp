#include "qus/core/arena.h"
#include "qus/core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw expected exception\n";
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_ptr(void* actual, void* expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    int failures = 0;
    qus::DeviceContext ctx(0);

    failures += expect_throws<std::invalid_argument>([] { (void)qus::DeviceArena(0); },
                                                     "zero arena capacity");
    failures += expect_throws<std::invalid_argument>([] { (void)qus::PinnedHostBuffer(0); },
                                                     "zero pinned size");

    qus::DeviceArena arena(1024);
    failures += expect_size(arena.capacity(), 1024, "arena.capacity");
    failures += expect_size(arena.used(), 0, "arena.used initial");
    if (arena.base() == nullptr) {
        ++failures;
        std::cerr << "arena base is null\n";
    }

    auto* base    = static_cast<unsigned char*>(arena.base());
    qus::Tensor a = arena.alloc(qus::DType::BF16, {3, 5});
    failures += expect_ptr(a.data, base, "first allocation pointer");
    failures += expect_size(a.bytes(), 30, "first allocation bytes");
    failures += expect_size(arena.used(), 30, "arena.used after first allocation");

    qus::Tensor b = arena.alloc(qus::DType::U8, {17}, 64);
    failures += expect_ptr(b.data, base + 64, "second allocation pointer");
    if (reinterpret_cast<std::uintptr_t>(b.data) % 64 != 0) {
        ++failures;
        std::cerr << "second allocation is not 64-byte aligned\n";
    }
    failures += expect_size(arena.used(), 81, "arena.used after second allocation");

    const std::size_t mark = arena.mark();
    qus::Tensor transient  = arena.alloc(qus::DType::U8, {11}, 128);
    if (arena.used() <= mark) {
        ++failures;
        std::cerr << "transient allocation did not advance arena mark\n";
    }
    arena.rewind(mark);
    failures += expect_size(arena.used(), mark, "arena.used after rewind");
    qus::Tensor reused = arena.alloc(qus::DType::U8, {5}, 128);
    failures += expect_ptr(reused.data, transient.data, "allocation after rewind pointer");
    const std::size_t used_before_future_rewind = arena.used();
    arena.rewind(arena.capacity());
    failures +=
        expect_size(arena.used(), used_before_future_rewind, "arena.used after future rewind");

    failures += expect_throws<std::invalid_argument>(
        [&] { (void)arena.alloc(qus::DType::U8, {1}, 3); }, "invalid alignment");
    failures += expect_throws<std::bad_alloc>(
        [&] { (void)arena.alloc(qus::DType::FP32, {300}, 256); }, "arena oom");
    failures += expect_throws<std::overflow_error>(
        [&] {
            (void)arena.alloc(qus::DType::U8, {std::numeric_limits<std::int32_t>::max(),
                                               std::numeric_limits<std::int32_t>::max(), 1, 4});
        },
        "arena oversized allocation");

    arena.reset();
    failures += expect_size(arena.used(), 0, "arena.used after reset");
    qus::Tensor c = arena.alloc(qus::DType::U8, {4});
    failures += expect_ptr(c.data, base, "allocation after reset pointer");

    qus::DeviceArena moved(std::move(arena));
    if (arena.base() != nullptr || arena.capacity() != 0 || arena.used() != 0) {
        ++failures;
        std::cerr << "move construction did not clear source arena\n";
    }
    failures += expect_size(moved.capacity(), 1024, "moved arena capacity");

    qus::DeviceArena assigned(128);
    assigned = std::move(moved);
    if (moved.base() != nullptr || moved.capacity() != 0 || moved.used() != 0) {
        ++failures;
        std::cerr << "move assignment did not clear source arena\n";
    }
    void* assigned_base = assigned.base();
    assigned            = std::move(assigned);
    failures += expect_ptr(assigned.base(), assigned_base, "arena self-move base");
    failures += expect_size(assigned.capacity(), 1024, "arena self-move capacity");

    qus::PinnedHostBuffer pinned(128);
    if (pinned.data() == nullptr) {
        ++failures;
        std::cerr << "pinned data is null\n";
    }
    failures += expect_size(pinned.size(), 128, "pinned.size");
    std::memset(pinned.data(), 0x5a, pinned.size());

    qus::PinnedHostBuffer pinned_other(64);
    pinned_other = std::move(pinned);
    if (pinned.data() != nullptr || pinned.size() != 0) {
        ++failures;
        std::cerr << "move assignment did not clear source pinned buffer\n";
    }
    if (pinned_other.data() == nullptr || pinned_other.size() != 128) {
        ++failures;
        std::cerr << "move-assigned pinned buffer is not usable\n";
    }
    void* pinned_self = pinned_other.data();
    pinned_other      = std::move(pinned_other);
    failures += expect_ptr(pinned_other.data(), pinned_self, "pinned self-move data");
    failures += expect_size(pinned_other.size(), 128, "pinned self-move size");
    std::memset(pinned_other.data(), 0xa5, pinned_other.size());

    return failures == 0 ? 0 : fail("arena test failed");
}
