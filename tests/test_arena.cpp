#include "core/arena.h"
#include "core/device.h"

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
    ninfer::DeviceContext ctx(0);

    failures += expect_throws<std::invalid_argument>([] { (void)ninfer::DeviceArena(0); },
                                                     "zero arena capacity");
    failures += expect_throws<std::invalid_argument>([] { (void)ninfer::PinnedHostBuffer(0); },
                                                     "zero pinned size");

    ninfer::DeviceArena arena(1024);
    failures += expect_size(arena.capacity(), 1024, "arena.capacity");
    failures += expect_size(arena.used(), 0, "arena.used initial");
    failures += expect_size(arena.peak_used(), 0, "arena.peak initial");
    if (arena.base() == nullptr) {
        ++failures;
        std::cerr << "arena base is null\n";
    }

    auto* base    = static_cast<unsigned char*>(arena.base());
    ninfer::Tensor a = arena.alloc(ninfer::DType::BF16, {3, 5});
    failures += expect_ptr(a.data, base, "first allocation pointer");
    failures += expect_size(a.bytes(), 30, "first allocation bytes");
    failures += expect_size(arena.used(), 30, "arena.used after first allocation");
    failures += expect_size(arena.peak_used(), 30, "arena.peak after first allocation");

    ninfer::Tensor b = arena.alloc(ninfer::DType::U8, {17}, 64);
    failures += expect_ptr(b.data, base + 64, "second allocation pointer");
    if (reinterpret_cast<std::uintptr_t>(b.data) % 64 != 0) {
        ++failures;
        std::cerr << "second allocation is not 64-byte aligned\n";
    }
    failures += expect_size(arena.used(), 81, "arena.used after second allocation");
    failures += expect_size(arena.peak_used(), 81, "arena.peak after second allocation");

    const std::size_t used_before_scope = arena.used();
    void* transient_ptr                 = nullptr;
    std::size_t peak_after_transient    = 0;
    {
        auto outer_scope       = arena.scope();
        ninfer::Tensor transient  = arena.alloc(ninfer::DType::U8, {11}, 128);
        transient_ptr          = transient.data;
        const std::size_t used = arena.used();
        if (used <= used_before_scope) {
            ++failures;
            std::cerr << "transient allocation did not advance arena cursor\n";
        }
        {
            auto inner_scope = arena.scope();
            (void)arena.alloc(ninfer::DType::U8, {7}, 64);
        }
        failures += expect_size(arena.used(), used, "arena.used after nested scope");
        peak_after_transient = arena.peak_used();
    }
    failures += expect_size(arena.peak_used(), peak_after_transient, "arena.peak after scope exit");
    failures += expect_size(arena.used(), used_before_scope, "arena.used after scope exit");
    ninfer::Tensor reused = arena.alloc(ninfer::DType::U8, {5}, 128);
    failures += expect_ptr(reused.data, transient_ptr, "allocation after scope pointer");

    const std::size_t used_before_exception_scope = arena.used();
    failures += expect_throws<std::runtime_error>(
        [&] {
            auto exception_scope = arena.scope();
            (void)arena.alloc(ninfer::DType::U8, {9}, 64);
            throw std::runtime_error("scope unwind");
        },
        "arena scope exception");
    failures +=
        expect_size(arena.used(), used_before_exception_scope, "arena.used after exception scope");

    const std::size_t used_before_failures = arena.used();
    const std::size_t peak_before_failures = arena.peak_used();
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)arena.alloc(ninfer::DType::U8, {1}, 3); }, "invalid alignment");
    failures +=
        expect_size(arena.used(), used_before_failures, "arena.used after invalid alignment");
    failures +=
        expect_size(arena.peak_used(), peak_before_failures, "arena.peak after invalid alignment");
    failures += expect_throws<std::bad_alloc>(
        [&] { (void)arena.alloc(ninfer::DType::FP32, {300}, 256); }, "arena oom");
    failures += expect_size(arena.used(), used_before_failures, "arena.used after oom");
    failures += expect_size(arena.peak_used(), peak_before_failures, "arena.peak after oom");
    failures += expect_throws<std::overflow_error>(
        [&] {
            (void)arena.alloc(ninfer::DType::U8, {std::numeric_limits<std::int32_t>::max(),
                                               std::numeric_limits<std::int32_t>::max(), 1, 4});
        },
        "arena oversized allocation");

    arena.reset();
    failures += expect_size(arena.used(), 0, "arena.used after reset");
    failures += expect_size(arena.peak_used(), peak_before_failures, "arena.peak after reset");
    arena.reset_peak();
    failures += expect_size(arena.peak_used(), 0, "arena.peak after reset_peak on empty arena");
    ninfer::Tensor c = arena.alloc(ninfer::DType::U8, {4});
    failures += expect_ptr(c.data, base, "allocation after reset pointer");
    failures += expect_size(arena.peak_used(), 4, "arena.peak after reset allocation");

    ninfer::DeviceArena moved(std::move(arena));
    if (arena.base() != nullptr || arena.capacity() != 0 || arena.used() != 0) {
        ++failures;
        std::cerr << "move construction did not clear source arena\n";
    }
    failures += expect_size(arena.peak_used(), 0, "moved-from arena peak");
    failures += expect_size(moved.capacity(), 1024, "moved arena capacity");
    failures += expect_size(moved.peak_used(), 4, "moved arena peak");

    ninfer::DeviceArena assigned(128);
    assigned = std::move(moved);
    if (moved.base() != nullptr || moved.capacity() != 0 || moved.used() != 0) {
        ++failures;
        std::cerr << "move assignment did not clear source arena\n";
    }
    failures += expect_size(moved.peak_used(), 0, "move-assigned-from arena peak");
    failures += expect_size(assigned.peak_used(), 4, "move-assigned arena peak");
    void* assigned_base = assigned.base();
    assigned            = std::move(assigned);
    failures += expect_ptr(assigned.base(), assigned_base, "arena self-move base");
    failures += expect_size(assigned.capacity(), 1024, "arena self-move capacity");
    failures += expect_size(assigned.peak_used(), 4, "arena self-move peak");

    ninfer::PinnedHostBuffer pinned(128);
    if (pinned.data() == nullptr) {
        ++failures;
        std::cerr << "pinned data is null\n";
    }
    failures += expect_size(pinned.size(), 128, "pinned.size");
    std::memset(pinned.data(), 0x5a, pinned.size());

    ninfer::PinnedHostBuffer pinned_other(64);
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
