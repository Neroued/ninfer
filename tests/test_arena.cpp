#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
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
    CUDA_CHECK(cudaSetDevice(0));

    ninfer::DeviceArena arena(1024);
    failures += expect_size(arena.capacity(), 1024, "arena.capacity");
    failures += expect_size(arena.used(), 0, "arena.used initial");
    failures += expect_size(arena.peak_used(), 0, "arena.peak initial");
    if (arena.base() == nullptr) {
        ++failures;
        std::cerr << "arena base is null\n";
    }

    auto* base       = static_cast<unsigned char*>(arena.base());
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
        auto outer_scope         = arena.scope();
        ninfer::Tensor transient = arena.alloc(ninfer::DType::U8, {11}, 128);
        transient_ptr            = transient.data;
        const std::size_t used   = arena.used();
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
    failures += expect_throws<std::bad_alloc>(
        [&] { (void)arena.alloc(ninfer::DType::FP32, {300}, 256); }, "arena oom");
    failures += expect_size(arena.used(), used_before_failures, "arena.used after oom");
    failures += expect_size(arena.peak_used(), peak_before_failures, "arena.peak after oom");

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

    void* external = nullptr;
    CUDA_CHECK(cudaMalloc(&external, 512));
    {
        ninfer::DeviceArena borrowed(ninfer::DeviceSpan{external, 512});
        failures += expect_ptr(borrowed.base(), external, "borrowed arena base");
        failures += expect_size(borrowed.capacity(), 512, "borrowed arena capacity");
        const ninfer::Tensor item = borrowed.alloc(ninfer::DType::U8, {17}, 64);
        failures += expect_ptr(item.data, external, "borrowed arena allocation");
        failures += expect_size(borrowed.used(), 17, "borrowed arena used");
    }
    CUDA_CHECK(cudaMemset(external, 0, 512));
    CUDA_CHECK(cudaFree(external));

    ninfer::PinnedHostBuffer pinned(128);
    if (pinned.data() == nullptr) {
        ++failures;
        std::cerr << "pinned data is null\n";
    }
    failures += expect_size(pinned.size(), 128, "pinned.size");
    std::memset(pinned.data(), 0x5a, pinned.size());

    return failures == 0 ? 0 : fail("arena test failed");
}
