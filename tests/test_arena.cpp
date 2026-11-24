#include "core/arena.h"
#include "core/buffer.h"
#include "core/device.h"
#include "core/transfer.h"

#include <cuda_runtime.h>

#include <array>
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
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;
    ninfer::DeviceContext device;

    ninfer::DeviceBuffer empty;
    failures += expect_ptr(empty.data(), nullptr, "empty device buffer pointer");
    failures += expect_size(empty.size(), 0, "empty device buffer size");

    const std::array<std::uint8_t, 8> host_source{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<std::uint8_t, 8> host_destination{};
    ninfer::DeviceBuffer buffer(device.device(), host_source.size());
    buffer.copy_from_host(host_source.data(), host_source.size());
    buffer.copy_to_host(host_destination.data(), host_destination.size());
    if (host_destination != host_source) {
        ++failures;
        std::cerr << "device buffer round trip changed payload\n";
    }
    failures += expect_throws<std::out_of_range>(
        [&] { buffer.copy_from_host(host_source.data(), 2, buffer.size() - 1); },
        "device buffer upload range");
    ninfer::DeviceBuffer moved_buffer(std::move(buffer));
    failures += expect_ptr(buffer.data(), nullptr, "moved-from device buffer pointer");
    failures += expect_size(buffer.size(), 0, "moved-from device buffer size");
    failures += expect_size(moved_buffer.size(), host_source.size(), "moved device buffer size");

    ninfer::DeviceBuffer backing(device.device(), 1024);
    ninfer::MemoryArena arena(backing.span());
    failures += expect_size(arena.capacity(), 1024, "arena.capacity");
    failures += expect_size(arena.used(), 0, "arena.used initial");
    failures += expect_size(arena.peak_used(), 0, "arena.peak initial");
    if (arena.storage().data == nullptr) {
        ++failures;
        std::cerr << "arena base is null\n";
    }

    auto* base       = static_cast<unsigned char*>(arena.storage().data);
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

    ninfer::MemoryArena moved(std::move(arena));
    if (arena.storage().data != nullptr || arena.capacity() != 0 || arena.used() != 0) {
        ++failures;
        std::cerr << "move construction did not clear source arena\n";
    }
    failures += expect_size(arena.peak_used(), 0, "moved-from arena peak");
    failures += expect_size(moved.capacity(), 1024, "moved arena capacity");
    failures += expect_size(moved.peak_used(), 4, "moved arena peak");

    void* external = nullptr;
    CUDA_CHECK(cudaMalloc(&external, 512));
    {
        ninfer::MemoryArena borrowed(ninfer::MemorySpan{external, 512, device.device()});
        failures += expect_ptr(borrowed.storage().data, external, "borrowed arena base");
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

    ninfer::PinnedHostBuffer result(128);
    ninfer::DeviceBuffer transferred(device.device(), 128);
    ninfer::copy_bytes_async(transferred.span(), pinned.span(), device.transfer_view());
    ninfer::CudaCompletionEvent uploaded(device);
    uploaded.record(device.transfer_stream());
    uploaded.wait(device.stream());
    ninfer::copy_bytes_async(result.span(), transferred.span(), device.execution_view());
    device.synchronize();
    if (std::memcmp(pinned.data(), result.data(), 128) != 0) {
        ++failures;
        std::cerr << "stream-ordered transfer changed payload\n";
    }
    ninfer::HostBuffer pageable(128);
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::copy_bytes_async(transferred.span(), pageable.span(), device.transfer_view());
        },
        "asynchronous pageable upload");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            auto wrong   = transferred.span();
            wrong.device = ninfer::Device::cuda(device.device().index() + 1);
            ninfer::copy_bytes(result.span(), wrong, device.execution_view());
        },
        "transfer execution device mismatch");

    ninfer::HostBuffer registered(4096, 4096);
    auto* source_bytes = static_cast<unsigned char*>(registered.data());
    for (std::size_t i = 0; i < registered.size(); ++i) {
        source_bytes[i] = (i * 13 + i / 31) % 251;
    }
    ninfer::CudaHostRegistration registration(registered.span(), device.device());
    const auto view         = registration.view();
    auto moved_registration = std::move(registration);
    failures += expect_size(registration.view().host.bytes, 0, "moved registration is empty");
    registration = std::move(moved_registration);
    ninfer::CudaHostCopyBatch batch(device.device(), 2);
    {
        ninfer::HostBuffer no_data;
        ninfer::CudaHostRegistration no_registration(no_data.span(), device.device());
        batch.append(transferred.span().subspan(0, 0), no_registration.view());
        batch.submit(device.transfer_view());
        failures += expect_size(batch.size(), 0, "empty registered transfer");
    }
    batch.append(transferred.span().subspan(0, 32), view.subspan(71, 32));
    batch.append(transferred.span().subspan(32, 96), view.subspan(1001, 96));
    batch.submit(device.transfer_view());
    uploaded.record(device.transfer_stream());
    uploaded.wait(device.stream());
    ninfer::copy_bytes_async(result.span(), transferred.span(), device.execution_view());
    device.synchronize();
    if (std::memcmp(result.data(), source_bytes + 71, 32) ||
        std::memcmp(static_cast<unsigned char*>(result.data()) + 32, source_bytes + 1001, 96)) {
        ++failures;
        std::cerr << "registered batch transfer changed bytes\n";
    }
    ninfer::copy_host_2d_async(transferred.span(), 32, view.subspan(111, 97), 27, 16, 4,
                               device.transfer_view());
    uploaded.record(device.transfer_stream());
    uploaded.wait(device.stream());
    ninfer::copy_bytes_async(result.span(), transferred.span(), device.execution_view());
    device.synchronize();
    for (std::size_t row = 0; row < 4; ++row) {
        if (std::memcmp(static_cast<unsigned char*>(result.data()) + row * 32,
                        source_bytes + 111 + row * 27, 16)) {
            ++failures;
        }
    }
    failures += expect_throws<std::out_of_range>(
        [&] {
            ninfer::copy_host_2d_async(transferred.span(), 32, view.subspan(0, 96), 27, 16, 4,
                                       device.transfer_view());
        },
        "registered strided transfer extent");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            auto wrong        = view;
            wrong.cuda_device = ninfer::Device::cuda(device.device().index() + 1);
            batch.clear();
            batch.append(transferred.span(), wrong.subspan(0, 128));
        },
        "registered transfer execution device mismatch");

    return failures == 0 ? 0 : fail("arena test failed");
}
