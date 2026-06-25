#include "qus/core/device.h"
#include "qus/core/kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

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

int check_shape(const qus::Tensor& t, const std::int32_t (&expected)[4], const char* label) {
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

int fill_slot(const qus::KVSlot& slot, unsigned char k_value, unsigned char v_value) {
    CUDA_CHECK(cudaMemset(slot.k.data, k_value, slot.k.bytes()));
    CUDA_CHECK(cudaMemset(slot.v.data, v_value, slot.v.bytes()));
    return 0;
}

int expect_device_bytes(const qus::Tensor& tensor, unsigned char expected, const char* label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (unsigned char value : host) {
        if (value != expected) {
            std::cerr << label << " expected byte 0x" << std::hex << static_cast<int>(expected)
                      << ", got 0x" << static_cast<int>(value) << std::dec << '\n';
            return 1;
        }
    }
    return 0;
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
    qus::DeviceArena cache_arena(8192);
    qus::KVCache cache(cache_arena, 2, 8, 4, 16, qus::DType::U8);

    failures += expect_size(cache.layer_count(), 2, "cache.layer_count");
    failures += expect_size(cache.pos, 0, "cache.pos initial");
    failures += expect_size(cache.k.size(), 2, "cache.k.size");
    failures += expect_size(cache.v.size(), 2, "cache.v.size");
    for (std::size_t layer = 0; layer < 2; ++layer) {
        failures += check_shape(cache.k[layer], {4, 16, 8, 1}, "cache.k");
        failures += check_shape(cache.v[layer], {4, 16, 8, 1}, "cache.v");
        if (cache.k[layer].data == cache.v[layer].data) {
            ++failures;
            std::cerr << "K/V tensors alias for layer " << layer << '\n';
        }
    }

    failures +=
        expect_throws<std::out_of_range>([&] { (void)cache.slot(0, 0); }, "slot before advance");

    qus::KVSlot l0p0 = cache.append_slot(0);
    qus::KVSlot l1p0 = cache.append_slot(1);
    failures += check_shape(l0p0.k, {4, 16, 1, 1}, "l0p0.k");
    failures += check_shape(l0p0.v, {4, 16, 1, 1}, "l0p0.v");
    failures += fill_slot(l0p0, 0x11, 0x12);
    failures += fill_slot(l1p0, 0x21, 0x22);
    cache.advance();
    failures += expect_size(cache.pos, 1, "cache.pos after first advance");
    failures += expect_device_bytes(cache.slot(0, 0).k, 0x11, "layer0 pos0 K");
    qus::Tensor l0p0_flat = cache.slot(0, 0).k.reshape({64});
    failures += check_shape(l0p0_flat, {64, 1, 1, 1}, "l0p0_flat");
    failures += expect_device_bytes(cache.slot(0, 0).v, 0x12, "layer0 pos0 V");
    failures += expect_device_bytes(cache.slot(1, 0).k, 0x21, "layer1 pos0 K");
    failures += expect_device_bytes(cache.slot(1, 0).v, 0x22, "layer1 pos0 V");

    qus::KVSlot l0p1 = cache.append_slot(0);
    qus::KVSlot l1p1 = cache.append_slot(1);
    failures += fill_slot(l0p1, 0x31, 0x32);
    failures += fill_slot(l1p1, 0x41, 0x42);
    cache.advance();
    failures += expect_device_bytes(cache.slot(0, 0).k, 0x11, "layer0 pos0 K after pos1");
    failures += expect_device_bytes(cache.slot(0, 1).k, 0x31, "layer0 pos1 K");
    failures += expect_device_bytes(cache.slot(1, 0).v, 0x22, "layer1 pos0 V after pos1");
    failures += expect_device_bytes(cache.slot(1, 1).v, 0x42, "layer1 pos1 V");

    void* k0_base = cache.k[0].data;
    void* v0_base = cache.v[0].data;
    cache.reset();
    failures += expect_size(cache.pos, 0, "cache.pos after reset");
    if (cache.k[0].data != k0_base || cache.v[0].data != v0_base) {
        ++failures;
        std::cerr << "reset changed layer base pointers\n";
    }

    failures += expect_throws<std::invalid_argument>(
        [&] { qus::KVCache invalid(cache_arena, 0, 8, 4, 16, qus::DType::U8); }, "zero layers");
    failures += expect_throws<std::invalid_argument>(
        [&] { qus::KVCache invalid(cache_arena, 1, 0, 4, 16, qus::DType::U8); },
        "zero max_context");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            qus::KVCache invalid(
                cache_arena, 1,
                static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) + 1U, 4, 16,
                qus::DType::U8);
        },
        "too large max_context");
    failures += expect_throws<std::invalid_argument>(
        [&] { qus::KVCache invalid(cache_arena, 1, 8, 0, 16, qus::DType::U8); }, "zero kv heads");
    failures += expect_throws<std::invalid_argument>(
        [&] { qus::KVCache invalid(cache_arena, 1, 8, 4, 0, qus::DType::U8); }, "zero head_dim");

    failures += expect_throws<std::out_of_range>([&] { (void)cache.append_slot(2); },
                                                 "invalid append layer");

    while (cache.pos < cache.max_context) { cache.advance(); }
    failures += expect_throws<std::out_of_range>([&] { (void)cache.slot(0, cache.pos); },
                                                 "read at current pos");
    failures += expect_throws<std::out_of_range>([&] { (void)cache.append_slot(0); },
                                                 "append at full capacity");
    failures +=
        expect_throws<std::out_of_range>([&] { cache.advance(); }, "advance at full capacity");

    qus::DeviceArena small_arena(512);
    const std::size_t before_used = small_arena.used();
    failures += expect_throws<std::bad_alloc>(
        [&] { qus::KVCache too_big(small_arena, 2, 8, 4, 16, qus::DType::U8); },
        "undersized cache arena");
    failures += expect_size(small_arena.used(), before_used, "small arena used after failed cache");

    return failures == 0 ? 0 : fail("kv cache test failed");
}
