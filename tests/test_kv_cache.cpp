#include "core/device.h"
#include "targets/qwen3_6_27b_rtx5090/impl/state/kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

namespace q27 = ninfer::targets::qwen3_6_27b_rtx5090::detail;

struct PlannedCache {
    q27::KVCacheLayout layout;
    std::size_t bytes = 0;
};

PlannedCache plan_cache(std::uint32_t layers, std::uint32_t max_context, std::int32_t heads,
                        std::int32_t head_dim, ninfer::DType dtype = ninfer::DType::BF16,
                        std::int32_t quant_group = 0) {
    ninfer::LayoutBuilder builder;
    auto layout =
        q27::plan_kv_cache(builder, layers, max_context, heads, head_dim, dtype, quant_group);
    return PlannedCache{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
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

int fill_slot(const q27::KVHeadSlot& slot, unsigned char k_value, unsigned char v_value) {
    CUDA_CHECK(cudaMemset(slot.k.data, k_value, slot.k.bytes()));
    CUDA_CHECK(cudaMemset(slot.v.data, v_value, slot.v.bytes()));
    return 0;
}

int fill_scales(const q27::KVHeadSlot& slot, unsigned char k_value, unsigned char v_value) {
    CUDA_CHECK(cudaMemset(slot.k_scale.data, k_value, slot.k_scale.bytes()));
    CUDA_CHECK(cudaMemset(slot.v_scale.data, v_value, slot.v_scale.bytes()));
    return 0;
}

int expect_device_bytes(const ninfer::Tensor& tensor, unsigned char expected, const char* label) {
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
    ninfer::DeviceContext ctx(0);
    auto cache_plan = plan_cache(2, 8, 4, 16);
    ninfer::DeviceArena cache_arena(cache_plan.bytes);
    q27::KVCache cache({cache_arena.base(), cache_arena.capacity()}, cache_plan.layout);

    failures += expect_size(cache.layer_count(), 2, "cache.layer_count");
    failures += expect_size(cache.pos, 0, "cache.pos initial");
    failures += expect_size(cache.padded_context, 128, "cache.padded_context");
    failures += expect_size(cache.quant_group, 0, "cache.quant_group bf16");
    failures += expect_size(cache.k.size(), 2, "cache.k.size");
    failures += expect_size(cache.v.size(), 2, "cache.v.size");
    failures += expect_size(cache.k_scale.size(), 0, "cache.k_scale.size bf16");
    failures += expect_size(cache.v_scale.size(), 0, "cache.v_scale.size bf16");
    for (std::size_t layer = 0; layer < 2; ++layer) {
        failures += check_shape(cache.k[layer], {16, 128, 4, 1}, "cache.k");
        failures += check_shape(cache.v[layer], {16, 128, 4, 1}, "cache.v");
        if (cache.k[layer].data == cache.v[layer].data) {
            ++failures;
            std::cerr << "K/V tensors alias for layer " << layer << '\n';
        }
    }

    for (std::int32_t head = 0; head < 4; ++head) {
        q27::KVHeadSlot l0p0 = cache.append_slot(0, head);
        q27::KVHeadSlot l1p0 = cache.append_slot(1, head);
        failures += check_shape(l0p0.k, {16, 1, 1, 1}, "l0p0.k");
        failures += check_shape(l0p0.v, {16, 1, 1, 1}, "l0p0.v");
        if (l0p0.k_scale.data != nullptr || l0p0.v_scale.data != nullptr) {
            ++failures;
            std::cerr << "BF16 slot exposed scale tensors\n";
        }
        failures += fill_slot(l0p0, static_cast<unsigned char>(0x10 + head),
                              static_cast<unsigned char>(0x20 + head));
        failures += fill_slot(l1p0, static_cast<unsigned char>(0x30 + head),
                              static_cast<unsigned char>(0x40 + head));
    }
    cache.advance();
    failures += expect_size(cache.pos, 1, "cache.pos after first advance");
    for (std::int32_t head = 0; head < 4; ++head) {
        failures += expect_device_bytes(cache.slot(0, 0, head).k,
                                        static_cast<unsigned char>(0x10 + head), "layer0 pos0 K");
        failures += expect_device_bytes(cache.slot(0, 0, head).v,
                                        static_cast<unsigned char>(0x20 + head), "layer0 pos0 V");
        failures += expect_device_bytes(cache.slot(1, 0, head).k,
                                        static_cast<unsigned char>(0x30 + head), "layer1 pos0 K");
        failures += expect_device_bytes(cache.slot(1, 0, head).v,
                                        static_cast<unsigned char>(0x40 + head), "layer1 pos0 V");
    }
    ninfer::Tensor l0p0_flat = cache.slot(0, 0, 0).k.reshape({16});
    failures += check_shape(l0p0_flat, {16, 1, 1, 1}, "l0p0_flat");

    for (std::int32_t head = 0; head < 4; ++head) {
        q27::KVHeadSlot l0p1 = cache.append_slot(0, head);
        q27::KVHeadSlot l1p1 = cache.append_slot(1, head);
        failures += fill_slot(l0p1, static_cast<unsigned char>(0x50 + head),
                              static_cast<unsigned char>(0x60 + head));
        failures += fill_slot(l1p1, static_cast<unsigned char>(0x70 + head),
                              static_cast<unsigned char>(0x80 + head));
    }
    cache.advance();
    for (std::int32_t head = 0; head < 4; ++head) {
        failures +=
            expect_device_bytes(cache.slot(0, 0, head).k, static_cast<unsigned char>(0x10 + head),
                                "layer0 pos0 K after pos1");
        failures += expect_device_bytes(cache.slot(0, 1, head).k,
                                        static_cast<unsigned char>(0x50 + head), "layer0 pos1 K");
        failures +=
            expect_device_bytes(cache.slot(1, 0, head).v, static_cast<unsigned char>(0x40 + head),
                                "layer1 pos0 V after pos1");
        failures += expect_device_bytes(cache.slot(1, 1, head).v,
                                        static_cast<unsigned char>(0x80 + head), "layer1 pos1 V");
    }

    cache.rewind(1);
    failures += expect_size(cache.pos, 1, "cache.pos after rewind");
    for (std::int32_t head = 0; head < 4; ++head) {
        q27::KVHeadSlot l0p1 = cache.append_slot(0, head);
        q27::KVHeadSlot l1p1 = cache.append_slot(1, head);
        failures += fill_slot(l0p1, static_cast<unsigned char>(0x90 + head),
                              static_cast<unsigned char>(0xa0 + head));
        failures += fill_slot(l1p1, static_cast<unsigned char>(0xb0 + head),
                              static_cast<unsigned char>(0xc0 + head));
    }
    cache.advance();
    failures += expect_size(cache.pos, 2, "cache.pos after rewrite advance");
    for (std::int32_t head = 0; head < 4; ++head) {
        failures +=
            expect_device_bytes(cache.slot(0, 0, head).k, static_cast<unsigned char>(0x10 + head),
                                "layer0 pos0 K after rewrite");
        failures +=
            expect_device_bytes(cache.slot(0, 1, head).k, static_cast<unsigned char>(0x90 + head),
                                "layer0 pos1 K rewritten");
        failures +=
            expect_device_bytes(cache.slot(1, 0, head).v, static_cast<unsigned char>(0x40 + head),
                                "layer1 pos0 V after rewrite");
        failures +=
            expect_device_bytes(cache.slot(1, 1, head).v, static_cast<unsigned char>(0xc0 + head),
                                "layer1 pos1 V rewritten");
    }

    void* k0_base = cache.k[0].data;
    void* v0_base = cache.v[0].data;
    cache.reset();
    failures += expect_size(cache.pos, 0, "cache.pos after reset");
    if (cache.k[0].data != k0_base || cache.v[0].data != v0_base) {
        ++failures;
        std::cerr << "reset changed layer base pointers\n";
    }

    auto int8_plan = plan_cache(1, 8, 2, 64, ninfer::DType::I8);
    ninfer::DeviceArena int8_arena(int8_plan.bytes);
    q27::KVCache int8_cache({int8_arena.base(), int8_arena.capacity()}, int8_plan.layout);
    failures += expect_size(int8_cache.layer_count(), 1, "int8 layer_count");
    failures += expect_size(int8_cache.quant_group, q27::kKvQuantGroup, "int8 quant_group");
    failures += expect_size(int8_cache.k.size(), 1, "int8 k.size");
    failures += expect_size(int8_cache.v.size(), 1, "int8 v.size");
    failures += expect_size(int8_cache.k_scale.size(), 1, "int8 k_scale.size");
    failures += expect_size(int8_cache.v_scale.size(), 1, "int8 v_scale.size");
    failures += check_shape(int8_cache.k[0], {64, 128, 2, 1}, "int8 k");
    failures += check_shape(int8_cache.v[0], {64, 128, 2, 1}, "int8 v");
    failures += check_shape(int8_cache.k_scale[0], {1, 128, 2, 1}, "int8 k_scale");
    failures += check_shape(int8_cache.v_scale[0], {1, 128, 2, 1}, "int8 v_scale");
    if (int8_cache.k[0].dtype != ninfer::DType::I8 || int8_cache.v[0].dtype != ninfer::DType::I8) {
        ++failures;
        std::cerr << "int8 code plane dtype mismatch\n";
    }
    if (int8_cache.k_scale[0].dtype != ninfer::DType::FP16 ||
        int8_cache.v_scale[0].dtype != ninfer::DType::FP16) {
        ++failures;
        std::cerr << "int8 scale plane dtype mismatch\n";
    }
    for (std::int32_t head = 0; head < 2; ++head) {
        q27::KVHeadSlot slot = int8_cache.append_slot(0, head);
        failures += check_shape(slot.k, {64, 1, 1, 1}, "int8 slot.k");
        failures += check_shape(slot.v, {64, 1, 1, 1}, "int8 slot.v");
        failures += check_shape(slot.k_scale, {1, 1, 1, 1}, "int8 slot.k_scale");
        failures += check_shape(slot.v_scale, {1, 1, 1, 1}, "int8 slot.v_scale");
        failures += fill_slot(slot, static_cast<unsigned char>(0x11 + head),
                              static_cast<unsigned char>(0x21 + head));
        failures += fill_scales(slot, static_cast<unsigned char>(0x31 + head),
                                static_cast<unsigned char>(0x41 + head));
    }
    int8_cache.advance();
    for (std::int32_t head = 0; head < 2; ++head) {
        const q27::KVHeadSlot slot = int8_cache.slot(0, 0, head);
        failures +=
            expect_device_bytes(slot.k, static_cast<unsigned char>(0x11 + head), "int8 slot K");
        failures +=
            expect_device_bytes(slot.v, static_cast<unsigned char>(0x21 + head), "int8 slot V");
        failures += expect_device_bytes(slot.k_scale, static_cast<unsigned char>(0x31 + head),
                                        "int8 slot K scale");
        failures += expect_device_bytes(slot.v_scale, static_cast<unsigned char>(0x41 + head),
                                        "int8 slot V scale");
    }

    return failures == 0 ? 0 : fail("kv cache test failed");
}
