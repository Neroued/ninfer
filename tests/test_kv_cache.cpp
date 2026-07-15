#include "core/device.h"
#include "core/kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <utility>

namespace {

struct PlannedCache {
    ninfer::KVCacheLayout layout;
    std::size_t bytes = 0;
};

PlannedCache plan_cache(std::uint32_t layers, std::uint32_t max_context, std::int32_t heads,
                        std::int32_t head_dim, ninfer::DType dtype = ninfer::DType::BF16,
                        std::int32_t quant_group = 0) {
    ninfer::LayoutBuilder builder;
    auto layout =
        ninfer::plan_kv_cache(builder, layers, max_context, heads, head_dim, dtype, quant_group);
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

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                      << tensor.ne[i] << '\n';
        }
    }
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    auto bf16_plan = plan_cache(2, 8, 4, 16);
    ninfer::DeviceArena bf16_arena(bf16_plan.bytes);
    ninfer::KVCache bf16_cache({bf16_arena.base(), bf16_arena.capacity()}, bf16_plan.layout);
    failures += expect_size(bf16_cache.layer_count(), 2, "bf16 layer count");
    const ninfer::KVCacheLayerView bf16_l0 = bf16_cache.layer_view(0);
    const ninfer::KVCacheLayerView bf16_l1 = bf16_cache.layer_view(1);
    failures += check_shape(bf16_l0.k, {16, 128, 4, 1}, "bf16 layer0 K");
    failures += check_shape(bf16_l0.v, {16, 128, 4, 1}, "bf16 layer0 V");
    failures += expect_size(bf16_l0.max_context, 8, "bf16 max context");
    failures += expect_size(bf16_l0.padded_context, 128, "bf16 padded context");
    failures += expect_size(bf16_l0.num_kv_heads, 4, "bf16 heads");
    failures += expect_size(bf16_l0.head_dim, 16, "bf16 head dim");
    failures += expect_size(bf16_l0.quant_group, 0, "bf16 quant group");
    if (bf16_l0.dtype != ninfer::DType::BF16 || bf16_l0.k_scale.data != nullptr ||
        bf16_l0.v_scale.data != nullptr) {
        ++failures;
        std::cerr << "BF16 layer view has invalid dtype or scale planes\n";
    }
    if (bf16_l0.k.data == bf16_l0.v.data || bf16_l0.k.data == bf16_l1.k.data ||
        bf16_l0.v.data == bf16_l1.v.data) {
        ++failures;
        std::cerr << "BF16 physical planes alias\n";
    }

    auto int8_plan = plan_cache(1, 8, 2, 64, ninfer::DType::I8);
    ninfer::DeviceArena int8_arena(int8_plan.bytes);
    ninfer::KVCache int8_cache({int8_arena.base(), int8_arena.capacity()}, int8_plan.layout);
    const ninfer::KVCacheLayerView int8 = int8_cache.layer_view(0);
    failures += check_shape(int8.k, {64, 128, 2, 1}, "int8 K");
    failures += check_shape(int8.v, {64, 128, 2, 1}, "int8 V");
    failures += check_shape(int8.k_scale, {1, 128, 2, 1}, "int8 K scale");
    failures += check_shape(int8.v_scale, {1, 128, 2, 1}, "int8 V scale");
    failures += expect_size(int8.quant_group, ninfer::kKvQuantGroup, "int8 quant group");
    if (int8.dtype != ninfer::DType::I8 || int8.k.dtype != ninfer::DType::I8 ||
        int8.v.dtype != ninfer::DType::I8 || int8.k_scale.dtype != ninfer::DType::FP16 ||
        int8.v_scale.dtype != ninfer::DType::FP16) {
        ++failures;
        std::cerr << "INT8 layer view plane dtype mismatch\n";
    }
    if (int8.k.data == int8.v.data || int8.k_scale.data == int8.v_scale.data) {
        ++failures;
        std::cerr << "INT8 physical planes alias\n";
    }

    return failures == 0 ? 0 : fail("kv cache test failed");
}
