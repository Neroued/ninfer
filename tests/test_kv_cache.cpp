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
                        std::int32_t head_dim, ninfer::DType k_dtype = ninfer::DType::BF16,
                        ninfer::DType v_dtype = ninfer::DType::BF16) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_kv_cache(
        builder, layers, max_context, heads, head_dim, k_dtype, v_dtype,
        k_dtype == ninfer::DType::I8 ? ninfer::kKvQuantGroup : 0,
        v_dtype == ninfer::DType::I8 ? ninfer::kKvQuantGroup : 0);
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

int test_payload_accounting() {
    constexpr struct {
        ninfer::DType k;
        ninfer::DType v;
        std::size_t expected;
        const char* label;
    } cases[] = {
        {ninfer::DType::BF16, ninfer::DType::BF16, 262'144, "BF16/BF16 payload"},
        {ninfer::DType::BF16, ninfer::DType::I8, 198'656, "BF16/INT8 payload"},
        {ninfer::DType::I8, ninfer::DType::BF16, 198'656, "INT8/BF16 payload"},
        {ninfer::DType::I8, ninfer::DType::I8, 135'168, "INT8/INT8 payload"},
    };
    int failures = 0;
    for (const auto& test_case : cases) {
        const PlannedCache plan = plan_cache(2, 129, 2, 64, test_case.k, test_case.v);
        failures += expect_size(plan.layout.padded_context, 256, "payload padded context");
        failures += expect_size(plan.layout.payload_bytes(), test_case.expected, test_case.label);
    }
    return failures;
}

} // namespace

int main() {
    int failures = test_payload_accounting();
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        if (failures != 0) { return fail("kv cache payload accounting failed"); }
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

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
    failures += expect_size(bf16_l0.k_quant_group, 0, "bf16 K quant group");
    failures += expect_size(bf16_l0.v_quant_group, 0, "bf16 V quant group");
    if (bf16_l0.k_dtype != ninfer::DType::BF16 || bf16_l0.v_dtype != ninfer::DType::BF16 ||
        bf16_l0.k_scale.data != nullptr || bf16_l0.v_scale.data != nullptr) {
        ++failures;
        std::cerr << "BF16 layer view has invalid dtype or scale planes\n";
    }
    if (bf16_l0.k.data == bf16_l0.v.data || bf16_l0.k.data == bf16_l1.k.data ||
        bf16_l0.v.data == bf16_l1.v.data) {
        ++failures;
        std::cerr << "BF16 physical planes alias\n";
    }

    auto int8_plan = plan_cache(1, 8, 2, 64, ninfer::DType::I8, ninfer::DType::I8);
    ninfer::DeviceArena int8_arena(int8_plan.bytes);
    ninfer::KVCache int8_cache({int8_arena.base(), int8_arena.capacity()}, int8_plan.layout);
    const ninfer::KVCacheLayerView int8 = int8_cache.layer_view(0);
    failures += check_shape(int8.k, {64, 128, 2, 1}, "int8 K");
    failures += check_shape(int8.v, {64, 128, 2, 1}, "int8 V");
    failures += check_shape(int8.k_scale, {1, 128, 2, 1}, "int8 K scale");
    failures += check_shape(int8.v_scale, {1, 128, 2, 1}, "int8 V scale");
    failures += expect_size(int8.k_quant_group, ninfer::kKvQuantGroup, "int8 K quant group");
    failures += expect_size(int8.v_quant_group, ninfer::kKvQuantGroup, "int8 V quant group");
    if (int8.k_dtype != ninfer::DType::I8 || int8.v_dtype != ninfer::DType::I8 ||
        int8.k.dtype != ninfer::DType::I8 || int8.v.dtype != ninfer::DType::I8 ||
        int8.k_scale.dtype != ninfer::DType::FP16 || int8.v_scale.dtype != ninfer::DType::FP16) {
        ++failures;
        std::cerr << "INT8 layer view plane dtype mismatch\n";
    }
    if (int8.k.data == int8.v.data || int8.k_scale.data == int8.v_scale.data) {
        ++failures;
        std::cerr << "INT8 physical planes alias\n";
    }

    // K BF16 with V INT8-G64: only the V side gets codes + scale planes.
    auto mixed_plan = plan_cache(2, 8, 2, 64, ninfer::DType::BF16, ninfer::DType::I8);
    ninfer::DeviceArena mixed_arena(mixed_plan.bytes);
    ninfer::KVCache mixed_cache({mixed_arena.base(), mixed_arena.capacity()}, mixed_plan.layout);
    const ninfer::KVCacheLayerView mixed = mixed_cache.layer_view(1);
    failures += check_shape(mixed.k, {64, 128, 2, 1}, "mixed K");
    failures += check_shape(mixed.v, {64, 128, 2, 1}, "mixed V");
    failures += check_shape(mixed.v_scale, {1, 128, 2, 1}, "mixed V scale");
    failures += expect_size(mixed.k_quant_group, 0, "mixed K quant group");
    failures += expect_size(mixed.v_quant_group, ninfer::kKvQuantGroup, "mixed V quant group");
    if (mixed.k_dtype != ninfer::DType::BF16 || mixed.v_dtype != ninfer::DType::I8 ||
        mixed.k.dtype != ninfer::DType::BF16 || mixed.v.dtype != ninfer::DType::I8 ||
        mixed.k_scale.data != nullptr || mixed.v_scale.dtype != ninfer::DType::FP16) {
        ++failures;
        std::cerr << "mixed K-BF16/V-INT8 layer view plane mismatch\n";
    }
    if (mixed_plan.layout.k_scale.size() != 0 || mixed_plan.layout.v_scale.size() != 2) {
        ++failures;
        std::cerr << "mixed K-BF16/V-INT8 layout scale plane counts wrong\n";
    }
    if (mixed.k.data == mixed.v.data) {
        ++failures;
        std::cerr << "mixed physical planes alias\n";
    }

    // K INT8-G64 with V BF16: only the K side gets codes + scale planes.
    auto reverse_plan = plan_cache(2, 8, 2, 64, ninfer::DType::I8, ninfer::DType::BF16);
    ninfer::DeviceArena reverse_arena(reverse_plan.bytes);
    ninfer::KVCache reverse_cache({reverse_arena.base(), reverse_arena.capacity()},
                                  reverse_plan.layout);
    const ninfer::KVCacheLayerView reverse = reverse_cache.layer_view(1);
    failures += check_shape(reverse.k, {64, 128, 2, 1}, "reverse mixed K");
    failures += check_shape(reverse.v, {64, 128, 2, 1}, "reverse mixed V");
    failures += check_shape(reverse.k_scale, {1, 128, 2, 1}, "reverse mixed K scale");
    failures += expect_size(reverse.k_quant_group, ninfer::kKvQuantGroup,
                            "reverse mixed K quant group");
    failures += expect_size(reverse.v_quant_group, 0, "reverse mixed V quant group");
    if (reverse.k_dtype != ninfer::DType::I8 || reverse.v_dtype != ninfer::DType::BF16 ||
        reverse.k.dtype != ninfer::DType::I8 || reverse.v.dtype != ninfer::DType::BF16 ||
        reverse.k_scale.dtype != ninfer::DType::FP16 || reverse.v_scale.data != nullptr) {
        ++failures;
        std::cerr << "mixed K-INT8/V-BF16 layer view plane mismatch\n";
    }
    if (reverse_plan.layout.k_scale.size() != 2 || !reverse_plan.layout.v_scale.empty()) {
        ++failures;
        std::cerr << "mixed K-INT8/V-BF16 layout scale plane counts wrong\n";
    }
    if (reverse.k.data == reverse.v.data) {
        ++failures;
        std::cerr << "reverse mixed physical planes alias\n";
    }

    return failures == 0 ? 0 : fail("kv cache test failed");
}
