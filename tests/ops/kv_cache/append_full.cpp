#include "ops/suite.h"
#include "../../../tools/support/paged_kv_fixture.h"
#include "kv_cache_storage.h"
#include "ops/float_codec.h"
#include "ops/fp8_cache_reference.h"
#include "ninfer/ops/kv_cache_append.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::float_codec;

namespace {

constexpr int kPage               = 64;
constexpr int kFullHeadDim        = 256;
constexpr int kFullGroup          = 64;
constexpr int kFullGroups         = kFullHeadDim / kFullGroup;
constexpr int kFullFp8Groups      = 2;
constexpr int kFullNvfp4Group     = 16;
constexpr int kFullNvfp4Groups    = kFullHeadDim / kFullNvfp4Group;
constexpr int kFullNvfp4CodeBytes = kFullHeadDim / 2;

struct TestVectorLayout {
    DType code_dtype;
    int code_extent;
    DType scale_dtype;
    int scale_extent;
};

struct TestCacheLayout {
    TestVectorLayout key;
    TestVectorLayout value;
};

TestCacheLayout test_cache_layout(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {{DType::BF16, kFullHeadDim, DType::U8, 0},
                {DType::FP16, kFullHeadDim, DType::U8, 0}};
    case KvCacheStorage::Int8Group64:
        return {{DType::I8, kFullHeadDim, DType::FP16, kFullGroups},
                {DType::I8, kFullHeadDim, DType::FP16, kFullGroups}};
    case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        return {{DType::FP8_E4M3FN, kFullHeadDim, DType::U8, kFullFp8Groups},
                {DType::FP8_E4M3FN, kFullHeadDim, DType::U8, kFullFp8Groups}};
    case KvCacheStorage::Nvfp4Group16:
        return {{DType::U8, kFullNvfp4CodeBytes, DType::U8, kFullNvfp4Groups},
                {DType::U8, kFullNvfp4CodeBytes, DType::U8, kFullNvfp4Groups}};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {{DType::FP8_E4M3FN, kFullHeadDim, DType::U8, kFullFp8Groups},
                {DType::U8, kFullNvfp4CodeBytes, DType::U8, kFullNvfp4Groups}};
    }
    throw std::invalid_argument("unsupported test KV storage");
}

std::vector<std::uint16_t> patterned_bits(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(count);
    std::uint32_t state = seed;
    for (auto& bit : bits) {
        state = state * 1664525u + 1013904223u;
        bit   = static_cast<std::uint16_t>(state >> 16);
    }
    return bits;
}

std::size_t full_input_index(int d, int head, int token, int kv_heads) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kFullHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(token));
}

std::size_t full_cache_index(int leading_extent, int leading, int head, int position,
                             int physical_page, int kv_heads) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPage) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(physical_page)));
}

std::uint16_t bf16_bits_to_f16_bits(std::uint16_t bits) {
    return f32_to_f16_bits(bf16_to_f32(bits));
}

void normalized_hadamard_d256_host(std::array<float, kFullHeadDim>& values) {
    for (int block = 0; block < 8; ++block) {
        const int block_begin = block * 32;
        for (int stride = 1; stride <= 16; stride <<= 1) {
            for (int base = 0; base < 32; base += 2 * stride) {
                for (int offset = 0; offset < stride; ++offset) {
                    const int low_index  = block_begin + base + offset;
                    const int high_index = low_index + stride;
                    const float low      = values[static_cast<std::size_t>(low_index)];
                    const float high     = values[static_cast<std::size_t>(high_index)];
                    values[static_cast<std::size_t>(low_index)]  = low + high;
                    values[static_cast<std::size_t>(high_index)] = low - high;
                }
            }
        }
    }
    for (int span = 1; span < 8; span <<= 1) {
        for (int base = 0; base < 8; base += 2 * span) {
            for (int offset = 0; offset < span; ++offset) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int low_index  = lane + 32 * (base + offset);
                    const int high_index = lane + 32 * (base + offset + span);
                    const float low      = values[static_cast<std::size_t>(low_index)];
                    const float high     = values[static_cast<std::size_t>(high_index)];
                    values[static_cast<std::size_t>(low_index)]  = low + high;
                    values[static_cast<std::size_t>(high_index)] = low - high;
                }
            }
        }
    }
    for (float& value : values) value *= 0x1p-4f;
}

void set_public_row_from_rotated(std::vector<float>& destination,
                                 std::array<float, kFullHeadDim> rotated, int head, int token,
                                 int kv_heads) {
    // R is its own inverse. Restricting the test vectors to binary fractions keeps the public
    // BF16 representation exact while exercising production FP32 rotation and cache quantization.
    normalized_hadamard_d256_host(rotated);
    for (int d = 0; d < kFullHeadDim; ++d) {
        const auto index   = full_input_index(d, head, token, kv_heads);
        destination[index] = bf16_to_f32(f32_to_bf16(rotated[static_cast<std::size_t>(d)]));
    }
}

std::array<float, kFullHeadDim> nvfp4_codec_rotated_vector() {
    std::array<float, kFullHeadDim> values{};
    const std::array<float, 16> table{
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    std::copy(table.begin(), table.end(), values.begin());
    const std::array<float, 16> ties{
        0.25f,  0.75f,  1.25f,  1.75f,  2.5f,  3.5f,  5.0f,  6.0f,
        -0.25f, -0.75f, -1.25f, -1.75f, -2.5f, -3.5f, -5.0f, -6.0f,
    };
    std::copy(ties.begin(), ties.end(), values.begin() + kFullNvfp4Group);
    for (int i = 0; i < kFullNvfp4Group; ++i) {
        values[2 * kFullNvfp4Group + i] = table[static_cast<std::size_t>(i)] * std::ldexp(1.0f, -9);
    }
    // raw scale 1.0625 is an E4M3 tie between scale codes 0x38 and 0x39.
    values[3 * kFullNvfp4Group]     = 6.375f;
    values[3 * kFullNvfp4Group + 1] = -3.1875f;
    // raw scale exceeds 448, locking finite scale and E2M1 saturation.
    values[4 * kFullNvfp4Group]     = 3072.0f;
    values[4 * kFullNvfp4Group + 1] = -3072.0f;
    return values;
}

void encode_full_fp8_row(const std::array<float, kFullHeadDim>& values,
                         std::vector<std::uint8_t>& codes, int head, int position,
                         int physical_page, int kv_heads, std::vector<std::uint8_t>& scales) {
    for (int group = 0; group < 2; ++group) {
        float maximum = 0;
        for (int d = group * 128; d < (group + 1) * 128; ++d)
            maximum = std::max(maximum, std::abs(values[d]));
        const auto scale        = fp8_cache_scale(maximum);
        const double multiplier = fp8_cache_multiplier(scale);
        for (int d = group * 128; d < (group + 1) * 128; ++d)
            codes[full_cache_index(kFullHeadDim, d, head, position, physical_page, kv_heads)] =
                encode_e4m3fn_rne_satfinite(float(values[d] / multiplier));
        scales[full_cache_index(kFullFp8Groups, group, head, position, physical_page, kv_heads)] =
            scale;
    }
}

void encode_full_nvfp4_row(const std::array<float, kFullHeadDim>& values,
                           std::vector<std::uint8_t>& codes, int head, int position,
                           int physical_page, int kv_heads, std::vector<std::uint8_t>& scales) {
    for (int group = 0; group < kFullNvfp4Groups; ++group) {
        const int d0 = group * kFullNvfp4Group;
        float absmax = 0.0f;
        for (int i = 0; i < kFullNvfp4Group; ++i) {
            absmax = std::max(absmax, std::abs(values[static_cast<std::size_t>(d0 + i)]));
        }
        std::uint8_t scale_code = 0;
        float scale             = 0.0f;
        if (absmax != 0.0f) {
            const float raw_scale = absmax / 6.0f;
            const float bounded   = std::clamp(raw_scale, std::ldexp(1.0f, -9), 448.0f);
            scale_code            = encode_e4m3fn_rne_satfinite(bounded);
            scale                 = decode_e4m3fn_positive(scale_code);
        }
        scales[full_cache_index(kFullNvfp4Groups, group, head, position, physical_page, kv_heads)] =
            scale_code;
        for (int pair = 0; pair < kFullNvfp4Group / 2; ++pair) {
            const int low_d  = d0 + 2 * pair;
            const int high_d = low_d + 1;
            const std::uint8_t low =
                scale == 0.0f
                    ? 0
                    : encode_e2m1_rne_satfinite(values[static_cast<std::size_t>(low_d)] / scale);
            const std::uint8_t high =
                scale == 0.0f
                    ? 0
                    : encode_e2m1_rne_satfinite(values[static_cast<std::size_t>(high_d)] / scale);
            const int byte                    = group * (kFullNvfp4Group / 2) + pair;
            codes[full_cache_index(kFullNvfp4CodeBytes, byte, head, position, physical_page,
                                   kv_heads)] = static_cast<std::uint8_t>(low | (high << 4));
        }
    }
}

void encode_full_group(const std::vector<float>& source, std::size_t source_base,
                       std::vector<std::int8_t>& codes, int head, int position, int physical_page,
                       int group, int kv_heads, std::vector<std::uint16_t>& scales) {
    float absmax = 0.0f;
    for (int i = 0; i < kFullGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }
    const std::uint16_t scale_bits = f32_to_f16_bits(absmax / 127.0f);
    const float scale              = f16_bits_to_f32(scale_bits);
    const float inverse            = scale == 0.0f ? 0.0f : 1.0f / scale;
    for (int i = 0; i < kFullGroup; ++i) {
        const int code =
            scale == 0.0f
                ? 0
                : std::clamp(round_even_to_i32(source[source_base + static_cast<std::size_t>(i)] *
                                               inverse),
                             -127, 127);
        const auto target = full_cache_index(kFullHeadDim, group * kFullGroup + i, head, position,
                                             physical_page, kv_heads);
        codes[target]     = static_cast<std::int8_t>(code);
    }
    scales[full_cache_index(kFullGroups, group, head, position, physical_page, kv_heads)] =
        scale_bits;
}

class FullCacheFixture {
public:
    FullCacheFixture(const TestCacheLayout& layout, std::int32_t heads, std::int32_t pages,
                     std::span<const std::int32_t> mapping, std::int32_t logical_pages,
                     std::array<MemorySpan, 4> planes)
        : packed_(planes[0].device, geometry(layout, heads), pages),
          expected_offsets_(packed_.table_offsets(mapping)), offsets_(to_device(expected_offsets_)),
          table_(offsets_.span(), DType::I64,
                 {logical_pages, static_cast<std::int32_t>(mapping.size()) / logical_pages}) {
        const std::array<TestVectorLayout, 2> vectors{layout.key, layout.value};
        for (int side = 0; side < 2; ++side) {
            const auto& vector = vectors[side];
            dense_[side] =
                Tensor(planes[side], vector.code_dtype, {vector.code_extent, kPage, heads, pages});
            if (vector.scale_extent)
                dense_[side + 2] = Tensor(planes[side + 2], vector.scale_dtype,
                                          {vector.scale_extent, kPage, heads, pages});
        }
    }

    PagedKVLayerView view(int heads, KvCacheStorage storage) {
        return {packed_.plane(0, table_),
                packed_.plane(1, table_),
                dense_[2].data ? packed_.plane(2, table_) : PagedPlaneView{},
                dense_[3].data ? packed_.plane(3, table_) : PagedPlaneView{},
                kFullHeadDim,
                heads,
                storage};
    }

    void upload() {
        for (int i = 0; i < 4; ++i)
            if (dense_[i].data) packed_.upload_plane(i, dense_[i]);
    }

    void copy_back() {
        for (int i = 0; i < 4; ++i)
            if (dense_[i].data) packed_.download_plane(i, dense_[i]);
    }

    bool guards_intact() const { return packed_.guards_intact(); }

    int verify_offsets() const {
        return verify_exact("page offsets readonly",
                            from_device<std::int64_t>(offsets_, expected_offsets_.size()),
                            expected_offsets_);
    }

private:
    static KVPageGeometry geometry(const TestCacheLayout& layout, int heads) {
        KVPageGeometry result;
        result.planes = {{layout.key.code_dtype, layout.key.code_extent, heads},
                         {layout.value.code_dtype, layout.value.code_extent, heads}};
        if (layout.key.scale_extent) {
            result.planes.push_back({layout.key.scale_dtype, layout.key.scale_extent, heads});
            result.planes.push_back({layout.value.scale_dtype, layout.value.scale_extent, heads});
        }
        return result;
    }

    tool::PackedKVFixture packed_;
    std::vector<std::int64_t> expected_offsets_;
    DeviceBuffer offsets_;
    Tensor table_;
    std::array<Tensor, 4> dense_;
};

int full_append_case(int kv_heads, KvCacheStorage storage, int tokens = 3) {
    const TestCacheLayout layout = test_cache_layout(storage);
    const int first_position     = tokens >= 128 ? 61 : 63;
    const int logical_pages      = (first_position + tokens + kPage - 1) / kPage;
    const int physical_pages     = 2 * logical_pages + 1;
    std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
    for (int page = 0; page < logical_pages; ++page) {
        mapping[static_cast<std::size_t>(page)] = 2 * page + 1;
    }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = first_position + token;
    }
    const std::size_t input_count = static_cast<std::size_t>(kFullHeadDim) * kv_heads * tokens;
    const auto plane_count        = [=](int leading_extent) {
        return static_cast<std::size_t>(leading_extent) * kPage * kv_heads * physical_pages;
    };
    const std::size_t k_code_count  = plane_count(layout.key.code_extent);
    const std::size_t v_code_count  = plane_count(layout.value.code_extent);
    const std::size_t k_scale_count = plane_count(layout.key.scale_extent);
    const std::size_t v_scale_count = plane_count(layout.value.scale_extent);

    std::vector<float> host_k(input_count);
    std::vector<float> host_v(input_count);
    fill_uniform(host_k, 0x13579u + static_cast<std::uint32_t>(kv_heads), -0.75f, 0.75f);
    fill_uniform(host_v, 0x24680u + static_cast<std::uint32_t>(kv_heads), -1.25f, 1.25f);
    round_to_bf16(host_k);
    round_to_bf16(host_v);
    for (int i = 0; i < kFullGroup; ++i) {
        host_k[full_input_index(i, 0, 0, kv_heads)]              = 0.0f;
        host_v[full_input_index(kFullGroup + i, 0, 0, kv_heads)] = 0.0f;
    }
    if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        for (int d = 0; d < kFullHeadDim; ++d) {
            host_k[full_input_index(d, 0, 0, kv_heads)] = 0.0f;
            host_v[full_input_index(d, 0, 0, kv_heads)] = 0.0f;
        }
        host_v[full_input_index(0, 1, 0, kv_heads)] = 448.0f;
        host_v[full_input_index(1, 1, 0, kv_heads)] = 1.0625f;
        host_v[full_input_index(0, 0, 1, kv_heads)] = std::ldexp(1.0f, -20);
        host_v[full_input_index(0, 1, 1, kv_heads)] = bf16_to_f32(0x7f7fU);
        // With absmax=49, d1/s is exactly an E4M3 midpoint while d1*FP32(1/s) is one
        // FP32 ulp above it. This row locks the codec's ordered reciprocal-multiply semantics.
        for (int d = 0; d < kFullHeadDim; ++d) {
            host_v[full_input_index(d, 0, 2, kv_heads)] = 0.0f;
        }
        host_v[full_input_index(0, 0, 2, kv_heads)] = 49.0f;
        host_v[full_input_index(1, 0, 2, kv_heads)] = 0.0013885498046875f;
    } else if (storage == KvCacheStorage::Nvfp4Group16) {
        for (int d = 0; d < kFullHeadDim; ++d) {
            host_k[full_input_index(d, 0, 0, kv_heads)] = 0.0f;
        }
        auto codec_vector = nvfp4_codec_rotated_vector();
        set_public_row_from_rotated(host_v, codec_vector, 0, 0, kv_heads);
        if (kv_heads > 1) {
            for (float& value : codec_vector) value = -value;
            set_public_row_from_rotated(host_k, codec_vector, 1, 0, kv_heads);
        }
    } else if (storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        for (int d = 0; d < kFullHeadDim; ++d) {
            host_k[full_input_index(d, 0, 0, kv_heads)] = 0.0f;
        }
        auto codec_vector = nvfp4_codec_rotated_vector();
        set_public_row_from_rotated(host_v, codec_vector, 0, 0, kv_heads);
        if (kv_heads > 1) {
            host_k[full_input_index(0, 1, 0, kv_heads)] = 448.0f;
            host_k[full_input_index(1, 1, 0, kv_heads)] = 1.0625f;
        }
    }
    std::vector<std::uint16_t> input_k(input_count);
    std::vector<std::uint16_t> input_v(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        input_k[i] = f32_to_bf16(host_k[i]);
        input_v[i] = f32_to_bf16(host_v[i]);
    }

    DeviceBuffer d_k         = to_device(input_k);
    DeviceBuffer d_v         = to_device(input_v);
    DeviceBuffer d_positions = to_device(positions);
    Tensor k(d_k.span(), DType::BF16, {kFullHeadDim, kv_heads, tokens});
    Tensor v(d_v.span(), DType::BF16, {kFullHeadDim, kv_heads, tokens});
    Tensor position_tensor(d_positions.span(), DType::I32, {tokens});

    GuardedDeviceBuffer cache_k(k_code_count * dtype_size(layout.key.code_dtype));
    GuardedDeviceBuffer cache_v(v_code_count * dtype_size(layout.value.code_dtype));
    GuardedDeviceBuffer scale_k(
        layout.key.scale_extent == 0 ? 1 : k_scale_count * dtype_size(layout.key.scale_dtype));
    GuardedDeviceBuffer scale_v(
        layout.value.scale_extent == 0 ? 1 : v_scale_count * dtype_size(layout.value.scale_dtype));
    FullCacheFixture fixture(layout, kv_heads, physical_pages, mapping, logical_pages,
                             {cache_k.span(), cache_v.span(), scale_k.span(), scale_v.span()});
    auto cache = fixture.view(kv_heads, storage);

    int failures = 0;
    if (storage == KvCacheStorage::BFloat16) {
        auto expected_k = patterned_bits(k_code_count, 0xabcdef01u);
        auto expected_v = patterned_bits(v_code_count, 0x10fedcbau);
        cache_k.copy_from_host(expected_k.data(), expected_k.size() * sizeof(std::uint16_t));
        cache_v.copy_from_host(expected_v.data(), expected_v.size() * sizeof(std::uint16_t));
        for (int token = 0; token < tokens; ++token) {
            const int position = positions[static_cast<std::size_t>(token)];
            const int page     = mapping[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kv_heads; ++head) {
                for (int d = 0; d < kFullHeadDim; ++d) {
                    const auto source = full_input_index(d, head, token, kv_heads);
                    const auto target =
                        full_cache_index(kFullHeadDim, d, head, position, page, kv_heads);
                    expected_k[target] = input_k[source];
                    expected_v[target] = bf16_bits_to_f16_bits(input_v[source]);
                }
            }
        }
        fixture.upload();
        ops::kv_cache_append(k, v, position_tensor, cache,
                             ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
        fixture.copy_back();
        const std::string label = "kv_cache_append full bf16 Hkv=" + std::to_string(kv_heads) +
                                  " T=" + std::to_string(tokens) +
                                  " P=" + std::to_string(first_position);
        failures +=
            verify_exact((label + " k").c_str(),
                         from_device<std::uint16_t>(cache_k.data(), k_code_count), expected_k);
        failures +=
            verify_exact((label + " v").c_str(),
                         from_device<std::uint16_t>(cache_v.data(), v_code_count), expected_v);
    } else if (storage == KvCacheStorage::Int8Group64) {
        std::vector<std::int8_t> expected_k(k_code_count, static_cast<std::int8_t>(0x55));
        std::vector<std::int8_t> expected_v(v_code_count, static_cast<std::int8_t>(0xaa));
        auto expected_scale_k = patterned_bits(k_scale_count, 0x01234567u);
        auto expected_scale_v = patterned_bits(v_scale_count, 0x89abcdefu);
        cache_k.copy_from_host(expected_k.data(), expected_k.size());
        cache_v.copy_from_host(expected_v.data(), expected_v.size());
        scale_k.copy_from_host(expected_scale_k.data(),
                               expected_scale_k.size() * sizeof(std::uint16_t));
        scale_v.copy_from_host(expected_scale_v.data(),
                               expected_scale_v.size() * sizeof(std::uint16_t));
        for (int token = 0; token < tokens; ++token) {
            const int position = positions[static_cast<std::size_t>(token)];
            const int page     = mapping[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kv_heads; ++head) {
                for (int group = 0; group < kFullGroups; ++group) {
                    const auto source = full_input_index(group * kFullGroup, head, token, kv_heads);
                    encode_full_group(host_v, source, expected_v, head, position, page, group,
                                      kv_heads, expected_scale_v);
                }
            }
        }
        fixture.upload();
        ops::kv_cache_append(k, v, position_tensor, cache,
                             ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
        fixture.copy_back();
        const std::string label = "kv_cache_append full int8-g64 Hkv=" + std::to_string(kv_heads) +
                                  " T=" + std::to_string(tokens) +
                                  " P=" + std::to_string(first_position);
        failures +=
            verify_exact((label + " v codes").c_str(),
                         from_device<std::int8_t>(cache_v.data(), v_code_count), expected_v);
        failures += verify_exact((label + " v scales").c_str(),
                                 from_device<std::uint16_t>(scale_v.data(), v_scale_count),
                                 expected_scale_v);
        failures += scale_k.verify_guards((label + " k scale guards").c_str());
        failures += scale_v.verify_guards((label + " v scale guards").c_str());
    } else if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        std::vector<std::uint8_t> expected_k(k_code_count, 0x55U);
        std::vector<std::uint8_t> expected_v(v_code_count, 0xaaU);
        std::vector<std::uint8_t> expected_scale_k(k_scale_count, 0x55);
        std::vector<std::uint8_t> expected_scale_v(v_scale_count, 0x55);
        cache_k.copy_from_host(expected_k.data(), expected_k.size());
        cache_v.copy_from_host(expected_v.data(), expected_v.size());
        scale_k.copy_from_host(expected_scale_k.data(),
                               expected_scale_k.size() * sizeof(std::uint8_t));
        scale_v.copy_from_host(expected_scale_v.data(),
                               expected_scale_v.size() * sizeof(std::uint8_t));
        for (int token = 0; token < tokens; ++token) {
            const int position = positions[static_cast<std::size_t>(token)];
            const int page     = mapping[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kv_heads; ++head) {
                std::array<float, kFullHeadDim> k_row{};
                std::array<float, kFullHeadDim> v_row{};
                for (int d = 0; d < kFullHeadDim; ++d) {
                    const auto source                  = full_input_index(d, head, token, kv_heads);
                    k_row[static_cast<std::size_t>(d)] = host_k[source];
                    v_row[static_cast<std::size_t>(d)] = host_v[source];
                }
                normalized_hadamard_d256_host(k_row);
                encode_full_fp8_row(k_row, expected_k, head, position, page, kv_heads,
                                    expected_scale_k);
                encode_full_fp8_row(v_row, expected_v, head, position, page, kv_heads,
                                    expected_scale_v);
            }
        }
        fixture.upload();
        ops::kv_cache_append(k, v, position_tensor, cache,
                             ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
        fixture.copy_back();
        const std::string label =
            "kv_cache_append full fp8-g128-ue8m0 Hkv=" + std::to_string(kv_heads) +
            " T=" + std::to_string(tokens) + " P=" + std::to_string(first_position);
        failures +=
            verify_exact((label + " k codes").c_str(),
                         from_device<std::uint8_t>(cache_k.data(), k_code_count), expected_k);
        failures +=
            verify_exact((label + " v codes").c_str(),
                         from_device<std::uint8_t>(cache_v.data(), v_code_count), expected_v);
        failures += verify_exact((label + " k scales").c_str(),
                                 from_device<std::uint8_t>(scale_k.data(), k_scale_count),
                                 expected_scale_k);
        failures += verify_exact((label + " v scales").c_str(),
                                 from_device<std::uint8_t>(scale_v.data(), v_scale_count),
                                 expected_scale_v);
        failures += scale_k.verify_guards((label + " k scale guards").c_str());
        failures += scale_v.verify_guards((label + " v scale guards").c_str());
    } else if (storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        std::vector<std::uint8_t> expected_k(k_code_count, 0x55U);
        std::vector<std::uint8_t> expected_v(v_code_count, 0xaaU);
        std::vector<std::uint8_t> expected_scale_k(k_scale_count, 0x55);
        std::vector<std::uint8_t> expected_scale_v(v_scale_count, 0x42U);
        cache_k.copy_from_host(expected_k.data(), expected_k.size());
        cache_v.copy_from_host(expected_v.data(), expected_v.size());
        scale_k.copy_from_host(expected_scale_k.data(),
                               expected_scale_k.size() * sizeof(std::uint8_t));
        scale_v.copy_from_host(expected_scale_v.data(), expected_scale_v.size());
        for (int token = 0; token < tokens; ++token) {
            const int position = positions[static_cast<std::size_t>(token)];
            const int page     = mapping[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kv_heads; ++head) {
                std::array<float, kFullHeadDim> k_row{};
                std::array<float, kFullHeadDim> v_row{};
                for (int d = 0; d < kFullHeadDim; ++d) {
                    const auto source                  = full_input_index(d, head, token, kv_heads);
                    k_row[static_cast<std::size_t>(d)] = host_k[source];
                    v_row[static_cast<std::size_t>(d)] = host_v[source];
                }
                normalized_hadamard_d256_host(k_row);
                normalized_hadamard_d256_host(v_row);
                encode_full_fp8_row(k_row, expected_k, head, position, page, kv_heads,
                                    expected_scale_k);
                encode_full_nvfp4_row(v_row, expected_v, head, position, page, kv_heads,
                                      expected_scale_v);
            }
        }
        fixture.upload();
        ops::kv_cache_append(k, v, position_tensor, cache,
                             ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
        fixture.copy_back();
        const std::string label = "kv_cache_append full k8v4 Hkv=" + std::to_string(kv_heads) +
                                  " T=" + std::to_string(tokens) +
                                  " P=" + std::to_string(first_position);
        failures +=
            verify_exact((label + " k codes").c_str(),
                         from_device<std::uint8_t>(cache_k.data(), k_code_count), expected_k);
        failures +=
            verify_exact((label + " v codes").c_str(),
                         from_device<std::uint8_t>(cache_v.data(), v_code_count), expected_v);
        failures += verify_exact((label + " k scales").c_str(),
                                 from_device<std::uint8_t>(scale_k.data(), k_scale_count),
                                 expected_scale_k);
        failures += verify_exact((label + " v scales").c_str(),
                                 from_device<std::uint8_t>(scale_v.data(), v_scale_count),
                                 expected_scale_v);
        failures += scale_k.verify_guards((label + " k scale guards").c_str());
        failures += scale_v.verify_guards((label + " v scale guards").c_str());
    } else {
        std::vector<std::uint8_t> expected_k(k_code_count, 0x55U);
        std::vector<std::uint8_t> expected_v(v_code_count, 0xaaU);
        std::vector<std::uint8_t> expected_scale_k(k_scale_count, 0x31U);
        std::vector<std::uint8_t> expected_scale_v(v_scale_count, 0x42U);
        cache_k.copy_from_host(expected_k.data(), expected_k.size());
        cache_v.copy_from_host(expected_v.data(), expected_v.size());
        scale_k.copy_from_host(expected_scale_k.data(), expected_scale_k.size());
        scale_v.copy_from_host(expected_scale_v.data(), expected_scale_v.size());
        for (int token = 0; token < tokens; ++token) {
            const int position = positions[static_cast<std::size_t>(token)];
            const int page     = mapping[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kv_heads; ++head) {
                std::array<float, kFullHeadDim> k_row{};
                std::array<float, kFullHeadDim> v_row{};
                for (int d = 0; d < kFullHeadDim; ++d) {
                    const auto source                  = full_input_index(d, head, token, kv_heads);
                    k_row[static_cast<std::size_t>(d)] = host_k[source];
                    v_row[static_cast<std::size_t>(d)] = host_v[source];
                }
                normalized_hadamard_d256_host(k_row);
                normalized_hadamard_d256_host(v_row);
                encode_full_nvfp4_row(k_row, expected_k, head, position, page, kv_heads,
                                      expected_scale_k);
                encode_full_nvfp4_row(v_row, expected_v, head, position, page, kv_heads,
                                      expected_scale_v);
            }
        }
        fixture.upload();
        ops::kv_cache_append(k, v, position_tensor, cache,
                             ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
        fixture.copy_back();
        const std::string label = "kv_cache_append full nvfp4-g16 Hkv=" + std::to_string(kv_heads) +
                                  " T=" + std::to_string(tokens) +
                                  " P=" + std::to_string(first_position);
        failures +=
            verify_exact((label + " k codes").c_str(),
                         from_device<std::uint8_t>(cache_k.data(), k_code_count), expected_k);
        failures +=
            verify_exact((label + " v codes").c_str(),
                         from_device<std::uint8_t>(cache_v.data(), v_code_count), expected_v);
        failures += verify_exact((label + " k scales").c_str(),
                                 from_device<std::uint8_t>(scale_k.data(), k_scale_count),
                                 expected_scale_k);
        failures += verify_exact((label + " v scales").c_str(),
                                 from_device<std::uint8_t>(scale_v.data(), v_scale_count),
                                 expected_scale_v);
        failures += scale_k.verify_guards((label + " k scale guards").c_str());
        failures += scale_v.verify_guards((label + " v scale guards").c_str());
    }
    const std::string label = "kv_cache_append full Hkv=" + std::to_string(kv_heads) +
                              " T=" + std::to_string(tokens) +
                              " P=" + std::to_string(first_position);
    failures += verify_exact((label + " input k unchanged").c_str(),
                             from_device<std::uint16_t>(d_k, input_count), input_k);
    failures += verify_exact((label + " input v unchanged").c_str(),
                             from_device<std::uint16_t>(d_v, input_count), input_v);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<std::int32_t>(d_positions, positions.size()), positions);
    failures += fixture.verify_offsets();
    failures += cache_k.verify_guards((label + " k guards").c_str());
    failures += cache_v.verify_guards((label + " v guards").c_str());
    failures += !fixture.guards_intact();
    return failures;
}

struct EncodedVector {
    std::array<std::uint8_t, 512> data{};
    std::array<std::uint8_t, 32> scales{};
};

EncodedVector encode_prefix_vector(std::span<const std::uint16_t, kFullHeadDim> input,
                                   KvCacheStorage storage, bool key) {
    EncodedVector encoded;
    const auto store_u16 = [](auto& bytes, int offset, std::uint16_t bits) {
        std::memcpy(bytes.data() + offset, &bits, sizeof(bits));
    };
    if (storage == KvCacheStorage::BFloat16) {
        for (int d = 0; d < kFullHeadDim; ++d)
            store_u16(encoded.data, 2 * d, key ? input[d] : bf16_bits_to_f16_bits(input[d]));
        return encoded;
    }
    std::array<float, kFullHeadDim> values;
    for (int d = 0; d < kFullHeadDim; ++d) values[d] = bf16_to_f32(input[d]);
    if (key || storage == KvCacheStorage::Nvfp4Group16 ||
        storage == KvCacheStorage::Fp8KeyNvfp4Value)
        normalized_hadamard_d256_host(values);
    if (storage == KvCacheStorage::Int8Group64) {
        for (int group = 0; group < kFullGroups; ++group) {
            float maximum = 0;
            for (int d = 0; d < kFullGroup; ++d)
                maximum = std::max(maximum, std::abs(values[group * kFullGroup + d]));
            const auto bits    = f32_to_f16_bits(maximum / 127.0f);
            const auto scale   = f16_bits_to_f32(bits);
            const auto inverse = scale == 0 ? 0.0f : 1.0f / scale;
            store_u16(encoded.scales, 2 * group, bits);
            for (int d = 0; d < kFullGroup; ++d) {
                const int component = group * kFullGroup + d;
                const int code =
                    std::clamp(round_even_to_i32(values[component] * inverse), -127, 127);
                encoded.data[component] = static_cast<std::uint8_t>(code);
            }
        }
    } else if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 ||
               (key && storage == KvCacheStorage::Fp8KeyNvfp4Value)) {
        for (int group = 0; group < 2; ++group) {
            float maximum = 0;
            for (int d = group * 128; d < (group + 1) * 128; ++d)
                maximum = std::max(maximum, std::abs(values[d]));
            const auto scale        = fp8_cache_scale(maximum);
            encoded.scales[group]   = scale;
            const double multiplier = fp8_cache_multiplier(scale);
            for (int d = group * 128; d < (group + 1) * 128; ++d)
                encoded.data[d] = encode_e4m3fn_rne_satfinite(float(values[d] / multiplier));
        }
    } else {
        for (int group = 0; group < kFullNvfp4Groups; ++group) {
            float maximum = 0;
            for (int d = 0; d < kFullNvfp4Group; ++d)
                maximum = std::max(maximum, std::abs(values[group * kFullNvfp4Group + d]));
            const auto bits =
                maximum == 0
                    ? std::uint8_t{0}
                    : encode_e4m3fn_rne_satfinite(std::clamp(maximum / 6.0f, 0x1p-9f, 448.0f));
            const auto scale      = decode_e4m3fn_positive(bits);
            encoded.scales[group] = bits;
            for (int pair = 0; pair < kFullNvfp4Group / 2; ++pair) {
                const auto index = group * kFullNvfp4Group + 2 * pair;
                const auto low = scale == 0 ? 0 : encode_e2m1_rne_satfinite(values[index] / scale);
                const auto high =
                    scale == 0 ? 0 : encode_e2m1_rne_satfinite(values[index + 1] / scale);
                encoded.data[index / 2] = static_cast<std::uint8_t>(low | (high << 4));
            }
        }
    }
    return encoded;
}

int full_prefix_case(int kv_heads, KvCacheStorage storage, int width, int batch, bool captured) {
    const auto layout        = test_cache_layout(storage);
    const int logical_pages  = (width + 70 + kPage - 1) / kPage;
    const int table_rows     = batch + 1;
    const int physical_pages = 2 * logical_pages * table_rows + 1;
    const auto input_count   = static_cast<std::size_t>(kFullHeadDim) * kv_heads * width * batch;
    std::vector<std::uint16_t> input_k(input_count), input_v(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        input_k[i] =
            f32_to_bf16(static_cast<float>(static_cast<int>((i * 37 + 19) % 257) - 128) / 64);
        input_v[i] =
            f32_to_bf16(static_cast<float>(static_cast<int>((i * 13 + 5) % 251) - 125) / 32);
    }
    std::fill_n(input_k.begin(), kFullHeadDim, std::uint16_t{0});
    std::vector<std::int32_t> counts(batch), rows(batch), positions(width * batch);
    std::vector<std::int32_t> mapping(logical_pages * table_rows);
    for (std::size_t i = 0; i < mapping.size(); ++i) mapping[i] = static_cast<int>(2 * i + 1);

    DeviceBuffer d_k = to_device(input_k), d_v = to_device(input_v);
    DeviceBuffer d_counts = to_device(counts), d_rows = to_device(rows);
    DeviceBuffer d_positions = to_device(positions);
    const auto plane_bytes   = [&](const TestVectorLayout& vector, bool scales) {
        return static_cast<std::size_t>(scales ? vector.scale_extent : vector.code_extent) * kPage *
               kv_heads * physical_pages *
               dtype_size(scales ? vector.scale_dtype : vector.code_dtype);
    };
    const std::array<std::size_t, 4> sizes{
        plane_bytes(layout.key, false), plane_bytes(layout.value, false),
        plane_bytes(layout.key, true), plane_bytes(layout.value, true)};
    std::array<GuardedDeviceBuffer, 4> storage_buffers{
        GuardedDeviceBuffer(sizes[0]), GuardedDeviceBuffer(sizes[1]),
        GuardedDeviceBuffer(std::max<std::size_t>(1, sizes[2])),
        GuardedDeviceBuffer(std::max<std::size_t>(1, sizes[3]))};
    FullCacheFixture fixture(layout, kv_heads, physical_pages, mapping, logical_pages,
                             {storage_buffers[0].span(), storage_buffers[1].span(),
                              storage_buffers[2].span(), storage_buffers[3].span()});
    const auto planes = fixture.view(kv_heads, storage);
    const PagedKVBatchLayerView cache{
        planes.k_pages, planes.v_pages, planes.k_scale_pages, planes.v_scale_pages, kFullHeadDim,
        kv_heads,       storage};
    Tensor k(d_k.span(), DType::BF16, {kFullHeadDim, kv_heads, width, batch});
    Tensor v(d_v.span(), DType::BF16, {kFullHeadDim, kv_heads, width, batch});
    Tensor p(d_positions.span(), DType::I32, {width, batch});
    Tensor c(d_counts.span(), DType::I32, {batch});
    Tensor r(d_rows.span(), DType::I32, {batch});
    auto& device      = cuda_environment().context;
    const auto launch = [&] {
        ops::kv_cache_append_prefix(k, v, p, c, r, {0, static_cast<std::uint32_t>(width)}, cache,
                                    device.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (captured) {
        definition.capture(device.stream(), launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int pass = 0; pass < 3; ++pass) {
        std::array<std::vector<std::uint8_t>, 4> expected;
        for (int plane = 0; plane < 4; ++plane) {
            expected[plane].assign(sizes[plane], static_cast<std::uint8_t>(0xa5 + plane));
            if (sizes[plane])
                storage_buffers[plane].copy_from_host(expected[plane].data(), sizes[plane]);
        }
        for (int row = 0; row < batch; ++row) {
            const int kind = (row + pass) % 4;
            counts[row]    = kind == 0 ? 0 : kind == 1 ? 1 : kind == 2 ? width - 2 : width;
            rows[row]      = counts[row] == 0 ? -1 : (batch - row + pass) % table_rows;
            for (int token = 0; token < width; ++token)
                positions[row * width + token] = counts[row] == 0 ? -1 : 61 + pass + token;
            for (int token = 0; token < counts[row]; ++token) {
                const int position = positions[row * width + token];
                const int page     = mapping[rows[row] * logical_pages + position / kPage];
                for (int head = 0; head < kv_heads; ++head) {
                    const auto source = full_input_index(0, head, row * width + token, kv_heads);
                    for (int side = 0; side < 2; ++side) {
                        const auto& vector = side == 0 ? layout.key : layout.value;
                        const auto& input  = side == 0 ? input_k : input_v;
                        const auto encoded =
                            encode_prefix_vector(std::span<const std::uint16_t, kFullHeadDim>(
                                                     input.data() + source, kFullHeadDim),
                                                 storage, side == 0);
                        const auto code_offset = full_cache_index(vector.code_extent, 0, head,
                                                                  position, page, kv_heads) *
                                                 dtype_size(vector.code_dtype);
                        const auto code_bytes = vector.code_extent * dtype_size(vector.code_dtype);
                        std::copy_n(encoded.data.begin(), code_bytes,
                                    expected[side].begin() + code_offset);
                        if (vector.scale_extent) {
                            const auto scale_offset = full_cache_index(vector.scale_extent, 0, head,
                                                                       position, page, kv_heads) *
                                                      dtype_size(vector.scale_dtype);
                            const auto scale_bytes =
                                vector.scale_extent * dtype_size(vector.scale_dtype);
                            std::copy_n(encoded.scales.begin(), scale_bytes,
                                        expected[side + 2].begin() + scale_offset);
                        }
                    }
                }
            }
        }
        d_counts.copy_from_host(counts.data(), counts.size() * sizeof(std::int32_t));
        d_rows.copy_from_host(rows.data(), rows.size() * sizeof(std::int32_t));
        d_positions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        fixture.upload();
        if (captured)
            graph.launch(device.stream());
        else
            launch();
        device.synchronize();
        fixture.copy_back();
        failures += !fixture.guards_intact();
        for (int plane = 0; plane < 4; ++plane) {
            if (!sizes[plane]) continue;
            const auto label = "D256 prefix " + std::string(kv_cache_storage_name(storage)) +
                               " Hkv=" + std::to_string(kv_heads) +
                               " pass=" + std::to_string(pass) + " plane=" + std::to_string(plane);
            failures +=
                verify_exact(label.c_str(),
                             from_device<std::uint8_t>(storage_buffers[plane].data(), sizes[plane]),
                             expected[plane]);
            failures += storage_buffers[plane].verify_guards(label.c_str());
        }
    }
    failures += verify_exact("prefix input K unchanged",
                             from_device<std::uint16_t>(d_k, input_count), input_k);
    failures += verify_exact("prefix input V unchanged",
                             from_device<std::uint16_t>(d_v, input_count), input_v);
    failures += fixture.verify_offsets();
    failures += verify_exact("prefix counts unchanged",
                             from_device<std::int32_t>(d_counts, counts.size()), counts);
    failures +=
        verify_exact("prefix rows unchanged", from_device<std::int32_t>(d_rows, rows.size()), rows);
    failures += verify_exact("prefix positions unchanged",
                             from_device<std::int32_t>(d_positions, positions.size()), positions);
    return failures;
}

} // namespace

void add_full_append_cases(ninfer::test::OpSuite& suite) {
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value}) {
        const std::string format = kv_cache_storage_name(storage);
        for (int heads : {4, 2})
            suite.add("append/full/" + format + "/h" + std::to_string(heads) + "/t3",
                      [=] { return full_append_case(heads, storage); }, {{"format", format}});
        if (storage != KvCacheStorage::BFloat16)
            suite.add("append/full/" + format + "/h2/t129",
                      [=] { return full_append_case(2, storage, 129); }, {{"format", format}});
        for (int heads : {2, 4}) {
            const auto base = "append/prefix/d256/" + format + "/h" + std::to_string(heads);
            suite.add(base + "/ragged-b8-graph",
                      [=] { return full_prefix_case(heads, storage, 16, 8, true); },
                      {{"format", format}, {"execution", "graph"}});
            suite.add(base + "/w129",
                      [=] { return full_prefix_case(heads, storage, 129, 1, false); },
                      {{"format", format}, {"execution", "eager"}});
        }
    }
}
