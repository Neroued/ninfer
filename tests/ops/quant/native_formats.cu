#include "core/buffer.h"
#include "core/device.h"
#include "core/weight_bank.h"
#include "ops/common/quant/csf32.cuh"
#include "ops/common/quant/fp8.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/float_codec.h"
#include "ops/suite.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace ninfer;
namespace codec     = ops::common;
namespace reference = test::float_codec;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
DeviceBuffer upload(Device device, const std::vector<T>& values) {
    DeviceBuffer out(device, values.size() * sizeof(T));
    out.copy_from_host(values.data(), out.size());
    return out;
}

template <class T>
std::vector<T> download(const DeviceBuffer& buffer, const DeviceContext& device) {
    device.synchronize();
    std::vector<T> values(buffer.size() / sizeof(T));
    buffer.copy_to_host(values.data(), buffer.size());
    return values;
}

std::uint16_t bf16_word(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

float represented_bf16(std::uint16_t value) {
    return std::bit_cast<float>(std::uint32_t(value) << 16);
}

float divide(float value, float divisor) {
    volatile float result = value / divisor;
    return result;
}

struct Quantized {
    std::uint8_t codes[32]{};
    std::uint8_t scale = 0;
};

template <bool Bf16>
__global__ void quantize_fp4(const void* inputs, const float* multipliers, Quantized* output,
                             int groups) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= groups) return;
    const auto value = [&] {
        if constexpr (Bf16)
            return codec::quantize_nvfp4_mul_k16(
                static_cast<const __nv_bfloat16*>(inputs) + index * 16, multipliers[index]);
        else
            return codec::quantize_nvfp4_mul_k16(static_cast<const float*>(inputs) + index * 16,
                                                 multipliers[index]);
    }();
    const auto* codes = reinterpret_cast<const std::uint8_t*>(&value.codes_lo);
    for (int i = 0; i < 8; ++i) output[index].codes[i] = codes[i];
    output[index].scale = value.scale;
}

template <bool Bf16>
__global__ void quantize_fp8(const void* inputs, Quantized* output, int groups) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= groups) return;
    const auto value = [&] {
        if constexpr (Bf16)
            return codec::quantize_mxfp8_k32(static_cast<const __nv_bfloat16*>(inputs) +
                                             index * 32);
        else
            return codec::quantize_mxfp8_k32(static_cast<const float*>(inputs) + index * 32);
    }();
    const auto* first  = reinterpret_cast<const std::uint8_t*>(&value.codes_lo);
    const auto* second = reinterpret_cast<const std::uint8_t*>(&value.codes_hi);
    for (int i = 0; i < 16; ++i) {
        output[index].codes[i]      = first[i];
        output[index].codes[16 + i] = second[i];
    }
    output[index].scale = value.scale;
}

void quantization(bool fp4) {
    DeviceContext device;
    const int width = fp4 ? 16 : 32;
    const std::vector<float> maxima =
        fp4 ? std::vector<float>{0, 0x1p-24F, 6, 12, 0.01371F, 137.25F, 1.0e8F}
            : std::vector<float>{0,
                                 0x1p-140F,
                                 448,
                                 std::nextafter(448.0F, 0.0F),
                                 std::nextafter(448.0F, 1000.0F),
                                 0x1.c00p-112F,
                                 0x1.c00p+108F};
    const std::vector<float> multipliers{1, 0.371F, 1, 0.019137F, 1.731F, 17.25F, 0.01F};
    std::vector<float> values(maxima.size() * width);
    constexpr float ties[]{0, 0.25F, 0.75F, 1.25F, 1.75F, 2.5F, 3.5F, 5, 6};
    for (std::size_t group = 0; group < maxima.size(); ++group) {
        for (int i = 0; i < width; ++i)
            values[group * width + i] = maxima[group] * (i - width / 2) / (width / 2);
        values[group * width] = maxima[group];
    }
    if (fp4) std::copy(std::begin(ties), std::end(ties), values.begin() + 2 * width);
    std::vector<std::uint16_t> bf16(values.size());
    std::transform(values.begin(), values.end(), bf16.begin(), bf16_word);
    auto f32_input = upload(device.device(), values), bf16_input = upload(device.device(), bf16);
    auto calibration = upload(device.device(), multipliers);
    DeviceBuffer output(device.device(), maxima.size() * sizeof(Quantized));
    for (const bool use_bf16 : {false, true}) {
        output.fill();
        const void* input = use_bf16 ? bf16_input.data() : f32_input.data();
        if (fp4) {
            const auto launch = [&](auto mode) {
                quantize_fp4<decltype(mode)::value><<<1, 32, 0, device.stream()>>>(
                    input, static_cast<const float*>(calibration.data()),
                    static_cast<Quantized*>(output.data()), maxima.size());
            };
            if (use_bf16)
                launch(std::true_type{});
            else
                launch(std::false_type{});
        } else {
            const auto launch = [&](auto mode) {
                quantize_fp8<decltype(mode)::value><<<1, 32, 0, device.stream()>>>(
                    input, static_cast<Quantized*>(output.data()), maxima.size());
            };
            if (use_bf16)
                launch(std::true_type{});
            else
                launch(std::false_type{});
        }
        CUDA_CHECK(cudaGetLastError());
        const auto actual = download<Quantized>(output, device);
        for (std::size_t group = 0; group < maxima.size(); ++group) {
            float maximum    = 0;
            const auto value = [&](int i) {
                const auto index = group * width + i;
                return use_bf16 ? represented_bf16(bf16[index]) : values[index];
            };
            for (int i = 0; i < width; ++i) maximum = std::max(maximum, std::abs(value(i)));
            std::uint8_t scale = 127;
            if (fp4) {
                scale = reference::encode_e4m3fn_rne_satfinite(
                    divide(divide(maximum, multipliers[group]), 6));
            } else if (maximum != 0) {
                for (scale = 0; scale < 254; ++scale)
                    if (double(maximum) <= 448.0 * reference::decode_ue8m0(scale)) break;
            }
            const auto label = std::string(fp4 ? "NVFP4_MUL" : "MXFP8") +
                               (use_bf16 ? " BF16" : " FP32") + " group=" + std::to_string(group);
            require(actual[group].scale == scale, label + " scale differs");
            for (int i = 0; i < width; ++i) {
                if (fp4) {
                    const auto expected = scale ? reference::encode_e2m1_rne_satfinite(
                                                      divide(divide(value(i), multipliers[group]),
                                                             reference::decode_e4m3fn(scale)))
                                                : 0;
                    const auto code     = (actual[group].codes[i / 2] >> (4 * (i % 2))) & 15;
                    require(code == expected, label + " E2M1 code=" + std::to_string(i));
                } else {
                    const float normalized =
                        static_cast<float>(double(value(i)) / reference::decode_ue8m0(scale));
                    const auto expected = reference::encode_e4m3fn_rne_satfinite(normalized);
                    require(actual[group].codes[i] == expected,
                            label + " E4M3 code=" + std::to_string(i));
                }
            }
        }
    }
}

struct Decoded {
    float lo, hi;
    std::uint32_t bf16;
};

struct DecodedFp4 {
    unsigned bf16[2];
    codec::Nvfp4F16x16 fp16[2];
};

__global__ void decode_fp4(const std::uint8_t* codes, DecodedFp4* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto scale      = static_cast<std::uint8_t>(index % 127);
    const auto* data      = codes + (index / 127) * 8;
    const auto scale_bits = codec::fp8_e4m3x2_to_bf16x2_bits(unsigned(scale) * 0x101U);
    output[index].bf16[0] = codec::nvfp4_scaled_pair_bf16(data[0], scale);
    output[index].bf16[1] = codec::nvfp4_pair_bf16_scaled(data[0], scale_bits);
    output[index].fp16[0] = codec::nvfp4_dequant_f16x16(data, scale);
    output[index].fp16[1] = codec::nvfp4_dequant_f16x16(codec::load_vec<int2>(data), scale);
}

void decode_nvfp4() {
    DeviceContext device;
    std::vector<std::uint8_t> codes(256 * 8);
    for (int code = 0; code < 256; ++code)
        for (int byte = 0; byte < 8; ++byte) codes[code * 8 + byte] = (code + 37 * byte) & 255;
    auto input          = upload(device.device(), codes);
    constexpr int count = 256 * 127;
    DeviceBuffer output(device.device(), count * sizeof(DecodedFp4));
    decode_fp4<<<(count + 127) / 128, 128, 0, device.stream()>>>(
        static_cast<const std::uint8_t*>(input.data()), static_cast<DecodedFp4*>(output.data()),
        count);
    CUDA_CHECK(cudaGetLastError());
    const auto actual = download<DecodedFp4>(output, device);
    for (int index = 0; index < count; ++index) {
        const float scale = reference::decode_e4m3fn(index % 127);
        std::array<std::uint16_t, 16> expected;
        unsigned expected_pair = 0;
        for (int word = 0; word < 16; ++word) {
            const auto code   = codes[(index / 127) * 8 + word / 2] >> ((word & 1) * 4);
            const float value = reference::decode_e2m1(code & 15) * scale;
            expected[word]    = reference::f32_to_f16_bits(value);
            if (word < 2) expected_pair |= unsigned(bf16_word(value)) << (word * 16);
        }
        for (int form = 0; form < 2; ++form) {
            require(actual[index].bf16[form] == expected_pair, "NVFP4 BF16 pair decode differs");
            std::array<std::uint16_t, 16> represented;
            std::memcpy(represented.data(), &actual[index].fp16[form], sizeof(represented));
            require(represented == expected, "NVFP4 FP16 block decode differs");
        }
    }
}

__global__ void decode_fp8(const std::uint16_t* codes, Decoded* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto scale = static_cast<std::uint8_t>(index % 255);
    const auto value = codec::mxfp8_dequant_f32x2(codes[index / 255], scale);
    output[index]    = {value.x, value.y, codec::fp8_dequant_bf16x2(codes[index / 255], scale)};
}

__global__ void decode_e4m3_all(float2* output) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 256) return;
    const auto codes = static_cast<std::uint16_t>(index | ((255 - index) << 8));
    output[index]    = codec::fp8_code2_to_float2(codes);
}

void decode_mxfp8() {
    DeviceContext device;
    const std::vector<std::uint16_t> pairs{0x8000, 0x8101, 0x8808, 0xb838, 0xfe7e, 0x2937};
    auto input       = upload(device.device(), pairs);
    const auto count = pairs.size() * 255;
    DeviceBuffer output(device.device(), count * sizeof(Decoded));
    decode_fp8<<<(count + 127) / 128, 128, 0, device.stream()>>>(
        static_cast<const std::uint16_t*>(input.data()), static_cast<Decoded*>(output.data()),
        count);
    CUDA_CHECK(cudaGetLastError());
    const auto actual = download<Decoded>(output, device);
    for (std::size_t i = 0; i < count; ++i)
        for (int half = 0; half < 2; ++half) {
            const auto code     = static_cast<std::uint8_t>(pairs[i / 255] >> (8 * half));
            const auto expected = static_cast<float>(double(reference::decode_e4m3fn(code)) *
                                                     reference::decode_ue8m0(i % 255));
            const auto got      = half ? actual[i].hi : actual[i].lo;
            require(std::bit_cast<std::uint32_t>(expected) == std::bit_cast<std::uint32_t>(got),
                    "MXFP8 FP32 decode differs at " + std::to_string(i));
            const auto word = static_cast<std::uint16_t>(actual[i].bf16 >> (16 * half));
            require(word == bf16_word(expected),
                    "MXFP8 BF16 decode differs at " + std::to_string(i));
        }
    DeviceBuffer raw_output(device.device(), 256 * sizeof(float2));
    decode_e4m3_all<<<2, 128, 0, device.stream()>>>(static_cast<float2*>(raw_output.data()));
    CUDA_CHECK(cudaGetLastError());
    const auto raw = download<float2>(raw_output, device);
    for (int index = 0; index < 256; ++index)
        for (int half = 0; half < 2; ++half) {
            const auto code     = static_cast<std::uint8_t>(half ? 255 - index : index);
            const auto expected = reference::decode_e4m3fn(code);
            const auto got      = half ? raw[index].y : raw[index].x;
            require(std::isnan(expected) ? std::isnan(got)
                                         : std::bit_cast<std::uint32_t>(expected) ==
                                               std::bit_cast<std::uint32_t>(got),
                    "E4M3 pair widening differs at code " + std::to_string(code));
        }
}

struct CsfQuery {
    std::uint32_t matrix, origin, row, column, destination_row, destination_column;
};

template <int Rows, int Columns>
struct CsfFixture {
    static constexpr int rows = Rows, columns = Columns;
    std::vector<std::uint8_t> bases, nibbles, values;
    std::vector<std::uint16_t> offsets, exceptions;
    std::vector<std::uint32_t> page_bases;

    std::vector<CsfQuery> queries;

    explicit CsfFixture(bool no_exceptions) {
        constexpr int matrices = 2;
        bases.resize(matrices * rows);
        values.resize(matrices * rows * columns);
        for (int matrix = 0; matrix < matrices; ++matrix) {
            for (int row = 0; row < rows; ++row)
                bases[matrix * rows + row] = 32 + row % 19 + matrix * 3;
            int tile = 0, block = 0, relative = 0;
            for (int row0 = 0; row0 < rows; row0 += 128)
                for (int column0 = 0; column0 < columns; column0 += 4, ++tile) {
                    if (tile % 32 == 0) {
                        page_bases.push_back(exceptions.size() * 2);
                        offsets.push_back(0);
                        relative = 0;
                    }
                    for (int pair = 0; pair < 2; ++pair)
                        for (int part = 0; part < 4; ++part) {
                            const int row_begin = row0 + part * 32, column = column0 + pair * 2;
                            if (row_begin >= rows || column >= columns) continue;
                            const int width  = std::min(2, columns - column);
                            const int origin = row_begin >= 64 ? 64 : 0;
                            queries.push_back(
                                {std::uint32_t(matrix), std::uint32_t(origin),
                                 std::uint32_t(row_begin - origin), std::uint32_t(column),
                                 std::uint32_t((block % 4) * 32), std::uint32_t((block % 4) * 2)});
                            std::uint8_t packed = 0;
                            for (int row = 0; row < 32; ++row)
                                for (int c = 0; c < width; ++c) {
                                    const auto delta =
                                        (row_begin + row + column + c * 5 + matrix) % 16;
                                    const auto index = row * width + c;
                                    if (index % 2 == 0)
                                        packed = delta;
                                    else
                                        nibbles.push_back(packed | (delta << 4));
                                    auto value = std::uint8_t(
                                        bases[matrix * rows + row_begin + row] + delta);
                                    if (!no_exceptions &&
                                        (block % 11 == 0 ||
                                         (block % 11 != 1 && (row * 7 + c) % 29 == 0))) {
                                        value = (row * 7 + column + c + matrix) % 127;
                                        exceptions.push_back(std::uint16_t(value) << 8 |
                                                             (row * 2 + c));
                                        ++relative;
                                    }
                                    values[(matrix * rows + row_begin + row) * columns + column +
                                           c] = value;
                                }
                            offsets.push_back(relative);
                            ++block;
                        }
                }
        }
    }
};

template <bool Static, int Rows, int Columns>
__global__ void read_csf(Csf32ScaleView view, const CsfQuery* queries, std::uint8_t* staged,
                         int offsets_stride, int pages_stride) {
    const auto query = queries[blockIdx.x];
    view.bases += query.matrix * view.rows;
    view.nibbles += query.matrix * view.rows * view.columns / 2;
    view.offsets += query.matrix * offsets_stride;
    view.page_bases += query.matrix * pages_stride;
    view.row_begin = query.origin;
    __shared__ __align__(16) std::uint8_t tile[1024];
    for (int i = threadIdx.x; i < 1024; i += 32) tile[i] = 0xa5;
    __syncwarp();
    if constexpr (Static)
        codec::csf32_stage_block<128, Rows, Columns>(
            view, query.row, query.column, tile, query.destination_row, query.destination_column);
    else
        codec::csf32_stage_block<128>(view, query.row, query.column, tile, query.destination_row,
                                      query.destination_column);
    __syncthreads();
    for (int i = threadIdx.x; i < 1024; i += 32) staged[blockIdx.x * 1024 + i] = tile[i];
}

struct CsfTileQuery {
    std::uint32_t matrix, origin;
    uint4 rows;
    std::uint32_t column, destination_column;
};

template <bool Static, int Rows, int Columns, bool LinearRows = false, int RowBlocks = 4>
__global__ void read_csf_tile(Csf32ScaleView view, const CsfTileQuery* queries,
                              std::uint8_t* output, int offsets_stride, int pages_stride) {
    const auto query = queries[blockIdx.x];
    view.bases += query.matrix * view.rows;
    view.nibbles += query.matrix * view.rows * view.columns / 2;
    view.offsets += query.matrix * offsets_stride;
    view.page_bases += query.matrix * pages_stride;
    view.row_begin = query.origin;
    __shared__ __align__(16) std::uint8_t tile[1024];
    __shared__ codec::Csf32RowBases128 row_bases;
    __shared__ codec::Csf32EncodedTile128x4 encoded;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) tile[i] = 0xa5;
    __syncthreads();
    if (threadIdx.x < 32) {
        codec::csf32_copy_row_bases_async<RowBlocks>(view, query.rows, row_bases);
        if constexpr (Static)
            codec::csf32_copy_tile128x4_async<Rows, Columns, RowBlocks>(view, query.rows,
                                                                        query.column, encoded);
        else
            codec::csf32_copy_tile128x4_async<0, 0, RowBlocks>(view, query.rows, query.column,
                                                               encoded);
        codec::cp_commit();
        codec::cp_wait<0>();
    }
    __syncthreads();
    if constexpr (LinearRows) {
        const unsigned row = threadIdx.x / 32;
        if (row < RowBlocks) {
            if constexpr (Static) {
                codec::csf32_decode_word<128, Rows, Columns, true>(
                    view, query.rows, query.column, row_bases, encoded, row, tile, row * 32,
                    query.destination_column);
            } else {
                codec::csf32_decode_word<128, 0, 0, true>(view, query.rows, query.column, row_bases,
                                                          encoded, row, tile, row * 32,
                                                          query.destination_column);
            }
        }
    } else {
        for (unsigned block = threadIdx.x / 32; block < RowBlocks * 2; block += 4) {
            const unsigned row = block / 2, pair = block % 2;
            if constexpr (Static)
                codec::csf32_decode_block<128, Rows, Columns, LinearRows>(
                    view, query.rows, query.column, row_bases, encoded, row, pair, tile, row * 32,
                    query.destination_column + pair * 2);
            else
                codec::csf32_decode_block<128, 0, 0, LinearRows>(
                    view, query.rows, query.column, row_bases, encoded, row, pair, tile, row * 32,
                    query.destination_column + pair * 2);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        // Read each represented row/column into the test's established native layout.
        const int source =
            LinearRows ? i / 512 * 512 + ((i % 16) / 4 * 32 + i % 512 / 16) * 4 + i % 4 : i;
        output[blockIdx.x * 1024 + i] = tile[source];
    }
}

template <int Rows, int Columns>
void csf_decode(bool no_exceptions) {
    DeviceContext device;
    CsfFixture<Rows, Columns> source(no_exceptions);
    auto bases      = upload(device.device(), source.bases),
         nibbles    = upload(device.device(), source.nibbles);
    auto offsets    = upload(device.device(), source.offsets),
         pages      = upload(device.device(), source.page_bases);
    auto exceptions = upload(device.device(), source.exceptions),
         queries    = upload(device.device(), source.queries);
    Csf32ScaleView view{static_cast<const std::uint8_t*>(bases.data()),
                        static_cast<const std::uint8_t*>(nibbles.data()),
                        static_cast<const std::uint16_t*>(offsets.data()),
                        static_cast<const std::uint32_t*>(pages.data()),
                        static_cast<const std::uint16_t*>(exceptions.data()),
                        std::uint64_t(source.rows),
                        std::uint64_t(source.columns),
                        0,
                        source.exceptions.size()};
    DeviceBuffer staged(device.device(), source.queries.size() * 1024);
    const auto separated = Rows >= 192   ? make_uint4(0, 32, 128, 160)
                           : Rows >= 160 ? make_uint4(0, 32, 96, 128)
                                         : make_uint4(0, 32, 64, 96);
    const std::vector<CsfTileQuery> tile_queries{
        {0, 0, separated, 0, 0},
        {1, 0, separated, 0, 0},
        {1, 0, separated, (Columns / 4 - 1) * 4, 4},
        {1, Rows >= 160 ? 32U : 0U, make_uint4(0, 32, 64, 96),
         std::min(64U, unsigned((Columns / 4 - 1) * 4)), 0}};
    auto tile_query_storage = upload(device.device(), tile_queries);
    DeviceBuffer tiles(device.device(), tile_queries.size() * 1024);
    for (bool use_static : {false, true}) {
        if (use_static)
            read_csf<true, Rows, Columns><<<source.queries.size(), 32, 0, device.stream()>>>(
                view, static_cast<const CsfQuery*>(queries.data()),
                static_cast<std::uint8_t*>(staged.data()), source.offsets.size() / 2,
                source.page_bases.size() / 2);
        else
            read_csf<false, Rows, Columns><<<source.queries.size(), 32, 0, device.stream()>>>(
                view, static_cast<const CsfQuery*>(queries.data()),
                static_cast<std::uint8_t*>(staged.data()), source.offsets.size() / 2,
                source.page_bases.size() / 2);
        CUDA_CHECK(cudaGetLastError());
        const auto actual = download<std::uint8_t>(staged, device);
        for (std::size_t i = 0; i < source.queries.size(); ++i) {
            const auto& q = source.queries[i];
            std::array<std::uint8_t, 1024> expected;
            expected.fill(0xa5);
            for (int lane = 0; lane < 32; ++lane) {
                for (int col = 0; col < std::min(2, source.columns - int(q.column)); ++col) {
                    const auto value =
                        source.values[(q.matrix * source.rows + q.origin + q.row + lane) *
                                          source.columns +
                                      q.column + col];
                    const int r = q.destination_row + lane, c = q.destination_column + col;
                    expected[(c / 4) * 512 + (r % 32) * 16 + (r / 32) * 4 + c % 4] = value;
                }
            }
            require(std::memcmp(expected.data(), actual.data() + i * 1024, 1024) == 0,
                    "CSF native scale staging or untouched bytes differ at " + std::to_string(i));
        }
        for (int row_blocks : {2, 4}) {
            for (bool linear_rows : {false, true}) {
                const auto launch_tile = [&]<bool LinearRows, int RowBlocks>() {
                    if (use_static)
                        read_csf_tile<true, Rows, Columns, LinearRows, RowBlocks>
                            <<<tile_queries.size(), 128, 0, device.stream()>>>(
                                view, static_cast<const CsfTileQuery*>(tile_query_storage.data()),
                                static_cast<std::uint8_t*>(tiles.data()), source.offsets.size() / 2,
                                source.page_bases.size() / 2);
                    else
                        read_csf_tile<false, Rows, Columns, LinearRows, RowBlocks>
                            <<<tile_queries.size(), 128, 0, device.stream()>>>(
                                view, static_cast<const CsfTileQuery*>(tile_query_storage.data()),
                                static_cast<std::uint8_t*>(tiles.data()), source.offsets.size() / 2,
                                source.page_bases.size() / 2);
                };
                const auto launch_rows = [&]<int RowBlocks>() {
                    if (linear_rows)
                        launch_tile.template operator()<true, RowBlocks>();
                    else
                        launch_tile.template operator()<false, RowBlocks>();
                };
                if (row_blocks == 2)
                    launch_rows.template operator()<2>();
                else
                    launch_rows.template operator()<4>();
                CUDA_CHECK(cudaGetLastError());
                const auto actual_tiles = download<std::uint8_t>(tiles, device);
                for (std::size_t i = 0; i < tile_queries.size(); ++i) {
                    const auto& q = tile_queries[i];
                    const std::array rows{q.rows.x, q.rows.y, q.rows.z, q.rows.w};
                    std::array<std::uint8_t, 1024> expected;
                    expected.fill(0xa5);
                    for (int block = 0; block < row_blocks; ++block)
                        for (int lane = 0; lane < 32; ++lane)
                            for (int column = 0; column < 4; ++column) {
                                const auto source_row =
                                    q.matrix * Rows + q.origin + rows[block] + lane;
                                expected[q.destination_column / 4 * 512 + lane * 16 + block * 4 +
                                         column] =
                                    source.values[source_row * Columns + q.column + column];
                            }
                    require(std::memcmp(expected.data(), actual_tiles.data() + i * 1024, 1024) == 0,
                            "CSF async block staging/guard differs at " + std::to_string(i));
                }
            }
        }
    }
}

struct Resolved {
    std::uint64_t code_offset, scale_offset, bases_offset, exceptions;
    float multiplier;
    std::uint32_t valid;
};

__global__ void resolve_experts(ExpertMatrixDeviceView view, Resolved* output, int count) {
    const int id = threadIdx.x;
    if (id >= count) return;
    const auto matrix = codec::nvfp4_mul_expert_matrix(view, id);
    Resolved result{};
    result.valid = matrix.valid;
    if (matrix.valid) {
        result.code_offset = reinterpret_cast<const std::byte*>(matrix.codes) - view.codes;
        result.multiplier  = matrix.multiplier;
        if (view.layout == QuantLayout::Csf32) {
            result.bases_offset = matrix.csf.bases - view.csf.first.bases;
            result.exceptions   = reinterpret_cast<std::uint64_t>(matrix.csf.exceptions);
        } else
            result.scale_offset = reinterpret_cast<const std::byte*>(matrix.scales) - view.scales;
    }
    output[id] = result;
}

void expert_addressing() {
    DeviceContext device;
    DeviceBuffer codes(device.device(), 4096), scales(device.device(), 4096),
        bases(device.device(), 128);
    DeviceBuffer exceptions(device.device(), 64);
    const std::vector<std::uint32_t> mapping{1, UINT32_MAX, UINT32_MAX, UINT32_MAX, 0};
    auto map = upload(device.device(), mapping);
    const std::vector<float> multipliers{.371F, 1.913F};
    auto scalar = upload(device.device(), multipliers);
    ExpertMatrixDeviceView view;
    view.codes                     = static_cast<const std::byte*>(codes.data());
    view.matrix_scales             = static_cast<const float*>(scalar.data());
    view.matrix_scale_stride       = 1;
    view.code_stride               = 2048;
    view.format                    = QType::NVFP4_MUL;
    view.layout                    = QuantLayout::Csf32;
    view.physical_count            = 2;
    view.expert_indices            = static_cast<const std::uint32_t*>(map.data());
    view.csf.first.bases           = static_cast<const std::uint8_t*>(bases.data());
    view.csf.first.nibbles         = static_cast<const std::uint8_t*>(scales.data());
    view.csf.first.offsets         = static_cast<const std::uint16_t*>(scales.data()) + 128;
    view.csf.first.page_bases      = static_cast<const std::uint32_t*>(scales.data()) + 80;
    view.csf.first.exceptions      = static_cast<const std::uint16_t*>(exceptions.data());
    view.csf.first.rows            = 64;
    view.csf.first.columns         = 4;
    view.csf.first.exception_count = 32;
    view.csf.bases_stride          = 64;
    view.csf.nibbles_stride        = 128;
    view.csf.offsets_stride        = 10;
    view.csf.page_bases_stride     = 4;
    DeviceBuffer output(device.device(), mapping.size() * sizeof(Resolved));
    resolve_experts<<<1, 32, 0, device.stream()>>>(view, static_cast<Resolved*>(output.data()),
                                                   mapping.size());
    CUDA_CHECK(cudaGetLastError());
    auto actual = download<Resolved>(output, device);
    for (std::size_t id = 0; id < mapping.size(); ++id) {
        const auto slot = mapping[id];
        require(actual[id].valid == (slot != UINT32_MAX), "CSF expert validity differs");
        if (slot == UINT32_MAX) continue;
        require(actual[id].code_offset == slot * 2048 && actual[id].bases_offset == slot * 64 &&
                    actual[id].exceptions == reinterpret_cast<std::uint64_t>(exceptions.data()) &&
                    actual[id].multiplier == multipliers[slot],
                "CSF expert physical addressing differs");
    }
    const std::vector<std::uint64_t> offsets{2048, UINT64_MAX, 0};
    const std::vector<float> indexed_multipliers{1.913F, 1, .371F};
    auto offsets_device  = upload(device.device(), offsets),
         indexed_scalars = upload(device.device(), indexed_multipliers);
    view.layout          = QuantLayout::BlockScaleK16M128x4;
    view.scales          = static_cast<const std::byte*>(scales.data());
    view.offsets         = static_cast<const std::uint64_t*>(offsets_device.data());
    view.expert_indices  = nullptr;
    view.physical_count  = 0;
    view.matrix_scales   = static_cast<const float*>(indexed_scalars.data());
    resolve_experts<<<1, 32, 0, device.stream()>>>(view, static_cast<Resolved*>(output.data()),
                                                   offsets.size());
    CUDA_CHECK(cudaGetLastError());
    actual = download<Resolved>(output, device);
    for (std::size_t id = 0; id < offsets.size(); ++id) {
        require(actual[id].valid == (offsets[id] != UINT64_MAX), "indexed expert validity differs");
        if (offsets[id] == UINT64_MAX) continue;
        require(actual[id].code_offset == offsets[id] && actual[id].scale_offset == offsets[id] &&
                    actual[id].multiplier == indexed_multipliers[id],
                "indexed expert scale domain differs");
    }
}

} // namespace

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    suite.add("csf32/scale_tiles",
              [] {
                  csf_decode<160, 133>(false);
                  csf_decode<160, 133>(true);
                  csf_decode<256, 128>(false);
                  csf_decode<256, 128>(true);
                  csf_decode<128, 4>(false);
                  return 0;
              },
              {{"format", "nvfp4_mul"}});
    suite.add("nvfp4_mul/g16",
              [] {
                  quantization(true);
                  decode_nvfp4();
                  return 0;
              },
              {{"format", "nvfp4_mul"}});
    suite.add("mxfp8/g32",
              [] {
                  quantization(false);
                  decode_mxfp8();
                  return 0;
              },
              {{"format", "mxfp8"}});
    suite.add("nvfp4_mul/expert_addressing",
              [] {
                  expert_addressing();
                  return 0;
              },
              {{"format", "nvfp4_mul"}});
    return suite.main(argc, argv);
}
