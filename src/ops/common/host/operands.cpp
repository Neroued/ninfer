#include "ops/common/host/operands.h"

#include <cmath>

namespace ninfer::ops::common {

namespace {
std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::overflow_error(std::string(operation) + ": weight geometry overflows");
    return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error(std::string(operation) + ": weight geometry overflows");
    return left + right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* operation) {
    return checked_mul(checked_add(value, alignment - 1, operation) / alignment, alignment,
                       operation);
}

bool matrix_shape(const Weight& weight) {
    return weight.ndim == 2 && weight.shape[0] == weight.n && weight.shape[1] == weight.k &&
           weight.shape[2] == 1 && weight.shape[3] == 1 && weight.padded_shape[0] == weight.n &&
           weight.padded_shape[1] == weight.k && weight.padded_shape[2] == 1 &&
           weight.padded_shape[3] == 1;
}

void weight_planes(const Weight& weight, std::uint64_t codes_bytes, std::uint64_t scales_bytes,
                   std::size_t scale_alignment, const char* operation) {
    if (!weight.payload || !operand::aligned(weight.qdata, 16) ||
        !operand::aligned(weight.scales, scale_alignment) || weight.qhigh ||
        weight.high_plane_bytes)
        throw std::invalid_argument(std::string(operation) + ": invalid encoded weight planes");
    const auto backing = operand::range(weight.payload, weight.payload_bytes, operation);
    const auto codes   = operand::range(weight.qdata, codes_bytes, operation);
    const auto scales  = operand::range(weight.scales, scales_bytes, operation);
    if (codes.begin < backing.begin || codes.end > backing.end || scales.begin < backing.begin ||
        scales.end > backing.end || operand::overlaps(codes, scales))
        throw std::invalid_argument(std::string(operation) +
                                    ": weight planes exceed their backing");
}

Fp8WeightGeometry grouped_fp8_weight(const Weight& weight, QType format, QuantLayout layout,
                                     DType scale_dtype, int group, int block_rows,
                                     std::size_t scale_word, const char* operation) {
    if (weight.n <= 0 || weight.k <= 0 || weight.n % block_rows || weight.k % group)
        throw std::invalid_argument(std::string(operation) + ": FP8 shape splits a scale group");
    Fp8WeightGeometry geometry{};
    geometry.code_plane_bytes   = checked_mul(weight.n, weight.k, operation);
    geometry.scale_plane_offset = align_up(geometry.code_plane_bytes, 256, operation);
    const auto scale_columns    = weight.k / group;
    const auto scale_rows       = weight.n / block_rows;
    const auto scale_stride     = checked_mul(scale_columns, scale_word, operation);
    geometry.scale_plane_bytes  = checked_mul(scale_rows, scale_stride, operation);
    geometry.required_payload_bytes =
        checked_add(geometry.scale_plane_offset, geometry.scale_plane_bytes, operation);
    if (weight.qtype != format || weight.layout != layout || weight.scale_dtype != scale_dtype ||
        weight.group_size != static_cast<std::uint32_t>(group) || weight.group != group ||
        !matrix_shape(weight) || weight.scale_ne[0] != scale_columns ||
        weight.scale_ne[1] != scale_rows || weight.scale_ne[2] != 1 || weight.scale_ne[3] != 1 ||
        weight.scale_nb[0] != static_cast<std::int64_t>(scale_word) ||
        weight.scale_nb[1] != static_cast<std::int64_t>(scale_stride) ||
        weight.scale_nb[2] != static_cast<std::int64_t>(geometry.scale_plane_bytes) ||
        weight.scale_nb[3] != static_cast<std::int64_t>(geometry.scale_plane_bytes) ||
        weight.payload_bytes < geometry.required_payload_bytes)
        throw std::invalid_argument(std::string(operation) + ": invalid grouped FP8 weight");
    weight_planes(weight, geometry.code_plane_bytes, geometry.scale_plane_bytes, scale_word,
                  operation);
    return geometry;
}
} // namespace

Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation) {
    if (weight.n <= 0 || weight.k <= 0) {
        throw std::invalid_argument(std::string(operation) + ": FP8 shape must be positive");
    }

    Fp8WeightGeometry geometry{};
    geometry.code_plane_bytes   = checked_mul(static_cast<std::uint64_t>(weight.n),
                                              static_cast<std::uint64_t>(weight.k), operation);
    geometry.scale_plane_offset = align_up(geometry.code_plane_bytes, 256, operation);
    geometry.scale_plane_bytes  = checked_mul(static_cast<std::uint64_t>(weight.n), 2, operation);
    geometry.required_payload_bytes =
        checked_add(geometry.scale_plane_offset, geometry.scale_plane_bytes, operation);

    const std::int64_t scale_stride = static_cast<std::int64_t>(weight.n) * 2;
    if (weight.qtype != QType::FP8_E4M3FN_ROW_BF16 || weight.layout != QuantLayout::RowScale ||
        weight.scale_dtype != DType::BF16 ||
        weight.group_size != static_cast<std::uint32_t>(weight.k) || weight.group != weight.k ||
        weight.ndim != 2 || weight.shape[0] != weight.n || weight.shape[1] != weight.k ||
        weight.shape[2] != 1 || weight.shape[3] != 1 || weight.padded_shape[0] != weight.n ||
        weight.padded_shape[1] != weight.k || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.scale_ne[0] != weight.n || weight.scale_ne[1] != 1 ||
        weight.scale_ne[2] != 1 || weight.scale_ne[3] != 1 || weight.scale_nb[0] != 2 ||
        weight.scale_nb[1] != scale_stride || weight.scale_nb[2] != scale_stride ||
        weight.scale_nb[3] != scale_stride || weight.payload == nullptr ||
        weight.qdata == nullptr || weight.scales == nullptr || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.payload_bytes < geometry.required_payload_bytes ||
        !operand::aligned(weight.qdata, 16) || !operand::aligned(weight.scales, 16)) {
        throw std::invalid_argument(std::string(operation) + ": invalid FP8 weight");
    }

    const auto* payload = static_cast<const std::byte*>(weight.payload);
    if (weight.qdata != payload || weight.scales != payload + geometry.scale_plane_offset) {
        throw std::invalid_argument(std::string(operation) + ": invalid FP8 plane geometry");
    }
    return geometry;
}

Fp8WeightGeometry validate_mxfp8_weight(const Weight& weight, const char* operation) {
    return grouped_fp8_weight(weight, QType::MXFP8, QuantLayout::RowGroupScale, DType::U8, 32, 1, 1,
                              operation);
}

Fp8WeightGeometry validate_fp8_block_fp32_weight(const Weight& weight, const char* operation) {
    return grouped_fp8_weight(weight, QType::FP8_E4M3FN_BLOCK128X128_FP32,
                              QuantLayout::BlockScaleM128K128, DType::FP32, 128, 128, 4, operation);
}

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation) {
    if (weight.n <= 0 || weight.k <= 0 || (weight.n % 128) != 0 || (weight.k % 64) != 0) {
        throw std::invalid_argument(std::string(operation) + ": NVFP4 requires N%128=0 and K%64=0");
    }

    const std::uint64_t elements = checked_mul(static_cast<std::uint64_t>(weight.n),
                                               static_cast<std::uint64_t>(weight.k), operation);
    Nvfp4WeightGeometry geometry{};
    geometry.code_plane_bytes   = elements / 2;
    geometry.scale_plane_offset = align_up(geometry.code_plane_bytes, 256, operation);
    geometry.scale_plane_bytes  = elements / 16;
    geometry.required_payload_bytes =
        checked_add(checked_add(geometry.scale_plane_offset, geometry.scale_plane_bytes, operation),
                    sizeof(float), operation);

    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.scale_dtype != DType::FP8_E4M3FN || weight.group_size != 16 || weight.group != 16 ||
        weight.ndim != 2 || weight.shape[0] != weight.n || weight.shape[1] != weight.k ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k ||
        weight.payload == nullptr || weight.qdata == nullptr || weight.scales == nullptr ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < geometry.required_payload_bytes ||
        !std::isfinite(weight.weight_scale) || weight.weight_scale <= 0.0F ||
        !std::isfinite(weight.input_scale) || weight.input_scale <= 0.0F ||
        !operand::aligned(weight.qdata, 16) || !operand::aligned(weight.scales, 16)) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 weight");
    }

    const auto backing = operand::range(weight.payload, weight.payload_bytes, operation);
    const auto codes   = operand::range(weight.qdata, geometry.code_plane_bytes, operation);
    const auto scales  = operand::range(weight.scales, geometry.scale_plane_bytes, operation);
    if (codes.begin < backing.begin || codes.end > backing.end || scales.begin < backing.begin ||
        scales.end > backing.end || operand::overlaps(codes, scales)) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 plane geometry");
    }
    return geometry;
}

Nvfp4WeightGeometry validate_nvfp4_mul_weight(const Weight& weight, const char* operation) {
    if (weight.n <= 0 || weight.k <= 0 || weight.n % 128 || weight.k % 64)
        throw std::invalid_argument(std::string(operation) +
                                    ": NVFP4 multiplier requires N%128=0 and K%64=0");
    const auto elements = checked_mul(weight.n, weight.k, operation);
    Nvfp4WeightGeometry geometry{};
    geometry.code_plane_bytes   = elements / 2;
    geometry.scale_plane_offset = align_up(geometry.code_plane_bytes, 256, operation);
    geometry.scale_plane_bytes  = elements / 16;
    geometry.required_payload_bytes =
        checked_add(checked_add(geometry.scale_plane_offset, geometry.scale_plane_bytes, operation),
                    4, operation);
    if (weight.qtype != QType::NVFP4_MUL || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.scale_dtype != DType::FP8_E4M3FN || weight.group_size != 16 || weight.group != 16 ||
        !matrix_shape(weight) || weight.payload_bytes < geometry.required_payload_bytes ||
        !std::isfinite(weight.weight_scale) || weight.weight_scale <= 0)
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 multiplier weight");
    weight_planes(weight, geometry.code_plane_bytes, geometry.scale_plane_bytes, 16, operation);
    return geometry;
}

} // namespace ninfer::ops::common
