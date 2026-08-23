#include "ops/linear/fp8/fp8_format.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(std::string(operation) + ": FP8 geometry overflows");
    }
    return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(operation) + ": FP8 geometry overflows");
    }
    return left + right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* operation) {
    return checked_mul(checked_add(value, alignment - 1, operation) / alignment, alignment,
                       operation);
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
    if (weight.qtype != QType::FP8_E4M3FN_ROW_BF16S || weight.layout != QuantLayout::RowScale ||
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
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string(operation) + ": invalid FP8 weight");
    }

    const auto* payload = static_cast<const std::byte*>(weight.payload);
    if (weight.qdata != payload || weight.scales != payload + geometry.scale_plane_offset) {
        throw std::invalid_argument(std::string(operation) + ": invalid FP8 plane geometry");
    }
    return geometry;
}

} // namespace ninfer::ops::detail
