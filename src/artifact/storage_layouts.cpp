#include "artifact/reader.h"

#include <limits>

namespace ninfer::artifact {
namespace {

constexpr std::uint64_t kTensorAlignment = 256;
constexpr std::uint64_t kKAlignment      = 128;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment,
                       std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

struct QuantGeometry {
    std::uint64_t group_size;
    std::uint64_t base_bytes_per_group;
    std::uint64_t high_bytes_per_group;
};

QuantGeometry quant_geometry(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
        return {64, 32, 0};
    case NumericFormat::Q5G64_F16S:
        return {64, 32, 8};
    case NumericFormat::Q6G64_F16S:
        return {64, 32, 16};
    case NumericFormat::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw ArtifactError("row-split-k128-v1 requires a grouped quantized format");
    }
}

std::uint64_t direct_word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return 2;
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return 4;
    default:
        throw ArtifactError("contiguous-le-v1 requires BF16, FP32, or I32");
    }
}

} // namespace

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16:
        return "BF16";
    case NumericFormat::FP32:
        return "FP32";
    case NumericFormat::I32:
        return "I32";
    case NumericFormat::Q4G64_F16S:
        return "Q4G64_F16S";
    case NumericFormat::Q5G64_F16S:
        return "Q5G64_F16S";
    case NumericFormat::Q6G64_F16S:
        return "Q6G64_F16S";
    case NumericFormat::W8G32_F16S:
        return "W8G32_F16S";
    }
    return {};
}

std::string_view layout_name(StorageLayout layout) noexcept {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        return "contiguous-le-v1";
    case StorageLayout::RowSplitK128V1:
        return "row-split-k128-v1";
    }
    return {};
}

std::string_view encoding_name(ResourceEncoding encoding) noexcept {
    switch (encoding) {
    case ResourceEncoding::RawBytesV1:
        return "raw-bytes-v1";
    }
    return {};
}

std::uint64_t tensor_alignment(StorageLayout) noexcept {
    return kTensorAlignment;
}

std::uint64_t resource_alignment(ResourceEncoding) noexcept {
    return 1;
}

std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape) {
    if (layout == StorageLayout::ContiguousLeV1) {
        if (shape.size() > 16) {
            throw ArtifactError("contiguous-le-v1 supports rank 0 through 16");
        }
        std::uint64_t elements = 1;
        for (const auto dim : shape) {
            if (dim == 0) {
                throw ArtifactError("tensor shape dimensions must be positive");
            }
            elements = checked_mul(elements, dim, "tensor element count");
        }
        return checked_mul(elements, direct_word_bytes(format), "tensor encoded size");
    }

    if (layout != StorageLayout::RowSplitK128V1) {
        throw ArtifactError("unknown tensor layout");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
    }

    return row_split_geometry(format, shape).encoded_bytes;
}

RowSplitGeometry row_split_geometry(NumericFormat format,
                                    std::span<const std::uint64_t> shape) {
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
    }
    const auto format_geometry = quant_geometry(format);
    RowSplitGeometry out;
    out.rows = shape[0];
    out.columns = shape[1];
    out.padded_columns = align_up(shape[1], kKAlignment, "padded K");
    out.group_size = format_geometry.group_size;
    out.groups_per_row = out.padded_columns / out.group_size;
    out.low_bytes_per_group = format_geometry.base_bytes_per_group;
    out.high_bytes_per_group = format_geometry.high_bytes_per_group;
    const auto groups = checked_mul(out.rows, out.groups_per_row, "physical group count");
    out.low_plane_bytes =
        checked_mul(groups, out.low_bytes_per_group, "base plane bytes");
    out.high_plane_bytes =
        checked_mul(groups, out.high_bytes_per_group, "high plane bytes");
    out.scale_plane_bytes = checked_mul(groups, 2, "scale plane bytes");
    out.high_plane_offset =
        align_up(out.low_plane_bytes, kTensorAlignment, "high plane offset");
    const auto aligned_high =
        align_up(out.high_plane_bytes, kTensorAlignment, "scale plane alignment");
    out.scale_plane_offset =
        checked_add(out.high_plane_offset, aligned_high, "scale plane offset");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "tensor encoded size");
    return out;
}

} // namespace ninfer::artifact
