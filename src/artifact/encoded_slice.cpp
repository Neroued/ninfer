#include "artifact/encoded_slice.h"

#include "artifact/reader.h"

#include <algorithm>

namespace ninfer::artifact {
namespace {
std::uint64_t columns(const WeightGeometry& g) { return g.shape.empty() ? 1 : g.shape.back(); }

bool nvfp4(QuantLayout layout) {
    return layout == QuantLayout::BlockScaleK16M128x4 ||
           layout == QuantLayout::BatchedBlockScaleK16M128x4;
}
} // namespace

EncodedTensor whole_tensor(const Reader& reader, ObjectHandle object) {
    const auto& g = reader.geometry(object);
    return {g, {{object, 0, 0, g.bytes}}, object};
}

EncodedTensor slice_tensor(const Reader& reader, std::span<const std::uint64_t> shape,
                           std::span<const MatrixSlice> slices) {
    if (slices.empty()) throw ArtifactError("encoded selection has no source slices");
    const auto& source = reader.geometry(slices.front().object);
    const auto format  = source.format;
    const auto layout =
        source.layout == QuantLayout::BatchedBlockScaleK16M128x4 && shape.size() == 2
            ? QuantLayout::BlockScaleK16M128x4
            : source.layout;
    EncodedTensor out{weight_geometry(format, layout, shape), {}, {}};
    const auto& d = out.geometry;
    if (layout == QuantLayout::RowPagedG16_4K)
        throw ArtifactError("paged NF4 selection requires complete source shards");
    const auto dk = columns(d), dn = d.elements / dk;
    std::uint64_t covered = 0;
    std::optional<std::pair<ObjectHandle, std::uint64_t>> divisor;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        const auto& s = slices[i];
        const auto& g = reader.geometry(s.object);
        const auto sk = columns(g), sn = g.elements / sk;
        if (g.format != format || (g.layout != layout && !(nvfp4(g.layout) && nvfp4(layout))) ||
            !s.rows || !s.columns || s.row > sn || s.rows > sn - s.row || s.column > sk ||
            s.columns > sk - s.column || s.destination_row > dn ||
            s.rows > dn - s.destination_row || s.destination_column > dk ||
            s.columns > dk - s.destination_column)
            throw ArtifactError("encoded selection exceeds its matrix or changes representation");
        for (std::size_t j = 0; j < i; ++j) {
            const auto& p = slices[j];
            if (s.destination_row < p.destination_row + p.rows &&
                p.destination_row < s.destination_row + s.rows &&
                s.destination_column < p.destination_column + p.columns &&
                p.destination_column < s.destination_column + s.columns)
                throw ArtifactError("encoded selection overlaps destination elements");
        }
        covered         = checked_add(covered, checked_mul(s.rows, s.columns, "selected elements"),
                                      "selected elements");
        const auto copy = [&](std::uint64_t source, std::uint64_t destination, std::uint64_t bytes,
                              std::uint64_t count, std::uint64_t source_stride,
                              std::uint64_t destination_stride) {
            if (!bytes || !count) return;
            if (source_stride == bytes && destination_stride == bytes) {
                bytes         = checked_mul(bytes, count, "encoded copy bytes");
                count         = 1;
                source_stride = destination_stride = 0;
            }
            out.copies.push_back(
                {s.object, source, destination, bytes, count, source_stride, destination_stride});
        };
        if (layout == QuantLayout::Contiguous) {
            const auto word = d.bytes / d.elements;
            copy((s.row * sk + s.column) * word,
                 (s.destination_row * dk + s.destination_column) * word, s.columns * word, s.rows,
                 sk * word, dk * word);
            continue;
        }
        if (layout == QuantLayout::RowScale) {
            copy(s.row * sk + s.column, s.destination_row * dk + s.destination_column, s.columns,
                 s.rows, sk, dk);
            if (s.destination_column != 0 || s.columns != dk)
                throw ArtifactError("row-scaled assembly requires complete destination rows");
            copy(g.scale_offset + s.row * 2, d.scale_offset + s.destination_row * 2, s.rows * 2, 1,
                 0, 0);
            continue;
        }
        const auto group = g.group_size;
        const bool full_columns =
            s.column == 0 && s.columns == sk && s.destination_column == 0 && dk == sk;
        const auto block_rows = nvfp4(layout) ? 128 : g.scale_block_rows;
        if (s.row % block_rows || s.rows % block_rows || s.destination_row % block_rows ||
            (!full_columns &&
             (s.column % group || s.columns % group || s.destination_column % group)) ||
            (nvfp4(layout) && (s.column % 64 || s.columns % 64 || s.destination_column % 64)))
            throw ArtifactError("encoded selection splits a scale block");
        const auto groups = full_columns ? g.padded_columns / group : s.columns / group;
        const auto plane  = [&](std::uint64_t source_base, std::uint64_t destination_base,
                               std::uint64_t source_row, std::uint64_t destination_row) {
            const auto word = source_row / (g.padded_columns / group);
            copy(source_base + s.row * source_row + s.column / group * word,
                  destination_base + s.destination_row * destination_row +
                      s.destination_column / group * word,
                  groups * word, s.rows, source_row, destination_row);
        };
        plane(0, 0, g.code_bytes_per_row, d.code_bytes_per_row);
        if (g.high_bytes)
            plane(g.high_offset, d.high_offset, g.high_bytes_per_row, d.high_bytes_per_row);
        if (nvfp4(layout)) {
            copy(g.scale_offset + (s.row / 128 * (sk / 64) + s.column / 64) * 512,
                 d.scale_offset +
                     (s.destination_row / 128 * (dk / 64) + s.destination_column / 64) * 512,
                 s.columns / 64 * 512, s.rows / 128, sk / 64 * 512, dk / 64 * 512);
            const auto matrix_rows = g.shape[g.shape.size() - 2];
            const auto matrix      = s.row / matrix_rows;
            if ((s.row + s.rows - 1) / matrix_rows != matrix || d.divisor_count != 1)
                throw ArtifactError("NVFP4 selection crosses matrix divisor domains");
            const auto source = std::pair{s.object, matrix};
            if (divisor && *divisor != source)
                throw ArtifactError("NVFP4 assembly mixes matrix divisor domains");
            if (!divisor) copy(g.divisor_offset + matrix * 4, d.divisor_offset, 4, 1, 0, 0);
            divisor = source;
        } else {
            const auto scale_word = g.scale_row_bytes / (g.padded_columns / group);
            copy(g.scale_offset + s.row / block_rows * g.scale_row_bytes +
                     s.column / group * scale_word,
                 d.scale_offset + s.destination_row / block_rows * d.scale_row_bytes +
                     s.destination_column / group * scale_word,
                 groups * scale_word, s.rows / block_rows, g.scale_row_bytes, d.scale_row_bytes);
        }
    }
    if (covered != d.elements) throw ArtifactError("encoded selection does not cover destination");
    return out;
}

} // namespace ninfer::artifact
