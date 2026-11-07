#include "artifact/csf_selection.h"

#include "artifact/framing.h"
#include "artifact/reader.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace ninfer::artifact {
namespace {

std::uint16_t read_u16(const std::byte* p) {
    return std::to_integer<std::uint16_t>(p[0]) | (std::to_integer<std::uint16_t>(p[1]) << 8);
}

void put_u16(std::byte* p, std::uint16_t n) {
    p[0] = std::byte(n);
    p[1] = std::byte(n >> 8);
}

std::uint64_t matrix_rows(const WeightGeometry& g) { return g.shape[g.shape.size() - 2]; }

class MetadataWindow {
public:
    void reset() { size_ = 0; }

    void read(const Reader& reader, std::uint64_t plane, std::uint64_t plane_bytes,
              std::uint64_t offset, std::span<std::byte> output) {
        while (!output.empty()) {
            if (offset < begin_ || offset >= begin_ + size_) {
                begin_ = offset / data_.size() * data_.size();
                size_  = std::min<std::uint64_t>(data_.size(), plane_bytes - begin_);
                reader.read_into(plane + begin_, std::span(data_).first(size_));
            }
            const auto count = std::min<std::uint64_t>(output.size(), begin_ + size_ - offset);
            std::memcpy(output.data(), data_.data() + offset - begin_, count);
            offset += count;
            output = output.subspan(count);
        }
    }

private:
    std::array<std::byte, 64 * 1024> data_{};
    std::uint64_t begin_ = 0, size_ = 0;
};

// A source page is bounded by 32 tiles: 256 blocks, 8192 nibble bytes, 32768 exception bytes.
class SourcePage {
public:
    explicit SourcePage(const Reader& reader) : reader_(reader) {}

    void load(ObjectHandle object, const WeightGeometry& g, std::uint64_t matrix,
              const Csf32Block& block, bool payload) {
        if (object == object_ && matrix == matrix_ && block.page == page_ &&
            (!payload || loaded_payload_))
            return;
        if (object != object_ || matrix != matrix_) {
            offsets_cache_.reset();
            page_bases_cache_.reset();
        }
        object_                  = object;
        matrix_                  = matrix;
        page_                    = block.page;
        loaded_payload_          = false;
        const auto& c            = g.csf;
        const auto page          = csf32_page(g, block.page);
        offset_begin_            = page.offset_begin;
        const auto words         = page.block_end - page.block_begin + 1;
        const auto object_offset = reader_.directory().tensor(object).offset;
        offsets_cache_.read(
            reader_, object_offset + c.offsets_offset + matrix * c.offsets_per_matrix * 2,
            c.offsets_per_matrix * 2, offset_begin_ * 2, std::span(offsets_).first(words * 2));
        std::array<std::byte, 4> base{};
        page_bases_cache_.read(
            reader_, object_offset + c.page_bases_offset + matrix * c.pages_per_matrix * 4,
            c.pages_per_matrix * 4, block.page * 4, base);
        const auto begin = read_u32_le(base.data());
        const auto last  = read_u16(offsets_.data() + (words - 1) * 2);
        if (begin % 2 || read_u16(offsets_.data()) || last > exceptions_.size() / 2 ||
            begin / 2 > c.exception_count || last > c.exception_count - begin / 2)
            throw ArtifactError("CSF source page exceeds its exception arena");
        if (!payload) return;
        const auto first = csf32_block(g, page.block_begin);
        const auto tail  = csf32_block(g, page.block_end - 1);
        nibble_begin_    = first.nibble_offset;
        const auto bytes = tail.nibble_offset + tail.columns * 16 - nibble_begin_;
        read(c.nibbles_offset + matrix * c.nibble_bytes_per_matrix + nibble_begin_,
             std::span(nibbles_).first(bytes));
        read(c.exceptions_offset + begin, std::span(exceptions_).first(last * 2));
        loaded_payload_ = true;
    }

    std::uint64_t count(const Csf32Block& first, std::uint64_t end_ordinal) const {
        const auto begin = read_u16(offsets_.data() + (first.offset_index - offset_begin_) * 2);
        const auto end = read_u16(offsets_.data() + (end_ordinal + first.page - offset_begin_) * 2);
        if (end < begin || end > 16384) throw ArtifactError("invalid CSF page exception offsets");
        return end - begin;
    }

    std::span<const std::byte> nibbles(const Csf32Block& first, const Csf32Block& last) const {
        return std::span(nibbles_).subspan(first.nibble_offset - nibble_begin_,
                                           last.nibble_offset + last.columns * 16 -
                                               first.nibble_offset);
    }

    std::span<const std::byte> exceptions(const Csf32Block& first, std::uint64_t count) const {
        const auto begin = read_u16(offsets_.data() + (first.offset_index - offset_begin_) * 2);
        return std::span(exceptions_)
            .subspan(begin * 2, this->count(first, first.ordinal + count) * 2);
    }

    std::uint16_t append_offsets(const Csf32Block& first, std::uint64_t count,
                                 std::byte* destination, std::uint16_t prefix) const {
        const auto* input = offsets_.data() + (first.offset_index - offset_begin_) * 2;
        const auto begin  = read_u16(input);
        auto previous     = begin;
        for (std::uint64_t i = 1; i <= count; ++i) {
            const auto value = read_u16(input + i * 2);
            if (value < previous || value - previous > 64 || value > 16384)
                throw ArtifactError("invalid CSF block exception offsets");
            put_u16(destination + (i - 1) * 2, prefix + value - begin);
            previous = value;
        }
        return prefix + previous - begin;
    }

private:
    void read(std::uint64_t offset, std::span<std::byte> output) {
        reader_.read_into(reader_.directory().tensor(object_).offset + offset, output);
    }

    const Reader& reader_;
    ObjectHandle object_;
    std::uint64_t matrix_ = UINT64_MAX, page_ = UINT64_MAX, offset_begin_ = 0, nibble_begin_ = 0;
    bool loaded_payload_ = false;
    MetadataWindow offsets_cache_, page_bases_cache_;
    std::array<std::byte, 514> offsets_{};
    std::array<std::byte, 8192> nibbles_{};
    std::array<std::byte, 32768> exceptions_{};
};

struct SourceRun {
    ObjectHandle object;
    const WeightGeometry& geometry;
    std::uint64_t matrix;
    Csf32Block block;
    std::uint64_t count;
};

SourceRun source_run(const Reader& reader, const EncodedTensor& tensor, std::uint64_t matrix,
                     const Csf32Block& block, std::uint64_t destination_end) {
    const auto row = matrix * matrix_rows(tensor.geometry) + block.row;
    auto it        = std::upper_bound(tensor.csf_slices.begin(), tensor.csf_slices.end(), row,
                                      [](auto r, const MatrixSlice& s) { return r < s.destination_row; });
    if (it == tensor.csf_slices.begin()) throw ArtifactError("CSF destination row has no source");
    const auto& slice = *--it;
    if (row + 32 > slice.destination_row + slice.rows)
        throw ArtifactError("CSF selection splits a compressed row block");
    const auto& g         = reader.geometry(slice.object);
    const auto source_row = slice.row + row - slice.destination_row;
    const auto rows       = matrix_rows(g);
    const auto ordinal = csf32_block_index(g, source_row % rows, slice.column / 16 + block.column);
    const auto source  = csf32_block(g, ordinal);
    if (source.columns != block.columns)
        throw ArtifactError("CSF selection changes a compressed column block");
    const auto destination_rows  = matrix_rows(tensor.geometry);
    const auto tile_begin        = block.row / 128 * 128;
    const auto tile_rows         = std::min<std::uint64_t>(128, destination_rows - tile_begin);
    const auto source_tile_begin = source.row / 128 * 128;
    const auto source_tile_rows  = std::min<std::uint64_t>(128, rows - source_tile_begin);
    auto count                   = std::min({(tile_begin + tile_rows - block.row) / 32,
                                             (source_tile_begin + source_tile_rows - source.row) / 32,
                                             (slice.destination_row + slice.rows - row) / 32});
    const auto global_tile_begin = matrix * destination_rows + tile_begin;
    if (slice.destination_row <= global_tile_begin &&
        slice.destination_row + slice.rows >= global_tile_begin + tile_rows &&
        source.row - source_tile_begin == block.row - tile_begin && source_tile_rows == tile_rows) {
        const auto pairs = (tensor.geometry.shape.back() / 16 + 1) / 2;
        count = (pairs - block.column / 2) * (tile_rows / 32) - (block.row - tile_begin) / 32;
    }
    count = std::min({count, destination_end - block.ordinal,
                      csf32_page(g, source.page).block_end - source.ordinal});
    return {slice.object, g, source_row / rows, source, count};
}

class PlaneWriter {
public:
    PlaneWriter(std::uint64_t offset,
                const std::function<void(std::uint64_t, std::span<const std::byte>)>& write)
        : offset_(offset), write_(write), data_(kCsfRepackBufferBytes) {}

    void append(std::span<const std::byte> data) {
        while (!data.empty()) {
            const auto count = std::min(data.size(), data_.size() - used_);
            std::memcpy(data_.data() + used_, data.data(), count);
            used_ += count;
            data = data.subspan(count);
            if (used_ == data_.size()) flush();
        }
    }

    void flush() {
        if (!used_) return;
        write_(offset_, std::span(data_).first(used_));
        offset_ += used_;
        used_ = 0;
    }

private:
    std::uint64_t offset_;
    const std::function<void(std::uint64_t, std::span<const std::byte>)>& write_;
    std::vector<std::byte> data_;
    std::size_t used_ = 0;
};

} // namespace

EncodedTensor select_csf_tensor(const Reader& reader, std::span<const std::uint64_t> shape,
                                std::span<const MatrixSlice> slices) {
    const auto read_begin = reader.payload_read_bytes();
    const auto& source    = reader.geometry(slices.front().object);
    EncodedTensor out{weight_geometry(source.format, QuantLayout::Csf32, shape, {0}),
                      {},
                      {},
                      {slices.begin(), slices.end()}};
    std::sort(out.csf_slices.begin(), out.csf_slices.end(),
              [](const auto& a, const auto& b) { return a.destination_row < b.destination_row; });
    const auto& d              = out.geometry;
    const auto dk              = d.shape.back();
    std::uint64_t covered_rows = 0;
    std::vector<std::optional<std::pair<ObjectHandle, std::uint64_t>>> scalars(
        d.matrix_scale_count);
    for (const auto& s : out.csf_slices) {
        const auto& g   = reader.geometry(s.object);
        const auto sk   = g.shape.back();
        const auto rows = g.elements / sk;
        if (g.format != d.format || g.layout != QuantLayout::Csf32 || !s.rows || s.row > rows ||
            s.rows > rows - s.row || s.column > sk || s.columns > sk - s.column ||
            s.destination_row != covered_rows || s.destination_column || s.columns != dk ||
            s.row % 32 || s.rows % 32 || s.column % 32 ||
            (s.columns % 32 && s.column + s.columns != sk))
            throw ArtifactError(
                "CSF selection requires complete destination rows and scale blocks");
        covered_rows = checked_add(covered_rows, s.rows, "CSF selected rows");
        if (covered_rows > d.elements / dk)
            throw ArtifactError("CSF selection exceeds destination");
        out.copies.push_back({s.object, s.row * sk / 2 + s.column / 2, s.destination_row * dk / 2,
                              dk / 2, s.rows, sk / 2, dk / 2});
        out.copies.push_back(
            {s.object, g.csf.bases_offset + s.row, d.csf.bases_offset + s.destination_row, s.rows});
        for (std::uint64_t offset = 0; offset < s.rows;) {
            const auto sr = s.row + offset, dr = s.destination_row + offset;
            const auto sm = sr / matrix_rows(g), dm = dr / matrix_rows(d);
            const auto origin = std::pair{s.object, sm};
            if (scalars.at(dm) && *scalars[dm] != origin)
                throw ArtifactError("CSF assembly mixes matrix scale domains");
            if (!scalars[dm])
                out.copies.push_back(
                    {s.object, g.matrix_scale_offset + sm * 4, d.matrix_scale_offset + dm * 4, 4});
            scalars[dm] = origin;
            offset += std::min({s.rows - offset, matrix_rows(g) - sr % matrix_rows(g),
                                matrix_rows(d) - dr % matrix_rows(d)});
        }
    }
    if (covered_rows != d.elements / dk) throw ArtifactError("CSF selection is incomplete");
    SourcePage page(reader);
    std::uint64_t exceptions = 0;
    for (const auto& s : out.csf_slices) {
        const auto& g           = reader.geometry(s.object);
        const auto rows         = matrix_rows(g);
        const auto count_blocks = [&](std::uint64_t matrix, std::uint64_t begin,
                                      std::uint64_t end) {
            while (begin < end) {
                const auto block = csf32_block(g, begin);
                const auto stop  = std::min(end, csf32_page(g, block.page).block_end);
                page.load(s.object, g, matrix, block, false);
                exceptions =
                    checked_add(exceptions, page.count(block, stop), "selected CSF exceptions");
                begin = stop;
            }
        };
        for (auto row = s.row; row < s.row + s.rows;) {
            const auto matrix = row / rows, local = row % rows;
            const auto tile_begin   = local / 128 * 128;
            const auto tile_end     = std::min(rows, tile_begin + 128);
            const auto end          = std::min(s.row + s.rows - matrix * rows, tile_end);
            const auto first_column = s.column / 16;
            const auto last_column  = (s.column + s.columns - 1) / 16 / 2 * 2;
            if (local == tile_begin && end == tile_end) {
                count_blocks(matrix, csf32_block_index(g, local, first_column),
                             csf32_block_index(g, end - 32, last_column) + 1);
            } else {
                for (auto column = first_column; column <= last_column; column += 2)
                    count_blocks(matrix, csf32_block_index(g, local, column),
                                 csf32_block_index(g, end - 32, column) + 1);
            }
            row += end - local;
        }
    }
    out.geometry    = weight_geometry(source.format, QuantLayout::Csf32, shape, {exceptions});
    out.preparation = std::make_shared<const EncodedPreparationStats>(
        EncodedPreparationStats{reader.payload_read_bytes() - read_begin});
    return out;
}

std::uint64_t csf_repack_bytes(const WeightGeometry& g) {
    return g.matrix_scale_count * (g.csf.nibble_bytes_per_matrix + g.csf.offsets_per_matrix * 2 +
                                   g.csf.pages_per_matrix * 4) +
           g.csf.exception_count * 2;
}

void repack_csf_scales(
    const Reader& reader, const EncodedTensor& tensor,
    const std::function<void(std::uint64_t, std::span<const std::byte>)>& write) {
    const auto& g = tensor.geometry;
    PlaneWriter nibbles(g.csf.nibbles_offset, write), offsets(g.csf.offsets_offset, write),
        pages(g.csf.page_bases_offset, write), exceptions(g.csf.exceptions_offset, write);
    SourcePage source_page(reader);
    std::uint64_t total_exceptions = 0;
    for (std::uint64_t matrix = 0; matrix < g.matrix_scale_count; ++matrix) {
        for (std::uint64_t page = 0; page < g.csf.pages_per_matrix; ++page) {
            const auto range = csf32_page(g, page);
            const auto base  = static_cast<std::uint32_t>(total_exceptions * 2);
            const std::array base_bytes{std::byte(base), std::byte(base >> 8),
                                        std::byte(base >> 16), std::byte(base >> 24)};
            pages.append(base_bytes);
            std::array<std::byte, 514> page_offsets{};
            std::uint16_t relative = 0;
            for (auto ordinal = range.block_begin; ordinal < range.block_end;) {
                const auto block  = csf32_block(g, ordinal);
                const auto source = source_run(reader, tensor, matrix, block, range.block_end);
                source_page.load(source.object, source.geometry, source.matrix, source.block, true);
                const auto last =
                    csf32_block(source.geometry, source.block.ordinal + source.count - 1);
                nibbles.append(source_page.nibbles(source.block, last));
                const auto words = source_page.exceptions(source.block, source.count);
                exceptions.append(words);
                relative = source_page.append_offsets(
                    source.block, source.count,
                    page_offsets.data() + (ordinal - range.block_begin + 1) * 2, relative);
                ordinal += source.count;
            }
            offsets.append(
                std::span(page_offsets).first((range.block_end - range.block_begin + 1) * 2));
            total_exceptions += relative;
        }
    }
    if (total_exceptions != g.csf.exception_count)
        throw ArtifactError("CSF exception count changed after selection planning");
    nibbles.flush();
    offsets.flush();
    pages.flush();
    exceptions.flush();
}

} // namespace ninfer::artifact
