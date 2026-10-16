#include "models/qwen4_exp/ple/disk/table.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::ple::disk {

struct Table::Impl {
    struct Shard {
        artifact::FileObjectView object;
        Nf4Bf16Lut lut;
    };

    LookupParameters parameters;
    artifact::FileBacking backing;
    std::vector<Shard> shards;
    std::size_t max_segments = 0;

    Impl(const artifact::Reader& reader, LookupParameters params) : parameters(std::move(params)) {
        std::vector<artifact::ObjectHandle> objects;
        objects.reserve(parameters.shards.size());
        for (const auto& shard : parameters.shards) objects.push_back(shard.object);
        backing = artifact::prepare_file_backing(reader, objects);
        shards.reserve(objects.size());
        for (const auto& shard : parameters.shards) {
            const auto view = backing.object(shard.object);
            max_segments    = std::max(max_segments, view.segments.size());
            try {
                shards.push_back({view, make_nf4_bf16_lut(shard.divisor)});
            } catch (const std::invalid_argument& e) {
                throw artifact::ArtifactError(reader.directory().tensor(shard.object).id + ": " +
                                              e.what());
            }
        }
    }
};

Table::Table(const artifact::Reader& reader, LookupParameters parameters)
    : impl_(std::make_unique<Impl>(reader, std::move(parameters))) {}

Table::~Table()                           = default;
Table::Table(Table&&) noexcept            = default;
Table& Table::operator=(Table&&) noexcept = default;

RowSource Table::row(std::uint32_t global_row) const {
    const auto& parameters = impl_->parameters;
    if (global_row >= parameters.rows) throw std::out_of_range("PLE global row exceeds table");
    const auto it = std::upper_bound(
        parameters.shards.begin(), parameters.shards.end(), global_row,
        [](std::uint32_t row, const TableShard& shard) { return row < shard.row_end; });
    const auto local   = global_row - (it->row_end - it->geometry.shape[0]);
    const auto& source = impl_->shards[it - parameters.shards.begin()];
    return {source.object, weight_paged_row_offset(it->geometry, local), &source.lut};
}

const LookupParameters& Table::parameters() const noexcept { return impl_->parameters; }

std::span<const io::FileView> Table::files() const noexcept { return impl_->backing.files(); }

TableGeometry Table::geometry() const noexcept {
    TableGeometry out{impl_->max_segments};
    for (const auto file : files()) {
        out.memory_alignment = std::max(out.memory_alignment, std::size_t(file.memory_alignment));
        out.read_alignment   = std::max({out.read_alignment, std::size_t(file.offset_alignment),
                                         std::size_t(file.memory_alignment)});
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp::ple::disk
