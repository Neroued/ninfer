#include "artifact/backing.h"

#include <algorithm>
#include <map>

namespace ninfer::artifact {

struct FileBacking::Impl {
    struct Object {
        std::uint64_t bytes;
        std::optional<WeightGeometry> geometry;
        std::vector<ReadSegment> segments;
    };

    std::vector<io::ReadOnlyFile> owners;
    std::vector<io::FileView> files;
    std::map<std::size_t, Object> objects;
};

FileBacking::FileBacking() : impl_(std::make_unique<Impl>()) {}

FileBacking::~FileBacking()                                 = default;
FileBacking::FileBacking(FileBacking&&) noexcept            = default;
FileBacking& FileBacking::operator=(FileBacking&&) noexcept = default;

FileObjectView FileBacking::object(ObjectHandle handle) const {
    if (!impl_) throw ArtifactError("moved file backing");
    const auto found = impl_->objects.find(handle.index);
    if (found == impl_->objects.end()) throw ArtifactError("object has no file backing");
    return {found->second.bytes, found->second.segments};
}

const WeightGeometry& FileBacking::geometry(ObjectHandle handle) const {
    (void)object(handle);
    const auto& geometry = impl_->objects.at(handle.index).geometry;
    if (!geometry) throw ArtifactError("file object is not a tensor");
    return *geometry;
}

std::span<const io::FileView> FileBacking::files() const noexcept {
    return impl_ ? std::span<const io::FileView>(impl_->files) : std::span<const io::FileView>{};
}

FileBacking prepare_file_backing(const Reader& reader, std::span<const ObjectHandle> objects) {
    FileBacking result;
    auto& out = *result.impl_;
    std::map<std::size_t, std::size_t> file_indices;
    for (const auto handle : objects) {
        if (out.objects.contains(handle.index)) continue;
        reader.validate_object(handle);
        const auto& object = reader.directory().object(handle);
        FileBacking::Impl::Object entry{
            object_bytes(object), {}, reader.segments(object_offset(object), object_bytes(object))};
        if (std::holds_alternative<TensorObject>(object)) entry.geometry = reader.geometry(handle);
        for (auto& segment : entry.segments) {
            auto [it, inserted] = file_indices.try_emplace(segment.file_index, out.owners.size());
            if (inserted) {
                out.owners.push_back(reader.open_file(segment.file_index, io::ReadMode::Direct));
                out.files.push_back(out.owners.back().view());
            }
            segment.file_index = it->second;
        }
        out.objects.emplace(handle.index, std::move(entry));
    }
    return result;
}

std::size_t map_file_range(FileObjectView object, std::uint64_t offset, std::uint64_t bytes,
                           std::span<ReadSegment> output) {
    if (offset > object.bytes || bytes > object.bytes - offset)
        throw ArtifactError("file range exceeds object");
    if (!bytes) return 0;
    const auto end = offset + bytes;
    auto first     = std::lower_bound(object.segments.begin(), object.segments.end(), offset,
                                      [](const ReadSegment& segment, std::uint64_t position) {
                                      return segment.destination_offset + segment.bytes <= position;
                                  });
    auto last      = first;
    while (last != object.segments.end() && last->destination_offset < end) ++last;
    const auto count = static_cast<std::size_t>(last - first);
    if (count > output.size()) throw ArtifactError("insufficient file range output capacity");
    std::size_t i         = 0;
    std::uint64_t covered = 0;
    for (auto it = first; it != last; ++it) {
        const auto begin  = std::max(offset, it->destination_offset);
        const auto finish = std::min(end, it->destination_offset + it->bytes);
        output[i++]       = {it->file_index, it->file_offset + begin - it->destination_offset,
                             begin - offset, finish - begin};
        covered += finish - begin;
    }
    if (covered != bytes) throw ArtifactError("incomplete file object mapping");
    return count;
}

} // namespace ninfer::artifact
