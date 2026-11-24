#pragma once

#include "artifact/reader.h"

namespace ninfer::artifact {

struct FileObjectView {
    std::uint64_t bytes = 0;
    // Complete object mapping: destination_offset is object-relative, file_index indexes
    // FileBacking::files(). Both the mapping and file views borrow the backing owner.
    std::span<const ReadSegment> segments;
};

// Move-only immutable backing for selected encoded objects. Files and view storage survive
// Reader destruction and owner moves. All borrowers and pending reads end before destruction.
class FileBacking {
public:
    FileBacking();
    ~FileBacking();
    FileBacking(FileBacking&&) noexcept;
    FileBacking& operator=(FileBacking&&) noexcept;
    FileBacking(const FileBacking&)            = delete;
    FileBacking& operator=(const FileBacking&) = delete;

    [[nodiscard]] FileObjectView object(ObjectHandle handle) const;
    [[nodiscard]] const WeightGeometry& geometry(ObjectHandle handle) const;
    [[nodiscard]] std::span<const io::FileView> files() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend FileBacking prepare_file_backing(const Reader&, std::span<const ObjectHandle>);
};

[[nodiscard]] FileBacking prepare_file_backing(const Reader& reader,
                                               std::span<const ObjectHandle> objects);

// Allocation-free range mapping. Returned segment destinations are relative to this request.
// Use object.segments.size() as an upper bound for output capacity; zero bytes yields zero parts.
[[nodiscard]] std::size_t map_file_range(FileObjectView object, std::uint64_t offset,
                                         std::uint64_t bytes, std::span<ReadSegment> output);

} // namespace ninfer::artifact
