#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninfer::io {

enum class ReadMode { Buffered, Direct };

struct FileView {
    int descriptor               = -1;
    std::uint64_t bytes          = 0;
    ReadMode mode                = ReadMode::Buffered;
    std::size_t memory_alignment = 1;
    std::size_t offset_alignment = 1;
};

// One immutable regular-file handle. Views borrow it; all I/O completes before destruction.
class ReadOnlyFile {
public:
    ReadOnlyFile(const std::filesystem::path& path, ReadMode mode = ReadMode::Buffered);
    ~ReadOnlyFile();
    ReadOnlyFile(ReadOnlyFile&&) noexcept;
    ReadOnlyFile& operator=(ReadOnlyFile&&) noexcept;
    ReadOnlyFile(const ReadOnlyFile&)            = delete;
    ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;

    [[nodiscard]] FileView view() const noexcept { return view_; }

    [[nodiscard]] std::uint64_t bytes() const noexcept { return view_.bytes; }

    // Buffered exact read. A missing required byte is an error.
    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    // One aligned direct read, including a possible short final block.
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;

private:
    std::filesystem::path path_;
    FileView view_;
#ifdef _WIN32
    // Windows HANDLE. The Linux build stores its descriptor in view_.descriptor instead;
    // handles are pointer-sized and cannot travel through that int field.
    void* handle_ = nullptr;
#endif
};

struct DirectReadWindow {
    std::uint64_t offset;
    std::size_t bytes;
    std::size_t prefix;   // Requested bytes start here within the aligned buffer.
    std::size_t required; // Minimum returned byte count covering the requested range.
};

// Maps a nonempty in-file range to one enclosing aligned read. Its end may exceed EOF.
[[nodiscard]] DirectReadWindow direct_read_window(FileView file, std::uint64_t offset,
                                                  std::size_t bytes);
void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination);

} // namespace ninfer::io
