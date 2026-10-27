// Windows implementation of core/io/file.h, mirroring the POSIX build's buffered and
// direct (O_DIRECT) read modes. Buffered files use a cache-friendly handle; direct files
// use FILE_FLAG_NO_BUFFERING, whose sector-alignment contract the engine's 4 KiB read
// unit satisfies. Positional reads use the synchronous OVERLAPPED form of ReadFile.

#include "core/io/file.h"

#include <windows.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <malloc.h>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer::io {
namespace {

constexpr std::size_t kMaxSingleRead =
    static_cast<std::size_t>(std::numeric_limits<DWORD>::max());

// FILE_FLAG_NO_BUFFERING requires sector-aligned buffers and offsets. Physical sector
// sizes are at most 4 KiB on current hardware, which is also the engine's read unit.
constexpr std::size_t kDirectAlignment = 4096;

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                            path.string() + ": " + operation);
}

void* open_handle(const std::filesystem::path& path, ReadMode mode) {
    const DWORD flags = mode == ReadMode::Direct ? FILE_FLAG_NO_BUFFERING : 0;
    void* handle       = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) fail(path, "open");
    return handle;
}

std::size_t read_at(void* handle, std::uint64_t offset, std::span<std::byte> destination,
                    const std::filesystem::path& path) {
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD bytes_read = 0;
    if (::ReadFile(handle, destination.data(), static_cast<DWORD>(destination.size()),
                   &bytes_read, &overlapped) == 0)
        fail(path, "ReadFile");
    return bytes_read;
}

} // namespace

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    // The descriptor is opaque on Windows: only the Linux io_uring read queue consumes
    // it, and that path is not built here. Views carry a nonnegative placeholder.
    view_.descriptor  = 0;
    view_.mode        = mode;
    view_.memory_alignment = mode == ReadMode::Direct ? kDirectAlignment : 1;
    view_.offset_alignment = mode == ReadMode::Direct ? kDirectAlignment : 1;
    void* handle      = open_handle(path, mode);
    try {
        LARGE_INTEGER size{};
        if (::GetFileSizeEx(handle, &size) == 0 || size.QuadPart < 0)
            fail(path, "expected regular file");
        view_.bytes = static_cast<std::uint64_t>(size.QuadPart);
        if (mode == ReadMode::Direct && view_.bytes) {
            // Probe the alignment contract once, exactly like the Linux DIOALIGN fallback.
            // _aligned_malloc's argument order is (size, alignment), the reverse of
            // std::aligned_alloc; MSVC's STL does not provide the C++17 spelling.
            std::unique_ptr<std::byte[], decltype(&::_aligned_free)> probe(
                static_cast<std::byte*>(::_aligned_malloc(kDirectAlignment, kDirectAlignment)),
                &::_aligned_free);
            if (!probe) throw std::bad_alloc();
            (void)read_at(handle, 0, {probe.get(), kDirectAlignment}, path);
        }
    } catch (...) {
        ::CloseHandle(handle);
        throw;
    }
    handle_ = handle;
}

ReadOnlyFile::~ReadOnlyFile() {
    if (handle_) ::CloseHandle(handle_);
}

ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)),
      view_(std::exchange(other.view_, {})),
      handle_(std::exchange(other.handle_, nullptr)) {}

ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        if (handle_) ::CloseHandle(handle_);
        path_   = std::move(other.path_);
        view_   = std::exchange(other.view_, {});
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || !handle_)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        throw std::system_error(ERROR_HANDLE_EOF, std::system_category(),
                                path_.string() + ": read exceeds file length");
    while (!destination.empty()) {
        const auto count = std::min(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = read_at(handle_, offset, destination.subspan(0, count), path_);
        if (!read)
            throw std::system_error(ERROR_HANDLE_EOF, std::system_category(),
                                     path_.string() + ": unexpected EOF");
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(read);
    }
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    if (destination.empty()) return 0;
    return read_at(handle_, offset, destination, path_);
}

void validate_direct_read(FileView file, std::uint64_t offset,
                          std::span<std::byte> destination) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct ||
        !std::has_single_bit(file.memory_alignment) ||
        !std::has_single_bit(file.offset_alignment) || offset % file.offset_alignment ||
        destination.size() % file.offset_alignment ||
        (!destination.empty() &&
         (!destination.data() ||
          reinterpret_cast<std::uintptr_t>(destination.data()) % file.memory_alignment)) ||
        destination.size() > kMaxSingleRead)
        throw std::invalid_argument("invalid direct read range or alignment");
    if (offset > std::numeric_limits<std::uint64_t>::max() - destination.size())
        throw std::overflow_error("direct read range overflow");
}

DirectReadWindow direct_read_window(FileView file, std::uint64_t offset, std::size_t bytes) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct || !bytes ||
        !std::has_single_bit(file.offset_alignment) || offset > file.bytes ||
        bytes > file.bytes - offset)
        throw std::invalid_argument("invalid direct read window");
    const auto alignment = file.offset_alignment;
    const auto prefix    = offset % alignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - prefix - (alignment - 1))
        throw std::overflow_error("direct read window size overflow");
    const auto required = bytes + prefix;
    const auto count    = (required + alignment - 1) / alignment * alignment;
    return {offset - prefix, count, prefix, required};
}

} // namespace ninfer::io