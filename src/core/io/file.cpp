#include "core/io/file.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::io {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       int error = errno) {
    throw std::system_error(error, std::generic_category(), path.string() + ": " + operation);
}

off_t position(std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::invalid_argument("file offset exceeds off_t");
    return static_cast<off_t>(value);
}

} // namespace

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    view_.descriptor =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (mode == ReadMode::Direct ? O_DIRECT : 0));
    if (view_.descriptor < 0) fail(path, "open");
    view_.mode = mode;
    try {
        struct stat status {};

        if (::fstat(view_.descriptor, &status)) fail(path, "fstat");
        if (status.st_size < 0 || !S_ISREG(status.st_mode))
            fail(path, "expected regular file", EINVAL);
        view_.bytes = static_cast<std::uint64_t>(status.st_size);
        if (mode == ReadMode::Direct) {
            struct statx alignment {};

            const bool reported = ::statx(view_.descriptor, "", AT_EMPTY_PATH | AT_STATX_DONT_SYNC,
                                          STATX_DIOALIGN, &alignment) == 0 &&
                                  (alignment.stx_mask & STATX_DIOALIGN);
            view_.memory_alignment = reported ? alignment.stx_dio_mem_align : 4096;
            view_.offset_alignment = reported ? alignment.stx_dio_offset_align : 4096;
            if (!std::has_single_bit(view_.memory_alignment) ||
                !std::has_single_bit(view_.offset_alignment))
                fail(path, "unsupported direct I/O alignment", EOPNOTSUPP);
            if (!reported && view_.bytes) {
                // Local filesystems without DIOALIGN use the engine's 4 KiB read unit.
                std::unique_ptr<void, decltype(&std::free)> probe(std::aligned_alloc(4096, 4096),
                                                                  &std::free);
                if (!probe) throw std::bad_alloc();
                (void)read_direct(0, {static_cast<std::byte*>(probe.get()), 4096});
            }
        }
    } catch (...) {
        ::close(view_.descriptor);
        view_.descriptor = -1;
        throw;
    }
}

ReadOnlyFile::~ReadOnlyFile() {
    if (view_.descriptor >= 0) ::close(view_.descriptor);
}

ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)), view_(std::exchange(other.view_, {})) {}

ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        if (view_.descriptor >= 0) ::close(view_.descriptor);
        path_ = std::move(other.path_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || view_.descriptor < 0)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        fail(path_, "read exceeds file length", ENODATA);
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto n     = ::pread(view_.descriptor, destination.data(), count, position(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            fail(path_, "pread");
        }
        if (!n) fail(path_, "unexpected EOF", ENODATA);
        offset += static_cast<std::uint64_t>(n);
        destination = destination.subspan(static_cast<std::size_t>(n));
    }
}

void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct ||
        !std::has_single_bit(file.memory_alignment) ||
        !std::has_single_bit(file.offset_alignment) || offset % file.offset_alignment ||
        destination.size() % file.offset_alignment ||
        (!destination.empty() &&
         (!destination.data() ||
          reinterpret_cast<std::uintptr_t>(destination.data()) % file.memory_alignment)) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()))
        throw std::invalid_argument("invalid direct read range or alignment");
    (void)position(offset);
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    if (destination.empty()) return 0;
    ssize_t n;
    do {
        n = ::pread(view_.descriptor, destination.data(), destination.size(), position(offset));
    } while (n < 0 && errno == EINTR);
    if (n < 0) fail(path_, "direct pread");
    return static_cast<std::size_t>(n);
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
    (void)position(offset - prefix);
    return {offset - prefix, count, prefix, required};
}

} // namespace ninfer::io
