#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#if defined(_WIN32)

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " +
                         std::system_category().message(static_cast<int>(::GetLastError())));
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

#if defined(_WIN32)

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fd_ == INVALID_HANDLE_VALUE) { fail(path_, "open"); }

    LARGE_INTEGER file_size{};
    if (::GetFileSizeEx(fd_, &file_size) == 0) {
        const auto error = ::GetLastError();
        ::CloseHandle(fd_);
        fd_ = INVALID_HANDLE_VALUE;
        ::SetLastError(error);
        fail(path_, "GetFileSizeEx");
    }
    if (file_size.QuadPart < 0) {
        ::CloseHandle(fd_);
        fd_ = INVALID_HANDLE_VALUE;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(file_size.QuadPart);
}

InputFile::~InputFile() {
    if (direct_fd_ != INVALID_HANDLE_VALUE) { ::CloseHandle(direct_fd_); }
    if (fd_ != INVALID_HANDLE_VALUE) { ::CloseHandle(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        OVERLAPPED overlapped{};
        overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
        overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
        DWORD bytes_read = 0;
        if (::ReadFile(fd_, destination.data(), static_cast<DWORD>(count), &bytes_read,
                       &overlapped) == 0) {
            fail(path_, "ReadFile");
        }
        if (!bytes_read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(bytes_read);
        destination = destination.subspan(static_cast<std::size_t>(bytes_read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<DWORD>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ == INVALID_HANDLE_VALUE) {
        // FILE_FLAG_NO_BUFFERING bypasses the system cache, matching POSIX O_DIRECT.
        // The 4096-byte alignment contract enforced above satisfies the flag's
        // sector-alignment requirements.
        direct_fd_ = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
                                    nullptr);
        if (direct_fd_ == INVALID_HANDLE_VALUE) { fail(path_, "open direct"); }
    }
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD bytes_read = 0;
    if (::ReadFile(direct_fd_, destination.data(), static_cast<DWORD>(destination.size()),
                   &bytes_read, &overlapped) == 0) {
        fail(path_, "direct ReadFile");
    }
    return static_cast<std::size_t>(bytes_read);
}

#else

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

#endif

} // namespace ninfer::artifact
