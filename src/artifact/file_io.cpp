#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <limits>
#include <utility>

#if defined(_WIN32)
#include <system_error>

#include <windows.h>
#else
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#if defined(_WIN32)

[[noreturn]] void fail_win32(const std::filesystem::path& path, const char* operation) {
    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                            path.string() + ": " + operation);
}

OVERLAPPED overlapped_offset(std::uint64_t offset) {
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    return overlapped;
}

// A single ReadFile call is capped by its DWORD byte-count parameter.
constexpr std::size_t kMaxSingleRead = std::numeric_limits<DWORD>::max();

#else // POSIX

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

// pread takes a size_t byte count and returns ssize_t.
constexpr std::size_t kMaxSingleRead = std::numeric_limits<ssize_t>::max();

#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
#if defined(_WIN32)
    // FILE_FLAG_NO_BUFFERING on the direct handle makes ReadFile bypass the system cache,
    // matching the POSIX O_DIRECT semantics. The 4096-byte alignment contract enforced in
    // read_direct() satisfies the flag's sector-alignment requirements.
    const HANDLE file = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { fail_win32(path_, "open"); }

    LARGE_INTEGER file_size{};
    if (::GetFileSizeEx(file, &file_size) == 0) {
        const auto error = ::GetLastError();
        ::CloseHandle(file);
        ::SetLastError(error);
        fail_win32(path_, "fstat");
    }
    if (file_size.QuadPart < 0) {
        ::CloseHandle(file);
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_    = static_cast<std::uint64_t>(file_size.QuadPart);
    file_     = file;
#else
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
#endif
}

InputFile::~InputFile() {
#if defined(_WIN32)
    if (direct_file_ != nullptr) { ::CloseHandle(direct_file_); }
    if (file_ != nullptr) { ::CloseHandle(file_); }
#else
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
#if defined(_WIN32)
        OVERLAPPED overlapped = overlapped_offset(offset);
        DWORD read            = 0;
        if (::ReadFile(file_, destination.data(), static_cast<DWORD>(count), &read,
                       &overlapped) == 0) {
            fail_win32(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
#else
        const auto read = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
#endif
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > kMaxSingleRead) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
#if defined(_WIN32)
    if (direct_file_ == nullptr) {
        // FILE_FLAG_NO_BUFFERING bypasses the system cache like O_DIRECT; ReadFile with an
        // OVERLAPPED offset provides the positional read semantics of pread.
        const HANDLE direct = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                            OPEN_EXISTING,
                                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
                                            nullptr);
        if (direct == INVALID_HANDLE_VALUE) { fail_win32(path_, "open direct"); }
        direct_file_ = direct;
    }
    OVERLAPPED overlapped = overlapped_offset(offset);
    DWORD read            = 0;
    if (::ReadFile(direct_file_, destination.data(), static_cast<DWORD>(destination.size()), &read,
                   &overlapped) == 0) {
        fail_win32(path_, "direct pread");
    }
    return static_cast<std::size_t>(read);
#else
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
#endif
}

} // namespace ninfer::artifact
