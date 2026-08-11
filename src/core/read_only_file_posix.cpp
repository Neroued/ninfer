#include "core/read_only_file.h"

#include <cerrno>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer {

struct ReadOnlyFile::Impl {
    int fd                = -1;
    const std::byte* data = nullptr;
    std::size_t size      = 0;

    explicit Impl(const std::filesystem::path& path) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status{};
        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            fd = -1;
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            fd = -1;
            throw std::overflow_error("file size does not fit the process address space");
        }

        size = static_cast<std::size_t>(status.st_size);
        if (size != 0) {
            void* mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                fd = -1;
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
            data = static_cast<const std::byte*>(mapping);
        }
    }

    ~Impl() {
        if (data != nullptr) { ::munmap(const_cast<std::byte*>(data), size); }
        if (fd >= 0) { ::close(fd); }
    }
};

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}

ReadOnlyFile::~ReadOnlyFile()                                  = default;
ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&&) noexcept            = default;
ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&&) noexcept = default;

std::span<const std::byte> ReadOnlyFile::mapped_bytes() const noexcept {
    return {impl_->data, impl_->size};
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw std::overflow_error("direct file read exceeds platform I/O limits");
    }

    ssize_t bytes = -1;
    do {
        bytes =
            ::pread(impl_->fd, destination.data(), destination.size(), static_cast<off_t>(offset));
    } while (bytes < 0 && errno == EINTR);
    if (bytes < 0) { throw std::system_error(errno, std::generic_category(), "direct file read"); }
    return static_cast<std::size_t>(bytes);
}

} // namespace ninfer
