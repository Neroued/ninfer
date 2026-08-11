#include "core/read_only_file.h"

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace ninfer {
namespace {

void close_handle(HANDLE& handle) noexcept {
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
        ::CloseHandle(handle);
        handle = nullptr;
    }
}

} // namespace

struct ReadOnlyFile::Impl {
    HANDLE mapping_file   = INVALID_HANDLE_VALUE;
    HANDLE direct_file    = INVALID_HANDLE_VALUE;
    HANDLE mapping        = nullptr;
    const std::byte* data = nullptr;
    std::size_t size      = 0;

    explicit Impl(const std::filesystem::path& path) {
        mapping_file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (mapping_file == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFileW " + path.string());
        }

        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(mapping_file, &file_size)) {
            const auto error = ::GetLastError();
            close_handle(mapping_file);
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "GetFileSizeEx " + path.string());
        }
        if (file_size.QuadPart < 0 ||
            static_cast<std::uint64_t>(file_size.QuadPart) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            close_handle(mapping_file);
            throw std::overflow_error("file size does not fit the process address space");
        }

        size = static_cast<std::size_t>(file_size.QuadPart);
        if (size != 0) {
            mapping = ::CreateFileMappingW(mapping_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping == nullptr) {
                const auto error = ::GetLastError();
                close_handle(mapping_file);
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "CreateFileMappingW " + path.string());
            }
            const void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
            if (view == nullptr) {
                const auto error = ::GetLastError();
                close_handle(mapping);
                close_handle(mapping_file);
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "MapViewOfFile " + path.string());
            }
            data = static_cast<const std::byte*>(view);
        }

        direct_file =
            ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED |
                              FILE_FLAG_SEQUENTIAL_SCAN,
                          nullptr);
        if (direct_file == INVALID_HANDLE_VALUE) {
            const auto error = ::GetLastError();
            if (data != nullptr) { ::UnmapViewOfFile(data); }
            data = nullptr;
            close_handle(mapping);
            close_handle(mapping_file);
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "CreateFileW direct " + path.string());
        }
    }

    ~Impl() {
        if (data != nullptr) { ::UnmapViewOfFile(data); }
        close_handle(mapping);
        close_handle(direct_file);
        close_handle(mapping_file);
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
    constexpr auto max_file_offset =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (offset > max_file_offset || destination.size() > max_file_offset - offset) {
        throw std::overflow_error("direct file read exceeds platform I/O limits");
    }

    std::size_t total = 0;
    while (total < destination.size()) {
        constexpr std::size_t max_read = 1ULL << 30;
        const auto amount = static_cast<DWORD>(std::min(max_read, destination.size() - total));
        const std::uint64_t absolute = offset + total;
        OVERLAPPED operation{};
        operation.Offset     = static_cast<DWORD>(absolute & 0xffffffffULL);
        operation.OffsetHigh = static_cast<DWORD>(absolute >> 32U);

        DWORD bytes = 0;
        const BOOL started =
            ::ReadFile(impl_->direct_file, destination.data() + total, amount, &bytes, &operation);
        if (!started) {
            const auto error = ::GetLastError();
            if (error == ERROR_HANDLE_EOF) { break; }
            if (error != ERROR_IO_PENDING ||
                !::GetOverlappedResult(impl_->direct_file, &operation, &bytes, TRUE)) {
                const auto final_error = error == ERROR_IO_PENDING ? ::GetLastError() : error;
                throw std::system_error(static_cast<int>(final_error), std::system_category(),
                                        "direct file read");
            }
        }
        total += bytes;
        if (bytes != amount) { break; }
    }
    return total;
}

} // namespace ninfer
