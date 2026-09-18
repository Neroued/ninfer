#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninfer::artifact {

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;

private:
    std::filesystem::path path_;
#if defined(_WIN32)
    void* file_            = nullptr;
    mutable void* direct_file_ = nullptr;
#else
    int fd_                = -1;
    mutable int direct_fd_ = -1;
#endif
    std::uint64_t bytes_   = 0;
};

} // namespace ninfer::artifact
