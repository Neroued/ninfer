#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::artifact {

inline constexpr std::array<std::byte, 8> kEntryMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
inline constexpr std::array<std::byte, 8> kPartMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'P'},
    std::byte{'R'}, std::byte{'T'}, std::byte{0},   std::byte{3},
};
inline constexpr std::size_t kHeaderBytes        = 32;
inline constexpr std::uint64_t kPayloadAlignment = 4096;
using ArtifactId                                 = std::array<std::byte, 16>;

inline std::uint64_t read_u64_le(const std::byte* bytes) noexcept {
    std::uint64_t out = 0;
    for (unsigned i = 0; i < 8; ++i) {
        out |= std::uint64_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return out;
}

inline std::uint32_t read_u32_le(const std::byte* bytes) noexcept {
    std::uint32_t out = 0;
    for (unsigned i = 0; i < 4; ++i) {
        out |= std::uint32_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return out;
}

} // namespace ninfer::artifact
