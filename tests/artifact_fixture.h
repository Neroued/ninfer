#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::test::artifact_fixture {

using Json = nlohmann::json;

inline constexpr std::array<std::uint8_t, 8> kMagic = {'N', 'I', 'N', 'F', 'E', 'R', 0, 1};

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline void write_u64_le(std::byte* output, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) { output[i] = std::byte((value >> (i * 8)) & 0xff); }
}

struct TemporaryArtifact {
    std::filesystem::path path;

    ~TemporaryArtifact() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

inline TemporaryArtifact write_fixture(const Json& directory, std::string_view suffix) {
    const std::string json         = directory.dump();
    const auto payload_offset      = align_up(16 + json.size(), 4096);
    const auto nonnegative_integer = [](const Json& value) {
        return value.is_number_unsigned() ||
               (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    };
    std::uint64_t payload_bytes = 0;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            payload_bytes = std::max(payload_bytes, object.at("offset").get<std::uint64_t>() +
                                                        object.at("bytes").get<std::uint64_t>());
        }
    }

    std::vector<std::byte> file(payload_offset + payload_bytes, std::byte{0});
    for (std::size_t i = 0; i < kMagic.size(); ++i) { file[i] = std::byte{kMagic[i]}; }
    write_u64_le(file.data() + 8, json.size());
    std::memcpy(file.data() + 16, json.data(), json.size());

    std::uint8_t marker = 1;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            const auto offset = object.at("offset").get<std::uint64_t>();
            const auto bytes  = object.at("bytes").get<std::uint64_t>();
            std::fill_n(file.data() + payload_offset + offset, bytes, std::byte{marker++});
        }
    }

    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_" + std::string(suffix) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    if (!output) { throw std::runtime_error("failed to write artifact fixture"); }
    return {std::move(path)};
}

} // namespace ninfer::test::artifact_fixture
