#pragma once

#include "artifact/schema.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::artifact_fixture {

using Json = nlohmann::json;

inline void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Error = ninfer::artifact::ArtifactError, class Fn>
void rejects(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const Error&) { return; }
    throw std::runtime_error(message);
}

inline void put_word(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value,
                     unsigned size) {
    for (unsigned i = 0; i < size; ++i) { bytes[offset + i] = std::byte((value >> (8 * i)) & 255); }
}

// Deliberately small independent byte fixture, including non-aligned file boundaries.
// This helper writes test files only; production writer interoperability is checked separately.
struct Fixture {
    std::filesystem::path directory;
    std::filesystem::path entry;
    Json root;
    std::vector<std::byte> payload;

    Fixture() : payload(1344) {
        auto pattern = (std::filesystem::temp_directory_path() / "ninfer-artifact-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const char* path = ::mkdtemp(buffer.data());
        if (!path) { throw std::runtime_error("cannot create fixture directory"); }
        directory = path;
        entry     = directory / "model.ninfer";
        root      = {
            {"components",
                  {{"text", {{"config", Json::object()}, {"resources", {{"tokenizer.json", "asset"}}}}},
                   {"vision", {{"config", {{"model_type", "unused"}}}, {"target", "text"}}}}},
            {"objects", Json::array({{{"id", "asset"},
                                           {"kind", "resource"},
                                           {"encoding", "raw_bytes_v1"},
                                           {"offset", 0},
                                           {"bytes", 5}},
                                          {{"id", "q5"},
                                           {"kind", "tensor"},
                                           {"shape", {2, 130}},
                                           {"format", "q5_g64_fp16"},
                                           {"layout", "row_split_k128_v1"},
                                           {"offset", 256},
                                           {"bytes", 528}},
                                          {{"id", "divisors"},
                                           {"kind", "tensor"},
                                           {"shape", {2}},
                                           {"format", "fp32"},
                                           {"layout", "contiguous_le_v1"},
                                           {"offset", 1024},
                                           {"bytes", 8}},
                                          {{"id", "unused"},
                                           {"kind", "tensor"},
                                           {"shape", {64}},
                                           {"format", "future_format"},
                                           {"layout", "future_layout"},
                                           {"offset", 1280},
                                           {"bytes", 64}}})},
            {"bindings",
                  {{"matrix", {{"object", "q5"}}},
                   {"row", {{"parts", Json::array({{{"object", "q5"}, {"range", {130, 260}}}})}}},
                   {"values", {{"object", "divisors"}}}}},
            {"uses", Json::array()},
            {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 500}},
                                        {{"path", "weights-second.bin"}, {"payload_bytes", 780}},
                                        {{"path", "optional.bin"}, {"payload_bytes", 64}}})},
            {"metadata", {{"name", "unregistered-training-instance"}}}};
        for (int i = 0; i < 2; ++i) {
            root["uses"].push_back({{"parameter", "row"},
                                    {"input", i ? "context" : "query"},
                                    {"activation_policy", i ? "AllowA8" : "AllowA4"},
                                    {"auxiliaries",
                                     {{"input_divisor",
                                       {{"parts", Json::array({{{"object", "divisors"},
                                                                {"range", {i, i + 1}}}})}}}}}});
        }
        const std::string resource = "hello";
        for (std::size_t i = 0; i < resource.size(); ++i) { payload[i] = std::byte(resource[i]); }
        payload[256]            = std::byte{0x31};
        payload[256 + 256]      = std::byte{2};
        payload[256 + 128]      = std::byte{0x52};
        const std::array scales = {0x3800, 0x3c00, 0x3c00, 0, 0x4000, 0x3c00, 0x3c00, 0};
        for (std::size_t i = 0; i < scales.size(); ++i) {
            put_word(payload, 768 + 2 * i, scales[i], 2);
        }
        put_word(payload, 1024, 0x40000000, 4);
        put_word(payload, 1028, 0x40400000, 4);
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    Fixture(const Fixture&)            = delete;
    Fixture& operator=(const Fixture&) = delete;

    void write(bool omit_optional = false, std::string text = {}) const {
        if (text.empty()) { text = root.dump(); }
        std::size_t logical = 0;
        for (std::size_t i = 0; i < root.at("files").size(); ++i) {
            const auto& record = root.at("files")[i];
            const auto count   = record.at("payload_bytes").get<std::size_t>();
            if (omit_optional && i == root.at("files").size() - 1) { break; }
            std::array<std::byte, 32> header{};
            const std::array<unsigned char, 8> magic =
                i == 0 ? std::array<unsigned char, 8>{'N', 'I', 'N', 'F', 'E', 'R', 0, 3}
                       : std::array<unsigned char, 8>{'N', 'I', 'N', 'P', 'R', 'T', 0, 3};
            for (std::size_t j = 0; j < magic.size(); ++j) { header[j] = std::byte(magic[j]); }
            put_word(header, 8, i == 0 ? text.size() : i, 8);
            header[16]      = std::byte{0x71};
            const auto path = i == 0 ? entry : directory / record.at("path").get<std::string>();
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.write(reinterpret_cast<const char*>(header.data()), header.size());
            if (i == 0) { file.write(text.data(), static_cast<std::streamsize>(text.size())); }
            const auto position = 32 + (i == 0 ? text.size() : 0);
            const auto aligned  = (position + 4095) / 4096 * 4096;
            const std::string padding(aligned - position, '\0');
            file.write(padding.data(), static_cast<std::streamsize>(padding.size()));
            require(logical + count <= payload.size(), "fixture payload is incomplete");
            file.write(reinterpret_cast<const char*>(payload.data() + logical), count);
            logical += count;
        }
    }
};

} // namespace ninfer::test::artifact_fixture
