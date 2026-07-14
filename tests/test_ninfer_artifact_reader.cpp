#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;

constexpr std::array<std::uint8_t, 8> kMagic = {'N', 'I', 'N', 'F', 'E', 'R', 0, 1};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void write_u64_le(std::byte* output, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        output[i] = std::byte((value >> (i * 8)) & 0xff);
    }
}

struct TemporaryArtifact {
    std::filesystem::path path;

    ~TemporaryArtifact() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

TemporaryArtifact write_fixture(const Json& directory, std::string_view suffix) {
    const std::string json = directory.dump();
    const auto payload_offset = align_up(16 + json.size(), 4096);
    const auto nonnegative_integer = [](const Json& value) {
        return value.is_number_unsigned() ||
               (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    };
    std::uint64_t payload_bytes = 0;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) &&
            nonnegative_integer(object.at("bytes"))) {
            payload_bytes = std::max(payload_bytes,
                                     object.at("offset").get<std::uint64_t>() +
                                         object.at("bytes").get<std::uint64_t>());
        }
    }

    std::vector<std::byte> file(payload_offset + payload_bytes, std::byte{0});
    for (std::size_t i = 0; i < kMagic.size(); ++i) {
        file[i] = std::byte{kMagic[i]};
    }
    write_u64_le(file.data() + 8, json.size());
    std::memcpy(file.data() + 16, json.data(), json.size());

    std::uint8_t marker = 1;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) &&
            nonnegative_integer(object.at("bytes"))) {
            const auto offset = object.at("offset").get<std::uint64_t>();
            const auto bytes  = object.at("bytes").get<std::uint64_t>();
            std::fill_n(file.data() + payload_offset + offset, bytes, std::byte{marker++});
        }
    }

    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_reader_" + std::string(suffix) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    if (!output) {
        throw std::runtime_error("failed to write artifact fixture");
    }
    return {std::move(path)};
}

Json normative_directory() {
    return {
        {"model_id", "fixture-model"},
        {"objects",
         Json::array({
             {{"name", "resource"},
              {"kind", "resource"},
              {"encoding", "raw-bytes-v1"},
              {"offset", 0},
              {"bytes", 3}},
             {{"name", "bf16"},
              {"kind", "tensor"},
              {"shape", {2, 3}},
              {"format", "BF16"},
              {"layout", "contiguous-le-v1"},
              {"offset", 256},
              {"bytes", 12}},
             {{"name", "fp32_scalar"},
              {"kind", "tensor"},
              {"shape", Json::array()},
              {"format", "FP32"},
              {"layout", "contiguous-le-v1"},
              {"offset", 512},
              {"bytes", 4}},
             {{"name", "i32"},
              {"kind", "tensor"},
              {"shape", {2}},
              {"format", "I32"},
              {"layout", "contiguous-le-v1"},
              {"offset", 768},
              {"bytes", 8}},
             {{"name", "q4"},
              {"kind", "tensor"},
              {"shape", {1, 1}},
              {"format", "Q4G64_F16S"},
              {"layout", "row-split-k128-v1"},
              {"offset", 1024},
              {"bytes", 260}},
             {{"name", "q5"},
              {"kind", "tensor"},
              {"shape", {2, 130}},
              {"format", "Q5G64_F16S"},
              {"layout", "row-split-k128-v1"},
              {"offset", 1536},
              {"bytes", 528}},
             {{"name", "q6"},
              {"kind", "tensor"},
              {"shape", {1, 64}},
              {"format", "Q6G64_F16S"},
              {"layout", "row-split-k128-v1"},
              {"offset", 2304},
              {"bytes", 516}},
             {{"name", "w8"},
              {"kind", "tensor"},
              {"shape", {1, 33}},
              {"format", "W8G32_F16S"},
              {"layout", "row-split-k128-v1"},
              {"offset", 3072},
              {"bytes", 264}},
         })},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) {
        return;
    }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows   = StorageLayout::RowSplitK128V1;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.model_id() != "fixture-model" || reader.objects().size() != 8 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 8> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4", "q5", "q6", "w8",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) ||
            payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5       = std::get_if<TensorDescriptor>(reader.find("q5"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S ||
        q5->layout != StorageLayout::RowSplitK128V1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory = normative_directory();
        directory["extra"] = true;
        auto fixture = write_fixture(directory, "extra_root_member");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "open root schema");
    }
    {
        auto directory = normative_directory();
        directory["objects"][1]["format"] = "F16";
        auto fixture = write_fixture(directory, "unknown_format");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "unknown format");
    }
    {
        auto directory = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
}

const TensorDescriptor& require_tensor(const Reader& reader, std::string_view name) {
    const auto* object = reader.find(name);
    const auto* tensor = object == nullptr ? nullptr : std::get_if<TensorDescriptor>(object);
    if (tensor == nullptr) {
        throw std::runtime_error("missing expected tensor " + std::string(name));
    }
    return *tensor;
}

void test_python_generated_real_artifact_if_present() {
    const auto path = std::filesystem::path(NINFER_SOURCE_DIR) /
                      "out/qwen3_6_27b_rtx5090.ninfer";
    if (!std::filesystem::exists(path)) {
        std::cout << "real artifact not present; fixture acceptance completed\n";
        return;
    }

    Reader reader(path);
    if (reader.model_id() != "qwen3.6-27b" || reader.objects().size() != 1172) {
        throw std::runtime_error("real artifact root signature mismatch");
    }

    std::size_t tensors = 0;
    std::size_t resources = 0;
    std::map<NumericFormat, std::size_t> format_counts;
    std::map<StorageLayout, std::size_t> layout_counts;
    for (const auto& object : reader.objects()) {
        if (const auto* tensor = std::get_if<TensorDescriptor>(&object)) {
            ++tensors;
            ++format_counts[tensor->format];
            ++layout_counts[tensor->layout];
        } else {
            ++resources;
        }
        const auto span = reader.payload(object);
        if (span.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            span.data.size() != ninfer::artifact::object_bytes(object)) {
            throw std::runtime_error("real artifact payload span mismatch");
        }
    }
    if (tensors != 1166 || resources != 6 ||
        format_counts[NumericFormat::BF16] != 582 ||
        format_counts[NumericFormat::FP32] != 96 ||
        format_counts[NumericFormat::I32] != 1 ||
        format_counts[NumericFormat::Q4G64_F16S] != 183 ||
        format_counts[NumericFormat::Q5G64_F16S] != 294 ||
        format_counts[NumericFormat::Q6G64_F16S] != 3 ||
        format_counts[NumericFormat::W8G32_F16S] != 7 ||
        layout_counts[StorageLayout::ContiguousLeV1] != 679 ||
        layout_counts[StorageLayout::RowSplitK128V1] != 487) {
        throw std::runtime_error("real artifact inventory counts differ from Python");
    }

    const auto& embedding = require_tensor(reader, "text/token_embedding");
    const auto& qk = require_tensor(reader, "text/layers/3/attention/query_key");
    const auto& draft_ids = require_tensor(reader, "text/draft_head_token_ids");
    const auto& mtp = require_tensor(reader, "mtp/layer/attention/query_key_gate_value");
    if (embedding.shape != std::vector<std::uint64_t>({248320, 5120}) ||
        embedding.format != NumericFormat::Q6G64_F16S ||
        qk.shape != std::vector<std::uint64_t>({7168, 5120}) ||
        qk.format != NumericFormat::Q4G64_F16S ||
        draft_ids.shape != std::vector<std::uint64_t>({131072}) ||
        draft_ids.format != NumericFormat::I32 ||
        mtp.shape != std::vector<std::uint64_t>({14336, 5120}) ||
        mtp.format != NumericFormat::W8G32_F16S) {
        throw std::runtime_error("real artifact key signatures differ from Python");
    }
    if (ninfer::artifact::object_name(reader.objects().front()) !=
            "frontend/tokenizer.json" ||
        ninfer::artifact::object_name(reader.objects().back()) !=
            "vision/merger/norm/bias") {
        throw std::runtime_error("real artifact object order differs from Python");
    }
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_normative_fixture();
        test_common_validation();
        test_python_generated_real_artifact_if_present();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
