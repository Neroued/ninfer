#include "qus/core/weight_store_parser.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t kModuleRecordSize = 64;
constexpr std::uint64_t kTensorEntrySize  = 128;
constexpr std::uint64_t kHeaderSize       = 4096;

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open fixture"); }
    const std::vector<char> chars{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    std::vector<std::byte> bytes(chars.size());
    std::memcpy(bytes.data(), chars.data(), chars.size());
    return bytes;
}

std::uint32_t read_u32(const std::vector<std::byte>& bytes, std::uint64_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

std::uint64_t read_u64(const std::vector<std::byte>& bytes, std::uint64_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void write_u16(std::vector<std::byte>& bytes, std::uint64_t offset, std::uint16_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void write_u32(std::vector<std::byte>& bytes, std::uint64_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void write_u64(std::vector<std::byte>& bytes, std::uint64_t offset, std::uint64_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "qus_q5090_parser_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(QUS_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command =
        "python3 \"" + script.string() + "\" --out \"" + path.string() + "\"";
    const int rc = std::system(command.c_str());
    if (rc != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

const qus::ParsedQ5090Tensor& find_tensor(const qus::ParsedQ5090File& parsed,
                                          std::string_view name) {
    for (const qus::ParsedQ5090Tensor& tensor : parsed.tensors) {
        if (tensor.name == name) { return tensor; }
    }
    throw std::runtime_error("tensor not found in parsed fixture");
}

template <typename Mutate>
int expect_parse_throws(const std::vector<std::byte>& valid, std::string_view label,
                        Mutate mutate) {
    std::vector<std::byte> bytes = valid;
    mutate(bytes);
    try {
        (void)qus::parse_q5090_file(bytes, qus::Q5090Expectations{});
    } catch (const std::exception&) { return 0; }
    std::cerr << label << " did not throw\n";
    return 1;
}

int check_valid_parse(const std::vector<std::byte>& bytes) {
    qus::Q5090Expectations expected;
    expected.layer_count             = 64;
    expected.hidden_size             = 5120;
    expected.intermediate_size       = 17408;
    expected.vocab_size              = 248320;
    expected.num_attention_heads     = 24;
    expected.num_key_value_heads     = 4;
    expected.head_dim                = 256;
    expected.gdn_key_heads           = 16;
    expected.gdn_value_heads         = 48;
    expected.gdn_key_head_dim        = 128;
    expected.gdn_value_head_dim      = 128;
    expected.gdn_conv_width          = 4;
    expected.full_attention_interval = 4;
    expected.max_position_embeddings = 262144;

    const qus::ParsedQ5090File parsed = qus::parse_q5090_file(bytes, expected);
    int failures                      = 0;
    failures += parsed.header.tensor_count == 10 ? 0 : fail("tensor_count mismatch");
    failures += parsed.header.module_count == 3 ? 0 : fail("module_count mismatch");
    failures += parsed.header.layer_count == 64 ? 0 : fail("layer_count mismatch");
    failures += parsed.modules.size() == 3 ? 0 : fail("parsed module size mismatch");
    failures += parsed.tensors.size() == 10 ? 0 : fail("parsed tensor size mismatch");

    failures += parsed.modules[0].module_kind == qus::ModuleKind::TextCore
                    ? 0
                    : fail("TEXT module mismatch");
    failures +=
        parsed.modules[0].tensor_index_begin == 0 && parsed.modules[0].tensor_index_count == 6
            ? 0
            : fail("TEXT module range mismatch");
    failures += parsed.modules[1].module_kind == qus::ModuleKind::MtpDraft
                    ? 0
                    : fail("MTP module mismatch");
    failures += parsed.modules[2].module_kind == qus::ModuleKind::VisionEncoder
                    ? 0
                    : fail("VISION module mismatch");
    failures += parsed.modules[2].load_policy == qus::LoadPolicy::LazyGpu
                    ? 0
                    : fail("VISION load policy mismatch");

    const auto& embed = find_tensor(parsed, "model.language_model.embed_tokens.weight");
    failures += embed.qtype == qus::QType::Q6G64_F16S ? 0 : fail("embed qtype mismatch");
    failures += embed.layout == qus::QuantLayout::RowGroupedG64 ? 0 : fail("embed layout mismatch");
    failures += embed.module_kind == qus::ModuleKind::TextCore ? 0 : fail("embed module mismatch");
    failures +=
        embed.shape == std::array<std::uint32_t, 4>{3, 5, 1, 1} ? 0 : fail("embed shape mismatch");
    failures += embed.padded_shape == std::array<std::uint32_t, 4>{3, 64, 1, 1}
                    ? 0
                    : fail("embed padded mismatch");
    failures += embed.group_size == 64 ? 0 : fail("embed group mismatch");
    failures += embed.scale_dtype == qus::ScaleDType::FP16 ? 0 : fail("embed scale dtype mismatch");
    failures += embed.payload_bytes == 150 ? 0 : fail("embed payload bytes mismatch");
    failures += embed.name_hash == qus::q5090_fnv1a64(embed.name) ? 0 : fail("embed hash mismatch");

    const auto& mtp = find_tensor(parsed, "mtp.fc.weight");
    failures += mtp.qtype == qus::QType::W8G128_F16S ? 0 : fail("mtp qtype mismatch");
    failures += mtp.layout == qus::QuantLayout::TileN64K128 ? 0 : fail("mtp layout mismatch");
    failures += mtp.module_kind == qus::ModuleKind::MtpDraft ? 0 : fail("mtp module mismatch");
    failures += mtp.padded_shape == std::array<std::uint32_t, 4>{64, 128, 1, 1}
                    ? 0
                    : fail("mtp padded mismatch");
    failures += mtp.payload_bytes == 8320 ? 0 : fail("mtp payload bytes mismatch");

    const auto& fp32 = find_tensor(parsed, "model.language_model.layers.0.linear_attn.A_log");
    failures += fp32.qtype == qus::QType::FP32_CTRL ? 0 : fail("fp32 qtype mismatch");
    failures += fp32.layout == qus::QuantLayout::Contiguous ? 0 : fail("fp32 layout mismatch");
    failures += fp32.scale_dtype == qus::ScaleDType::None ? 0 : fail("fp32 scale dtype mismatch");
    failures += fp32.payload_bytes == 12 ? 0 : fail("fp32 payload bytes mismatch");

    const auto& vision = find_tensor(parsed, "model.visual.patch_embed.proj.weight");
    failures +=
        vision.module_kind == qus::ModuleKind::VisionEncoder ? 0 : fail("vision module mismatch");
    failures += vision.source_kind == static_cast<std::uint32_t>(qus::SourceKind::VisPatchEmbed)
                    ? 0
                    : fail("vision source kind mismatch");
    return failures;
}

} // namespace

int main() {
    int failures                             = 0;
    const std::filesystem::path fixture_path = make_fixture();
    const std::vector<std::byte> valid       = read_file(fixture_path);
    failures += check_valid_parse(valid);

    const std::uint64_t module_offset       = read_u64(valid, 48);
    const std::uint64_t tensor_offset       = read_u64(valid, 64);
    const std::uint64_t string_offset       = read_u64(valid, 80);
    const std::uint64_t string_bytes        = read_u64(valid, 88);
    const std::uint64_t payload_base        = read_u64(valid, 96);
    const std::uint64_t first_entry         = tensor_offset;
    const std::uint64_t second_entry        = tensor_offset + kTensorEntrySize;
    const std::uint64_t first_payload       = read_u64(valid, first_entry + 64);
    const std::uint64_t first_payload_bytes = read_u64(valid, first_entry + 72);
    const std::uint64_t second_module       = module_offset + kModuleRecordSize;

    failures += expect_parse_throws(valid, "bad magic", [](auto& b) { b[0] = std::byte{0x58}; });
    failures += expect_parse_throws(valid, "bad version", [](auto& b) { write_u32(b, 16, 2); });
    failures +=
        expect_parse_throws(valid, "bad endian", [](auto& b) { write_u32(b, 20, 0x04030201U); });
    failures +=
        expect_parse_throws(valid, "bad header size", [](auto& b) { write_u32(b, 24, 2048); });
    failures +=
        expect_parse_throws(valid, "unknown header flags", [](auto& b) { write_u32(b, 40, 0x10); });
    failures +=
        expect_parse_throws(valid, "module flags mismatch", [](auto& b) { write_u32(b, 40, 0x1); });
    failures += expect_parse_throws(valid, "header reserved", [](auto& b) { write_u32(b, 44, 1); });
    failures += expect_parse_throws(valid, "tensor index non-adjacent", [module_offset](auto& b) {
        write_u64(b, 64, module_offset + kModuleRecordSize);
    });
    failures += expect_parse_throws(valid, "payload bytes mismatch",
                                    [](auto& b) { write_u64(b, 104, read_u64(b, 104) - 1); });
    failures += expect_parse_throws(
        valid, "nonzero metadata padding",
        [string_offset, string_bytes](auto& b) { b[string_offset + string_bytes] = std::byte{1}; });
    failures += expect_parse_throws(valid, "bad module kind",
                                    [module_offset](auto& b) { write_u32(b, module_offset, 99); });
    failures += expect_parse_throws(valid, "bad module span", [module_offset](auto& b) {
        write_u64(b, module_offset + 24, read_u64(b, module_offset + 24) + 256);
    });
    failures +=
        expect_parse_throws(valid, "module span overlap", [module_offset, second_module](auto& b) {
            write_u64(b, second_module + 24, read_u64(b, module_offset + 24));
        });
    failures += expect_parse_throws(valid, "bad qtype",
                                    [first_entry](auto& b) { write_u16(b, first_entry + 16, 99); });
    failures += expect_parse_throws(valid, "bad layout",
                                    [first_entry](auto& b) { write_u16(b, first_entry + 18, 99); });
    failures += expect_parse_throws(valid, "bad ndim",
                                    [first_entry](auto& b) { write_u16(b, first_entry + 22, 5); });
    failures += expect_parse_throws(valid, "bad name hash",
                                    [first_entry](auto& b) { write_u64(b, first_entry + 8, 123); });
    failures += expect_parse_throws(valid, "duplicate name", [first_entry, second_entry](auto& b) {
        write_u32(b, second_entry, read_u32(b, first_entry));
        write_u32(b, second_entry + 4, read_u32(b, first_entry + 4));
        write_u64(b, second_entry + 8, read_u64(b, first_entry + 8));
    });
    failures +=
        expect_parse_throws(valid, "duplicate source", [first_entry, second_entry](auto& b) {
            write_u32(b, second_entry + 80, read_u32(b, first_entry + 80));
            write_u32(b, second_entry + 84, read_u32(b, first_entry + 84));
        });
    failures +=
        expect_parse_throws(valid, "bad string terminator", [first_entry, string_offset](auto& b) {
            const std::uint32_t name_len = read_u32(b, first_entry + 4);
            b[string_offset + name_len]  = std::byte{'x'};
        });
    failures += expect_parse_throws(valid, "unaligned payload", [first_entry](auto& b) {
        write_u64(b, first_entry + 64, read_u64(b, first_entry + 64) + 1);
    });
    failures +=
        expect_parse_throws(valid, "payload before region", [first_entry, payload_base](auto& b) {
            write_u64(b, first_entry + 64, payload_base - 256);
        });
    failures +=
        expect_parse_throws(valid, "payload overlap", [second_entry, first_payload](auto& b) {
            write_u64(b, second_entry + 64, first_payload);
        });
    failures += expect_parse_throws(valid, "nonzero payload padding",
                                    [first_payload, first_payload_bytes](auto& b) {
                                        b[first_payload + first_payload_bytes] = std::byte{1};
                                    });
    failures += expect_parse_throws(
        valid, "crc mismatch", [first_payload](auto& b) { b[first_payload] ^= std::byte{0x1}; });
    failures += expect_parse_throws(valid, "shape payload mismatch", [first_entry](auto& b) {
        write_u32(b, first_entry + 40, 128);
    });
    try {
        qus::Q5090Expectations expected;
        expected.hidden_size = 1;
        (void)qus::parse_q5090_file(valid, expected);
        failures += fail("dims mismatch did not throw");
    } catch (const std::exception&) {}

    return failures == 0 ? 0 : fail("q5090 parser test failed");
}
