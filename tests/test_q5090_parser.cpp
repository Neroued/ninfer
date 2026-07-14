#include "ninfer/core/weight_store_parser.h"
#include "ninfer/core/weight_store.h"
#include "ninfer/text/tokenizer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t kModuleRecordSize  = 64;
constexpr std::uint64_t kTensorEntrySize   = 128;
constexpr std::uint64_t kSegmentRecordSize = 32;
constexpr std::uint64_t kHeaderSize        = 4096;

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

void write_file(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("failed to create fixture"); }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) { throw std::runtime_error("failed to write fixture"); }
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

std::filesystem::path make_fixture(std::string_view profile = "default") {
    const auto path = std::filesystem::temp_directory_path() /
                      ("ninfer_q5090_parser_fixture_" + std::string(profile) + ".qus");
    const std::filesystem::path script =
        std::filesystem::path(NINFER_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    std::string command = "python3 \"" + script.string() + "\" --out \"" + path.string() + "\"";
    if (profile != "default") { command += " --profile " + std::string(profile); }
    const int rc = std::system(command.c_str());
    if (rc != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

const ninfer::ParsedQ5090Tensor& find_tensor(const ninfer::ParsedQ5090File& parsed,
                                          std::string_view name) {
    for (const ninfer::ParsedQ5090Tensor& tensor : parsed.tensors) {
        if (tensor.name == name) { return tensor; }
    }
    throw std::runtime_error("tensor not found in parsed fixture");
}

const ninfer::ParsedQ5090Segment& find_segment(const ninfer::ParsedQ5090File& parsed,
                                            std::string_view name) {
    for (const ninfer::ParsedQ5090Segment& segment : parsed.segments) {
        if (segment.name == name) { return segment; }
    }
    throw std::runtime_error("segment not found in parsed fixture");
}

template <typename Mutate>
int expect_parse_throws(const std::vector<std::byte>& valid, std::string_view label,
                        Mutate mutate) {
    std::vector<std::byte> bytes = valid;
    mutate(bytes);
    try {
        (void)ninfer::parse_q5090_file(bytes);
    } catch (const std::exception&) { return 0; }
    std::cerr << label << " did not throw\n";
    return 1;
}

int check_valid_parse(const std::vector<std::byte>& bytes) {
    const ninfer::ParsedQ5090File parsed = ninfer::parse_q5090_file(bytes);
    int failures                      = 0;
    failures += parsed.header.tensor_count == 20 ? 0 : fail("tensor_count mismatch");
    failures += parsed.header.module_count == 3 ? 0 : fail("module_count mismatch");
    failures += parsed.header.segment_count == 25 ? 0 : fail("segment_count mismatch");
    failures += parsed.header.fusion_group_count == 3 ? 0 : fail("fusion_group_count mismatch");
    failures += parsed.header.layer_count == 64 ? 0 : fail("layer_count mismatch");
    failures += parsed.modules.size() == 3 ? 0 : fail("parsed module size mismatch");
    failures += parsed.tensors.size() == 20 ? 0 : fail("parsed tensor size mismatch");
    failures += parsed.segments.size() == 25 ? 0 : fail("parsed segment size mismatch");
    failures += parsed.fusion_groups.size() == 3 ? 0 : fail("parsed fusion group size mismatch");
    failures += parsed.header.format_minor == 2 ? 0 : fail("format minor mismatch");
    failures += parsed.tokenizer_records.size() == 3 ? 0 : fail("tokenizer record size mismatch");
    failures += parsed.tokenizer.tokenizer_json.find("\"model\"") != std::string::npos
                    ? 0
                    : fail("embedded tokenizer.json mismatch");
    failures +=
        parsed.tokenizer.merges_txt == "#version: 0.2\n" ? 0 : fail("embedded merges.txt mismatch");
    failures += parsed.tokenizer.generation_config_json == "{\"eos_token_id\":[0]}\n"
                    ? 0
                    : fail("embedded generation_config.json mismatch");
    const ninfer::text::QwenTokenizer tokenizer(parsed.tokenizer);
    failures += tokenizer.encode("a") == std::vector<int>{0}
                    ? 0
                    : fail("embedded artifact tokenizer encode mismatch");
    failures += tokenizer.default_stop_token_ids() == std::vector<int>{0}
                    ? 0
                    : fail("embedded artifact tokenizer stop ids mismatch");

    failures += parsed.modules[0].module_kind == ninfer::ModuleKind::TextCore
                    ? 0
                    : fail("TEXT module mismatch");
    failures +=
        parsed.modules[0].tensor_index_begin == 0 && parsed.modules[0].tensor_index_count == 6
            ? 0
            : fail("TEXT module range mismatch");
    failures += parsed.modules[1].module_kind == ninfer::ModuleKind::MtpDraft
                    ? 0
                    : fail("MTP module mismatch");
    failures += parsed.modules[1].tensor_index_count == 12 ? 0 : fail("MTP module range mismatch");
    failures += parsed.modules[2].module_kind == ninfer::ModuleKind::VisionEncoder
                    ? 0
                    : fail("VISION module mismatch");

    const auto& embed = find_tensor(parsed, "model.language_model.embed_tokens.weight");
    failures += embed.qtype == ninfer::QType::Q6G64_F16S ? 0 : fail("embed qtype mismatch");
    failures += embed.layout == ninfer::QuantLayout::RowSplit ? 0 : fail("embed layout mismatch");
    failures += embed.module_kind == ninfer::ModuleKind::TextCore ? 0 : fail("embed module mismatch");
    failures +=
        embed.shape == std::array<std::uint32_t, 4>{3, 5, 1, 1} ? 0 : fail("embed shape mismatch");
    failures += embed.padded_shape == std::array<std::uint32_t, 4>{3, 128, 1, 1}
                    ? 0
                    : fail("embed padded mismatch");
    failures += embed.group_size == 64 ? 0 : fail("embed group mismatch");
    failures += embed.scale_dtype == ninfer::ScaleDType::FP16 ? 0 : fail("embed scale dtype mismatch");
    failures += embed.payload_bytes == 524 ? 0 : fail("embed payload bytes mismatch");
    failures += embed.nibble_plane_bytes == 192 ? 0 : fail("embed nibble bytes mismatch");
    failures += embed.high_plane_bytes == 96 ? 0 : fail("embed high bytes mismatch");
    failures += embed.scale_plane_bytes == 12 ? 0 : fail("embed scale bytes mismatch");
    failures += embed.segment_count == 1 ? 0 : fail("embed segment count mismatch");
    failures += embed.name_hash == ninfer::q5090_fnv1a64(embed.name) ? 0 : fail("embed hash mismatch");
    const auto& embed_segment = find_segment(parsed, "model.language_model.embed_tokens.weight");
    failures += embed_segment.row_begin == 0 && embed_segment.row_count == 3
                    ? 0
                    : fail("embed segment range mismatch");

    const auto& gateup = find_tensor(parsed, "layers.0.mlp.gateup");
    failures += gateup.segment_count == 2 ? 0 : fail("gateup segment count mismatch");
    failures += gateup.source_kind == static_cast<std::uint32_t>(ninfer::SourceKind::Other)
                    ? 0
                    : fail("gateup source kind mismatch");
    const auto& gate = find_segment(parsed, "layers.0.mlp.gate_proj.weight");
    const auto& up   = find_segment(parsed, "layers.0.mlp.up_proj.weight");
    failures += gate.row_begin == 0 && gate.row_count == 5 ? 0 : fail("gate segment mismatch");
    failures += up.row_begin == 5 && up.row_count == 4 ? 0 : fail("up segment mismatch");

    const auto& fusion = parsed.fusion_groups[0];
    failures += fusion.group_id == 3 ? 0 : fail("fusion group id mismatch");
    failures += fusion.first_block_tensor_index == 1 ? 0 : fail("fusion first block mismatch");
    failures += fusion.block_count == 1 ? 0 : fail("fusion block count mismatch");
    failures += fusion.total_n == 9 ? 0 : fail("fusion total_n mismatch");
    failures += fusion.shared_k == 7 ? 0 : fail("fusion shared_k mismatch");
    const auto& mtp_attn_fusion = parsed.fusion_groups[1];
    failures += mtp_attn_fusion.group_id == 1 ? 0 : fail("mtp attn fusion id mismatch");
    failures += mtp_attn_fusion.first_block_tensor_index == 10
                    ? 0
                    : fail("mtp attn fusion first block mismatch");
    failures += mtp_attn_fusion.block_count == 1 ? 0 : fail("mtp attn fusion block count mismatch");
    failures += mtp_attn_fusion.total_n == 24 ? 0 : fail("mtp attn fusion total_n mismatch");
    failures += mtp_attn_fusion.shared_k == 8 ? 0 : fail("mtp attn fusion shared_k mismatch");
    const auto& mtp_mlp_fusion = parsed.fusion_groups[2];
    failures += mtp_mlp_fusion.group_id == 3 ? 0 : fail("mtp mlp fusion id mismatch");
    failures += mtp_mlp_fusion.first_block_tensor_index == 15
                    ? 0
                    : fail("mtp mlp fusion first block mismatch");
    failures += mtp_mlp_fusion.block_count == 1 ? 0 : fail("mtp mlp fusion block count mismatch");
    failures += mtp_mlp_fusion.total_n == 20 ? 0 : fail("mtp mlp fusion total_n mismatch");
    failures += mtp_mlp_fusion.shared_k == 8 ? 0 : fail("mtp mlp fusion shared_k mismatch");

    const auto& mtp = find_tensor(parsed, "mtp.fc.weight");
    failures += mtp.qtype == ninfer::QType::W8G32_F16S ? 0 : fail("mtp qtype mismatch");
    failures += mtp.layout == ninfer::QuantLayout::RowSplit ? 0 : fail("mtp layout mismatch");
    failures += mtp.module_kind == ninfer::ModuleKind::MtpDraft ? 0 : fail("mtp module mismatch");
    failures +=
        mtp.shape == std::array<std::uint32_t, 4>{8, 16, 1, 1} ? 0 : fail("mtp shape mismatch");
    failures += mtp.padded_shape == std::array<std::uint32_t, 4>{8, 128, 1, 1}
                    ? 0
                    : fail("mtp padded mismatch");
    failures += mtp.group_size == 32 ? 0 : fail("mtp group mismatch");
    failures += mtp.payload_bytes == 1088 ? 0 : fail("mtp payload bytes mismatch");
    failures += mtp.nibble_plane_bytes == 1024 ? 0 : fail("mtp nibble bytes mismatch");
    failures += mtp.high_plane_bytes == 0 ? 0 : fail("mtp high bytes mismatch");
    failures += mtp.scale_plane_bytes == 64 ? 0 : fail("mtp scale bytes mismatch");

    const auto& mtp_attn = find_tensor(parsed, "mtp.layers.0.attn_in.w8");
    failures += mtp_attn.qtype == ninfer::QType::W8G32_F16S ? 0 : fail("mtp attn qtype mismatch");
    failures += mtp_attn.segment_count == 4 ? 0 : fail("mtp attn segment count mismatch");
    failures += mtp_attn.source_kind == static_cast<std::uint32_t>(ninfer::SourceKind::Other)
                    ? 0
                    : fail("mtp attn source kind mismatch");
    const auto& mtp_attn_gate = find_segment(parsed, "mtp.layers.0.self_attn.q_proj.gate");
    failures += mtp_attn_gate.row_begin == 12 && mtp_attn_gate.row_count == 8
                    ? 0
                    : fail("mtp attn gate segment mismatch");

    const auto& mtp_mlp = find_tensor(parsed, "mtp.layers.0.mlp.gateup.w8");
    failures += mtp_mlp.qtype == ninfer::QType::W8G32_F16S ? 0 : fail("mtp mlp qtype mismatch");
    failures += mtp_mlp.segment_count == 2 ? 0 : fail("mtp mlp segment count mismatch");

    const auto& fp32 = find_tensor(parsed, "layers.0.linear_attn.A_log");
    failures += fp32.qtype == ninfer::QType::FP32_CTRL ? 0 : fail("fp32 qtype mismatch");
    failures += fp32.layout == ninfer::QuantLayout::Contiguous ? 0 : fail("fp32 layout mismatch");
    failures += fp32.scale_dtype == ninfer::ScaleDType::None ? 0 : fail("fp32 scale dtype mismatch");
    failures += fp32.payload_bytes == 12 ? 0 : fail("fp32 payload bytes mismatch");

    const auto& vision = find_tensor(parsed, "model.visual.patch_embed.proj.weight");
    failures +=
        vision.module_kind == ninfer::ModuleKind::VisionEncoder ? 0 : fail("vision module mismatch");
    failures += vision.source_kind == static_cast<std::uint32_t>(ninfer::SourceKind::VisPatchEmbed)
                    ? 0
                    : fail("vision source kind mismatch");
    return failures;
}

int check_model_bind_conv1d_parse() {
    const std::filesystem::path fixture_path = make_fixture("model-bind");
    const std::vector<std::byte> bytes       = read_file(fixture_path);
    const ninfer::ParsedQ5090File parsed        = ninfer::parse_q5090_file(bytes);
    const auto& conv = find_tensor(parsed, "layers.0.linear_attn.conv1d.weight");

    int failures = 0;
    failures += conv.qtype == ninfer::QType::BF16_CTRL ? 0 : fail("conv1d qtype mismatch");
    failures += conv.layout == ninfer::QuantLayout::Contiguous ? 0 : fail("conv1d layout mismatch");
    failures += conv.module_kind == ninfer::ModuleKind::TextCore ? 0 : fail("conv1d module mismatch");
    failures += conv.source_kind == static_cast<std::uint32_t>(ninfer::SourceKind::GdnConv1d)
                    ? 0
                    : fail("conv1d source kind mismatch");
    failures += conv.shape == std::array<std::uint32_t, 4>{10240, 4, 1, 1}
                    ? 0
                    : fail("conv1d canonical shape mismatch");
    failures += conv.padded_shape == std::array<std::uint32_t, 4>{10240, 4, 1, 1}
                    ? 0
                    : fail("conv1d padded shape mismatch");
    failures +=
        conv.payload_bytes == 10240ULL * 4ULL * 2ULL ? 0 : fail("conv1d payload bytes mismatch");
    return failures;
}

int check_draft_head_parse() {
    int failures = 0;

    // Positive: a v4.2 file carrying the independent Q4 draft head + I32 id-map module parses,
    // sets LM_HEAD_DRAFT_PRESENT, and both draft blocks report the expected schema.
    const std::filesystem::path good_path = make_fixture("draft-head");
    const std::vector<std::byte> good     = read_file(good_path);
    const ninfer::ParsedQ5090File parsed     = ninfer::parse_q5090_file(good);

    failures += (parsed.header.flags & (1U << 4)) != 0
                    ? 0
                    : fail("draft-head LM_HEAD_DRAFT_PRESENT flag not set");
    failures += parsed.modules.size() == 4 ? 0 : fail("draft module count mismatch");
    failures += parsed.modules[1].module_kind == ninfer::ModuleKind::LmHeadDraft
                    ? 0
                    : fail("draft ModuleRecord kind mismatch");

    const auto& weights = find_tensor(parsed, "lm_head_draft");
    failures += weights.qtype == ninfer::QType::Q4G64_F16S ? 0 : fail("draft weights qtype mismatch");
    failures +=
        weights.layout == ninfer::QuantLayout::RowSplit ? 0 : fail("draft weights layout mismatch");
    failures += weights.module_kind == ninfer::ModuleKind::LmHeadDraft
                    ? 0
                    : fail("draft weights module mismatch");
    failures += weights.source_kind == static_cast<std::uint32_t>(ninfer::SourceKind::LmHeadDraft)
                    ? 0
                    : fail("draft weights source kind mismatch");
    failures += weights.shape == std::array<std::uint32_t, 4>{131072, 5120, 1, 1}
                    ? 0
                    : fail("draft weights fixed v4.2 shape mismatch");

    const auto& idmap = find_tensor(parsed, "lm_head_draft.idmap");
    failures += idmap.qtype == ninfer::QType::I32_CTRL ? 0 : fail("draft idmap qtype mismatch");
    failures +=
        idmap.layout == ninfer::QuantLayout::Contiguous ? 0 : fail("draft idmap layout mismatch");
    failures += idmap.ndim == 1 ? 0 : fail("draft idmap ndim mismatch");
    failures += idmap.source_kind == static_cast<std::uint32_t>(ninfer::SourceKind::LmHeadDraftIdmap)
                    ? 0
                    : fail("draft idmap source kind mismatch");
    failures += idmap.shape[0] == 131072 ? 0 : fail("draft idmap length mismatch");
    failures +=
        idmap.payload_bytes == 131072ULL * 4ULL ? 0 : fail("draft idmap payload bytes mismatch");

    // Negative: clearing LM_HEAD_DRAFT_PRESENT while the draft module remains must be
    // rejected (flag/block coupling).
    failures += expect_parse_throws(good, "draft flag cleared", [](auto& b) {
        write_u32(b, 40, read_u32(b, 40) & ~(1U << 4));
    });

    // Negative: an id-map whose length disagrees with the draft weight row count is
    // valid per-block but must be rejected (guards against OOB token remap).
    const std::vector<std::byte> bad_n = read_file(make_fixture("draft-head-bad-n"));
    failures += expect_parse_throws(bad_n, "draft idmap length mismatch", [](auto&) {});

    return failures;
}

int check_header_first_prepare(const std::filesystem::path& valid_path,
                               const std::vector<std::byte>& valid) {
    int failures = 0;
    ninfer::WeightStore prepared;
    prepared.prepare(valid_path.c_str());
    const ninfer::Q5090LoadPlan& plan = prepared.load_plan();
    failures += plan.modules.size() == 1 ? 0 : fail("CPU-only plan selected wrong module count");
    failures += plan.modules[0].module == ninfer::ModuleKind::TextCore
                    ? 0
                    : fail("CPU-only plan did not select TEXT_CORE");
    failures += plan.tensors.size() == 6 ? 0 : fail("CPU-only plan tensor count mismatch");
    failures += plan.file_read_bytes == read_u64(valid, 96) + plan.modules[0].file_bytes
                    ? 0
                    : fail("CPU-only plan file byte count mismatch");
    failures += prepared.module_arena(ninfer::ModuleKind::TextCore) == nullptr
                    ? 0
                    : fail("CPU-only prepare allocated a device arena");
    const ninfer::Q5090LoadStats& good_stats = prepared.load_stats();
    failures += good_stats.header_read_bytes == kHeaderSize
                    ? 0
                    : fail("CPU-only prepare header byte count mismatch");
    failures += good_stats.total_file_read_bytes == read_u64(valid, 96)
                    ? 0
                    : fail("CPU-only prepare read beyond metadata/tokenizer");

    std::vector<std::byte> bad = valid;
    write_u32(bad, 16, 99);
    const auto bad_path =
        std::filesystem::temp_directory_path() / "ninfer_q5090_header_first_bad_version.qus";
    write_file(bad_path, bad);
    ninfer::WeightStore rejected;
    try {
        rejected.prepare(bad_path.c_str());
        failures += fail("header-first bad version did not throw");
    } catch (const std::runtime_error&) {
        failures += rejected.module_arena(ninfer::ModuleKind::TextCore) == nullptr
                        ? 0
                        : fail("bad version allocated a device arena");
    }
    return failures;
}

} // namespace

int main() {
    int failures                             = 0;
    const std::filesystem::path fixture_path = make_fixture();
    const std::vector<std::byte> valid       = read_file(fixture_path);
    failures += check_valid_parse(valid);
    failures += check_header_first_prepare(fixture_path, valid);
    failures += check_model_bind_conv1d_parse();
    failures += check_draft_head_parse();

    const std::uint64_t module_offset       = read_u64(valid, 48);
    const std::uint64_t tensor_offset       = read_u64(valid, 64);
    const std::uint64_t string_offset       = read_u64(valid, 80);
    const std::uint64_t string_bytes        = read_u64(valid, 88);
    const std::uint64_t payload_base        = read_u64(valid, 96);
    const std::uint64_t segment_offset      = read_u64(valid, 200);
    const std::uint64_t fusion_offset       = read_u64(valid, 216);
    const std::uint64_t tokenizer_index     = read_u64(valid, 248);
    const std::uint64_t tokenizer_data      = read_u64(valid, 264);
    const std::uint64_t first_entry         = tensor_offset;
    const std::uint64_t second_entry        = tensor_offset + kTensorEntrySize;
    const std::uint64_t mtp_entry           = tensor_offset + 6 * kTensorEntrySize;
    const std::uint64_t first_payload       = read_u64(valid, first_entry + 64);
    const std::uint64_t first_payload_bytes = read_u64(valid, first_entry + 72);
    const std::uint64_t second_module       = module_offset + kModuleRecordSize;
    const std::uint64_t gate_segment        = segment_offset + kSegmentRecordSize;
    const std::uint64_t up_segment          = segment_offset + 2 * kSegmentRecordSize;
    const ninfer::ParsedQ5090File parsed_valid = ninfer::parse_q5090_file(valid);
    const auto& mtp_attn                    = find_tensor(parsed_valid, "mtp.layers.0.attn_in.w8");
    const std::uint32_t mtp_attn_n          = mtp_attn.shape[0];
    const std::uint64_t mtp_attn_segments =
        segment_offset + static_cast<std::uint64_t>(mtp_attn.segment_begin) * kSegmentRecordSize;

    failures += expect_parse_throws(valid, "bad magic", [](auto& b) { b[0] = std::byte{0x58}; });
    failures += expect_parse_throws(valid, "bad version", [](auto& b) { write_u32(b, 16, 1); });
    failures +=
        expect_parse_throws(valid, "bad endian", [](auto& b) { write_u32(b, 20, 0x04030201U); });
    failures +=
        expect_parse_throws(valid, "bad header size", [](auto& b) { write_u32(b, 24, 2048); });
    failures +=
        expect_parse_throws(valid, "retired v4.1 format minor", [](auto& b) { write_u32(b, 232, 1); });
    failures +=
        expect_parse_throws(valid, "unknown header flags", [](auto& b) { write_u32(b, 40, 0x20); });
    failures += expect_parse_throws(valid, "zero full-attention interval",
                                    [](auto& b) { write_u32(b, 156, 0); });
    failures +=
        expect_parse_throws(valid, "module flags mismatch", [](auto& b) { write_u32(b, 40, 0x1); });
    failures +=
        expect_parse_throws(valid, "header reserved", [](auto& b) { write_u32(b, 280, 1); });
    failures +=
        expect_parse_throws(valid, "tokenizer record count", [](auto& b) { write_u32(b, 236, 2); });
    failures += expect_parse_throws(valid, "tokenizer reserved", [tokenizer_index](auto& b) {
        write_u32(b, tokenizer_index + 28, 1);
    });
    failures += expect_parse_throws(valid, "tokenizer bad kind", [tokenizer_index](auto& b) {
        write_u32(b, tokenizer_index, 2);
    });
    failures += expect_parse_throws(valid, "tokenizer bad encoding", [tokenizer_index](auto& b) {
        write_u32(b, tokenizer_index + 4, 1);
    });
    failures += expect_parse_throws(valid, "tokenizer crc mismatch", [tokenizer_data](auto& b) {
        b[tokenizer_data] ^= std::byte{1};
    });
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
    failures += expect_parse_throws(valid, "retired qtype tag",
                                    [first_entry](auto& b) { write_u16(b, first_entry + 16, 7); });
    failures += expect_parse_throws(valid, "bad layout",
                                    [first_entry](auto& b) { write_u16(b, first_entry + 18, 99); });
    failures += expect_parse_throws(valid, "bad ndim",
                                    [first_entry](auto& b) { write_u16(b, first_entry + 22, 5); });
    failures += expect_parse_throws(valid, "bad name hash",
                                    [first_entry](auto& b) { write_u64(b, first_entry + 8, 123); });
    failures += expect_parse_throws(valid, "catalog name length cap", [first_entry](auto& b) {
        write_u32(b, first_entry + 4, 4097);
    });
    failures += expect_parse_throws(valid, "duplicate name", [first_entry, second_entry](auto& b) {
        write_u32(b, second_entry, read_u32(b, first_entry));
        write_u32(b, second_entry + 4, read_u32(b, first_entry + 4));
        write_u64(b, second_entry + 8, read_u64(b, first_entry + 8));
    });
    failures += expect_parse_throws(valid, "duplicate source", [gate_segment, up_segment](auto& b) {
        write_u32(b, up_segment + 0, read_u32(b, gate_segment + 0));
        write_u32(b, up_segment + 4, read_u32(b, gate_segment + 4));
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
    failures += expect_parse_throws(valid, "shape payload mismatch", [first_entry](auto& b) {
        write_u32(b, first_entry + 40, 128);
    });
    failures += expect_parse_throws(valid, "nibble plane mismatch", [first_entry](auto& b) {
        write_u64(b, first_entry + 100, read_u64(b, first_entry + 100) + 1);
    });
    failures += expect_parse_throws(valid, "high plane mismatch", [first_entry](auto& b) {
        write_u64(b, first_entry + 108, read_u64(b, first_entry + 108) + 1);
    });
    failures += expect_parse_throws(valid, "scale plane mismatch", [first_entry](auto& b) {
        write_u64(b, first_entry + 116, read_u64(b, first_entry + 116) + 1);
    });
    failures += expect_parse_throws(valid, "W8G32 bad group",
                                    [mtp_entry](auto& b) { write_u32(b, mtp_entry + 56, 64); });
    failures += expect_parse_throws(valid, "W8G32 nonzero high plane",
                                    [mtp_entry](auto& b) { write_u64(b, mtp_entry + 108, 32); });
    failures += expect_parse_throws(valid, "W8G32 bad scale plane", [mtp_entry](auto& b) {
        write_u64(b, mtp_entry + 116, read_u64(b, mtp_entry + 116) + 2);
    });
    failures += expect_parse_throws(valid, "segment partition mismatch",
                                    [up_segment](auto& b) { write_u32(b, up_segment + 8, 6); });
    failures += expect_parse_throws(
        valid, "segment partition uint32 wrap", [mtp_attn_segments, mtp_attn_n](auto& b) {
            const auto set_range = [&](std::uint32_t index, std::uint32_t begin,
                                       std::uint32_t count) {
                const std::uint64_t record =
                    mtp_attn_segments + static_cast<std::uint64_t>(index) * kSegmentRecordSize;
                write_u32(b, record + 8, begin);
                write_u32(b, record + 12, count);
            };
            set_range(0, 0, 1);
            set_range(1, 1, std::numeric_limits<std::uint32_t>::max());
            set_range(2, 0, 1);
            set_range(3, 1, mtp_attn_n - 1);
        });
    failures += expect_parse_throws(valid, "fusion total_n mismatch", [fusion_offset](auto& b) {
        write_u32(b, fusion_offset + 40, read_u32(b, fusion_offset + 40) + 1);
    });
    return failures == 0 ? 0 : fail("q5090 parser test failed");
}
