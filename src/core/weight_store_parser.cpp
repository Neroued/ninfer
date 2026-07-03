#include "qus/core/weight_store_parser.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace qus {
namespace {

constexpr std::array<std::byte, 16> kMagic = {
    std::byte{0x51}, std::byte{0x35}, std::byte{0x30}, std::byte{0x39},
    std::byte{0x30}, std::byte{0x4D}, std::byte{0x49}, std::byte{0x58},
    std::byte{0x45}, std::byte{0x44}, std::byte{0x56}, std::byte{0x33},
    std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
};

constexpr std::uint32_t kVersion               = 3;
constexpr std::uint32_t kEndianTag             = 0x01020304U;
constexpr std::uint32_t kHeaderSize            = 4096;
constexpr std::uint32_t kModuleRecordSize      = 64;
constexpr std::uint32_t kTensorEntrySize       = 128;
constexpr std::uint32_t kSegmentRecordSize     = 32;
constexpr std::uint32_t kFusionGroupRecordSize = 64;
constexpr std::uint64_t kPayloadAlign          = 256;
constexpr std::uint64_t kRegionAlign           = 4096;

struct Range {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

[[noreturn]] void parse_error(const char* message) { throw std::runtime_error(message); }

void require(bool ok, const char* message) {
    if (!ok) { parse_error(message); }
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        parse_error("q5090 integer addition overflow");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        parse_error("q5090 integer multiplication overflow");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t align) {
    if (align == 0) { parse_error("q5090 invalid alignment"); }
    const std::uint64_t add = align - 1;
    return (checked_add(value, add) / align) * align;
}

void require_range(std::span<const std::byte> file, std::uint64_t offset, std::uint64_t size,
                   const char* label) {
    const std::uint64_t end = checked_add(offset, size);
    if (offset > file.size() || end > file.size()) {
        throw std::runtime_error(std::string("q5090 range outside file: ") + label);
    }
}

std::uint16_t read_u16(std::span<const std::byte> file, std::uint64_t offset) {
    require_range(file, offset, 2, "u16");
    const auto* p = file.data() + offset;
    return static_cast<std::uint16_t>(std::to_integer<unsigned char>(p[0])) |
           (static_cast<std::uint16_t>(std::to_integer<unsigned char>(p[1])) << 8);
}

std::uint32_t read_u32(std::span<const std::byte> file, std::uint64_t offset) {
    require_range(file, offset, 4, "u32");
    const auto* p = file.data() + offset;
    return static_cast<std::uint32_t>(std::to_integer<unsigned char>(p[0])) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(p[1])) << 8) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(p[2])) << 16) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(p[3])) << 24);
}

std::uint64_t read_u64(std::span<const std::byte> file, std::uint64_t offset) {
    require_range(file, offset, 8, "u64");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(
                     std::to_integer<unsigned char>(file[static_cast<std::size_t>(offset) + i]))
                 << (8 * i);
    }
    return value;
}

bool all_zero(std::span<const std::byte> bytes) {
    return std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0}; });
}

void require_zero_range(std::span<const std::byte> file, std::uint64_t offset, std::uint64_t size,
                        const char* label) {
    require_range(file, offset, size, label);
    require(all_zero(file.subspan(static_cast<std::size_t>(offset),
                                  static_cast<std::size_t>(size))),
            label);
}

template <typename T>
void validate_expected(std::uint32_t actual, const std::optional<T>& expected, const char* name) {
    if (expected.has_value() && actual != static_cast<std::uint32_t>(*expected)) {
        throw std::runtime_error(std::string("q5090 model metadata mismatch: ") + name);
    }
}

QType qtype_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0: return QType::Q4G64_F16S;
    case 1: return QType::Q5G64_F16S;
    case 2: return QType::Q6G64_F16S;
    case 3: return QType::W8G128_F16S;
    case 4: return QType::BF16_CTRL;
    case 5: return QType::FP32_CTRL;
    case 6: return QType::W8G32_F16S;
    default: parse_error("q5090 invalid qtype tag");
    }
}

QuantLayout layout_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0: return QuantLayout::RowSplit;
    case 1: return QuantLayout::Contiguous;
    default: parse_error("q5090 invalid layout tag");
    }
}

ModuleKind module_from_tag(std::uint32_t tag) {
    switch (tag) {
    case 0: return ModuleKind::TextCore;
    case 1: return ModuleKind::MtpDraft;
    case 2: return ModuleKind::VisionEncoder;
    default: parse_error("q5090 invalid module tag");
    }
}

ScaleDType scale_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0: return ScaleDType::None;
    case 1: return ScaleDType::FP16;
    default: parse_error("q5090 invalid scale dtype tag");
    }
}

LoadPolicy load_policy_from_tag(std::uint32_t tag) {
    switch (tag) {
    case 0: return LoadPolicy::Resident;
    case 1: return LoadPolicy::LazyGpu;
    case 2: return LoadPolicy::CpuPinnedThenGpu;
    default: parse_error("q5090 invalid load policy tag");
    }
}

bool valid_source_kind(std::uint32_t kind) {
    switch (kind) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 30:
    case 31:
    case 32:
    case 33:
    case 34:
    case 35:
    case 36:
    case 40:
    case 41:
    case 42:
    case 50:
    case 51:
    case 52:
    case 53:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
    case 67:
    case 68:
    case 69:
    case 70:
    case 71:
    case 72:
    case 73:
    case 74:
    case 75:
    case 76:
    case 77:
    case 78:
    case 79:
    case 80:
        return true;
    default:
        return false;
    }
}

bool valid_source_kind(ModuleKind module, std::uint32_t kind) {
    if (kind == static_cast<std::uint32_t>(SourceKind::Other)) { return true; }
    switch (module) {
    case ModuleKind::TextCore:
        return (kind >= 1 && kind <= 5) || (kind >= 10 && kind <= 20) ||
               (kind >= 30 && kind <= 36) || (kind >= 40 && kind <= 42);
    case ModuleKind::MtpDraft:
        return (kind >= 50 && kind <= 53) || kind == 4 || kind == 5 ||
               (kind >= 30 && kind <= 36) || (kind >= 40 && kind <= 42);
    case ModuleKind::VisionEncoder:
        return kind >= 60 && kind <= 80;
    }
    return false;
}

bool valid_fusion_group_id(std::uint32_t group_id) {
    return group_id >= 1 && group_id <= 3;
}

bool is_quant_qtype(QType qtype) {
    return qtype == QType::Q4G64_F16S || qtype == QType::Q5G64_F16S ||
           qtype == QType::Q6G64_F16S || qtype == QType::W8G128_F16S ||
           qtype == QType::W8G32_F16S;
}

std::uint32_t quant_group_size(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
        return 64;
    case QType::W8G128_F16S:
        return 128;
    case QType::W8G32_F16S:
        return 32;
    default:
        parse_error("q5090 qtype has no quant group size");
    }
}

std::uint64_t nibble_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
        return 32;
    case QType::W8G128_F16S:
        return 128;
    case QType::W8G32_F16S:
        return 32;
    default:
        parse_error("q5090 qtype has no nibble plane bytes per group");
    }
}

std::uint64_t high_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::W8G128_F16S:
    case QType::W8G32_F16S:
        return 0;
    case QType::Q5G64_F16S:
        return 8;
    case QType::Q6G64_F16S:
        return 16;
    default:
        parse_error("q5090 qtype has no high plane bytes per group");
    }
}

std::uint64_t element_size(QType qtype) {
    switch (qtype) {
    case QType::BF16_CTRL: return 2;
    case QType::FP32_CTRL: return 4;
    default: parse_error("q5090 qtype has no contiguous element size");
    }
}

std::uint64_t numel(const std::array<std::uint32_t, 4>& shape, std::uint32_t ndim) {
    std::uint64_t result = 1;
    for (std::uint32_t i = 0; i < ndim; ++i) { result = checked_mul(result, shape[i]); }
    return result;
}

void validate_shapes(const ParsedQ5090Tensor& tensor) {
    require(tensor.ndim >= 1 && tensor.ndim <= 4, "q5090 invalid tensor rank");
    for (std::uint32_t i = 0; i < 4; ++i) {
        if (i < tensor.ndim) {
            require(tensor.shape[i] != 0, "q5090 zero tensor shape dim");
            require(tensor.padded_shape[i] != 0, "q5090 zero padded tensor shape dim");
        } else {
            require(tensor.shape[i] == 1, "q5090 inactive shape dim must be one");
            require(tensor.padded_shape[i] == 1, "q5090 inactive padded dim must be one");
        }
    }
}

std::uint64_t expected_payload_bytes(const ParsedQ5090Tensor& tensor) {
    validate_shapes(tensor);
    switch (tensor.layout) {
    case QuantLayout::RowSplit: {
        require(tensor.ndim == 2, "q5090 ROW_SPLIT tensor must be 2D");
        require(is_quant_qtype(tensor.qtype), "q5090 ROW_SPLIT qtype mismatch");
        const std::uint32_t group = quant_group_size(tensor.qtype);
        require(tensor.group_size == group, "q5090 ROW_SPLIT group mismatch");
        require(tensor.scale_dtype == ScaleDType::FP16, "q5090 ROW_SPLIT scale mismatch");
        require(tensor.padded_shape[0] == tensor.shape[0],
                "q5090 ROW_SPLIT padded N mismatch");
        require(tensor.shape[2] == 1 && tensor.shape[3] == 1,
                "q5090 ROW_SPLIT trailing shape mismatch");
        require(tensor.padded_shape[1] == align_up(tensor.shape[1], 128),
                "q5090 ROW_SPLIT padded K mismatch");
        const std::uint64_t groups = tensor.padded_shape[1] / group;
        const std::uint64_t nibble =
            checked_mul(checked_mul(tensor.shape[0], groups), nibble_bytes_per_group(tensor.qtype));
        const std::uint64_t high =
            checked_mul(checked_mul(tensor.shape[0], groups), high_bytes_per_group(tensor.qtype));
        const std::uint64_t scale = checked_mul(checked_mul(tensor.shape[0], groups), 2);
        const std::uint64_t high_rel = align_up(nibble, kPayloadAlign);
        const std::uint64_t scale_rel = checked_add(high_rel, align_up(high, kPayloadAlign));
        const std::uint64_t payload = checked_add(scale_rel, scale);
        require(tensor.nibble_plane_bytes == nibble,
                "q5090 ROW_SPLIT nibble plane byte mismatch");
        require(tensor.high_plane_bytes == high, "q5090 ROW_SPLIT high plane byte mismatch");
        require(tensor.scale_plane_bytes == scale, "q5090 ROW_SPLIT scale plane byte mismatch");
        return payload;
    }
    case QuantLayout::Contiguous: {
        require(tensor.qtype == QType::BF16_CTRL || tensor.qtype == QType::FP32_CTRL,
                "q5090 CONTIGUOUS qtype mismatch");
        require(tensor.group_size == 0, "q5090 CONTIGUOUS group mismatch");
        require(tensor.scale_dtype == ScaleDType::None, "q5090 CONTIGUOUS scale mismatch");
        require(tensor.padded_shape == tensor.shape, "q5090 CONTIGUOUS padded shape mismatch");
        const std::uint64_t raw = checked_mul(numel(tensor.shape, tensor.ndim),
                                              element_size(tensor.qtype));
        require(tensor.nibble_plane_bytes == raw, "q5090 CONTIGUOUS payload byte mismatch");
        require(tensor.high_plane_bytes == 0, "q5090 CONTIGUOUS high byte mismatch");
        require(tensor.scale_plane_bytes == 0, "q5090 CONTIGUOUS scale byte mismatch");
        return raw;
    }
    }
    parse_error("q5090 invalid layout");
}

ParsedQ5090Header parse_header(std::span<const std::byte> file,
                               const Q5090Expectations& expected) {
    require(file.size() >= kHeaderSize, "q5090 file too small for header");
    require(std::equal(kMagic.begin(), kMagic.end(), file.begin()), "q5090 bad magic");

    require(read_u32(file, 16) == kVersion, "q5090 bad version");
    require(read_u32(file, 20) == kEndianTag, "q5090 endian mismatch");
    require(read_u32(file, 24) == kHeaderSize, "q5090 bad header size");

    ParsedQ5090Header h;
    h.tensor_count              = read_u32(file, 28);
    h.module_count              = read_u32(file, 32);
    h.layer_count               = read_u32(file, 36);
    h.flags                     = read_u32(file, 40);
    h.segment_count             = read_u32(file, 44);
    h.module_index_offset       = read_u64(file, 48);
    h.module_index_bytes        = read_u64(file, 56);
    h.tensor_index_offset       = read_u64(file, 64);
    h.tensor_index_bytes        = read_u64(file, 72);
    h.string_table_offset       = read_u64(file, 80);
    h.string_table_bytes        = read_u64(file, 88);
    h.payload_offset            = read_u64(file, 96);
    h.payload_bytes             = read_u64(file, 104);
    h.hidden_size               = read_u32(file, 112);
    h.intermediate_size         = read_u32(file, 116);
    h.vocab_size                = read_u32(file, 120);
    h.num_attention_heads       = read_u32(file, 124);
    h.num_key_value_heads       = read_u32(file, 128);
    h.head_dim                  = read_u32(file, 132);
    h.gdn_key_heads             = read_u32(file, 136);
    h.gdn_value_heads           = read_u32(file, 140);
    h.gdn_key_head_dim          = read_u32(file, 144);
    h.gdn_value_head_dim        = read_u32(file, 148);
    h.gdn_conv_width            = read_u32(file, 152);
    h.full_attention_interval   = read_u32(file, 156);
    h.max_position_embeddings   = read_u32(file, 160);
    h.fusion_group_count        = read_u32(file, 164);
    for (std::size_t i = 0; i < h.sha256_safetensors_index.size(); ++i) {
        h.sha256_safetensors_index[i] = std::to_integer<std::uint8_t>(file[168 + i]);
    }
    h.segment_index_offset      = read_u64(file, 200);
    h.segment_index_bytes       = read_u64(file, 208);
    h.fusion_group_index_offset = read_u64(file, 216);
    h.fusion_group_index_bytes  = read_u64(file, 224);
    h.format_minor              = read_u32(file, 232);

    require(h.module_count >= 1 && h.module_count <= 3, "q5090 invalid module count");
    require(h.tensor_count > 0, "q5090 tensor count is zero");
    require(h.segment_count > 0, "q5090 segment count is zero");
    require(h.layer_count == expected.layer_count.value_or(64), "q5090 layer count mismatch");
    require((h.flags & ~0x0FU) == 0, "q5090 unknown header flags");
    require(h.format_minor == 0, "q5090 unsupported format minor");
    require(all_zero(file.subspan(236, kHeaderSize - 236)), "q5090 header padding nonzero");

    require(h.module_index_offset == kHeaderSize, "q5090 bad module index offset");
    require(h.module_index_bytes == checked_mul(h.module_count, kModuleRecordSize),
            "q5090 bad module index bytes");
    require(h.tensor_index_offset == checked_add(h.module_index_offset, h.module_index_bytes),
            "q5090 bad tensor index offset");
    require(h.tensor_index_bytes == checked_mul(h.tensor_count, kTensorEntrySize),
            "q5090 bad tensor index bytes");
    require(h.segment_index_offset == checked_add(h.tensor_index_offset, h.tensor_index_bytes),
            "q5090 bad segment index offset");
    require(h.segment_index_bytes == checked_mul(h.segment_count, kSegmentRecordSize),
            "q5090 bad segment index bytes");
    require(h.fusion_group_index_offset == checked_add(h.segment_index_offset,
                                                       h.segment_index_bytes),
            "q5090 bad fusion group index offset");
    require(h.fusion_group_index_bytes == checked_mul(h.fusion_group_count,
                                                      kFusionGroupRecordSize),
            "q5090 bad fusion group index bytes");
    require(h.string_table_offset == checked_add(h.fusion_group_index_offset,
                                                 h.fusion_group_index_bytes),
            "q5090 bad string table offset");
    const std::uint64_t string_end = checked_add(h.string_table_offset, h.string_table_bytes);
    require(h.payload_offset >= string_end, "q5090 payload begins before string table end");
    require(h.payload_offset % kRegionAlign == 0, "q5090 payload region is not aligned");
    require(checked_add(h.payload_offset, h.payload_bytes) == file.size(),
            "q5090 payload bytes do not match file size");

    require_range(file, h.module_index_offset, h.module_index_bytes, "module index");
    require_range(file, h.tensor_index_offset, h.tensor_index_bytes, "tensor index");
    require_range(file, h.segment_index_offset, h.segment_index_bytes, "segment index");
    require_range(file, h.fusion_group_index_offset, h.fusion_group_index_bytes,
                  "fusion group index");
    require_range(file, h.string_table_offset, h.string_table_bytes, "string table");
    require_range(file, h.payload_offset, h.payload_bytes, "payload region");
    require(all_zero(file.subspan(static_cast<std::size_t>(string_end),
                                  static_cast<std::size_t>(h.payload_offset - string_end))),
            "q5090 metadata padding nonzero");

    validate_expected(h.hidden_size, expected.hidden_size, "hidden_size");
    validate_expected(h.intermediate_size, expected.intermediate_size, "intermediate_size");
    validate_expected(h.vocab_size, expected.vocab_size, "vocab_size");
    validate_expected(h.num_attention_heads, expected.num_attention_heads, "num_attention_heads");
    validate_expected(h.num_key_value_heads, expected.num_key_value_heads, "num_key_value_heads");
    validate_expected(h.head_dim, expected.head_dim, "head_dim");
    validate_expected(h.gdn_key_heads, expected.gdn_key_heads, "gdn_key_heads");
    validate_expected(h.gdn_value_heads, expected.gdn_value_heads, "gdn_value_heads");
    validate_expected(h.gdn_key_head_dim, expected.gdn_key_head_dim, "gdn_key_head_dim");
    validate_expected(h.gdn_value_head_dim, expected.gdn_value_head_dim, "gdn_value_head_dim");
    validate_expected(h.gdn_conv_width, expected.gdn_conv_width, "gdn_conv_width");
    validate_expected(h.full_attention_interval, expected.full_attention_interval,
                      "full_attention_interval");
    validate_expected(h.max_position_embeddings, expected.max_position_embeddings,
                      "max_position_embeddings");
    return h;
}

std::vector<ParsedQ5090Module> parse_modules(std::span<const std::byte> file,
                                             const ParsedQ5090Header& h) {
    std::vector<ParsedQ5090Module> modules;
    modules.reserve(h.module_count);
    std::uint64_t expected_begin       = 0;
    std::uint32_t previous_kind        = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t previous_payload_end = h.payload_offset;
    for (std::uint32_t i = 0; i < h.module_count; ++i) {
        const std::uint64_t off = h.module_index_offset + checked_mul(i, kModuleRecordSize);
        ParsedQ5090Module m;
        const std::uint32_t raw_kind = read_u32(file, off);
        m.module_kind                = module_from_tag(raw_kind);
        m.module_version             = read_u32(file, off + 4);
        m.tensor_index_begin         = read_u64(file, off + 8);
        m.tensor_index_count         = read_u64(file, off + 16);
        m.payload_offset             = read_u64(file, off + 24);
        m.payload_bytes              = read_u64(file, off + 32);
        m.load_policy                = load_policy_from_tag(read_u32(file, off + 40));
        m.flags                      = read_u32(file, off + 44);
        require(all_zero(file.subspan(static_cast<std::size_t>(off + 48), 16)),
                "q5090 module reserved bytes nonzero");
        require(m.module_version == kVersion, "q5090 bad module version");
        require(m.flags == 0, "q5090 module flags nonzero");
        require(m.tensor_index_count > 0, "q5090 empty module");
        require(m.tensor_index_begin == expected_begin,
                "q5090 module tensor ranges are not contiguous");
        expected_begin = checked_add(expected_begin, m.tensor_index_count);
        require(checked_add(m.tensor_index_begin, m.tensor_index_count) <= h.tensor_count,
                "q5090 module tensor range outside tensor index");
        if (i == 0) {
            require(m.module_kind == ModuleKind::TextCore, "q5090 first module is not TEXT_CORE");
        } else {
            require(raw_kind > previous_kind, "q5090 module kinds are not strictly ordered");
        }
        previous_kind = raw_kind;
        require(m.payload_bytes > 0, "q5090 module payload span is empty");
        require(m.payload_offset >= h.payload_offset,
                "q5090 module payload begins before payload region");
        require(m.payload_offset >= previous_payload_end,
                "q5090 module payload spans are not ordered");
        require(checked_add(m.payload_offset, m.payload_bytes) <=
                    checked_add(h.payload_offset, h.payload_bytes),
                "q5090 module payload outside payload region");
        previous_payload_end = checked_add(m.payload_offset, m.payload_bytes);
        modules.push_back(m);
    }
    require(expected_begin == h.tensor_count,
            "q5090 module tensor ranges do not cover tensor index");
    return modules;
}

std::uint32_t module_flags(const std::vector<ParsedQ5090Module>& modules) {
    std::uint32_t flags = 0;
    for (const ParsedQ5090Module& module : modules) {
        flags |= 1U << static_cast<std::uint32_t>(module.module_kind);
    }
    return flags;
}

const ParsedQ5090Module& module_for_tensor_index(const std::vector<ParsedQ5090Module>& modules,
                                                 std::uint64_t index) {
    for (const ParsedQ5090Module& module : modules) {
        const std::uint64_t end = checked_add(module.tensor_index_begin, module.tensor_index_count);
        if (index >= module.tensor_index_begin && index < end) { return module; }
    }
    parse_error("q5090 tensor index not covered by any module");
}

std::string read_name(std::span<const std::byte> file, const ParsedQ5090Header& h,
                      std::uint32_t name_offset, std::uint32_t name_len) {
    const std::uint64_t name_end = checked_add(name_offset, name_len);
    require(checked_add(name_end, 1) <= h.string_table_bytes, "q5090 name outside string table");
    const std::uint64_t absolute = checked_add(h.string_table_offset, name_offset);
    require(file[static_cast<std::size_t>(absolute + name_len)] == std::byte{0},
            "q5090 name is not NUL-terminated");
    const auto* chars = reinterpret_cast<const char*>(file.data() + absolute);
    return std::string(chars, chars + name_len);
}

std::vector<ParsedQ5090Tensor> parse_tensors(std::span<const std::byte> file,
                                             const ParsedQ5090Header& h,
                                             const std::vector<ParsedQ5090Module>& modules) {
    std::vector<ParsedQ5090Tensor> tensors;
    tensors.reserve(h.tensor_count);
    std::vector<Range> ranges;
    ranges.reserve(h.tensor_count);
    std::set<std::string> seen_names;
    std::uint64_t previous_payload_end = h.payload_offset;
    for (std::uint32_t i = 0; i < h.tensor_count; ++i) {
        const std::uint64_t off = h.tensor_index_offset + checked_mul(i, kTensorEntrySize);
        ParsedQ5090Tensor t;
        t.name_offset      = read_u32(file, off);
        t.name_len         = read_u32(file, off + 4);
        t.name_hash        = read_u64(file, off + 8);
        t.qtype            = qtype_from_tag(read_u16(file, off + 16));
        t.layout           = layout_from_tag(read_u16(file, off + 18));
        t.module_kind      = module_from_tag(read_u16(file, off + 20));
        t.ndim             = read_u16(file, off + 22);
        for (int d = 0; d < 4; ++d) {
            t.shape[d]        = read_u32(file, off + 24 + d * 4);
            t.padded_shape[d] = read_u32(file, off + 40 + d * 4);
        }
        t.group_size        = read_u32(file, off + 56);
        t.scale_dtype       = scale_from_tag(read_u16(file, off + 60));
        t.segment_count     = read_u16(file, off + 62);
        t.payload_offset    = read_u64(file, off + 64);
        t.payload_bytes     = read_u64(file, off + 72);
        t.source_layer      = read_u32(file, off + 80);
        t.source_kind       = read_u32(file, off + 84);
        t.crc32             = read_u32(file, off + 88);
        t.segment_begin     = read_u32(file, off + 92);
        t.fusion_group_id   = read_u16(file, off + 96);
        t.fusion_index      = read_u16(file, off + 98);
        t.nibble_plane_bytes = read_u64(file, off + 100);
        t.high_plane_bytes   = read_u64(file, off + 108);
        t.scale_plane_bytes  = read_u64(file, off + 116);
        require(all_zero(file.subspan(static_cast<std::size_t>(off + 124), 4)),
                "q5090 tensor reserved bytes nonzero");

        const ParsedQ5090Module& module = module_for_tensor_index(modules, i);
        require(t.module_kind == module.module_kind,
                "q5090 tensor module does not match module range");
        t.name = read_name(file, h, t.name_offset, t.name_len);
        require(q5090_fnv1a64(t.name) == t.name_hash, "q5090 tensor name hash mismatch");
        require(t.segment_count > 0, "q5090 tensor has no segments");
        require(t.payload_bytes > 0, "q5090 tensor payload is empty");
        require(t.payload_offset % kPayloadAlign == 0, "q5090 tensor payload is not aligned");
        require(t.payload_offset >= h.payload_offset,
                "q5090 tensor payload begins before payload region");
        const std::uint64_t payload_end = checked_add(t.payload_offset, t.payload_bytes);
        require(payload_end <= checked_add(h.payload_offset, h.payload_bytes),
                "q5090 tensor payload outside payload region");
        require(t.payload_offset >= module.payload_offset &&
                    payload_end <= checked_add(module.payload_offset, module.payload_bytes),
                "q5090 tensor payload outside module span");
        require(t.payload_offset >= previous_payload_end,
                "q5090 tensor payload offsets are not ordered");
        require_zero_range(file, previous_payload_end, t.payload_offset - previous_payload_end,
                           "q5090 payload padding nonzero");
        previous_payload_end = payload_end;
        require(t.source_layer <= 63 || t.source_layer == kQ5090NoLayer,
                "q5090 invalid source layer");
        require(valid_source_kind(t.module_kind, t.source_kind), "q5090 invalid source kind");
        require(t.fusion_group_id == 0 || valid_fusion_group_id(t.fusion_group_id),
                "q5090 invalid tensor fusion group id");
        if (t.fusion_group_id == 0) {
            require(t.fusion_index == 0, "q5090 standalone tensor has nonzero fusion index");
        }
        require(seen_names.insert(t.name).second, "q5090 duplicate tensor name");
        require(expected_payload_bytes(t) == t.payload_bytes, "q5090 tensor payload byte mismatch");
        ranges.push_back(Range{t.payload_offset, payload_end});
        tensors.push_back(std::move(t));
    }
    require_zero_range(file, previous_payload_end,
                       checked_add(h.payload_offset, h.payload_bytes) - previous_payload_end,
                       "q5090 payload padding nonzero");

    std::sort(ranges.begin(), ranges.end(),
              [](const Range& a, const Range& b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        require(ranges[i].begin >= ranges[i - 1].end, "q5090 tensor payload ranges overlap");
    }

    for (const ParsedQ5090Module& module : modules) {
        std::uint64_t min_begin = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t max_end   = 0;
        for (std::uint64_t i = module.tensor_index_begin;
             i < checked_add(module.tensor_index_begin, module.tensor_index_count); ++i) {
            const ParsedQ5090Tensor& tensor = tensors[static_cast<std::size_t>(i)];
            min_begin                       = std::min(min_begin, tensor.payload_offset);
            max_end = std::max(max_end, checked_add(tensor.payload_offset, tensor.payload_bytes));
        }
        require(module.payload_offset == min_begin, "q5090 module payload offset mismatch");
        require(checked_add(module.payload_offset, module.payload_bytes) == max_end,
                "q5090 module payload span mismatch");
    }
    return tensors;
}

std::vector<ParsedQ5090Segment> parse_segments(std::span<const std::byte> file,
                                               const ParsedQ5090Header& h) {
    std::vector<ParsedQ5090Segment> segments;
    segments.reserve(h.segment_count);
    std::set<std::string> seen_names;
    for (std::uint32_t i = 0; i < h.segment_count; ++i) {
        const std::uint64_t off = h.segment_index_offset + checked_mul(i, kSegmentRecordSize);
        ParsedQ5090Segment s;
        s.source_kind  = read_u32(file, off);
        s.source_layer = read_u32(file, off + 4);
        s.row_begin    = read_u32(file, off + 8);
        s.row_count    = read_u32(file, off + 12);
        s.name_offset  = read_u32(file, off + 16);
        s.name_len     = read_u32(file, off + 20);
        s.name_hash    = read_u64(file, off + 24);
        s.name         = read_name(file, h, s.name_offset, s.name_len);
        require(q5090_fnv1a64(s.name) == s.name_hash, "q5090 segment name hash mismatch");
        require(s.source_layer <= 63 || s.source_layer == kQ5090NoLayer,
                "q5090 invalid segment source layer");
        require(valid_source_kind(s.source_kind), "q5090 invalid segment source kind");
        require(s.row_count > 0, "q5090 segment row_count is zero");
        require(seen_names.insert(s.name).second, "q5090 duplicate segment name");
        segments.push_back(std::move(s));
    }
    return segments;
}

void validate_tensor_segments(const std::vector<ParsedQ5090Tensor>& tensors,
                              const std::vector<ParsedQ5090Segment>& segments) {
    std::set<std::tuple<std::uint16_t, std::uint32_t, std::uint32_t>> seen_sources;
    for (const ParsedQ5090Tensor& t : tensors) {
        const std::uint64_t end = checked_add(t.segment_begin, t.segment_count);
        require(end <= segments.size(), "q5090 tensor segment range outside table");
        const auto module = static_cast<std::uint16_t>(t.module_kind);
        std::uint32_t row = 0;
        for (std::uint32_t j = 0; j < t.segment_count; ++j) {
            const ParsedQ5090Segment& s = segments[static_cast<std::size_t>(t.segment_begin + j)];
            require(valid_source_kind(t.module_kind, s.source_kind),
                    "q5090 invalid segment source kind for module");
            if (t.layout == QuantLayout::RowSplit) {
                require(s.row_begin == row, "q5090 ROW_SPLIT segments are not contiguous");
                row = static_cast<std::uint32_t>(checked_add(row, s.row_count));
                require(row <= t.shape[0], "q5090 ROW_SPLIT segment exceeds N");
            } else {
                require(t.segment_count == 1, "q5090 CONTIGUOUS block must have one segment");
                require(s.row_begin == 0, "q5090 CONTIGUOUS segment row_begin must be zero");
                require(s.row_count == t.shape[0],
                        "q5090 CONTIGUOUS segment row_count mismatch");
            }
            if (t.segment_count > 1) {
                require(s.source_layer == t.source_layer,
                        "q5090 fused block segment source_layer mismatch");
            }
            if (s.source_kind != static_cast<std::uint32_t>(SourceKind::Other)) {
                const auto key = std::make_tuple(module, s.source_kind, s.source_layer);
                require(seen_sources.insert(key).second, "q5090 duplicate segment source id");
            }
        }
        if (t.layout == QuantLayout::RowSplit) {
            require(row == t.shape[0], "q5090 ROW_SPLIT segments do not partition N");
        }

        const ParsedQ5090Segment& first = segments[static_cast<std::size_t>(t.segment_begin)];
        if (t.segment_count == 1) {
            require(t.source_kind == first.source_kind && t.source_layer == first.source_layer,
                    "q5090 single-segment source identity mismatch");
            if (t.fusion_group_id == 0) {
                require(t.name == first.name, "q5090 standalone segment name mismatch");
            }
        } else {
            require(t.source_kind == static_cast<std::uint32_t>(SourceKind::Other),
                    "q5090 fused block source_kind must be OTHER");
        }
    }
}

std::vector<ParsedQ5090FusionGroup> parse_fusion_groups(std::span<const std::byte> file,
                                                        const ParsedQ5090Header& h) {
    std::vector<ParsedQ5090FusionGroup> groups;
    groups.reserve(h.fusion_group_count);
    for (std::uint32_t i = 0; i < h.fusion_group_count; ++i) {
        const std::uint64_t off =
            h.fusion_group_index_offset + checked_mul(i, kFusionGroupRecordSize);
        ParsedQ5090FusionGroup g;
        g.group_id                 = read_u32(file, off);
        g.source_layer             = read_u32(file, off + 4);
        g.block_count              = read_u32(file, off + 8);
        g.shared_input_kind        = read_u32(file, off + 12);
        g.first_block_tensor_index = read_u64(file, off + 16);
        g.payload_offset           = read_u64(file, off + 24);
        g.payload_bytes            = read_u64(file, off + 32);
        g.total_n                  = read_u32(file, off + 40);
        g.shared_k                 = read_u32(file, off + 44);
        require(all_zero(file.subspan(static_cast<std::size_t>(off + 48), 16)),
                "q5090 fusion group reserved bytes nonzero");
        require(valid_fusion_group_id(g.group_id), "q5090 invalid fusion group id");
        require(g.source_layer <= 63, "q5090 invalid fusion group source layer");
        require(g.block_count > 0, "q5090 empty fusion group");
        require(valid_source_kind(g.shared_input_kind), "q5090 invalid fusion shared input kind");
        require(checked_add(g.first_block_tensor_index, g.block_count) <= h.tensor_count,
                "q5090 fusion group tensor range outside table");
        require(g.payload_offset >= h.payload_offset,
                "q5090 fusion group payload begins before payload region");
        require(checked_add(g.payload_offset, g.payload_bytes) <=
                    checked_add(h.payload_offset, h.payload_bytes),
                "q5090 fusion group payload outside payload region");
        groups.push_back(g);
    }
    return groups;
}

void validate_fusion_groups(const std::vector<ParsedQ5090Tensor>& tensors,
                            const std::vector<ParsedQ5090FusionGroup>& groups) {
    std::vector<bool> covered(tensors.size(), false);
    std::uint64_t previous_first = 0;
    bool have_previous           = false;
    for (const ParsedQ5090FusionGroup& g : groups) {
        if (have_previous) {
            require(g.first_block_tensor_index > previous_first,
                    "q5090 fusion groups are not ordered");
        }
        have_previous = true;
        previous_first = g.first_block_tensor_index;

        std::uint32_t total_n = 0;
        const std::uint64_t first = g.first_block_tensor_index;
        const std::uint64_t last = first + g.block_count - 1;
        const ParsedQ5090Tensor& first_tensor = tensors[static_cast<std::size_t>(first)];
        const ParsedQ5090Tensor& last_tensor  = tensors[static_cast<std::size_t>(last)];
        require(g.payload_offset == first_tensor.payload_offset,
                "q5090 fusion payload offset mismatch");
        require(checked_add(g.payload_offset, g.payload_bytes) ==
                    checked_add(last_tensor.payload_offset, last_tensor.payload_bytes),
                "q5090 fusion payload span mismatch");

        for (std::uint32_t j = 0; j < g.block_count; ++j) {
            const std::size_t index = static_cast<std::size_t>(g.first_block_tensor_index + j);
            require(!covered[index], "q5090 overlapping fusion groups");
            covered[index] = true;
            const ParsedQ5090Tensor& t = tensors[index];
            require(t.layout == QuantLayout::RowSplit, "q5090 fusion member must be ROW_SPLIT");
            require(t.fusion_group_id == g.group_id, "q5090 fusion member group id mismatch");
            require(t.fusion_index == j, "q5090 fusion member index mismatch");
            require(t.source_layer == g.source_layer, "q5090 fusion member layer mismatch");
            require(t.shape[1] == g.shared_k, "q5090 fusion member K mismatch");
            total_n = static_cast<std::uint32_t>(checked_add(total_n, t.shape[0]));
        }
        require(total_n == g.total_n, "q5090 fusion total_n mismatch");
    }

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].fusion_group_id != 0) {
            require(covered[i], "q5090 tensor references missing fusion group");
        }
    }
}

} // namespace

std::uint64_t q5090_fnv1a64(std::string_view name) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (unsigned char b : name) {
        h ^= b;
        h *= 0x100000001B3ULL;
    }
    return h;
}

ParsedQ5090File parse_q5090_file(std::span<const std::byte> file,
                                 const Q5090Expectations& expected,
                                 Q5090Progress* progress) {
    (void)progress;
    ParsedQ5090File parsed;
    parsed.header = parse_header(file, expected);
    parsed.modules = parse_modules(file, parsed.header);
    require((parsed.header.flags & 0x07U) == module_flags(parsed.modules),
            "q5090 header module flags mismatch");
    parsed.tensors = parse_tensors(file, parsed.header, parsed.modules);
    parsed.segments = parse_segments(file, parsed.header);
    validate_tensor_segments(parsed.tensors, parsed.segments);
    parsed.fusion_groups = parse_fusion_groups(file, parsed.header);
    validate_fusion_groups(parsed.tensors, parsed.fusion_groups);
    return parsed;
}

} // namespace qus
