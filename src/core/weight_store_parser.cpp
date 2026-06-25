#include "qus/core/weight_store_parser.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace qus {
namespace {

constexpr std::array<std::byte, 16> kMagic = {
    std::byte{0x51}, std::byte{0x35}, std::byte{0x30}, std::byte{0x39},
    std::byte{0x30}, std::byte{0x4D}, std::byte{0x49}, std::byte{0x58},
    std::byte{0x45}, std::byte{0x44}, std::byte{0x56}, std::byte{0x31},
    std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
};

constexpr std::uint32_t kVersion          = 1;
constexpr std::uint32_t kEndianTag        = 0x01020304U;
constexpr std::uint32_t kHeaderSize       = 4096;
constexpr std::uint32_t kModuleRecordSize = 64;
constexpr std::uint32_t kTensorEntrySize  = 128;
constexpr std::uint64_t kPayloadAlign     = 256;
constexpr std::uint64_t kRegionAlign      = 4096;

struct Range {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

class ProgressReporter {
public:
    explicit ProgressReporter(Q5090Progress* progress) : progress_(progress) {}

    void report(std::string_view phase, std::uint64_t done, std::uint64_t total,
                bool force = false) {
        if (progress_ == nullptr || !progress_->callback) { return; }
        if (!force && done < last_done_ + progress_->min_interval_bytes && done < total) { return; }
        last_done_ = done;
        progress_->callback(phase, done, total);
    }

private:
    Q5090Progress* progress_ = nullptr;
    std::uint64_t last_done_ = 0;
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
    require(
        all_zero(file.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(size))),
        label);
}

const std::array<std::uint32_t, 256>& crc32_table() {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> out{};
        for (std::uint32_t i = 0; i < out.size(); ++i) {
            std::uint32_t crc = i;
            for (int bit = 0; bit < 8; ++bit) {
                const std::uint32_t mask = 0U - (crc & 1U);
                crc                      = (crc >> 1) ^ (0xEDB88320U & mask);
            }
            out[i] = crc;
        }
        return out;
    }();
    return table;
}

template <typename T>
void validate_expected(std::uint32_t actual, const std::optional<T>& expected, const char* name) {
    if (expected.has_value() && actual != static_cast<std::uint32_t>(*expected)) {
        throw std::runtime_error(std::string("q5090 model metadata mismatch: ") + name);
    }
}

QType qtype_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0:
        return QType::Q4G64_F16S;
    case 1:
        return QType::Q5G64_F16S;
    case 2:
        return QType::Q6G64_F16S;
    case 3:
        return QType::W8G128_F16S;
    case 4:
        return QType::BF16_CTRL;
    case 5:
        return QType::FP32_CTRL;
    default:
        parse_error("q5090 invalid qtype tag");
    }
}

QuantLayout layout_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0:
        return QuantLayout::TileN64K64;
    case 1:
        return QuantLayout::TileN64K128;
    case 2:
        return QuantLayout::RowGroupedG64;
    case 3:
        return QuantLayout::Contiguous;
    default:
        parse_error("q5090 invalid layout tag");
    }
}

ModuleKind module_from_tag(std::uint32_t tag) {
    switch (tag) {
    case 0:
        return ModuleKind::TextCore;
    case 1:
        return ModuleKind::MtpDraft;
    case 2:
        return ModuleKind::VisionEncoder;
    default:
        parse_error("q5090 invalid module tag");
    }
}

ScaleDType scale_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0:
        return ScaleDType::None;
    case 1:
        return ScaleDType::FP16;
    default:
        parse_error("q5090 invalid scale dtype tag");
    }
}

LoadPolicy load_policy_from_tag(std::uint32_t tag) {
    switch (tag) {
    case 0:
        return LoadPolicy::Resident;
    case 1:
        return LoadPolicy::LazyGpu;
    case 2:
        return LoadPolicy::CpuPinnedThenGpu;
    default:
        parse_error("q5090 invalid load policy tag");
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

std::uint64_t element_size(QType qtype) {
    switch (qtype) {
    case QType::BF16_CTRL:
        return 2;
    case QType::FP32_CTRL:
        return 4;
    default:
        parse_error("q5090 qtype has no contiguous element size");
    }
}

std::uint64_t tile_bytes(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return 2176;
    case QType::Q5G64_F16S:
        return 2688;
    case QType::Q6G64_F16S:
        return 3200;
    case QType::W8G128_F16S:
        return 8320;
    default:
        parse_error("q5090 qtype has no tile bytes");
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
    case QuantLayout::TileN64K64: {
        require(tensor.ndim == 2, "q5090 TILE_N64_K64 tensor must be 2D");
        require(tensor.qtype == QType::Q4G64_F16S || tensor.qtype == QType::Q5G64_F16S ||
                    tensor.qtype == QType::Q6G64_F16S,
                "q5090 TILE_N64_K64 qtype mismatch");
        require(tensor.group_size == 64, "q5090 TILE_N64_K64 group mismatch");
        require(tensor.scale_dtype == ScaleDType::FP16, "q5090 TILE_N64_K64 scale mismatch");
        require(tensor.padded_shape[0] == align_up(tensor.shape[0], 64),
                "q5090 TILE_N64_K64 padded N mismatch");
        require(tensor.padded_shape[1] == align_up(tensor.shape[1], 64),
                "q5090 TILE_N64_K64 padded K mismatch");
        return checked_mul(checked_mul(tensor.padded_shape[0] / 64, tensor.padded_shape[1] / 64),
                           tile_bytes(tensor.qtype));
    }
    case QuantLayout::TileN64K128: {
        require(tensor.ndim == 2, "q5090 TILE_N64_K128 tensor must be 2D");
        require(tensor.qtype == QType::W8G128_F16S, "q5090 TILE_N64_K128 qtype mismatch");
        require(tensor.group_size == 128, "q5090 TILE_N64_K128 group mismatch");
        require(tensor.scale_dtype == ScaleDType::FP16, "q5090 TILE_N64_K128 scale mismatch");
        require(tensor.padded_shape[0] == align_up(tensor.shape[0], 64),
                "q5090 TILE_N64_K128 padded N mismatch");
        require(tensor.padded_shape[1] == align_up(tensor.shape[1], 128),
                "q5090 TILE_N64_K128 padded K mismatch");
        return checked_mul(checked_mul(tensor.padded_shape[0] / 64, tensor.padded_shape[1] / 128),
                           tile_bytes(tensor.qtype));
    }
    case QuantLayout::RowGroupedG64:
        require(tensor.ndim == 2, "q5090 ROW_GROUPED_G64 tensor must be 2D");
        require(tensor.qtype == QType::Q6G64_F16S, "q5090 ROW_GROUPED_G64 qtype mismatch");
        require(tensor.group_size == 64, "q5090 ROW_GROUPED_G64 group mismatch");
        require(tensor.scale_dtype == ScaleDType::FP16, "q5090 ROW_GROUPED_G64 scale mismatch");
        require(tensor.padded_shape[0] == tensor.shape[0],
                "q5090 ROW_GROUPED_G64 padded N mismatch");
        require(tensor.padded_shape[1] == align_up(tensor.shape[1], 64),
                "q5090 ROW_GROUPED_G64 padded K mismatch");
        return checked_mul(checked_mul(tensor.shape[0], tensor.padded_shape[1] / 64), 50);
    case QuantLayout::Contiguous:
        require(tensor.qtype == QType::BF16_CTRL || tensor.qtype == QType::FP32_CTRL,
                "q5090 CONTIGUOUS qtype mismatch");
        require(tensor.group_size == 0, "q5090 CONTIGUOUS group mismatch");
        require(tensor.scale_dtype == ScaleDType::None, "q5090 CONTIGUOUS scale mismatch");
        require(tensor.padded_shape == tensor.shape, "q5090 CONTIGUOUS padded shape mismatch");
        return checked_mul(numel(tensor.shape, tensor.ndim), element_size(tensor.qtype));
    default:
        parse_error("q5090 invalid layout");
    }
}

ParsedQ5090Header parse_header(std::span<const std::byte> file, const Q5090Expectations& expected) {
    require(file.size() >= kHeaderSize, "q5090 file too small for header");
    require(std::equal(kMagic.begin(), kMagic.end(), file.begin()), "q5090 bad magic");

    require(read_u32(file, 16) == kVersion, "q5090 bad version");
    require(read_u32(file, 20) == kEndianTag, "q5090 endian mismatch");
    require(read_u32(file, 24) == kHeaderSize, "q5090 bad header size");
    require(read_u32(file, 44) == 0, "q5090 reserved header field nonzero");
    require(read_u32(file, 164) == 0, "q5090 reserved header field nonzero");
    require(all_zero(file.subspan(200, kHeaderSize - 200)), "q5090 header padding nonzero");

    ParsedQ5090Header h;
    h.tensor_count            = read_u32(file, 28);
    h.module_count            = read_u32(file, 32);
    h.layer_count             = read_u32(file, 36);
    h.flags                   = read_u32(file, 40);
    h.module_index_offset     = read_u64(file, 48);
    h.module_index_bytes      = read_u64(file, 56);
    h.tensor_index_offset     = read_u64(file, 64);
    h.tensor_index_bytes      = read_u64(file, 72);
    h.string_table_offset     = read_u64(file, 80);
    h.string_table_bytes      = read_u64(file, 88);
    h.payload_offset          = read_u64(file, 96);
    h.payload_bytes           = read_u64(file, 104);
    h.hidden_size             = read_u32(file, 112);
    h.intermediate_size       = read_u32(file, 116);
    h.vocab_size              = read_u32(file, 120);
    h.num_attention_heads     = read_u32(file, 124);
    h.num_key_value_heads     = read_u32(file, 128);
    h.head_dim                = read_u32(file, 132);
    h.gdn_key_heads           = read_u32(file, 136);
    h.gdn_value_heads         = read_u32(file, 140);
    h.gdn_key_head_dim        = read_u32(file, 144);
    h.gdn_value_head_dim      = read_u32(file, 148);
    h.gdn_conv_width          = read_u32(file, 152);
    h.full_attention_interval = read_u32(file, 156);
    h.max_position_embeddings = read_u32(file, 160);
    for (std::size_t i = 0; i < h.sha256_safetensors_index.size(); ++i) {
        h.sha256_safetensors_index[i] = std::to_integer<std::uint8_t>(file[168 + i]);
    }

    require(h.module_count >= 1 && h.module_count <= 3, "q5090 invalid module count");
    require(h.tensor_count > 0, "q5090 tensor count is zero");
    require(h.layer_count == expected.layer_count.value_or(64), "q5090 layer count mismatch");
    require((h.flags & ~0x0FU) == 0, "q5090 unknown header flags");
    require(h.module_index_offset == kHeaderSize, "q5090 bad module index offset");
    require(h.module_index_bytes == checked_mul(h.module_count, kModuleRecordSize),
            "q5090 bad module index bytes");
    require(h.tensor_index_bytes == checked_mul(h.tensor_count, kTensorEntrySize),
            "q5090 bad tensor index bytes");
    require(h.tensor_index_offset == checked_add(h.module_index_offset, h.module_index_bytes),
            "q5090 bad tensor index offset");
    require(h.string_table_offset == checked_add(h.tensor_index_offset, h.tensor_index_bytes),
            "q5090 bad string table offset");
    const std::uint64_t string_end = checked_add(h.string_table_offset, h.string_table_bytes);
    require(h.payload_offset >= string_end, "q5090 payload begins before string table end");
    require(h.payload_offset % kRegionAlign == 0, "q5090 payload region is not aligned");
    require(checked_add(h.payload_offset, h.payload_bytes) == file.size(),
            "q5090 payload bytes do not match file size");
    require_range(file, h.module_index_offset, h.module_index_bytes, "module index");
    require_range(file, h.tensor_index_offset, h.tensor_index_bytes, "tensor index");
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
        require(m.module_version == 1, "q5090 bad module version");
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
            "q5090 tensor name is not NUL-terminated");
    const auto* chars = reinterpret_cast<const char*>(file.data() + absolute);
    return std::string(chars, chars + name_len);
}

std::vector<ParsedQ5090Tensor> parse_tensors(std::span<const std::byte> file,
                                             const ParsedQ5090Header& h,
                                             const std::vector<ParsedQ5090Module>& modules,
                                             Q5090Progress* progress) {
    ProgressReporter reporter(progress);
    std::vector<ParsedQ5090Tensor> tensors;
    tensors.reserve(h.tensor_count);
    std::vector<Range> ranges;
    ranges.reserve(h.tensor_count);
    std::set<std::string> seen_names;
    std::set<std::tuple<std::uint16_t, std::uint32_t, std::uint32_t>> seen_sources;
    std::uint64_t crc_done             = 0;
    std::uint64_t previous_payload_end = h.payload_offset;
    reporter.report("parse crc", 0, h.payload_bytes, true);
    for (std::uint32_t i = 0; i < h.tensor_count; ++i) {
        const std::uint64_t off = h.tensor_index_offset + checked_mul(i, kTensorEntrySize);
        ParsedQ5090Tensor t;
        t.name_offset = read_u32(file, off);
        t.name_len    = read_u32(file, off + 4);
        t.name_hash   = read_u64(file, off + 8);
        t.qtype       = qtype_from_tag(read_u16(file, off + 16));
        t.layout      = layout_from_tag(read_u16(file, off + 18));
        t.module_kind = module_from_tag(read_u16(file, off + 20));
        t.ndim        = read_u16(file, off + 22);
        for (int d = 0; d < 4; ++d) {
            t.shape[d]        = read_u32(file, off + 24 + d * 4);
            t.padded_shape[d] = read_u32(file, off + 40 + d * 4);
        }
        t.group_size  = read_u32(file, off + 56);
        t.scale_dtype = scale_from_tag(read_u16(file, off + 60));
        require(read_u16(file, off + 62) == 0, "q5090 tensor reserved field nonzero");
        t.payload_offset = read_u64(file, off + 64);
        t.payload_bytes  = read_u64(file, off + 72);
        t.source_layer   = read_u32(file, off + 80);
        t.source_kind    = read_u32(file, off + 84);
        t.crc32          = read_u32(file, off + 88);
        require(read_u32(file, off + 92) == 0, "q5090 tensor reserved field nonzero");
        require(all_zero(file.subspan(static_cast<std::size_t>(off + 96), 32)),
                "q5090 tensor reserved bytes nonzero");

        const ParsedQ5090Module& module = module_for_tensor_index(modules, i);
        require(t.module_kind == module.module_kind,
                "q5090 tensor module does not match module range");
        t.name = read_name(file, h, t.name_offset, t.name_len);
        require(q5090_fnv1a64(t.name) == t.name_hash, "q5090 tensor name hash mismatch");
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
        require(valid_source_kind(t.source_kind), "q5090 invalid source kind");
        require(seen_names.insert(t.name).second, "q5090 duplicate tensor name");
        if (t.source_kind != static_cast<std::uint32_t>(SourceKind::Other)) {
            const auto source_key = std::make_tuple(static_cast<std::uint16_t>(t.module_kind),
                                                    t.source_kind, t.source_layer);
            require(seen_sources.insert(source_key).second, "q5090 duplicate tensor source id");
        }
        require(expected_payload_bytes(t) == t.payload_bytes, "q5090 tensor payload byte mismatch");
        require(q5090_crc32(file.subspan(static_cast<std::size_t>(t.payload_offset),
                                         static_cast<std::size_t>(t.payload_bytes))) == t.crc32,
                "q5090 tensor crc mismatch");
        crc_done = std::min(h.payload_bytes, checked_add(crc_done, t.payload_bytes));
        reporter.report("parse crc", crc_done, h.payload_bytes);
        ranges.push_back(Range{t.payload_offset, payload_end});
        tensors.push_back(std::move(t));
    }
    require_zero_range(file, previous_payload_end,
                       checked_add(h.payload_offset, h.payload_bytes) - previous_payload_end,
                       "q5090 payload padding nonzero");
    reporter.report("parse crc", h.payload_bytes, h.payload_bytes, true);

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

} // namespace

std::uint64_t q5090_fnv1a64(std::string_view name) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (unsigned char b : name) {
        h ^= b;
        h *= 0x100000001B3ULL;
    }
    return h;
}

std::uint32_t q5090_crc32(std::span<const std::byte> bytes) {
    const auto& table = crc32_table();
    std::uint32_t crc = 0xFFFFFFFFU;
    for (std::byte byte : bytes) {
        const auto value = static_cast<std::uint32_t>(std::to_integer<unsigned char>(byte));
        crc              = table[(crc ^ value) & 0xFFU] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFU;
}

ParsedQ5090File parse_q5090_file(std::span<const std::byte> file, const Q5090Expectations& expected,
                                 Q5090Progress* progress) {
    ParsedQ5090File parsed;
    parsed.header  = parse_header(file, expected);
    parsed.modules = parse_modules(file, parsed.header);
    require((parsed.header.flags & 0x07U) == module_flags(parsed.modules),
            "q5090 header module flags mismatch");
    parsed.tensors = parse_tensors(file, parsed.header, parsed.modules, progress);
    return parsed;
}

} // namespace qus
