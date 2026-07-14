#pragma once

#include "ninfer/core/weight_store_parser.h"
#include "ninfer/model/config.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::parity {
namespace detail {

using Json = nlohmann::json;

inline std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open q5090 file: " + path.string()); }
    const std::vector<char> chars{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    std::vector<std::byte> bytes(chars.size());
    std::memcpy(bytes.data(), chars.data(), chars.size());
    return bytes;
}

inline std::string hex_bytes(const std::uint8_t* data, std::size_t n) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < n; ++i) { out << std::setw(2) << static_cast<unsigned>(data[i]); }
    return out.str();
}

inline std::string sha_hex(const std::array<std::uint8_t, 32>& sha) {
    return hex_bytes(sha.data(), sha.size());
}

inline std::uint32_t tag(ModuleKind v) { return static_cast<std::uint32_t>(v); }

inline std::uint32_t tag(QType v) { return static_cast<std::uint32_t>(v); }

inline std::uint32_t tag(QuantLayout v) { return static_cast<std::uint32_t>(v); }

inline std::uint32_t tag(ScaleDType v) { return static_cast<std::uint32_t>(v); }

inline const char* qtype_name(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return "Q4G64_F16S";
    case QType::Q5G64_F16S:
        return "Q5G64_F16S";
    case QType::Q6G64_F16S:
        return "Q6G64_F16S";
    case QType::BF16_CTRL:
        return "BF16_CTRL";
    case QType::FP32_CTRL:
        return "FP32_CTRL";
    case QType::W8G32_F16S:
        return "W8G32_F16S";
    case QType::I32_CTRL:
        return "I32_CTRL";
    }
    return "unknown";
}

inline const char* layout_name(QuantLayout layout) {
    switch (layout) {
    case QuantLayout::RowSplit:
        return "ROW_SPLIT";
    case QuantLayout::Contiguous:
        return "CONTIGUOUS";
    }
    return "unknown";
}

inline const char* fusion_group_name(std::uint32_t group_id) {
    switch (group_id) {
    case 1:
        return "ATTN_IN";
    case 2:
        return "GDN_IN";
    case 3:
        return "MLP_GATEUP";
    default:
        return "NONE";
    }
}

inline std::vector<std::uint32_t> active_shape(const std::array<std::uint32_t, 4>& shape,
                                               std::uint32_t ndim) {
    std::vector<std::uint32_t> out;
    out.reserve(ndim);
    for (std::uint32_t i = 0; i < ndim; ++i) { out.push_back(shape[i]); }
    return out;
}

inline std::uint64_t prod(const std::vector<std::uint32_t>& shape, std::size_t begin = 0) {
    std::uint64_t out = 1;
    for (std::size_t i = begin; i < shape.size(); ++i) { out *= shape[i]; }
    return out;
}

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t align) {
    return ((value + align - 1) / align) * align;
}

inline std::uint16_t read_u16_le(const std::byte* p) {
    return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(p[0])) |
           (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(p[1])) << 8);
}

inline std::uint32_t read_u32_le(const std::byte* p) {
    return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[0])) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[1])) << 8) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[2])) << 16) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[3])) << 24);
}

inline float bits_float(std::uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline float bf16_to_f32(std::uint16_t h) {
    return bits_float(static_cast<std::uint32_t>(h) << 16);
}

inline float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp        = (static_cast<std::uint32_t>(h) >> 10) & 0x1fu;
    std::uint32_t mant       = static_cast<std::uint32_t>(h) & 0x03ffu;
    if (exp == 0) {
        if (mant == 0) { return bits_float(sign); }
        int e = -14;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03ffu;
        return bits_float(sign | (static_cast<std::uint32_t>(e + 127) << 23) | (mant << 13));
    }
    if (exp == 31) { return bits_float(sign | 0x7f800000u | (mant << 13)); }
    exp = exp - 15 + 127;
    return bits_float(sign | (exp << 23) | (mant << 13));
}

inline std::uint32_t group_size(QType qtype) {
    switch (qtype) {
    case QType::W8G32_F16S:
        return 32;
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
        return 64;
    default:
        throw std::runtime_error("qtype is not ROW_SPLIT quantized");
    }
}

inline std::uint32_t bit_width(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return 4;
    case QType::Q5G64_F16S:
        return 5;
    case QType::Q6G64_F16S:
        return 6;
    case QType::W8G32_F16S:
        return 8;
    default:
        throw std::runtime_error("qtype is not ROW_SPLIT quantized");
    }
}

inline std::uint32_t nibble_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
    case QType::W8G32_F16S:
        return 32;
    default:
        throw std::runtime_error("qtype is not ROW_SPLIT quantized");
    }
}

inline std::uint32_t high_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::W8G32_F16S:
        return 0;
    case QType::Q5G64_F16S:
        return 8;
    case QType::Q6G64_F16S:
        return 16;
    default:
        throw std::runtime_error("qtype is not ROW_SPLIT quantized");
    }
}

inline int unpack_code(const std::byte* nibble, const std::byte* high, QType qtype,
                       std::uint32_t lane) {
    const std::uint32_t bits = bit_width(qtype);
    if (bits == 8) {
        const auto u = std::to_integer<std::uint8_t>(nibble[lane]);
        return u >= 128 ? static_cast<int>(u) - 256 : static_cast<int>(u);
    }

    const std::uint8_t low_byte = std::to_integer<std::uint8_t>(nibble[lane >> 1]);
    const std::uint32_t low     = (lane & 1u) ? (low_byte >> 4) : (low_byte & 0x0fu);
    std::uint32_t hi            = 0;
    if (bits == 5) {
        hi = (std::to_integer<std::uint8_t>(high[lane >> 3]) >> (lane & 7u)) & 0x01u;
    } else if (bits == 6) {
        const std::uint32_t bit_pos = lane * 2;
        hi = (std::to_integer<std::uint8_t>(high[bit_pos >> 3]) >> (bit_pos & 7u)) & 0x03u;
    }
    const std::uint32_t u    = low | (hi << 4);
    const std::uint32_t sign = 1u << (bits - 1);
    const std::uint32_t span = 1u << bits;
    return (u & sign) ? static_cast<int>(u) - static_cast<int>(span) : static_cast<int>(u);
}

inline bool is_quantized(QType qtype) {
    return qtype == QType::Q4G64_F16S || qtype == QType::Q5G64_F16S || qtype == QType::Q6G64_F16S ||
           qtype == QType::W8G32_F16S;
}

inline Json row_split_probes(const std::vector<std::byte>& file, const ParsedQ5090Tensor& tensor) {
    const std::uint32_t n                 = tensor.shape[0];
    const std::uint32_t k                 = tensor.shape[1];
    const std::uint32_t group             = group_size(tensor.qtype);
    const std::uint32_t groups            = tensor.padded_shape[1] / group;
    const std::uint32_t nibble_bpr        = nibble_bytes_per_group(tensor.qtype);
    const std::uint32_t high_bpr          = high_bytes_per_group(tensor.qtype);
    const std::uint64_t high_plane_offset = align_up(tensor.nibble_plane_bytes, 256);
    const std::uint64_t scale_plane_offset =
        high_plane_offset + align_up(tensor.high_plane_bytes, 256);
    const std::byte* payload = file.data() + tensor.payload_offset;

    Json probes                                              = Json::array();
    const std::array<std::array<std::uint32_t, 2>, 3> coords = {
        std::array<std::uint32_t, 2>{0, 0},
        std::array<std::uint32_t, 2>{n / 2, k / 2},
        std::array<std::uint32_t, 2>{n - 1, k - 1},
    };
    for (const auto& coord : coords) {
        const std::uint32_t row         = coord[0];
        const std::uint32_t col         = coord[1];
        const std::uint32_t g           = col / group;
        const std::uint32_t lane        = col % group;
        const std::uint64_t group_index = static_cast<std::uint64_t>(row) * groups + g;
        const std::byte* nibble         = payload + group_index * nibble_bpr;
        const std::byte* high =
            high_bpr == 0 ? nullptr : payload + high_plane_offset + group_index * high_bpr;
        const std::byte* scale_ptr = payload + scale_plane_offset + group_index * 2;
        const int q                = unpack_code(nibble, high, tensor.qtype, lane);
        const float scale          = f16_to_f32(read_u16_le(scale_ptr));
        probes.push_back(Json{{"row", row},
                              {"col", col},
                              {"scale", scale},
                              {"q", q},
                              {"value", scale * static_cast<float>(q)}});
    }
    return probes;
}

inline Json contiguous_probes(const std::vector<std::byte>& file, const ParsedQ5090Tensor& tensor) {
    const std::vector<std::uint32_t> shape = active_shape(tensor.shape, tensor.ndim);
    const std::uint64_t numel              = prod(shape);
    Json probes                            = Json::array();
    if (numel == 0) { return probes; }

    const std::byte* payload                   = file.data() + tensor.payload_offset;
    const std::uint32_t elem_bytes             = tensor.qtype == QType::BF16_CTRL ? 2u : 4u;
    const std::array<std::uint64_t, 3> indexes = {0, numel / 2, numel - 1};
    for (std::uint64_t index : indexes) {
        std::uint64_t row = index;
        std::uint64_t col = 0;
        if (shape.size() != 1) {
            const std::uint64_t cols = prod(shape, 1);
            row                      = index / cols;
            col                      = index % cols;
        }
        const std::byte* ptr = payload + index * elem_bytes;
        float value          = 0.0f;
        if (tensor.qtype == QType::BF16_CTRL) {
            value = bf16_to_f32(read_u16_le(ptr));
        } else if (tensor.qtype == QType::I32_CTRL) {
            const std::uint32_t raw = read_u32_le(ptr);
            std::int32_t integer    = 0;
            std::memcpy(&integer, &raw, sizeof(integer));
            value = static_cast<float>(integer);
        } else {
            value = bits_float(read_u32_le(ptr));
        }
        probes.push_back(Json{{"row", row}, {"col", col}, {"value", value}});
    }
    return probes;
}

inline Json header_json(const ParsedQ5090Header& h) {
    static constexpr std::array<std::uint8_t, 16> kMagic = {
        0x51, 0x35, 0x30, 0x39, 0x30, 0x4d, 0x49, 0x58,
        0x45, 0x44, 0x56, 0x34, 0x00, 0x00, 0x00, 0x00,
    };
    return Json{
        {"magic", hex_bytes(kMagic.data(), kMagic.size())},
        {"version", 4},
        {"endian", 0x01020304},
        {"header_size", 4096},
        {"tensor_count", h.tensor_count},
        {"module_count", h.module_count},
        {"layer_count", h.layer_count},
        {"flags", h.flags},
        {"segment_count", h.segment_count},
        {"module_index_offset", h.module_index_offset},
        {"module_index_bytes", h.module_index_bytes},
        {"tensor_index_offset", h.tensor_index_offset},
        {"tensor_index_bytes", h.tensor_index_bytes},
        {"string_table_offset", h.string_table_offset},
        {"string_table_bytes", h.string_table_bytes},
        {"payload_offset", h.payload_offset},
        {"payload_bytes", h.payload_bytes},
        {"hidden_size", h.hidden_size},
        {"intermediate_size", h.intermediate_size},
        {"vocab_size", h.vocab_size},
        {"num_attention_heads", h.num_attention_heads},
        {"num_key_value_heads", h.num_key_value_heads},
        {"head_dim", h.head_dim},
        {"gdn_key_heads", h.gdn_key_heads},
        {"gdn_value_heads", h.gdn_value_heads},
        {"gdn_key_head_dim", h.gdn_key_head_dim},
        {"gdn_value_head_dim", h.gdn_value_head_dim},
        {"gdn_conv_width", h.gdn_conv_width},
        {"full_attention_interval", h.full_attention_interval},
        {"max_position_embeddings", h.max_position_embeddings},
        {"fusion_group_count", h.fusion_group_count},
        {"sha256_safetensors_index", sha_hex(h.sha256_safetensors_index)},
        {"segment_index_offset", h.segment_index_offset},
        {"segment_index_bytes", h.segment_index_bytes},
        {"fusion_group_index_offset", h.fusion_group_index_offset},
        {"fusion_group_index_bytes", h.fusion_group_index_bytes},
        {"format_minor", h.format_minor},
        {"tokenizer_record_count", h.tokenizer_record_count},
        {"tokenizer_record_size", h.tokenizer_record_size},
        {"tokenizer_flags", h.tokenizer_flags},
        {"tokenizer_index_offset", h.tokenizer_index_offset},
        {"tokenizer_index_bytes", h.tokenizer_index_bytes},
        {"tokenizer_data_offset", h.tokenizer_data_offset},
        {"tokenizer_data_bytes", h.tokenizer_data_bytes},
    };
}

inline Json module_json(const ParsedQ5090Module& module) {
    return Json{{"module_kind", tag(module.module_kind)},
                {"module_version", module.module_version},
                {"tensor_index_begin", module.tensor_index_begin},
                {"tensor_index_count", module.tensor_index_count},
                {"payload_offset", module.payload_offset},
                {"payload_bytes", module.payload_bytes},
                {"reserved0", 0},
                {"reserved1", 0}};
}

inline const char* tokenizer_kind_name(Q5090TokenizerKind kind) {
    switch (kind) {
    case Q5090TokenizerKind::TokenizerJson:
        return "TOKENIZER_JSON";
    case Q5090TokenizerKind::MergesTxt:
        return "MERGES_TXT";
    case Q5090TokenizerKind::GenerationConfig:
        return "GENERATION_CONFIG_JSON";
    }
    return "unknown";
}

inline std::string hex_u32(std::uint32_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(8) << value;
    return out.str();
}

inline Json tokenizer_json(const ParsedQ5090TokenizerRecord& record) {
    return Json{{"kind", tokenizer_kind_name(record.kind)}, {"encoding", record.encoding},
                {"data_offset", record.data_offset},        {"data_bytes", record.data_bytes},
                {"crc32", hex_u32(record.crc32)},           {"sha256", sha_hex(record.sha256)}};
}

inline Json segment_json(const ParsedQ5090Segment& segment, std::uint64_t index) {
    return Json{{"segment_index", index},
                {"name", segment.name},
                {"source_kind", segment.source_kind},
                {"source_layer", segment.source_layer},
                {"row_begin", segment.row_begin},
                {"row_count", segment.row_count}};
}

inline Json block_json(const std::vector<std::byte>& file, const ParsedQ5090File& parsed,
                       const ParsedQ5090Tensor& tensor, std::uint64_t index) {
    Json segments = Json::array();
    for (std::uint32_t j = 0; j < tensor.segment_count; ++j) {
        const std::uint64_t seg_index = tensor.segment_begin + j;
        segments.push_back(
            segment_json(parsed.segments.at(static_cast<std::size_t>(seg_index)), seg_index));
    }

    const Json probes = is_quantized(tensor.qtype) ? row_split_probes(file, tensor)
                                                   : contiguous_probes(file, tensor);
    return Json{{"block_index", index},
                {"name", tensor.name},
                {"source_kind", tensor.source_kind},
                {"source_layer", tensor.source_layer},
                {"qtype", qtype_name(tensor.qtype)},
                {"layout", layout_name(tensor.layout)},
                {"shape", active_shape(tensor.shape, tensor.ndim)},
                {"padded_shape", active_shape(tensor.padded_shape, tensor.ndim)},
                {"payload_offset", tensor.payload_offset},
                {"payload_bytes", tensor.payload_bytes},
                {"nibble_plane_bytes", tensor.nibble_plane_bytes},
                {"high_plane_bytes", tensor.high_plane_bytes},
                {"scale_plane_bytes", tensor.scale_plane_bytes},
                {"crc32", tensor.crc32},
                {"fusion_group_id", tensor.fusion_group_id},
                {"fusion_index", tensor.fusion_index},
                {"segments", segments},
                {"dequant_probes", probes}};
}

inline Json fusion_json(const ParsedQ5090File& parsed, const ParsedQ5090FusionGroup& fusion) {
    Json members            = Json::array();
    const std::uint64_t end = std::min<std::uint64_t>(
        parsed.tensors.size(), fusion.first_block_tensor_index + fusion.block_count);
    for (std::uint64_t i = fusion.first_block_tensor_index; i < end; ++i) {
        members.push_back(parsed.tensors.at(static_cast<std::size_t>(i)).name);
    }
    return Json{{"group_id", fusion_group_name(fusion.group_id)},
                {"source_layer", fusion.source_layer},
                {"block_count", fusion.block_count},
                {"shared_input_kind", fusion.shared_input_kind},
                {"first_block_tensor_index", fusion.first_block_tensor_index},
                {"payload_offset", fusion.payload_offset},
                {"payload_bytes", fusion.payload_bytes},
                {"total_n", fusion.total_n},
                {"shared_k", fusion.shared_k},
                {"members", members}};
}

inline Json structural_dump(const std::filesystem::path& path) {
    const std::vector<std::byte> file = read_file(path);
    const ParsedQ5090File parsed      = parse_q5090_file(file);

    Json modules = Json::array();
    for (const ParsedQ5090Module& module : parsed.modules) {
        modules.push_back(module_json(module));
    }

    Json blocks = Json::array();
    for (std::size_t i = 0; i < parsed.tensors.size(); ++i) {
        blocks.push_back(block_json(file, parsed, parsed.tensors[i], i));
    }

    Json fusions = Json::array();
    for (const ParsedQ5090FusionGroup& fusion : parsed.fusion_groups) {
        fusions.push_back(fusion_json(parsed, fusion));
    }

    Json tokenizer = Json::array();
    for (const ParsedQ5090TokenizerRecord& record : parsed.tokenizer_records) {
        tokenizer.push_back(tokenizer_json(record));
    }

    return Json{{"format", "q5090_w4g64_mixed_v4_2"},
                {"file", path.string()},
                {"header", header_json(parsed.header)},
                {"modules", modules},
                {"blocks", blocks},
                {"fusion_groups", fusions},
                {"tokenizer", tokenizer}};
}

inline void write_json(const std::filesystem::path& path, const Json& value) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) { std::filesystem::create_directories(parent); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open output: " + path.string()); }
    out << value.dump(2) << '\n';
}

} // namespace detail

inline int structural_dump_main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <weights.qus> <out.json>\n";
        return 2;
    }
    try {
        detail::write_json(argv[2], detail::structural_dump(argv[1]));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}

} // namespace ninfer::parity
