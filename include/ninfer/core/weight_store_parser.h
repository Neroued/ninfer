#pragma once

#include "ninfer/core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {

inline constexpr std::uint32_t kQ5090NoLayer = 0xFFFFFFFFU;

struct ParsedQ5090Header {
    std::uint32_t tensor_count              = 0;
    std::uint32_t module_count              = 0;
    std::uint32_t layer_count               = 0;
    std::uint32_t flags                     = 0;
    std::uint32_t segment_count             = 0;
    std::uint64_t module_index_offset       = 0;
    std::uint64_t module_index_bytes        = 0;
    std::uint64_t tensor_index_offset       = 0;
    std::uint64_t tensor_index_bytes        = 0;
    std::uint64_t segment_index_offset      = 0;
    std::uint64_t segment_index_bytes       = 0;
    std::uint64_t fusion_group_index_offset = 0;
    std::uint64_t fusion_group_index_bytes  = 0;
    std::uint64_t string_table_offset       = 0;
    std::uint64_t string_table_bytes        = 0;
    std::uint64_t payload_offset            = 0;
    std::uint64_t payload_bytes             = 0;
    std::uint32_t hidden_size               = 0;
    std::uint32_t intermediate_size         = 0;
    std::uint32_t vocab_size                = 0;
    std::uint32_t num_attention_heads       = 0;
    std::uint32_t num_key_value_heads       = 0;
    std::uint32_t head_dim                  = 0;
    std::uint32_t gdn_key_heads             = 0;
    std::uint32_t gdn_value_heads           = 0;
    std::uint32_t gdn_key_head_dim          = 0;
    std::uint32_t gdn_value_head_dim        = 0;
    std::uint32_t gdn_conv_width            = 0;
    std::uint32_t full_attention_interval   = 0;
    std::uint32_t max_position_embeddings   = 0;
    std::uint32_t fusion_group_count        = 0;
    std::uint32_t format_minor              = 0;
    std::uint32_t tokenizer_record_count    = 0;
    std::uint32_t tokenizer_record_size     = 0;
    std::uint32_t tokenizer_flags           = 0;
    std::uint64_t tokenizer_index_offset    = 0;
    std::uint64_t tokenizer_index_bytes     = 0;
    std::uint64_t tokenizer_data_offset     = 0;
    std::uint64_t tokenizer_data_bytes      = 0;
    std::array<std::uint8_t, 32> sha256_safetensors_index{};
};

struct ParsedQ5090Module {
    ModuleKind module_kind           = ModuleKind::TextCore;
    std::uint32_t module_version     = 0;
    std::uint64_t tensor_index_begin = 0;
    std::uint64_t tensor_index_count = 0;
    std::uint64_t payload_offset     = 0;
    std::uint64_t payload_bytes      = 0;
};

enum class Q5090TokenizerKind : std::uint32_t {
    TokenizerJson    = 1,
    MergesTxt        = 2,
    GenerationConfig = 3,
};

struct ParsedQ5090TokenizerRecord {
    Q5090TokenizerKind kind   = Q5090TokenizerKind::TokenizerJson;
    std::uint32_t encoding    = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t data_bytes  = 0;
    std::uint32_t crc32       = 0;
    std::array<std::uint8_t, 32> sha256{};
};

struct Q5090TokenizerBundle {
    std::string tokenizer_json;
    std::string merges_txt;
    std::string generation_config_json;

    [[nodiscard]] bool empty() const noexcept {
        return tokenizer_json.empty() && merges_txt.empty() && generation_config_json.empty();
    }
};

struct ParsedQ5090Tensor {
    std::string name;
    std::uint32_t name_offset                 = 0;
    std::uint32_t name_len                    = 0;
    std::uint64_t name_hash                   = 0;
    QType qtype                               = QType::Q4G64_F16S;
    QuantLayout layout                        = QuantLayout::RowSplit;
    ModuleKind module_kind                    = ModuleKind::TextCore;
    std::uint32_t ndim                        = 0;
    std::array<std::uint32_t, 4> shape        = {1, 1, 1, 1};
    std::array<std::uint32_t, 4> padded_shape = {1, 1, 1, 1};
    std::uint32_t group_size                  = 0;
    ScaleDType scale_dtype                    = ScaleDType::None;
    std::uint32_t segment_count               = 0;
    std::uint64_t payload_offset              = 0;
    std::uint64_t payload_bytes               = 0;
    std::uint32_t source_layer                = kQ5090NoLayer;
    std::uint32_t source_kind                 = 0;
    std::uint32_t crc32                       = 0;
    std::uint32_t segment_begin               = 0;
    std::uint16_t fusion_group_id             = 0;
    std::uint16_t fusion_index                = 0;
    std::uint64_t nibble_plane_bytes          = 0;
    std::uint64_t high_plane_bytes            = 0;
    std::uint64_t scale_plane_bytes           = 0;
};

struct ParsedQ5090Segment {
    std::string name;
    std::uint32_t source_kind  = 0;
    std::uint32_t source_layer = kQ5090NoLayer;
    std::uint32_t row_begin    = 0;
    std::uint32_t row_count    = 0;
    std::uint32_t name_offset  = 0;
    std::uint32_t name_len     = 0;
    std::uint64_t name_hash    = 0;
};

struct ParsedQ5090FusionGroup {
    std::uint32_t group_id                 = 0;
    std::uint32_t source_layer             = 0;
    std::uint32_t block_count              = 0;
    std::uint32_t shared_input_kind        = 0;
    std::uint64_t first_block_tensor_index = 0;
    std::uint64_t payload_offset           = 0;
    std::uint64_t payload_bytes            = 0;
    std::uint32_t total_n                  = 0;
    std::uint32_t shared_k                 = 0;
};

struct ParsedQ5090File {
    ParsedQ5090Header header;
    std::vector<ParsedQ5090Module> modules;
    std::vector<ParsedQ5090Tensor> tensors;
    std::vector<ParsedQ5090Segment> segments;
    std::vector<ParsedQ5090FusionGroup> fusion_groups;
    std::vector<ParsedQ5090TokenizerRecord> tokenizer_records;
    Q5090TokenizerBundle tokenizer;
};

struct Q5090Progress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
    std::uint64_t min_interval_bytes = 256ULL * 1024ULL * 1024ULL;
};

std::uint64_t q5090_fnv1a64(std::string_view name);
ParsedQ5090Header parse_q5090_header(std::span<const std::byte> header, std::uint64_t file_size);
ParsedQ5090File parse_q5090_catalog(std::span<const std::byte> metadata, std::uint64_t file_size);
ParsedQ5090File parse_q5090_file(std::span<const std::byte> file,
                                 Q5090Progress* progress = nullptr);

} // namespace ninfer
