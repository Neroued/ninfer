#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Immutable declarations and parameter codecs shared by the grammar builder and output decoder.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        InferredTypes,
        DeclaredTypes,
    };

    enum class Encoding : std::uint8_t { NonStrict, RawString, Json };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::InferredTypes;
        TypeSet types;
        Encoding encoding = Encoding::NonStrict;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        std::string schema_json;
        std::size_t declaration_index = 0;
        bool strict                   = false;
    };

    std::vector<Tool> tools;
    bool constrained = false;
    bool required    = false;
    bool parallel    = true;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons);

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
select_tool_call_contract(const std::shared_ptr<const ToolCallOutputContract>& declarations,
                          const ToolChoice& choice);

} // namespace ninfer::models::qwen3_5::frontend
