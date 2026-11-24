#pragma once

#include "ninfer/types.h"
#include "text/json_schema.h"
#include "models/common/frontend/tools.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen::frontend {

// Immutable declarations and parameter codecs shared by the grammar builder and output decoder.
struct ToolCallOutputContract {
    enum class NormalizationPolicy : std::uint8_t {
        InferredTypes,
        DeclaredTypes,
    };

    enum class Encoding : std::uint8_t { BufferedNormalize, RawString, JsonValue };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::InferredTypes;
        text::SchemaTypes types;
        Encoding encoding = Encoding::BufferedNormalize;
    };

    struct Tool : models::frontend::ToolDeclaration {
        std::vector<Parameter> parameters;
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

} // namespace ninfer::models::qwen::frontend
