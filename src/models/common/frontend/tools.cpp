#include "models/common/frontend/tools.h"
#include "text/json_input.h"
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

namespace ninfer::models::frontend {
std::vector<ToolDeclaration> parse_tool_declarations(std::span<const std::string> tool_jsons) {
    using Json = nlohmann::ordered_json;
    std::vector<ToolDeclaration> result;
    result.reserve(tool_jsons.size());
    std::set<std::string> names;
    for (std::size_t i = 0; i < tool_jsons.size(); ++i) {
        try {
            const auto parsed    = text::parse_json_numbers(tool_jsons[i]);
            const auto& function = parsed.value.at("function");
            ToolDeclaration tool;
            tool.name = function.at("name").get<std::string>();
            if (tool.name.empty() || !names.insert(tool.name).second)
                throw RequestError(RequestErrorKind::InvalidToolConstraint,
                                   "tool names must be nonempty and unique",
                                   "/" + std::to_string(i));
            tool.strict = function.value("strict", false);
            if (tool.strict) {
                if (const auto pointer =
                        text::inexact_schema_number(parsed, "/function/parameters"))
                    throw RequestError(
                        RequestErrorKind::UnsupportedJsonSchema,
                        "numeric schema value cannot be preserved by the JSON number "
                        "representation",
                        "/" + std::to_string(i) + "/parameters" +
                            pointer->substr(std::string_view("/function/parameters").size()),
                        RequestErrorSource::Tools);
            }
            tool.schema_json =
                function
                    .value("parameters", Json{{"type", "object"}, {"properties", Json::object()}})
                    .dump();
            tool.declaration_index = i;
            result.push_back(std::move(tool));
        } catch (const Json::exception& error) {
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               std::string("invalid tool declaration: ") + error.what(),
                               "/" + std::to_string(i));
        }
    }
    return result;
}
} // namespace ninfer::models::frontend
