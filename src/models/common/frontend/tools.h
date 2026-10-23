#pragma once

#include "ninfer/types.h"
#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::frontend {

struct ToolDeclaration {
    std::string name;
    std::string schema_json;
    std::size_t declaration_index = 0;
    bool strict                   = false;
};

std::vector<ToolDeclaration> parse_tool_declarations(std::span<const std::string> tool_jsons);

struct ToolSelection {
    std::vector<std::size_t> indices;
    bool required    = false;
    bool parallel    = true;
    bool constrained = false;
};

// Format contracts extend ToolDeclaration; selection depends only on the declarations.
template <class Tool>
ToolSelection select_tools(std::span<const Tool> tools, const ToolChoice& choice) {
    if (choice.allowed_names)
        for (const auto& name : *choice.allowed_names)
            if (std::none_of(tools.begin(), tools.end(),
                             [&](const auto& t) { return t.name == name; }))
                throw RequestError(RequestErrorKind::InvalidToolConstraint,
                                   "tool choice refers to undeclared tool: " + name);
    ToolSelection result;
    result.required = choice.mode == ToolChoiceMode::Required;
    result.parallel = choice.parallel;
    if (choice.mode != ToolChoiceMode::None)
        for (std::size_t i = 0; i < tools.size(); ++i)
            if (!choice.allowed_names ||
                std::find(choice.allowed_names->begin(), choice.allowed_names->end(),
                          tools[i].name) != choice.allowed_names->end())
                result.indices.push_back(i);
    if (result.indices.empty()) {
        if (result.required)
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "required tool choice has no callable tools");
        result.constrained = !tools.empty();
    } else {
        result.constrained = result.required || !result.parallel ||
                             choice.allowed_names.has_value() ||
                             choice.constraints == ToolConstraintMode::Basic ||
                             std::any_of(result.indices.begin(), result.indices.end(),
                                         [&](auto i) { return tools[i].strict; });
    }
    return result;
}

} // namespace ninfer::models::frontend
