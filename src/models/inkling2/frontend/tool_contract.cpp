#include "models/inkling2/frontend/tool_contract.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace ninfer::models::inkling2 {

bool valid_tool_name(std::string_view name, std::size_t maximum) {
    if (name.empty() || name.size() > maximum) return false;
    for (const unsigned char c : name) {
        if (c < 0x20U || c == 0x7fU || c == '<' || c == '>' || c == '|' || std::isspace(c) != 0) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons) {
    if (tool_jsons.empty()) return nullptr;
    auto contract   = std::make_shared<ToolCallOutputContract>();
    contract->tools = models::frontend::parse_tool_declarations(tool_jsons);
    return contract;
}

std::shared_ptr<const ToolCallOutputContract>
select_tool_call_contract(const std::shared_ptr<const ToolCallOutputContract>& declarations,
                          const ToolChoice& choice) {
    if (!declarations || declarations->tools.empty()) {
        if (choice.mode == ToolChoiceMode::Required || choice.allowed_names) {
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "tool choice requires declared tools", {},
                               RequestErrorSource::Tools);
        }
        return nullptr;
    }
    const auto selection = models::frontend::select_tools<models::frontend::ToolDeclaration>(
        declarations->tools, choice);
    auto result         = std::make_shared<ToolCallOutputContract>();
    result->required    = selection.required;
    result->parallel    = selection.parallel;
    result->constrained = selection.constrained;
    result->tools.reserve(selection.indices.size());
    for (const std::size_t index : selection.indices) {
        result->tools.push_back(declarations->tools[index]);
    }
    return result;
}

} // namespace ninfer::models::inkling2
