#pragma once

#include "models/common/frontend/tools.h"
#include "ninfer/types.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2 {

bool valid_tool_name(std::string_view name, std::size_t maximum);

struct ToolCallOutputContract {
    std::vector<models::frontend::ToolDeclaration> tools;
    bool constrained = false;
    bool required    = false;
    bool parallel    = true;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons);

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
select_tool_call_contract(const std::shared_ptr<const ToolCallOutputContract>& declarations,
                          const ToolChoice& choice);

} // namespace ninfer::models::inkling2
