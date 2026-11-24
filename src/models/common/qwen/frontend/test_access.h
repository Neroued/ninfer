#pragma once

#include "models/common/qwen/frontend/frontend.h"
#include "models/common/qwen/frontend/resources.h"
#include "models/common/qwen/frontend/prepared_prompt.h"

namespace ninfer::models::qwen {

class FrontendTestAccess {
public:
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace ninfer::models::qwen
