#pragma once

#include "models/common/qwen/frontend/frontend.h"
#include "models/common/qwen/frontend/resources.h"
#include "models/common/qwen/frontend/prepared_prompt.h"

namespace ninfer::models::qwen {

class FrontendTestAccess {
public:
    [[nodiscard]] static PreparedPrompt from_data(PreparedPromptData data) {
        return PreparedPrompt(std::make_unique<PreparedPromptData>(std::move(data)));
    }

    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace ninfer::models::qwen
