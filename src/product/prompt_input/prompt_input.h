#pragma once

#include "ninfer/types.h"

#include <filesystem>
#include <string>

namespace ninfer::product {

[[nodiscard]] PromptInput prompt_from_text(std::string text, std::optional<bool> enable_thinking);
[[nodiscard]] PromptInput prompt_from_messages(const std::filesystem::path& path,
                                               std::optional<bool> enable_thinking,
                                               bool vision_enabled);

// Make the requested final-response contract visible to the model before preparation.
// Grammar enforcement remains an execution responsibility.
void apply_structured_output_instruction(PromptInput& input,
                                         const StructuredOutputOptions& options);

} // namespace ninfer::product
