#pragma once

#include "models/common/frontend/resources.h"
#include "text/jinja.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace ninfer::models::frontend {

struct ChatTemplate {
    text::JinjaTemplate program;
    nlohmann::ordered_json special_tokens;
};

ChatTemplate load_chat_template(const TextResources& resources,
                                const std::filesystem::path& override_path);
nlohmann::ordered_json template_context(std::string_view kwargs,
                                        const nlohmann::ordered_json& special_tokens);
nlohmann::ordered_json parse_tool_history_arguments(std::string_view arguments,
                                                    std::string_view name,
                                                    std::size_t message_index,
                                                    std::size_t call_index);
void merge_template_option(nlohmann::ordered_json& context, std::string_view key,
                           const nlohmann::ordered_json& value);
std::optional<text::ByteSpan> unique_output_region(const text::TemplateOutput& rendered,
                                                   std::uint32_t tag);
std::optional<std::size_t> source_boundary(const text::TemplateOutput& rendered, std::uint32_t tag,
                                           std::size_t offset);
bool same_prefix(const text::TemplateOutput& lhs, const text::TemplateOutput& rhs,
                 std::size_t size);

} // namespace ninfer::models::frontend
