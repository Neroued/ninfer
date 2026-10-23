#include "models/common/frontend/chat_template.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace ninfer::models::frontend {
using Json = nlohmann::ordered_json;

ChatTemplate load_chat_template(const TextResources& resources,
                                const std::filesystem::path& override_path) {
    nlohmann::ordered_json tokens = nlohmann::ordered_json::object();
    const auto config             = nlohmann::ordered_json::parse(resources.tokenizer_config_json);
    for (const char* key : {"bos_token", "eos_token", "pad_token", "unk_token", "sep_token",
                            "cls_token", "mask_token"}) {
        if (!config.contains(key) || config[key].is_null()) continue;
        const auto& value = config[key];
        if (value.is_string())
            tokens[key] = value;
        else if (value.is_object() && value.contains("content"))
            tokens[key] = value.at("content");
    }
    if (config.contains("additional_special_tokens"))
        tokens["additional_special_tokens"] = config["additional_special_tokens"];
    std::string source(resources.chat_template_jinja);
    std::string name = "artifact:chat_template.jinja";
    if (!override_path.empty()) {
        name = override_path.string();
        std::ifstream file(override_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::invalid_argument("cannot read chat template: " + name);
        const auto size = file.tellg();
        if (size <= 0 || size > 16 * 1024 * 1024) {
            throw std::invalid_argument(
                "chat template must be a nonempty UTF-8 source file up to 16 MiB: " + name);
        }
        source.resize(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(source.data(), static_cast<std::streamsize>(source.size()))) {
            throw std::invalid_argument("cannot read chat template: " + name);
        }
    }
    return {text::JinjaTemplate(std::move(source), std::move(name)), std::move(tokens)};
}

Json template_context(std::string_view kwargs, const Json& special_tokens) {
    Json context = kwargs.empty() ? Json::object() : Json::parse(kwargs);
    if (!context.is_object()) throw std::invalid_argument("chat_template_kwargs must be an object");
    for (const auto& item : context.items()) {
        const auto& key = item.key();
        if (key == "messages" || key == "tools" || key == "add_generation_prompt" ||
            key == "continue_final_message" || key == "bos_token" || key == "eos_token" ||
            key == "pad_token" || key == "unk_token" || key == "sep_token" || key == "cls_token" ||
            key == "mask_token" || key == "additional_special_tokens" ||
            special_tokens.contains(key)) {
            throw std::invalid_argument("chat_template_kwargs cannot override " + key);
        }
    }
    return context;
}

void merge_template_option(Json& context, std::string_view key, const Json& value) {
    const std::string name(key);
    if (context.contains(name) && !context[name].is_null() && context[name] != value) {
        throw std::invalid_argument("conflicting chat template parameter: " + name);
    }
    context[name] = value;
}

// Equal text alone does not prove equal control-token interpretation.
bool same_prefix(const text::TemplateOutput& lhs, const text::TemplateOutput& rhs,
                 std::size_t size) {
    if (lhs.text.size() < size || rhs.text.size() < size ||
        std::string_view(lhs.text).substr(0, size) != std::string_view(rhs.text).substr(0, size))
        return false;
    for (std::size_t i = 0;; ++i) {
        const bool left  = i < lhs.literal_spans.size() && lhs.literal_spans[i].begin < size;
        const bool right = i < rhs.literal_spans.size() && rhs.literal_spans[i].begin < size;
        if (!left || !right) return left == right;
        const auto a = lhs.literal_spans[i];
        const auto b = rhs.literal_spans[i];
        if (a.begin != b.begin || std::min(a.end, size) != std::min(b.end, size)) return false;
    }
}

std::optional<text::ByteSpan> unique_output_region(const text::TemplateOutput& output,
                                                   std::uint32_t tag) {
    std::optional<text::ByteSpan> result;
    for (const auto& region : output.regions) {
        if (region.tag != tag) continue;
        if (result) return std::nullopt;
        result = text::ByteSpan{region.begin, region.end};
    }
    return result;
}

std::optional<std::size_t> source_boundary(const text::TemplateOutput& output, std::uint32_t tag,
                                           std::size_t offset) {
    std::optional<std::size_t> result;
    for (const auto& region : output.regions) {
        if (region.tag != tag || !region.source_offset) continue;
        const auto begin = *region.source_offset;
        if (offset < begin || offset - begin > region.end - region.begin) continue;
        const auto mapped = region.begin + offset - begin;
        if (result && *result != mapped) return std::nullopt;
        result = mapped;
    }
    for (const auto& mapping : output.boundary_mappings) {
        if (mapping.tag != tag || offset < mapping.source_begin || offset > mapping.source_end)
            continue;
        if (result && *result != mapping.offset) return std::nullopt;
        result = mapping.offset;
    }
    return result;
}

} // namespace ninfer::models::frontend
