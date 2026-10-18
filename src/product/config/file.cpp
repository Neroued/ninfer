#include "product/config/schema.h"

#include <toml.hpp>

#include <algorithm>
#include <stdexcept>

namespace ninfer::product::config_detail {
namespace {

ConfigSource source_of(const toml::source_region& source) {
    return {
        .kind = ConfigSourceKind::File, .line = source.begin.line, .column = source.begin.column};
}

RawValue read_value(const toml::node& node, Field field) {
    if (node.is_string()) return node.as_string()->get();
    if (node.is_boolean()) return node.as_boolean()->get();
    if (node.is_integer()) return node.as_integer()->get();
    if (node.is_floating_point()) return node.as_floating_point()->get();
    if (const auto* array = node.as_array()) {
        if (field == Field::StopIds) {
            std::vector<std::int64_t> result;
            for (const auto& item : *array) {
                if (!item.is_integer()) throw std::invalid_argument("expected integer token IDs");
                result.push_back(item.as_integer()->get());
            }
            return result;
        }
        std::vector<std::string> result;
        for (const auto& item : *array) {
            if (!item.is_string()) throw std::invalid_argument("expected string array elements");
            result.push_back(item.as_string()->get());
        }
        return result;
    }
    throw std::invalid_argument("unsupported value type");
}

void read_table(const toml::table& table, std::string_view prefix,
                const std::filesystem::path& path, StartupPatch& patch) {
    for (const auto& [key, value] : table) {
        const std::string name = std::string(prefix) + std::string(key.str());
        const auto catalog     = fields();
        const auto field       = std::find_if(catalog.begin(), catalog.end(),
                                              [&](const auto& spec) { return spec.key == name; });
        const auto source      = source_of(value.source());
        if (field != catalog.end()) {
            try {
                assignment(patch, field->field) =
                    Assignment{.value  = parse_field_value(field->field,
                                                           read_value(value, field->field), false),
                               .source = source};
            } catch (const std::invalid_argument& error) {
                field_error(*field, source, path, error.what());
            }
        } else {
            const std::string child_prefix = name + ".";
            const bool section = std::any_of(catalog.begin(), catalog.end(), [&](const auto& spec) {
                return spec.key.starts_with(child_prefix);
            });
            if (!section || !value.is_table()) {
                const FieldSpec unknown{.field = Field::Count, .key = name};
                field_error(unknown, source_of(key.source()), path,
                            section ? "expected a table"
                                    : "unknown configuration field or section");
            }
            read_table(*value.as_table(), child_prefix, path, patch);
        }
    }
}

} // namespace

StartupPatch read_config_file(const std::filesystem::path& path) {
    toml::table table;
    try {
        table = toml::parse_file(path.string());
    } catch (const toml::parse_error& error) {
        throw std::invalid_argument(
            path.string() + ":" + std::to_string(error.source().begin.line) + ":" +
            std::to_string(error.source().begin.column) + ": " + std::string(error.description()));
    }
    StartupPatch patch;
    read_table(table, "", path, patch);
    return patch;
}

} // namespace ninfer::product::config_detail
