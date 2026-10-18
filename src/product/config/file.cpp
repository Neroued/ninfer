#include "product/config/schema.h"

#include <toml.hpp>

#include <algorithm>
#include <climits>
#include <limits>
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
        if (prefix.empty() && key.str() == "distributed") continue;
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

class DeploymentReader {
public:
    DeploymentReader(const std::filesystem::path& path, FileConfiguration& result)
        : path_(path), result_(result) {}

    DeploymentOptions read(const toml::node& node) {
        const auto& t =
            table(node, "distributed",
                  {"coordinator_node", "control_host", "control_port", "connect_timeout",
                   "shutdown_timeout", "parallel", "nodes", "ranks"});
        DeploymentOptions out;
        out.coordinator_node    = number(t, "coordinator_node", "distributed", 0);
        out.control_host        = string(t, "control_host", "distributed", true);
        out.control_port        = number(t, "control_port", "distributed", 0, UINT16_MAX);
        out.connect_timeout_ms  = duration(t, "connect_timeout", out.connect_timeout_ms);
        out.shutdown_timeout_ms = duration(t, "shutdown_timeout", out.shutdown_timeout_ms);
        if (const auto* p = t.get("parallel")) {
            const auto& values = table(*p, "distributed.parallel", {"tp", "ep", "dp", "pp"});
            out.parallel.tp    = number(values, "tp", "distributed.parallel", 1);
            out.parallel.ep    = number(values, "ep", "distributed.parallel", 1);
            out.parallel.dp    = number(values, "dp", "distributed.parallel", 1);
            out.parallel.pp    = number(values, "pp", "distributed.parallel", 1);
        }
        const auto& nodes = array(t, "nodes");
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const auto prefix = "distributed.nodes[" + std::to_string(i) + "]";
            const auto& n =
                table(nodes[i], prefix, {"id", "ssh", "worker", "artifact", "environment"});
            DeploymentNode value;
            value.id       = required_number(n, "id", prefix);
            value.ssh      = string(n, "ssh", prefix, false);
            value.worker   = string(n, "worker", prefix, true);
            value.artifact = string(n, "artifact", prefix, true);
            for (const auto key : {"worker", "artifact"})
                if (!std::filesystem::path(n[key].value<std::string>().value()).is_absolute())
                    error(*n.get(key), prefix + "." + key, "expected an absolute path on the node");
            if (const auto* env = n.get("environment")) {
                if (!env->is_table()) error(*env, prefix + ".environment", "expected a table");
                for (const auto& [key, item] : *env->as_table()) {
                    const auto name = prefix + ".environment." + std::string(key.str());
                    remember(item, name);
                    if (!item.is_string()) error(item, name, "expected a string");
                    value.environment.emplace(key.str(), item.as_string()->get());
                }
            }
            out.nodes.push_back(std::move(value));
        }
        const auto& ranks = array(t, "ranks");
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            const auto prefix = "distributed.ranks[" + std::to_string(i) + "]";
            const auto& r     = table(ranks[i], prefix, {"rank", "node", "device"});
            out.ranks.push_back({required_number(r, "rank", prefix),
                                 required_number(r, "node", prefix),
                                 static_cast<int>(number(r, "device", prefix, 0, INT_MAX))});
        }
        try {
            validate_deployment(out);
        } catch (const std::invalid_argument& e) { error(node, "distributed", e.what()); }
        return out;
    }

private:
    [[noreturn]] void error(const toml::node& node, std::string_view name,
                            std::string_view message) const {
        field_error({.field = Field::Count, .key = name}, source_of(node.source()), path_, message);
    }

    void remember(const toml::node& node, const std::string& name) {
        result_.deployment_sources.emplace(name, source_of(node.source()));
    }

    const toml::table& table(const toml::node& node, const std::string& name,
                             std::initializer_list<std::string_view> keys) {
        if (!node.is_table()) error(node, name, "expected a table");
        for (const auto& [key, value] : *node.as_table())
            if (std::find(keys.begin(), keys.end(), key.str()) == keys.end())
                error(value, name + "." + std::string(key.str()), "unknown configuration field");
        return *node.as_table();
    }

    const toml::array& array(const toml::table& t, std::string_view key) {
        const auto* node = t.get(key);
        if (!node)
            error(t, "distributed." + std::string(key), "required array of tables is absent");
        if (!node->is_array()) error(*node, "distributed." + std::string(key), "expected an array");
        return *node->as_array();
    }

    std::string string(const toml::table& t, std::string_view key, const std::string& prefix,
                       bool required) {
        const auto name  = prefix + "." + std::string(key);
        const auto* node = t.get(key);
        if (!node) {
            if (required) error(t, name, "required field is absent");
            return {};
        }
        remember(*node, name);
        if (!node->is_string()) error(*node, name, "expected a string");
        auto value = node->as_string()->get();
        if (required && value.empty()) error(*node, name, "must not be empty");
        return value;
    }

    std::uint32_t number(const toml::table& t, std::string_view key, const std::string& prefix,
                         std::uint32_t fallback, std::uint32_t maximum = UINT32_MAX) {
        const auto* node = t.get(key);
        if (!node) return fallback;
        const auto name = prefix + "." + std::string(key);
        remember(*node, name);
        const auto value = node->value<std::int64_t>();
        if (!node->is_integer() || !value || *value < 0 || std::uint64_t(*value) > maximum)
            error(*node, name, "integer is outside its allowed range");
        return static_cast<std::uint32_t>(*value);
    }

    std::uint32_t required_number(const toml::table& t, std::string_view key,
                                  const std::string& prefix) {
        if (!t.contains(key)) error(t, prefix + "." + std::string(key), "required field is absent");
        return number(t, key, prefix, 0);
    }

    std::uint32_t duration(const toml::table& t, std::string_view key, std::uint32_t fallback) {
        const auto* node = t.get(key);
        if (!node) return fallback;
        const auto name = "distributed." + std::string(key);
        remember(*node, name);
        try {
            if (node->is_integer()) return parse_duration_ms(node->as_integer()->get(), true);
            if (node->is_string()) return parse_duration_ms(node->as_string()->get(), true);
            throw std::invalid_argument("expected milliseconds or an ms/s duration");
        } catch (const std::invalid_argument& e) { error(*node, name, e.what()); }
    }

    const std::filesystem::path& path_;
    FileConfiguration& result_;
};

} // namespace

FileConfiguration read_config_file(const std::filesystem::path& path) {
    toml::table table;
    try {
        table = toml::parse_file(path.string());
    } catch (const toml::parse_error& error) {
        throw std::invalid_argument(
            path.string() + ":" + std::to_string(error.source().begin.line) + ":" +
            std::to_string(error.source().begin.column) + ": " + std::string(error.description()));
    }
    FileConfiguration result;
    read_table(table, "", path, result.fields);
    if (const auto* deployment = table.get("distributed"))
        result.deployment = DeploymentReader(path, result).read(*deployment);
    return result;
}

} // namespace ninfer::product::config_detail

namespace ninfer::product {
DeploymentOptions read_deployment_file(const std::filesystem::path& path) {
    auto file = config_detail::read_config_file(path);
    if (!file.deployment)
        throw std::invalid_argument("configuration has no distributed deployment");
    return std::move(*file.deployment);
}
} // namespace ninfer::product
