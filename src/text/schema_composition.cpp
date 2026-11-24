#include "text/schema_composition.h"
#include "ninfer/types.h"
#include "text/unicode.h"

#include "grammar_functor.h"
#include "json_string_grammar.h"
#include "json_number.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>

namespace ninfer::text {
namespace {
using Json              = nlohmann::ordered_json;
using Kind              = RequestErrorKind;
constexpr unsigned kAll = 127, kInteger = 8, kNumber = 16, kObject = 64;
constexpr const char* kTypeNames[] = {"null",   "boolean", "string", "integer",
                                      "number", "array",   "object"};

std::string child(const std::string& path, const std::string& key) {
    std::string result = path + '/';
    for (char c : key) result += c == '~' ? "~0" : c == '/' ? "~1" : std::string(1, c);
    return result;
}

bool annotation(const std::string& key) {
    static const std::set<std::string> keys{"title",    "description", "default",  "examples",
                                            "$comment", "deprecated",  "readOnly", "writeOnly",
                                            "$schema",  "$id",         "$defs",    "definitions"};
    return keys.contains(key);
}

xgrammar::NumberRange number_range(const Json& node) {
    xgrammar::NumberRange range;
    for (const char* key : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
        if (!node.contains(key)) continue;
        auto value = xgrammar::DecimalNumber::Parse(node[key].dump());
        const std::string_view name(key);
        if (name == "minimum" || name == "exclusiveMinimum")
            range.Lower(std::move(value), name.starts_with("exclusive"));
        else
            range.Upper(std::move(value), name.starts_with("exclusive"));
    }
    return range;
}

unsigned types(const Json& schema) {
    if (schema.is_boolean()) return schema.get<bool>() ? kAll : 0;
    if (!schema.contains("type")) return kAll;
    unsigned result = 0;
    const auto add  = [&](const Json& value) {
        for (unsigned i = 0; i < 7; ++i)
            if (value == kTypeNames[i]) result |= (1u << i) | (i == 4 ? kInteger : 0);
    };
    if (schema["type"].is_array())
        for (const auto& type : schema["type"]) add(type);
    else
        add(schema["type"]);
    return result;
}

unsigned value_type(const Json& value) {
    if (value.is_null()) return 1;
    if (value.is_boolean()) return 2;
    if (value.is_string()) return 4;
    if (value.is_number_integer()) return kInteger;
    if (value.is_number_float())
        return std::floor(value.get<double>()) == value.get<double>() ? kInteger : kNumber;
    return value.is_array() ? 32 : kObject;
}

bool equal_value(const Json& a, const Json& b) {
    if (a.is_number() && b.is_number()) {
        return xgrammar::DecimalNumber::Parse(a.dump()).Compare(
                   xgrammar::DecimalNumber::Parse(b.dump())) == 0;
    }
    if (a.type() != b.type()) return false;
    if (a.is_array()) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (!equal_value(a[i], b[i])) return false;
        return true;
    }
    if (a.is_object()) {
        if (a.size() != b.size()) return false;
        for (const auto& [key, value] : a.items())
            if (!b.contains(key) || !equal_value(value, b[key])) return false;
        return true;
    }
    return a == b;
}

bool needs_composition(const Json& node) {
    if (!node.is_object()) return false;
    for (const char* key : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"})
        if (node.contains(key)) return true;
    for (const char* op : {"$ref", "const", "enum", "anyOf", "oneOf", "allOf"}) {
        if (!node.contains(op)) continue;
        if (std::string_view(op) == "allOf" && node[op].size() > 1) return true;
        for (const auto& [key, value] : node.items())
            if (key != op && !annotation(key) &&
                !(key == "type" &&
                  (std::string_view(op) == "const" || std::string_view(op) == "enum")))
                return true;
    }
    for (const char* map : {"properties", "$defs", "definitions"})
        if (node.contains(map))
            for (const auto& value : node[map])
                if (needs_composition(value)) return true;
    for (const char* array : {"anyOf", "oneOf", "allOf", "prefixItems"})
        if (node.contains(array))
            for (const auto& value : node[array])
                if (needs_composition(value)) return true;
    for (const char* key : {"items", "additionalProperties"})
        if (node.contains(key) && needs_composition(node[key])) return true;
    return false;
}

struct Part {
    Json value;
    std::string path;
};

class Composition {
public:
    explicit Composition(const Json& source, bool draft7) : source_(source), draft7_(draft7) {}

    NormalizedSchema run() {
        auto root = join({{source_, ""}});
        if (root.is_boolean()) return {root, {}};
        root["$defs"] = std::move(definitions_);
        return {std::move(root), std::move(origins_)};
    }

private:
    const char* prefix_key(const Json& schema) const {
        if (draft7_)
            return schema.contains("items") && schema["items"].is_array() ? "items" : nullptr;
        return schema.contains("prefixItems") ? "prefixItems" : nullptr;
    }

    Part item_at(const Part& part, std::size_t position) const {
        const auto* prefix = prefix_key(part.value);
        if (prefix && position < part.value[prefix].size())
            return {part.value[prefix][position],
                    child(child(part.path, prefix), std::to_string(position))};
        const char* tail = draft7_ && prefix ? "additionalItems" : "items";
        return part.value.contains(tail) ? Part{part.value[tail], child(part.path, tail)}
                                         : Part{true, part.path};
    }

    [[noreturn]] void fail(const std::string& path, const std::string& message) const {
        throw RequestError(Kind::UnsupportedJsonSchema,
                           message + " at " + (path.empty() ? "/" : path), path);
    }

    void expand(const Part& part, std::vector<Part>& output, std::set<std::string> refs = {}) {
        if (!part.value.is_object()) {
            output.push_back(part);
            return;
        }
        auto own = part.value;
        if (own.contains("$ref")) {
            const auto ref = own["$ref"].get<std::string>();
            if (!refs.insert(ref).second)
                fail(part.path, "reference cycle does not descend into a value");
            expand({source_.at(Json::json_pointer(ref.substr(1))), ref.substr(1)}, output, refs);
            own.erase("$ref");
        }
        if (own.contains("allOf")) {
            for (std::size_t i = 0; i < own["allOf"].size(); ++i)
                expand({own["allOf"][i], child(child(part.path, "allOf"), std::to_string(i))},
                       output, refs);
            own.erase("allOf");
        }
        for (auto it = own.begin(); it != own.end();) {
            if (annotation(it.key()))
                it = own.erase(it);
            else
                ++it;
        }
        if (!own.empty()) output.push_back({std::move(own), part.path});
    }

    const Json& resolve(const Json& value) const {
        const Json* node = &value;
        std::set<std::string> seen;
        while (node->is_object() && node->contains("$ref")) {
            const auto key = (*node)["$ref"].get<std::string>().substr(8);
            if (!seen.insert(key).second) return *node;
            node = &definitions_.at(key);
        }
        return *node;
    }

    bool is_false(const Json& value) const {
        const auto& node = resolve(value);
        return node.is_boolean() && !node.get<bool>();
    }

    bool matches(const Json& schema, const Json& value, std::size_t depth = 0) {
        if (depth > 256) fail({}, "finite-value validation exceeds schema depth limit");
        if (schema.is_boolean()) return schema.get<bool>();
        if (schema.contains("$ref") &&
            !matches(source_.at(Json::json_pointer(schema["$ref"].get<std::string>().substr(1))),
                     value, depth + 1))
            return false;
        if (!(types(schema) & value_type(value))) return false;
        if (schema.contains("const") && !equal_value(schema["const"], value)) return false;
        if (schema.contains("enum") &&
            std::none_of(schema["enum"].begin(), schema["enum"].end(),
                         [&](const auto& v) { return equal_value(v, value); }))
            return false;
        for (const char* op : {"anyOf", "oneOf", "allOf"}) {
            if (!schema.contains(op)) continue;
            std::size_t accepted = 0;
            for (const auto& branch : schema[op]) accepted += matches(branch, value, depth + 1);
            if ((std::string_view(op) == "anyOf" && accepted == 0) ||
                (std::string_view(op) == "oneOf" && accepted != 1) ||
                (std::string_view(op) == "allOf" && accepted != schema[op].size()))
                return false;
        }
        if (value.is_string()) {
            const auto& text  = value.get_ref<const std::string&>();
            std::size_t count = 0;
            for (std::size_t pos = 0; pos < text.size(); ++count)
                pos += unicode_internal::utf8_codepoint_at(text, pos, "schema literal").length;
            if (schema.contains("minLength") && count < schema["minLength"].get<std::size_t>())
                return false;
            if (schema.contains("maxLength") && count > schema["maxLength"].get<std::size_t>())
                return false;
            if (schema.contains("pattern")) {
                const auto pattern = schema["pattern"].get<std::string>();
                auto found         = patterns_.find(pattern);
                if (found == patterns_.end()) {
                    auto parsed = xgrammar::GrammarFSMBuilder::Regex(
                        xgrammar::SchemaStringPattern(pattern), false);
                    if (parsed.IsErr()) fail({}, std::move(parsed).UnwrapErr().what());
                    found = patterns_.emplace(pattern, std::move(parsed).Unwrap()).first;
                }
                if (!found->second.AcceptString(text)) return false;
            }
        }
        if (value.is_number()) {
            if (!number_range(schema).Contains(xgrammar::DecimalNumber::Parse(value.dump())))
                return false;
        }
        if (value.is_array()) {
            if (schema.contains("minItems") && value.size() < schema["minItems"].get<std::size_t>())
                return false;
            if (schema.contains("maxItems") && value.size() > schema["maxItems"].get<std::size_t>())
                return false;
            for (std::size_t i = 0; i < value.size(); ++i)
                if (!matches(item_at({schema, {}}, i).value, value[i], depth + 1)) return false;
        }
        if (value.is_object()) {
            if (schema.contains("required"))
                for (const auto& key : schema["required"])
                    if (!value.contains(key.get<std::string>())) return false;
            for (const auto& [key, item] : value.items()) {
                if (schema.contains("properties") && schema["properties"].contains(key)) {
                    if (!matches(schema["properties"][key], item, depth + 1)) return false;
                } else if (schema.contains("additionalProperties") &&
                           !matches(schema["additionalProperties"], item, depth + 1))
                    return false;
            }
        }
        return true;
    }

    bool disjoint(const Json& lhs, const Json& rhs) const {
        const auto& a = resolve(lhs);
        const auto& b = resolve(rhs);
        if (!(types(a) & types(b))) return true;
        const auto finite = [](const Json& v) -> std::optional<Json> {
            if (!v.is_object()) return {};
            if (v.contains("enum")) return v["enum"];
            if (v.contains("const")) return Json::array({v["const"]});
            return {};
        };
        const auto av = finite(a), bv = finite(b);
        if (av && bv) {
            for (const auto& x : *av)
                for (const auto& y : *bv)
                    if (equal_value(x, y)) return false;
            return true;
        }
        if (av || bv) {
            for (const auto& value : *(av ? av : bv))
                if (value_type(value) & types(av ? b : a)) return false;
            return true;
        }
        if (!a.is_object() || !b.is_object() || !a.contains("required") || !b.contains("required"))
            return false;
        for (const auto& key : a["required"]) {
            if (std::find(b["required"].begin(), b["required"].end(), key) == b["required"].end())
                continue;
            const auto name = key.get<std::string>();
            if (!a.contains("properties") || !b.contains("properties") ||
                !a["properties"].contains(name) || !b["properties"].contains(name))
                continue;
            const auto x = finite(resolve(a["properties"][name])),
                       y = finite(resolve(b["properties"][name]));
            if (!x || !y) continue;
            bool overlap = false;
            for (const auto& v : *x)
                for (const auto& w : *y) overlap |= equal_value(v, w);
            if (!overlap) return true;
        }
        return false;
    }

    Json join(const std::vector<Part>& incoming) {
        if (++operations_ > 16384)
            fail(incoming.empty() ? "" : incoming[0].path,
                 "schema conjunction exceeds compilation limit");
        std::vector<Part> parts;
        for (const auto& part : incoming) expand(part, parts);
        std::string key;
        std::set<std::string> seen;
        for (auto it = parts.begin(); it != parts.end();) {
            if (it->value.is_boolean()) {
                if (!it->value.get<bool>()) return false;
                it = parts.erase(it);
                continue;
            }
            const auto text = it->value.dump();
            if (!seen.insert(text).second) {
                it = parts.erase(it);
                continue;
            }
            key += std::to_string(text.size()) + ":" + text;
            ++it;
        }
        if (parts.empty()) return true;
        if (auto found = memo_.find(key); found != memo_.end())
            return Json{{"$ref", "#/$defs/" + found->second}};
        const auto name    = "s" + std::to_string(memo_.size());
        memo_[key]         = name;
        definitions_[name] = true;
        origins_.emplace_back("/$defs/" + name, parts.front().path);
        auto result = reduce(parts);
        if (result.is_object()) {
            for (const auto& [field, value] : result.items()) {
                for (const auto& part : parts) {
                    if (part.value.contains(field) && part.value[field] == value) {
                        origins_.emplace_back(child("/$defs/" + name, field),
                                              child(part.path, field));
                        break;
                    }
                }
            }
        }
        definitions_[name] = std::move(result);
        if (is_false(Json{{"$ref", "#/$defs/" + name}})) return false;
        return Json{{"$ref", "#/$defs/" + name}};
    }

    Json reduce(const std::vector<Part>& parts) {
        for (const auto& part : parts) {
            if (!part.value.contains("const") && !part.value.contains("enum")) continue;
            const auto candidates = part.value.contains("const")
                                        ? Json::array({part.value["const"]})
                                        : part.value["enum"];
            Json admitted         = Json::array();
            for (const auto& candidate : candidates)
                if (std::all_of(parts.begin(), parts.end(),
                                [&](const auto& p) { return matches(p.value, candidate); }) &&
                    std::none_of(admitted.begin(), admitted.end(),
                                 [&](const auto& v) { return equal_value(v, candidate); }))
                    admitted.push_back(candidate);
            if (admitted.empty()) return false;
            return Json{{"enum", std::move(admitted)}};
        }
        for (std::size_t i = 0; i < parts.size(); ++i)
            for (const char* op : {"anyOf", "oneOf"}) {
                if (!parts[i].value.contains(op)) continue;
                auto common = parts;
                common[i].value.erase(op);
                Json branches = Json::array();
                for (std::size_t n = 0; n < parts[i].value[op].size(); ++n) {
                    auto selected = common;
                    selected.push_back({parts[i].value[op][n],
                                        child(child(parts[i].path, op), std::to_string(n))});
                    auto branch = join(selected);
                    if (!is_false(branch)) branches.push_back(std::move(branch));
                }
                if (std::string_view(op) == "oneOf")
                    for (std::size_t a = 0; a < branches.size(); ++a)
                        for (std::size_t b = 0; b < a; ++b)
                            if (!disjoint(branches[a], branches[b]))
                                fail(child(parts[i].path, op),
                                     "oneOf branches are not provably disjoint after conjunction");
                if (branches.empty()) return false;
                return Json{{"anyOf", std::move(branches)}};
            }
        unsigned domain = kAll;
        for (const auto& part : parts) domain &= types(part.value);
        if (!domain) return false;
        std::vector<unsigned> kinds;
        for (unsigned i = 0; i < 7; ++i)
            if ((domain & (1u << i)) && !(i == 3 && (domain & kNumber))) kinds.push_back(i);
        if (kinds.size() > 1) {
            Json branches = Json::array();
            for (auto kind : kinds) {
                auto selected = parts;
                selected.push_back({Json{{"type", kTypeNames[kind]}}, parts[0].path});
                auto branch = join(selected);
                if (!is_false(branch)) branches.push_back(std::move(branch));
            }
            return branches.empty() ? Json(false) : Json{{"anyOf", std::move(branches)}};
        }
        const auto kind = kinds.front();
        Json result{{"type", kTypeNames[kind]}};
        const auto bound = [&](const char* name, bool lower) {
            for (const auto& part : parts)
                if (part.value.contains(name)) {
                    const int cmp =
                        !result.contains(name)
                            ? (lower ? 1 : -1)
                            : xgrammar::DecimalNumber::Parse(part.value[name].dump())
                                  .Compare(xgrammar::DecimalNumber::Parse(result[name].dump()));
                    if (lower ? cmp > 0 : cmp < 0) result[name] = part.value[name];
                }
        };
        if (kind == 2 || kind == 5) {
            const char* lo = kind == 2 ? "minLength" : "minItems";
            const char* hi = kind == 2 ? "maxLength" : "maxItems";
            bound(lo, true);
            bound(hi, false);
            if (result.contains(lo) && result.contains(hi) && result[lo] > result[hi]) return false;
        }
        if (kind == 2) {
            Json patterns = Json::array();
            for (const auto& part : parts)
                if (part.value.contains("pattern") &&
                    std::find(patterns.begin(), patterns.end(), part.value["pattern"]) ==
                        patterns.end())
                    patterns.push_back(part.value["pattern"]);
            if (patterns.size() == 1)
                result["pattern"] = patterns[0];
            else if (patterns.size() > 1) {
                Json terms = Json::array({result});
                for (const auto& pattern : patterns)
                    terms.push_back(Json{{"type", "string"}, {"pattern", pattern}});
                result = Json{{"allOf", std::move(terms)}};
            }
        }
        if (kind == 3 || kind == 4) {
            for (const auto* name : {"minimum", "exclusiveMinimum"}) bound(name, true);
            for (const auto* name : {"maximum", "exclusiveMaximum"}) bound(name, false);
            const auto range = number_range(result);
            if (range.Empty()) return false;
            if (kind == 3 && (range.lower || range.upper)) {
                const auto integers = range.Integers();
                if (!integers) {
                    // Distinguish an interval between integers from one beyond the int64 domain.
                    const auto first = xgrammar::DecimalNumber::Parse("-9223372036854775808");
                    const auto last  = xgrammar::DecimalNumber::Parse("9223372036854775807");
                    if ((range.lower && range.lower->value.Compare(last) >= 0) ||
                        (range.upper && range.upper->value.Compare(first) <= 0))
                        fail(parts.front().path, "integer interval exceeds signed 64-bit range");
                    return false;
                }
                result = {{"type", "integer"},
                          {"minimum", integers->first},
                          {"maximum", integers->second}};
            }
        }
        if (kind == 5) {
            std::size_t prefix_size = 0;
            for (const auto& part : parts)
                if (const auto* key = prefix_key(part.value))
                    prefix_size = std::max(prefix_size, part.value[key].size());
            Json prefix = Json::array();
            for (std::size_t i = 0; i <= prefix_size; ++i) {
                std::vector<Part> conditions;
                for (const auto& part : parts) conditions.push_back(item_at(part, i));
                auto item = join(conditions);
                if (is_false(item)) {
                    // An impossible position caps the length; it need not occur at all.
                    const auto cap = static_cast<std::int64_t>(i);
                    if (result.value("minItems", std::int64_t{0}) > cap) return false;
                    if (!result.contains("maxItems") || result["maxItems"] > cap)
                        result["maxItems"] = cap;
                    result["items"] = false;
                    break;
                }
                if (i == prefix_size)
                    result["items"] = std::move(item);
                else
                    prefix.push_back(std::move(item));
            }
            if (!prefix.empty()) result["prefixItems"] = std::move(prefix);
        }
        if (kind == 6) {
            Json properties = Json::object(), required = Json::array();
            for (const auto& part : parts) {
                if (part.value.contains("properties"))
                    for (const auto& [key, value] : part.value["properties"].items())
                        properties[key] = true;
                if (part.value.contains("required"))
                    for (const auto& key : part.value["required"])
                        if (std::find(required.begin(), required.end(), key) == required.end())
                            required.push_back(key);
            }
            for (const auto& key : required)
                if (!properties.contains(key.get<std::string>()))
                    properties[key.get<std::string>()] = true;
            for (auto& [key, value] : properties.items()) {
                std::vector<Part> conditions;
                for (const auto& part : parts) {
                    if (part.value.contains("properties") && part.value["properties"].contains(key))
                        conditions.push_back({part.value["properties"][key],
                                              child(child(part.path, "properties"), key)});
                    else if (part.value.contains("additionalProperties"))
                        conditions.push_back({part.value["additionalProperties"],
                                              child(part.path, "additionalProperties")});
                }
                value = join(conditions);
                if (is_false(value) &&
                    std::find(required.begin(), required.end(), Json(key)) != required.end())
                    return false;
            }
            std::vector<Part> extra;
            for (const auto& part : parts)
                if (part.value.contains("additionalProperties"))
                    extra.push_back({part.value["additionalProperties"],
                                     child(part.path, "additionalProperties")});
            result["properties"]           = std::move(properties);
            result["required"]             = std::move(required);
            result["additionalProperties"] = join(extra);
        }
        return result;
    }

    const Json& source_;
    bool draft7_;
    Json definitions_ = Json::object();
    std::unordered_map<std::string, std::string> memo_;
    std::map<std::string, xgrammar::FSMWithStartEnd> patterns_;
    std::vector<std::pair<std::string, std::string>> origins_;
    unsigned operations_ = 0;
};
} // namespace

NormalizedSchema normalize_schema_composition(const Json& source, bool draft7) {
    if (!draft7 && !needs_composition(source)) return {source, {}};
    return Composition(source, draft7).run();
}
} // namespace ninfer::text
