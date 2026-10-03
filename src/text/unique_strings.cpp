#include "text/unique_strings.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;
using Nodes = std::vector<const Json*>;
// The outer native validator admits $id only for a resource subtree with no
// schema references, so it is metadata in the normalized plan consumed here.
const std::set<std::string> annotations = {"$schema", "$id", "$defs", "definitions", "title", "description",
    "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly"};

bool annotation(const std::string& key) { return annotations.contains(key); }
bool whitespace(unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
int hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
std::string utf8(std::uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) throw std::invalid_argument("invalid Unicode scalar");
    std::string out;
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 63)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 63)));
        out.push_back(static_cast<char>(0x80 | (cp & 63)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 63)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 63)));
        out.push_back(static_cast<char>(0x80 | (cp & 63)));
    }
    return out;
}
std::pair<std::uint32_t, std::size_t> scalar(std::string_view text) {
    if (text.empty()) throw std::invalid_argument("missing Unicode scalar");
    const auto first = static_cast<unsigned char>(text[0]);
    std::size_t n;
    std::uint32_t cp;
    if (first < 0x80) { n = 1; cp = first; }
    else if (first >= 0xc2 && first <= 0xdf) { n = 2; cp = first & 31; }
    else if (first >= 0xe0 && first <= 0xef) { n = 3; cp = first & 15; }
    else if (first >= 0xf0 && first <= 0xf4) { n = 4; cp = first & 7; }
    else throw std::invalid_argument("invalid UTF-8 scalar");
    if (text.size() < n) throw std::invalid_argument("incomplete UTF-8 scalar");
    for (std::size_t i = 1; i < n; ++i) {
        auto c = static_cast<unsigned char>(text[i]);
        if (c < 0x80 || c > 0xbf) throw std::invalid_argument("invalid UTF-8 continuation");
        cp = (cp << 6) | (c & 63);
    }
    if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000))
        throw std::invalid_argument("overlong UTF-8 scalar");
    (void)utf8(cp);
    return {cp, n};
}
void validate_scalars(std::string_view text) {
    while (!text.empty()) { auto [cp, n] = scalar(text); (void)cp; text.remove_prefix(n); }
}
std::string hex_escape(std::uint32_t cp) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result = "\\u0000";
    for (int i = 0; i < 4; ++i) result[5 - i] = digits[(cp >> (i * 4)) & 15];
    return result;
}
bool prefix_of(std::string_view prefix, std::string_view value, bool hex_case = false) {
    if (prefix.size() > value.size()) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (hex_case) {
            if (std::tolower(static_cast<unsigned char>(prefix[i])) !=
                std::tolower(static_cast<unsigned char>(value[i]))) return false;
        } else if (prefix[i] != value[i]) return false;
    }
    return true;
}
bool scalar_spelling_prefix(std::string_view pending, std::uint32_t cp) {
    if (cp >= 0x20 && cp != '"' && cp != '\\' && prefix_of(pending, utf8(cp))) return true;
    const std::string chars = "\"\\/\b\f\n\r\t";
    const std::string escapes = "\"\\/bfnrt";
    for (std::size_t i = 0; i < chars.size(); ++i)
        if (cp == static_cast<unsigned char>(chars[i]) && prefix_of(pending, std::string("\\") + escapes[i])) return true;
    if (cp <= 0xffff) return prefix_of(pending, hex_escape(cp), true);
    cp -= 0x10000;
    return prefix_of(pending, hex_escape(0xd800 | (cp >> 10)) + hex_escape(0xdc00 | (cp & 1023)), true);
}

struct StringDomain {
    bool finite = false;
    std::vector<std::string> values;
};
struct Rule {
    const Json* schema = nullptr;
    bool vacuous = false;
    bool unique = false;
    StringDomain domain;
    std::int64_t minimum = 0;
    std::optional<std::int64_t> maximum;
};

struct Plan {
    Json root;
    Json yes = true;
    Json no = false;
    bool constrained = false;

    explicit Plan(const std::string& schema) : root(Json::parse(schema)) {
        std::set<const Json*> scanned;
        scan(root, scanned);
        if (constrained) {
            std::set<std::pair<std::vector<const Json*>, bool>> visited;
            walk({&root}, false, visited);
        }
    }
    const Json& ref(const Json& node) const {
        const auto value = node.at("$ref").get<std::string>();
        if (value == "#") return root;
        if (!value.starts_with("#/")) throw std::invalid_argument("unique strings require local schema references");
        try { return root.at(Json::json_pointer(value.substr(1))); }
        catch (const Json::exception&) { throw std::invalid_argument("unresolved unique-string schema reference"); }
    }
    void expand_one(const Json* node, Nodes& result, std::set<const Json*>& active) const {
        if (node->is_boolean()) { if (node->get<bool>()) result.push_back(node); return; }
        if (!node->is_object()) throw std::invalid_argument("unique string schema must be an object or boolean");
        if (!active.insert(node).second) throw std::invalid_argument("cyclic alias in unique string schema");
        if (node->contains("$ref")) {
            for (const auto& [key, value] : node->items()) {
                (void)value;
                if (key != "$ref" && !annotation(key)) throw std::invalid_argument("unique string reference conjunction must be normalized");
            }
            expand_one(&ref(*node), result, active);
        } else if (node->contains("anyOf")) {
            for (const auto& [key, value] : node->items()) {
                (void)value;
                if (key != "anyOf" && !annotation(key)) throw std::invalid_argument("unique string anyOf conjunction must be normalized");
            }
            for (const auto& child : node->at("anyOf")) expand_one(&child, result, active);
        } else result.push_back(node);
        active.erase(node);
    }
    Nodes expand(const Nodes& nodes) const {
        Nodes result;
        for (auto node : nodes) { std::set<const Json*> active; expand_one(node, result, active); }
        // Reaching the identical immutable schema node twice is the same union
        // interpretation. Preserve first-arm order; never deduplicate distinct
        // nodes merely because their values look similar.
        std::unordered_set<const Json*> seen;
        std::erase_if(result, [&](const Json* node) { return !seen.insert(node).second; });
        return result;
    }
    bool type(const Json& node, const std::string& wanted) const {
        if (node.is_boolean()) return node.get<bool>();
        const auto value_type = [&](const Json& value) {
            if (wanted == "string") return value.is_string();
            if (wanted == "array") return value.is_array();
            if (wanted == "object") return value.is_object();
            if (wanted == "number") return value.is_number();
            if (wanted == "boolean") return value.is_boolean();
            if (wanted == "null") return value.is_null();
            return false;
        };
        if (node.contains("const")) {
            if (!value_type(node.at("const"))) return false;
        } else if (node.contains("enum") &&
                   std::none_of(node.at("enum").begin(), node.at("enum").end(), value_type)) return false;
        if (!node.contains("type")) return true;
        const auto matches = [&](const Json& declared) {
            return declared == wanted || (wanted == "number" && declared == "integer");
        };
        const auto& declared = node.at("type");
        if (declared.is_array()) return std::any_of(declared.begin(), declared.end(), matches);
        return matches(declared);
    }
    bool universal(const Json& node, const std::string& wanted) const {
        if (node.is_boolean()) return node.get<bool>();
        if (!type(node, wanted)) return false;
        for (const auto& [key, value] : node.items()) {
            (void)value;
            if (key == "type" || annotation(key)) continue;
            // Assertions for another instance type cannot constrain this type.
            if (wanted != "array" && (key == "items" || key == "prefixItems" || key == "minItems" || key == "maxItems" || key == "uniqueItems")) continue;
            if (wanted != "object" && (key == "properties" || key == "required" || key == "additionalProperties" || key == "minProperties" || key == "maxProperties")) continue;
            if (wanted != "string" && (key == "pattern" || key == "format" || key == "minLength" || key == "maxLength")) continue;
            if ((wanted == "array" && key == "uniqueItems" && value == false) ||
                (wanted == "array" && key == "minItems" && value == 0) ||
                (wanted == "object" && key == "required" && value.empty()) ||
                (wanted == "object" && key == "minProperties" && value == 0)) continue;
            return false;
        }
        return true;
    }
    Nodes typed(const Nodes& nodes, const std::string& wanted) const {
        Nodes result;
        for (auto node : expand(nodes)) if (type(*node, wanted)) {
            if (universal(*node, wanted)) return {node};
            result.push_back(node);
        }
        return result;
    }
    const Json* property(const Json& node, const std::string& key) const {
        if (node.is_boolean()) return node.get<bool>() ? &yes : &no;
        if (node.contains("properties") && node.at("properties").contains(key)) return &node.at("properties").at(key);
        if (node.contains("additionalProperties")) return &node.at("additionalProperties");
        return &yes;
    }
    const Json* item(const Json& node, std::size_t index) const {
        if (node.is_boolean()) return node.get<bool>() ? &yes : &no;
        if (node.contains("prefixItems") && index < node.at("prefixItems").size()) return &node.at("prefixItems").at(index);
        if (node.contains("items")) return &node.at("items");
        return &yes;
    }
    StringDomain strings(const Nodes& nodes, bool simple_only = false) const {
        StringDomain result{true, {}};
        for (auto node : typed(nodes, "string")) {
            if (node->is_boolean()) {
                if (simple_only) throw std::invalid_argument("unique string items require explicit string type");
                result.finite = false; return result;
            }
            if (simple_only) {
                for (const auto& [key, value] : node->items()) {
                    (void)value;
                    if (key != "type" && key != "enum" && key != "const" && !annotation(key))
                        throw std::invalid_argument("unique string items support only string type, enum and const assertions");
                }
                if (!provably_string(*node))
                    throw std::invalid_argument("unique string items require a provable string-only domain");
            }
            if (node->contains("const")) {
                if (node->at("const").is_string()) result.values.push_back(node->at("const").get<std::string>());
            } else if (node->contains("enum")) {
                for (const auto& value : node->at("enum")) {
                    if (value.is_string()) result.values.push_back(value.get<std::string>());
                    else if (simple_only) throw std::invalid_argument("unique string enum values must be strings");
                }
            } else { result.finite = false; return result; }
        }
        std::sort(result.values.begin(), result.values.end());
        result.values.erase(std::unique(result.values.begin(), result.values.end()), result.values.end());
        for (const auto& value : result.values) validate_scalars(value);
        return result;
    }
    bool provably_string(const Json& node) const {
        if (!node.is_object()) return false;
        if (node.contains("const")) return node.at("const").is_string();
        if (node.contains("enum")) {
            const auto& values = node.at("enum");
            return values.is_array() && !values.empty() &&
                   std::all_of(values.begin(), values.end(), [](const auto& value) { return value.is_string(); });
        }
        return node.contains("type") && node.at("type") == "string";
    }
    Rule rule(const Json& node) const {
        Rule result;
        result.schema = &node;
        if (node.is_boolean()) return result;
        if (node.contains("maxItems")) result.maximum = node.at("maxItems").get<std::int64_t>();
        result.minimum = node.value("minItems", std::int64_t{0});
        if (result.minimum < 0 || (result.maximum && *result.maximum < 0) ||
            (result.maximum && result.minimum > *result.maximum))
            throw std::invalid_argument("invalid unique string array length bounds");
        result.vacuous = result.maximum && *result.maximum <= 1;
        result.unique = node.value("uniqueItems", false);
        if (result.unique && !result.vacuous) {
            for (const auto& [key, value] : node.items()) {
                (void)value;
                if (key != "type" && key != "items" && key != "minItems" && key != "maxItems" && key != "uniqueItems" && !annotation(key))
                    throw std::invalid_argument("unsupported assertion on unique string array: " + key);
            }
            if (!node.contains("items")) throw std::invalid_argument("unbounded unique arrays require explicit string items");
            for (auto child : expand({&node.at("items")}))
                if (!provably_string(*child))
                    throw std::invalid_argument("unique array items require a string-only domain on every alternative");
            auto children = typed({&node.at("items")}, "string");
            if (children.empty()) throw std::invalid_argument("unique array items must accept strings");
            result.domain = strings({&node.at("items")}, true);
            if (result.domain.finite && result.minimum > static_cast<std::int64_t>(result.domain.values.size()))
                throw std::invalid_argument("unique string array minimum exceeds its enum cardinality");
        }
        return result;
    }
    void scan(const Json& node, std::set<const Json*>& visited) {
        if (!visited.insert(&node).second || node.is_boolean()) return;
        if (!node.is_object()) return;
        if (node.contains("uniqueItems")) {
            if (!node.at("uniqueItems").is_boolean()) throw std::invalid_argument("uniqueItems must be boolean");
            if (node.at("uniqueItems") == true && type(node, "array")) {
                auto r = rule(node);
                constrained = constrained || !r.vacuous;
            }
        }
        for (const auto& [key, value] : node.items()) {
            if (key == "properties" || key == "$defs" || key == "definitions") {
                if (value.is_object()) for (const auto& child : value) scan(child, visited);
            } else if (key == "items" || key == "additionalProperties") scan(value, visited);
            else if (key == "prefixItems" || key == "anyOf") {
                if (value.is_array()) for (const auto& child : value) scan(child, visited);
            }
        }
    }
    bool has_unique(const Json& node, std::set<const Json*>& visited) const {
        if (!visited.insert(&node).second || node.is_boolean()) return false;
        if (node.contains("$ref")) return has_unique(ref(node), visited);
        if (node.value("uniqueItems", false) && !(node.contains("maxItems") && node.at("maxItems") <= 1)) return true;
        for (const auto& [key, value] : node.items()) {
            if (key == "properties") { for (const auto& child : value) if (has_unique(child, visited)) return true; }
            else if (key == "items" || key == "additionalProperties") { if (has_unique(value, visited)) return true; }
            else if (key == "prefixItems" || key == "anyOf") { for (const auto& child : value) if (has_unique(child, visited)) return true; }
        }
        return false;
    }
    bool early_discriminator(const Nodes& objects) const {
        if (objects.size() < 2 || !objects[0]->is_object() || !objects[0]->contains("properties")) return false;
        for (const auto& [name, value] : objects[0]->at("properties").items()) {
            (void)value;
            std::vector<std::set<std::string>> domains;
            bool proved = true;
            for (auto object : objects) {
                if (!object->is_object() || !object->contains("properties") || !object->at("properties").contains(name) ||
                    !object->contains("required") || std::find(object->at("required").begin(), object->at("required").end(), Json(name)) == object->at("required").end()) { proved = false; break; }
                auto domain = strings({&object->at("properties").at(name)});
                if (!domain.finite || domain.values.empty()) { proved = false; break; }
                std::set<std::string> values(domain.values.begin(), domain.values.end());
                for (const auto& prior : domains) for (const auto& s : values) if (prior.contains(s)) proved = false;
                domains.push_back(std::move(values));
                if (object->contains("additionalProperties")) {
                    std::set<const Json*> visited;
                    if (has_unique(object->at("additionalProperties"), visited)) proved = false;
                }
                bool before = false;
                for (const auto& [key, child] : object->at("properties").items()) {
                    if (key == name) before = true;
                    std::set<const Json*> visited;
                    if (!before && has_unique(child, visited)) proved = false;
                }
                if (!proved) break;
            }
            if (proved) return true;
        }
        return false;
    }
    void walk(const Nodes& input, bool safe_split,
              std::set<std::pair<std::vector<const Json*>, bool>>& visited) const {
        auto nodes = expand(input);
        auto signature = nodes;
        std::sort(signature.begin(), signature.end());
        if (!visited.insert({signature, safe_split}).second) return;
        // An enclosing discriminator separates its own alternatives, not a fresh
        // nested anyOf. Check each child interpretation on its own as well.
        if (safe_split) for (auto node : input) walk({node}, false, visited);
        Nodes raw_arrays;
        for (auto node : nodes) if (type(*node, "array")) raw_arrays.push_back(node);
        bool raw_active = false;
        for (auto node : raw_arrays) { auto r = rule(*node); raw_active = raw_active || (r.unique && !r.vacuous); }
        // A universal arm of this very value's own anyOf subsumes its peers.
        // A universal child reached through another object interpretation does
        // not: that object's required/discriminator constraints still matter.
        const bool local_universal = input.size() == 1 &&
            std::any_of(raw_arrays.begin(), raw_arrays.end(), [&](auto node) { return universal(*node, "array"); });
        if (raw_active && !safe_split && !local_universal)
            for (auto node : raw_arrays) { auto r = rule(*node); if (!r.unique && !r.vacuous)
                throw std::invalid_argument("unique string anyOf overlaps an unconstrained array; require an early disjoint string discriminator"); }
        auto arrays = typed(nodes, "array");
        std::vector<Rule> rules;
        bool active = false;
        for (auto n : arrays) { auto r = rule(*n); active = active || (r.unique && !r.vacuous); rules.push_back(std::move(r)); }
        if (active && !safe_split) {
            for (const auto& r : rules) if (!r.unique && !r.vacuous)
                throw std::invalid_argument("unique string anyOf overlaps an unconstrained array; require an early disjoint string discriminator");
            std::optional<StringDomain> common;
            for (const auto& r : rules) if (r.unique && !r.vacuous) {
                if (common && common->finite && r.domain.finite && common->values != r.domain.values)
                    throw std::invalid_argument("different finite unique domains require an early disjoint string discriminator");
                if (!common) common = r.domain;
            }
        }
        std::size_t prefix_count = 0;
        for (auto n : arrays) if (n->is_object() && n->contains("prefixItems")) prefix_count = std::max(prefix_count, n->at("prefixItems").size());
        for (std::size_t index = 0; index <= prefix_count; ++index) {
            Nodes children;
            for (auto n : arrays) children.push_back(item(*n, index));
            if (!children.empty()) walk(children, safe_split, visited);
        }
        auto objects = typed(nodes, "object");
        bool split = safe_split || early_discriminator(objects);
        std::set<std::string> names;
        for (auto n : objects) if (n->is_object() && n->contains("properties"))
            for (const auto& [key, value] : n->at("properties").items()) { (void)value; names.insert(key); }
        for (const auto& name : names) {
            Nodes children;
            for (auto n : objects) children.push_back(property(*n, name));
            walk(children, split, visited);
        }
        Nodes additional;
        for (auto n : objects) additional.push_back(n->is_object() && n->contains("additionalProperties") ? &n->at("additionalProperties") : &yes);
        if (!additional.empty()) walk(additional, split, visited);
    }
    bool matches_string(const Json& node, const std::string& value) const {
        if (!type(node, "string")) return false;
        if (node.is_boolean()) return node.get<bool>();
        if (node.contains("const")) return node.at("const") == Json(value);
        if (node.contains("enum")) return std::find(node.at("enum").begin(), node.at("enum").end(), Json(value)) != node.at("enum").end();
        return true;
    }
};

enum class Mode { Reasoning, WaitingJSON, JSON, Opaque, Done };
enum class Phase { KeyOrEnd, Key, Colon, ValueOrEnd, Value, CommaOrEnd };
enum class Kind { Object, Array };
struct Frame {
    Kind kind;
    Phase phase;
    Nodes nodes;
    std::vector<Rule> rules;
    bool guarded = false;
    std::shared_ptr<std::unordered_set<std::string>> seen = std::make_shared<std::unordered_set<std::string>>();
    std::size_t count = 0;
    std::string key;
    Frame(Kind k, Phase p, Nodes n) : kind(k), phase(p), nodes(std::move(n)) {}
};

struct Parser {
    std::shared_ptr<const Plan> plan;
    std::string reasoning_close;
    std::string reasoning_tail;
    Mode mode;
    std::vector<Frame> frames;
    bool in_string = false;
    bool key_string = false;
    std::string decoded;
    std::string pending;
    StringDomain options;
    std::string primitive;

    Parser(std::shared_ptr<const Plan> p, std::string close)
        : plan(std::move(p)), reasoning_close(std::move(close)),
          mode(reasoning_close.empty() ? Mode::WaitingJSON : Mode::Reasoning) {}

    Nodes child_nodes() const {
        if (frames.empty()) return {&plan->root};
        const auto& frame = frames.back();
        Nodes result;
        if (frame.kind == Kind::Object) {
            for (auto n : frame.nodes) result.push_back(plan->property(*n, frame.key));
        } else {
            for (const auto& rule : frame.rules)
                if (!rule.maximum || frame.count < static_cast<std::size_t>(*rule.maximum))
                    result.push_back(plan->item(*rule.schema, frame.count));
        }
        return result;
    }
    bool remaining(const Frame& frame) const {
        for (const auto& r : frame.rules) {
            if (r.maximum && frame.count >= static_cast<std::size_t>(*r.maximum)) continue;
            auto candidates = plan->expand({plan->item(*r.schema, frame.count)});
            if (candidates.empty()) continue;
            if (!r.unique || r.vacuous || !r.domain.finite) return true;
            for (const auto& value : r.domain.values) if (!frame.seen->contains(value)) return true;
        }
        return false;
    }
    bool select_type(const std::string& kind, Nodes& nodes) {
        nodes = plan->typed(child_nodes(), kind);
        if (nodes.empty()) return false;
        if (!frames.empty() && frames.back().kind == Kind::Array) {
            auto& parent = frames.back();
            std::erase_if(parent.rules, [&](const Rule& r) {
                return (r.maximum && parent.count >= static_cast<std::size_t>(*r.maximum)) ||
                       plan->typed({plan->item(*r.schema, parent.count)}, kind).empty();
            });
            if (parent.rules.empty()) return false;
        }
        return true;
    }
    bool value_finished(const std::optional<std::string>& string_value = std::nullopt) {
        if (frames.empty()) { mode = Mode::Done; return true; }
        auto& parent = frames.back();
        if (parent.kind == Kind::Array) {
            if (string_value && parent.guarded) {
                if (parent.seen->contains(*string_value)) return false;
                std::erase_if(parent.rules, [&](const Rule& r) {
                    return r.unique && !r.vacuous && r.domain.finite &&
                           std::find(r.domain.values.begin(), r.domain.values.end(), *string_value) == r.domain.values.end();
                });
                if (parent.rules.empty()) return false;
                if (!parent.seen.unique()) parent.seen = std::make_shared<std::unordered_set<std::string>>(*parent.seen);
                parent.seen->insert(*string_value);
            }
            ++parent.count;
        } else if (string_value) {
            std::erase_if(parent.nodes, [&](const Json* n) {
                for (auto child : plan->expand({plan->property(*n, parent.key)}))
                    if (plan->matches_string(*child, *string_value)) return false;
                return true;
            });
            if (parent.nodes.empty()) return false;
        }
        parent.phase = Phase::CommaOrEnd;
        return true;
    }
    bool start_value(unsigned char c) {
        std::string type;
        if (c == '{') type = "object";
        else if (c == '[') type = "array";
        else if (c == '"') type = "string";
        else if (c == 't' || c == 'f') type = "boolean";
        else if (c == 'n') type = "null";
        else if (c == '-' || (c >= '0' && c <= '9')) type = "number";
        else return false;
        if (c == '[' && !frames.empty() && frames.back().kind == Kind::Object && frames.back().nodes.size() > 1) {
            bool active = false, unguarded = false;
            for (auto child : plan->expand(child_nodes())) if (plan->type(*child, "array")) {
                auto r = plan->rule(*child);
                active = active || (r.unique && !r.vacuous);
                unguarded = unguarded || (!r.unique && !r.vacuous);
            }
            if (active && unguarded) return false;
        }
        Nodes nodes;
        if (!select_type(type, nodes)) return false;
        if (c == '{') {
            frames.push_back({Kind::Object, Phase::KeyOrEnd, std::move(nodes)});
        } else if (c == '[') {
            Frame frame{Kind::Array, Phase::ValueOrEnd, std::move(nodes)};
            for (auto n : frame.nodes) {
                auto r = plan->rule(*n);
                frame.guarded = frame.guarded || (r.unique && !r.vacuous);
                frame.rules.push_back(std::move(r));
            }
            // A proved early discriminator must already have selected its branch
            // when an ambiguous unique array begins. Never silently skip its rule.
            if (frame.guarded) for (const auto& r : frame.rules)
                if (!r.unique && !r.vacuous) return false;
            frames.push_back(std::move(frame));
        } else if (c == '"') {
            in_string = true;
            key_string = false;
            decoded.clear();
            pending.clear();
            options = plan->strings(nodes);
            if (!frames.empty() && frames.back().kind == Kind::Array && frames.back().guarded && options.finite) {
                const auto& frame = frames.back();
                std::erase_if(options.values, [&](const auto& value) { return frame.seen->contains(value); });
            }
            if (options.finite && options.values.empty()) return false;
        } else primitive.assign(1, static_cast<char>(c));
        return true;
    }
    StringDomain key_options(const Frame& frame) const {
        StringDomain result{true, {}};
        for (auto node : frame.nodes) {
            if (node->is_boolean() || !node->contains("additionalProperties") || node->at("additionalProperties") != false)
                return {};
            if (node->contains("properties"))
                for (const auto& [name, value] : node->at("properties").items()) {
                    (void)value;
                    result.values.push_back(name);
                }
        }
        std::sort(result.values.begin(), result.values.end());
        result.values.erase(std::unique(result.values.begin(), result.values.end()), result.values.end());
        return result;
    }
    bool prefix_valid() const {
        if (!options.finite) return true;
        for (const auto& value : options.values) {
            if (!prefix_of(decoded, value)) continue;
            if (pending.empty()) return true;
            if (decoded.size() == value.size()) continue;
            auto [cp, size] = scalar(std::string_view(value).substr(decoded.size()));
            (void)size;
            if (scalar_spelling_prefix(pending, cp)) return true;
        }
        return false;
    }
    bool append_scalar(std::uint32_t cp) {
        try { decoded += utf8(cp); }
        catch (const std::invalid_argument&) { return false; }
        pending.clear();
        return prefix_valid();
    }
    bool string_byte(unsigned char c) {
        if (pending.empty()) {
            if (c == '"') {
                if (options.finite &&
                    std::find(options.values.begin(), options.values.end(), decoded) == options.values.end()) return false;
                in_string = false;
                if (key_string) {
                    if (frames.empty() || frames.back().kind != Kind::Object) return false;
                    auto& frame = frames.back();
                    frame.key = decoded;
                    std::erase_if(frame.nodes, [&](const Json* node) { return plan->expand({plan->property(*node, frame.key)}).empty(); });
                    if (frame.nodes.empty()) return false;
                    frame.phase = Phase::Colon;
                    return true;
                }
                return value_finished(decoded);
            }
            if (c == '\\' || c >= 0x80) {
                if (c >= 0x80 && !(c >= 0xc2 && c <= 0xf4)) return false;
                pending.assign(1, static_cast<char>(c)); return prefix_valid();
            }
            if (c < 0x20) return false;
            decoded.push_back(static_cast<char>(c));
            return prefix_valid();
        }
        pending.push_back(static_cast<char>(c));
        if (pending[0] == '\\') {
            if (pending.size() == 2 && pending[1] != 'u') {
                const std::string escapes = "\"\\/bfnrt";
                const std::string chars = "\"\\/\b\f\n\r\t";
                auto pos = escapes.find(pending[1]);
                if (pos == std::string::npos) return false;
                return append_scalar(static_cast<unsigned char>(chars[pos]));
            }
            if (pending[1] != 'u') return false;
            if (pending.size() == 2) return prefix_valid();
            if (pending.size() <= 6) {
                if (hex(pending.back()) < 0) return false;
                if (pending.size() < 6) return prefix_valid();
                std::uint32_t cp = 0;
                for (std::size_t i = 2; i < 6; ++i) cp = (cp << 4) | static_cast<std::uint32_t>(hex(pending[i]));
                if (cp >= 0xdc00 && cp <= 0xdfff) return false;
                if (cp < 0xd800 || cp > 0xdbff) return append_scalar(cp);
                return prefix_valid();
            }
            if ((pending.size() == 7 && pending[6] != '\\') ||
                (pending.size() == 8 && pending[7] != 'u') ||
                (pending.size() > 8 && hex(pending.back()) < 0)) return false;
            if (pending.size() < 12) return prefix_valid();
            if (pending.size() > 12) return false;
            std::uint32_t high = 0, low = 0;
            for (std::size_t i = 2; i < 6; ++i) high = (high << 4) | static_cast<std::uint32_t>(hex(pending[i]));
            for (std::size_t i = 8; i < 12; ++i) low = (low << 4) | static_cast<std::uint32_t>(hex(pending[i]));
            if (low < 0xdc00 || low > 0xdfff) return false;
            return append_scalar(0x10000 + ((high - 0xd800) << 10) + low - 0xdc00);
        }
        const auto first = static_cast<unsigned char>(pending[0]);
        std::size_t n = first >= 0xc2 && first <= 0xdf ? 2 : first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
        if (!n || c < 0x80 || c > 0xbf || pending.size() > n) return false;
        if (pending.size() < n) return prefix_valid();
        try { auto [cp, size] = scalar(pending); (void)size; return append_scalar(cp); }
        catch (const std::invalid_argument&) { return false; }
    }
    bool close_frame(unsigned char c) {
        if (frames.empty()) return false;
        const auto& frame = frames.back();
        if ((frame.kind == Kind::Array && c != ']') || (frame.kind == Kind::Object && c != '}')) return false;
        frames.pop_back();
        return value_finished();
    }
    bool json_byte(unsigned char c) {
        if (in_string) return string_byte(c);
        if (!primitive.empty()) {
            if (whitespace(c) || c == ',' || c == ']' || c == '}') {
                if (!(primitive == "true" || primitive == "false" || primitive == "null" ||
                      primitive[0] == '-' || (primitive[0] >= '0' && primitive[0] <= '9'))) return false;
                primitive.clear();
                if (!value_finished()) return false;
                return json_byte(c);
            }
            primitive.push_back(static_cast<char>(c));
            return true;
        }
        if (whitespace(c)) return true;
        if (mode == Mode::Done) return false;
        if (frames.empty()) return start_value(c);
        auto& frame = frames.back();
        switch (frame.phase) {
        case Phase::KeyOrEnd:
            if (c == '}') return close_frame(c);
            [[fallthrough]];
        case Phase::Key:
            if (c != '"') return false;
            options = key_options(frame);
            if (options.finite && options.values.empty()) return false;
            in_string = true; key_string = true; decoded.clear(); pending.clear();
            return true;
        case Phase::Colon:
            if (c != ':') return false;
            frame.phase = Phase::Value;
            return true;
        case Phase::ValueOrEnd:
            if (c == ']') return close_frame(c);
            [[fallthrough]];
        case Phase::Value:
            return start_value(c);
        case Phase::CommaOrEnd:
            if (c == ']' || c == '}') return close_frame(c);
            if (c != ',') return false;
            if (frame.kind == Kind::Array && frame.guarded && !remaining(frame)) return false;
            frame.phase = frame.kind == Kind::Array ? Phase::Value : Phase::Key;
            return true;
        }
        return false;
    }
    bool token(std::string_view text) {
        for (auto raw : text) {
            const auto c = static_cast<unsigned char>(raw);
            if (mode == Mode::Opaque) return true;
            if (mode == Mode::Reasoning) {
                reasoning_tail.push_back(raw);
                while (!reasoning_close.starts_with(reasoning_tail)) reasoning_tail.erase(0, 1);
                if (reasoning_tail == reasoning_close) { mode = Mode::WaitingJSON; reasoning_tail.clear(); }
                continue;
            }
            if (mode == Mode::WaitingJSON) {
                if (whitespace(c)) continue;
                // Native tool envelopes are separate alternatives, intentionally
                // outside the final-content JSON contract.
                if (c != '{' && c != '[') { mode = Mode::Opaque; return true; }
                mode = Mode::JSON;
            }
            if (!json_byte(c)) return false;
        }
        return true;
    }
};
} // namespace

struct UniqueStringState::Impl {
    std::shared_ptr<const Plan> plan;
    Parser parser;
    Impl(std::shared_ptr<const Plan> p, const std::string& close) : plan(std::move(p)), parser(plan, close) {}
};
void UniqueStringState::AnalyzeSchema(const std::string& schema) { (void)Plan(schema); }
UniqueStringState::UniqueStringState(const std::string& schema, const std::string& close)
    : impl_(std::make_unique<Impl>(std::make_shared<Plan>(schema), close)) {}
UniqueStringState::~UniqueStringState() = default;
UniqueStringState::UniqueStringState(const UniqueStringState& other) : impl_(std::make_unique<Impl>(*other.impl_)) {}
UniqueStringState& UniqueStringState::operator=(const UniqueStringState& other) {
    if (this != &other) impl_ = std::make_unique<Impl>(*other.impl_);
    return *this;
}
UniqueStringState::UniqueStringState(UniqueStringState&&) noexcept = default;
UniqueStringState& UniqueStringState::operator=(UniqueStringState&&) noexcept = default;
bool UniqueStringState::has_constraints() const noexcept { return impl_->plan->constrained; }
bool UniqueStringState::accepts(std::string_view token) const {
    if (!has_constraints()) return true;
    auto shadow = impl_->parser;
    return shadow.token(token);
}
bool UniqueStringState::accept(std::string_view token) {
    if (!has_constraints()) return true;
    auto shadow = impl_->parser;
    if (!shadow.token(token)) return false;
    impl_->parser = std::move(shadow);
    return true;
}
bool UniqueStringState::needs_check(std::string_view token) const noexcept {
    if (!has_constraints() || impl_->parser.mode == Mode::Opaque) return false;
    if (impl_->parser.mode == Mode::Reasoning)
        return token.find(impl_->parser.reasoning_close[0]) != std::string_view::npos || !impl_->parser.reasoning_tail.empty();
    if (!impl_->parser.in_string) return true;
    if (impl_->parser.options.finite) return true;
    // Within an unrestricted string only a close/escape can finish a duplicate
    // value. Outside strings primitive tokens can select a narrowed branch or
    // begin the wrong instance type, so structural-byte filtering is unsafe.
    return token.find_first_of("\"\\") != std::string_view::npos;
}
} // namespace ninfer::text
