#include "text/constraint_limits.h"

#include "text/json_input.h"

#include "grammar_parser.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;

[[noreturn]] void exceeded(std::string message, std::string pointer = {}) {
    throw RequestError(RequestErrorKind::ConstraintLimitExceeded, std::move(message),
                       std::move(pointer));
}

std::string segment(std::string_view key) {
    std::string result = "/";
    for (const char c : key) {
        if (c == '~')
            result += "~0";
        else if (c == '/')
            result += "~1";
        else
            result += c;
    }
    return result;
}

void check_bytes(std::size_t bytes, const OutputConstraintLimits& limits, const char* what) {
    if (bytes > limits.max_bytes) {
        exceeded(std::string(what) + " has " + std::to_string(bytes) + " bytes; the limit is " +
                 std::to_string(limits.max_bytes) + " (--constraint-max-kib)");
    }
}

[[noreturn]] void too_deep(const OutputConstraintLimits& limits, const char* what,
                           std::string where, std::string pointer = {}) {
    exceeded(std::string(what) + " nesting exceeds " + std::to_string(limits.max_depth) +
                 " levels (--constraint-max-depth)" + where,
             std::move(pointer));
}

// Objects and arrays outside strings; no parse, so nothing deep is ever built.
void check_json_text_depth(std::string_view text, const OutputConstraintLimits& limits) {
    if (const auto offset = json_nesting_exceeds(text, limits.max_depth))
        too_deep(limits, "JSON schema", " at byte " + std::to_string(*offset));
}

// Groups outside escapes and character classes.
void check_regex_depth(std::string_view pattern, const OutputConstraintLimits& limits) {
    std::uint32_t depth = 0;
    bool in_class       = false;
    for (std::size_t offset = 0; offset < pattern.size(); ++offset) {
        const char c = pattern[offset];
        if (c == '\\') {
            ++offset;
        } else if (in_class) {
            if (c == ']') in_class = false;
        } else if (c == '[') {
            in_class = true;
            // A ']' right after '[' or '[^' is a literal member.
            std::size_t next = offset + 1;
            if (next < pattern.size() && pattern[next] == '^') ++next;
            if (next < pattern.size() && pattern[next] == ']') ++next;
            offset = next - 1;
        } else if (c == '(') {
            if (++depth > limits.max_depth)
                too_deep(limits, "regex group", " at byte " + std::to_string(offset));
        } else if (c == ')' && depth != 0) {
            --depth;
        }
    }
}

// The vendor lexer (linear, not recursive) gives exact groups and rule definitions.
void check_gbnf(const std::string& source, const OutputConstraintLimits& limits) {
    using Type          = xgrammar::EBNFLexer::TokenType;
    const auto tokens   = xgrammar::EBNFLexer().Tokenize(source);
    std::uint32_t depth = 0;
    std::size_t rules   = 0;
    for (const auto& token : tokens) {
        if (token.type == Type::LParen || token.type == Type::LookaheadLParen) {
            if (++depth > limits.max_depth)
                too_deep(limits, "GBNF group",
                         " at line " + std::to_string(token.line) + ", column " +
                             std::to_string(token.column));
        } else if (token.type == Type::RParen && depth != 0) {
            --depth;
        } else if (token.type == Type::Assign && ++rules > limits.max_items) {
            exceeded("GBNF grammar defines more than " + std::to_string(limits.max_items) +
                     " rules (--constraint-max-items)");
        }
    }
}

} // namespace

void check_json_schema_limits(const Json& schema, const OutputConstraintLimits& limits) {
    struct Frame {
        const Json* node;
        Json::const_iterator next;
        std::size_t index = 0;
        std::string segment;
    };

    std::vector<Frame> stack;
    std::size_t values = 0;
    const auto pointer = [&](const std::string& last) {
        std::string result;
        for (std::size_t i = 1; i < stack.size(); ++i) result += stack[i].segment;
        return result + last;
    };
    const auto visit = [&](const Json& node, std::string where) {
        // Every value serializes to at least one byte.
        if (++values > limits.max_bytes) {
            exceeded("JSON schema exceeds " + std::to_string(limits.max_bytes) +
                     " bytes (--constraint-max-kib)");
        }
        if (!node.is_object() && !node.is_array()) return;
        if (stack.size() + 1 > limits.max_depth)
            too_deep(limits, "JSON schema", {}, pointer(where));
        if (node.size() > limits.max_items) {
            exceeded("JSON schema " + std::string(node.is_object() ? "object" : "array") + " has " +
                         std::to_string(node.size()) + " members; the limit is " +
                         std::to_string(limits.max_items) + " (--constraint-max-items)",
                     pointer(where));
        }
        stack.push_back(Frame{&node, node.cbegin(), 0, std::move(where)});
    };
    visit(schema, {});
    while (!stack.empty()) {
        Frame& top = stack.back();
        if (top.next == top.node->cend()) {
            stack.pop_back();
            continue;
        }
        const Json& child = *top.next;
        std::string where =
            top.node->is_object() ? segment(top.next.key()) : "/" + std::to_string(top.index);
        ++top.next;
        ++top.index;
        visit(child, std::move(where)); // may reallocate the stack: `top` is not used after
    }
}

void check_constraint_limits(const OutputConstraint& constraint,
                             const OutputConstraintLimits& limits) {
    switch (constraint.kind) {
    case OutputConstraintKind::JsonObject:
        return;
    case OutputConstraintKind::JsonSchema:
        check_bytes(constraint.source.size(), limits, "JSON schema");
        check_json_text_depth(constraint.source, limits);
        return;
    case OutputConstraintKind::Regex:
        check_bytes(constraint.source.size(), limits, "regex");
        check_regex_depth(constraint.source, limits);
        return;
    case OutputConstraintKind::Grammar:
        check_bytes(constraint.source.size(), limits, "GBNF grammar");
        check_gbnf(constraint.source, limits);
        return;
    case OutputConstraintKind::Choice: {
        if (constraint.choices.size() > limits.max_items) {
            exceeded("choice has " + std::to_string(constraint.choices.size()) +
                     " entries; the limit is " + std::to_string(limits.max_items) +
                     " (--constraint-max-items)");
        }
        std::size_t bytes = 0;
        for (const auto& value : constraint.choices) bytes += value.size();
        check_bytes(bytes, limits, "choice");
        return;
    }
    }
}

} // namespace ninfer::text
