#include "text/structured_output.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::text;

// CPU-only protocol regression: byte tokens expose malformed UTF-8
// transitions that complete-character vocabularies could hide.
int main() {
    try {
        std::vector<std::string> vocab(257);
        for (int i = 0; i < 256; ++i) {
            vocab[i] = std::string(1, static_cast<char>(i));
        }
        StructuredCompiler compiler(vocab, {256});
        const auto describe_bytes = [](const std::string& body) {
            std::ostringstream result;
            result << std::hex << std::setfill('0');
            for (unsigned char byte : body) {
                result << std::setw(2) << static_cast<int>(byte) << ' ';
            }
            return result.str();
        };
        const auto accepts = [&](const std::string& schema, const std::string& body) {
            // Normal JSON strings carry a closing-context assertion; test a
            // property value to include the surrounding JSON punctuation.
            const std::string object_schema =
                R"({"type":"object","properties":{"value":)" + schema +
                R"(},"required":["value"],"additionalProperties":false})";
            auto grammar = compiler.compile({StructuredOutputKind::JsonSchema, object_schema});
            std::vector<TokenId> tokens;
            for (unsigned char byte : std::string(R"({"value":")")) {
                tokens.push_back(byte);
            }
            for (unsigned char byte : body) { tokens.push_back(byte); }
            tokens.push_back('"');
            tokens.push_back('}');
            tokens.push_back(256);
            try {
                grammar->accept(tokens);
                return true;
            } catch (const std::logic_error&) { return false; }
        };
        const std::vector<std::string> schemas = {
            R"({"type":"string"})",
            R"({"type":"string","minLength":1,"maxLength":1})",
            R"({"type":"string","pattern":"^\\D$"})"};
        const std::vector<std::string> scalar_boundaries = {
            "\x7F", "\xC2\x80", "\xDF\xBF", "\xE0\xA0\x80",
            "\xED\x9F\xBF", "\xEE\x80\x80", "\xEF\xBF\xBF",
            "\xF0\x90\x80\x80", "\xF4\x8F\xBF\xBF"};
        const std::vector<std::string> malformed = {
            "\x80", "\xC0\xA0", "\xC1\xBF", "\xE0\x80\xA0",
            "\xED\x01\xBF", "\xED\x9F\x01", "\xED\xA0\x80", "\xED\xBF\xBF", "\xF0\x80\x80\xA0",
            "\xF4\x90\x80\x80", "\xF5\x80\x80\x80", "\xF7\xBF\xBF\xBF"};
        for (const auto& schema : schemas) {
            for (const auto& body : scalar_boundaries) {
                if (!accepts(schema, body)) {
                    throw std::runtime_error("valid Unicode scalar boundary rejected: " + schema + " bytes=" + describe_bytes(body));
                }
            }
            for (const auto& body : malformed) {
                if (accepts(schema, body)) {
                    throw std::runtime_error("malformed UTF-8 accepted: " + schema + " bytes=" + describe_bytes(body));
                }
            }
        }
        const auto escaped_scalar = R"({"type":"string","pattern":"^\\D$"})";
        for (const auto& body : std::vector<std::string>{
                 R"(\u007f)", R"(\u0080)", R"(\u07FF)", R"(\u0800)",
                 R"(\uD7fF)", R"(\uE000)", R"(\uFFFF)",
                 R"(\ud800\udc00)", R"(\uDBFF\uDFFF)"}) {
            if (!accepts(escaped_scalar, body)) {
                throw std::runtime_error("valid escaped scalar rejected: " + body);
            }
        }
        for (const auto& body : std::vector<std::string>{
                 R"(\u0031)", R"(\ud800)", R"(\udc00)", R"(\udc00\ud800)",
                 R"(\ud800\u0061)", R"(\udbff\ue000)", R"(\u12)", R"(\uZZZZ)"}) {
            if (accepts(escaped_scalar, body)) {
                throw std::runtime_error("invalid escaped scalar or constraint accepted: " + body);
            }
        }
        const auto newline = R"({"type":"string","pattern":"^\\n$"})";
        if (!accepts(newline, R"(\n)") || !accepts(newline, R"(\u000A)") ||
            !accepts(newline, R"(\u000a)") || accepts(newline, "\n") ||
            accepts(newline, "n")) {
            throw std::runtime_error("JSON newline spellings do not match the same logical character");
        }
        const auto quote = R"({"type":"string","pattern":"^\"$"})";
        const auto slash = R"({"type":"string","pattern":"^\\\\$"})";
        if (!accepts(quote, R"(\")") || !accepts(quote, R"(\u0022)") ||
            accepts(quote, "'") || !accepts(slash, R"(\\)") ||
            !accepts(slash, R"(\u005c)") || accepts(slash, R"(\/)")) {
            throw std::runtime_error("JSON quote or reverse-solidus escaping changed pattern semantics");
        }
        // A schema cannot silently lose an unsupported regex branch.
        // Reject isolated surrogate escapes even when another alternative is valid.
        for (const auto& schema : std::vector<std::string>{
                 R"({"type":"string","pattern":"^\\uD800$"})",
                 R"({"type":"string","pattern":"^[\\uD800a]$"})"}) {
            bool unsupported_surrogate = false;
            try {
                compiler.compile({StructuredOutputKind::JsonSchema, schema});
            } catch (const std::invalid_argument&) { unsupported_surrogate = true; }
            if (!unsupported_surrogate) {
                throw std::runtime_error("unpaired-surrogate pattern did not fail explicitly: " + schema);
            }
        }
        std::cout << "Unicode scalar and UTF-8 protocol boundaries passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
