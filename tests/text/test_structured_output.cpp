#include "text/structured_output.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
using namespace ninfer;
using namespace ninfer::text;

void require(bool value, const char* what) {
    if (!value) { throw std::runtime_error(what); }
}

int main() {
    try {
        std::vector<std::string> vocab(257);
        for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
        StructuredCompiler compiler(vocab, {256});
        const auto schema =
            R"({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2}},"required":["a","b"],"additionalProperties":false})";
        auto state          = compiler.compile({StructuredOutputKind::JsonSchema, schema});
        constexpr int words = 9;
        std::vector<std::uint32_t> before(words), after(words);
        state->fill_masks(before, {});
        require((before['{' / 32] & (1U << ('{' % 32))) != 0, "object start masked");
        require((before['[' / 32] & (1U << ('[' % 32))) == 0, "schema permits wrong root");
        require((before[8] & 1U) == 0, "premature EOS permitted");
        auto preview           = state->fork();
        const std::string json = "{\"a\":\"é😀\\n\\\"\",\"b\":[-1,23]}";
        std::vector<TokenId> ids;
        for (unsigned char ch : json) { ids.push_back(ch); }
        preview->accept(ids);
        state->fill_masks(after, {});
        require(before == after, "preview advanced committed grammar");
        *state = std::move(*preview);
        state->fill_masks(after, {});
        require((after[8] & 1U) != 0, "completed JSON cannot stop");
        state->accept(std::vector<TokenId>{256});
        require(nlohmann::json::parse(json).at("b").size() == 2, "invalid JSON fixture");

        const auto open_schema =
            R"({"type":"object","properties":{"a":{"type":"integer"},"東京":{"type":"boolean"},"quote\"":{"type":"boolean"}},"required":["a","東京"],"additionalProperties":{"type":"string"}})";
        const auto accepts = [&](const std::string& value) {
            auto grammar = compiler.compile({StructuredOutputKind::JsonSchema, open_schema});
            std::vector<TokenId> tokens;
            for (unsigned char ch : value) { tokens.push_back(ch); }
            tokens.push_back(256);
            try {
                grammar->accept(tokens);
                return true;
            } catch (const std::logic_error&) { return false; }
        };
        require(accepts(R"({"a":1,"東京":true,"other":"x"})"), "open object lost valid extra key");
        require(!accepts(R"({"a":1,"東京":true,"\u0061":"bad"})"),
                "escaped key bypasses property type");
        require(!accepts(R"({"a":1,"東京":true,"東京":"bad"})"),
                "Unicode key bypasses property type");
        require(!accepts(R"({"a":1,"東京":true,"quote"":"bad"})"), "invalid key escape accepted");

        auto recursive = compiler.compile(
            {StructuredOutputKind::JsonSchema, R"({"type":"array","items":{"$ref":"#"}})"});
        std::vector<TokenId> recursive_tokens;
        for (unsigned char ch : std::string("[[],[[]]]")) { recursive_tokens.push_back(ch); }
        recursive_tokens.push_back(256);
        recursive->accept(recursive_tokens);

        auto bounded = compiler.compile(
            {StructuredOutputKind::JsonSchema, R"({"type":"string","minLength":2,"maxLength":2})"});
        bounded->accept(std::vector<TokenId>{'"'});
        bounded->fill_masks(after, {});
        for (int control = 0; control < 32; ++control) {
            require((after[control / 32] & (1U << (control % 32))) == 0,
                    "bounded string permits unescaped control character");
        }
        std::vector<TokenId> unicode;
        for (unsigned char ch : std::string("é😀\"")) { unicode.push_back(ch); }
        bounded->accept(unicode);
        bounded->fill_masks(after, {});
        require((after[8] & 1U) != 0, "bounded string did not count Unicode code points");

        auto whitespace          = compiler.compile({StructuredOutputKind::JsonObject, {}});
        bool too_much_whitespace = false;
        try {
            whitespace->accept(std::vector<TokenId>(9, ' '));
        } catch (const std::logic_error&) { too_much_whitespace = true; }
        require(too_much_whitespace, "unbounded whitespace run accepted");

        auto object = compiler.compile({StructuredOutputKind::JsonObject, {}});
        std::vector<TokenId> drafts{'{', '"', 'x', '"', ':', '[', '1', ',', '2', ']', '}'};
        std::vector<std::uint32_t> masks(words * (drafts.size() + 1));
        object->fill_masks(masks, drafts);
        for (std::size_t i = 0; i < drafts.size(); ++i) {
            require(masks[i * words + drafts[i] / 32] & (1U << (drafts[i] % 32)),
                    "valid speculative draft masked");
        }
        require(masks[drafts.size() * words + 8] & 1U, "bonus mask missing EOS");
        object->fill_masks(before, {});
        require(before[0] == masks[0], "draft traversal advanced committed grammar");
        for (const char* bad :
             {R"({"type":"array","uniqueItems":true})",
              R"({"$ref":"#/$defs/a~1b","$defs":{"a/b":{"const":1},"a~1b":{"const":2}}})",
              R"({"oneOf":[{},{}]})", R"({"$ref":"https://example.org/schema"})",
              R"({"const":1,"type":"string"})", R"({"anyOf":[{}],"type":"object"})",
              R"({"type":"integer","minimum":0})"}) {
            bool failed = false;
            try {
                compiler.compile({StructuredOutputKind::JsonSchema, bad});
            } catch (const std::invalid_argument&) { failed = true; }
            require(failed, "unsupported constraint silently accepted");
        }
        std::cout << "OK structured grammar: masks, transaction, UTF-8, schema, EOS\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
