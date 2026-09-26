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
        const auto tokens_for = [](std::string_view value) {
            std::vector<TokenId> tokens;
            for (unsigned char ch : value) { tokens.push_back(ch); }
            return tokens;
        };
        const auto accepts_value = [&](const std::shared_ptr<GrammarState>& grammar,
                                       const std::string& value) {
            auto trial  = grammar->fork();
            auto tokens = tokens_for(value);
            tokens.push_back(256);
            try {
                trial->accept(tokens);
                return true;
            } catch (const std::logic_error&) { return false; }
        };
        auto rating = compiler.compile(
            {StructuredOutputKind::JsonSchema, R"({"type":"number","minimum":0,"maximum":10})"});
        require(accepts_value(rating, "0") && accepts_value(rating, "10") &&
                    accepts_value(rating, "9.5"),
                "inclusive number bounds lost valid values");
        for (int i = -80; i <= 240; ++i) {
            const double value  = i / 16.0;
            const auto spelling = nlohmann::json(value).dump();
            if (accepts_value(rating, spelling)) {
                const double decoded = nlohmann::json::parse(spelling).get<double>();
                require(decoded >= 0 && decoded <= 10, "number escaped independent range oracle");
            }
        }
        auto exclusive = compiler.compile(
            {StructuredOutputKind::JsonSchema,
             R"({"type":"number","exclusiveMinimum":-0.25,"exclusiveMaximum":0.25})"});
        require(accepts_value(exclusive, "0") && accepts_value(exclusive, "0.249999") &&
                    !accepts_value(exclusive, "0.25") && !accepts_value(exclusive, "-0.25") &&
                    !accepts_value(exclusive, "1e2"),
                "exclusive number bounds weakened");
        auto integer =
            compiler.compile({StructuredOutputKind::JsonSchema,
                              R"({"type":"integer","minimum":-2,"exclusiveMaximum":3})"});
        for (int i = -10; i <= 10; ++i) {
            require(accepts_value(integer, std::to_string(i)) == (i >= -2 && i < 3),
                    "integer range disagrees with independent oracle");
        }
        require(!accepts_value(integer, "1.5"), "integer accepted a fraction");
        auto tiny =
            compiler.compile({StructuredOutputKind::JsonSchema,
                              R"({"type":"number","minimum":0.000001,"maximum":0.000002})"});
        require(accepts_value(tiny, "0.000001") && accepts_value(tiny, "0.000002") &&
                    !accepts_value(tiny, "0") && !accepts_value(tiny, "0.000003"),
                "number precision boundary escaped bounds");

        auto reasoning = compiler.compile({StructuredOutputKind::JsonObject, {}},
                                          {.reasoning_close = "</think>"});
        reasoning->accept(tokens_for("reasoning with << overlap </thi"));
        const auto crossing = tokens_for("nk>{\"ok\":true}");
        std::vector<std::uint32_t> crossing_masks(words * (crossing.size() + 1));
        reasoning->fill_masks(crossing_masks, crossing);
        for (std::size_t i = 0; i < crossing.size(); ++i) {
            require(crossing_masks[i * words + crossing[i] / 32] & (1U << (crossing[i] % 32)),
                    "reasoning-to-JSON speculative transition masked a valid token");
        }
        require(!(crossing_masks[3 * words + 'p' / 32] & (1U << ('p' % 32))),
                "reasoning grammar escaped into final prose");
        require(crossing_masks[crossing.size() * words + 8] & 1U, "final bonus EOS missing");
        require(accepts_value(reasoning, "nk>{}"), "draft masks advanced committed phase");
        require(accepts_value(reasoning, "nk>\n\n{}"), "canonical close whitespace rejected");
        require(!accepts_value(reasoning, "nk>plain text</think>{}"),
                "a second reasoning close bypassed the first boundary");
        auto mixed_vocab = vocab;
        mixed_vocab.push_back("</think>{");
        mixed_vocab.push_back("</think>prose");
        StructuredCompiler mixed_compiler(mixed_vocab, {256});
        auto mixed = mixed_compiler.compile({StructuredOutputKind::JsonObject, {}},
                                            {.reasoning_close = "</think>"});
        mixed->fill_masks(before, {});
        require((before[8] & 2U) && !(before[8] & 4U),
                "single token spanning reasoning and content escaped grammar");
        mixed->accept(std::vector<TokenId>{257, '}', 256});
        for (const char* bad :
             {R"({"type":"array","uniqueItems":true})",
              R"({"$ref":"#/$defs/a~1b","$defs":{"a/b":{"const":1},"a~1b":{"const":2}}})",
              R"({"oneOf":[{},{}]})", R"({"$ref":"https://example.org/schema"})",
              R"({"const":1,"type":"string"})", R"({"anyOf":[{}],"type":"object"})",
              R"({"type":"integer","minimum":3,"maximum":2})", R"({"type":"number","minimum":"0"})",
              R"({"type":"number","minimum":0.0000001,"maximum":0.0000002})", R"({"minimum":0})",
              R"({"type":"number","minimum":1e30})"}) {
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
