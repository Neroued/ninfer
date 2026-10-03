#include "text/structured_output.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::text;
namespace {
void require(bool condition, const char* expression, int line) {
    if (!condition) throw std::runtime_error("line " + std::to_string(line) + ": " + expression);
}
#define REQUIRE(expression) require(static_cast<bool>(expression), #expression, __LINE__)
void feed(const std::shared_ptr<GrammarState>& state, const std::string& bytes) {
    std::vector<TokenId> tokens;
    for (unsigned char byte : bytes) tokens.push_back(byte);
    state->accept(tokens);
}
bool allowed(const std::vector<std::uint32_t>& masks, TokenId id) {
    return (masks[id / 32] & (std::uint32_t{1} << (id % 32))) != 0;
}
void consistency(const std::shared_ptr<GrammarState>& state, std::size_t vocabulary_size) {
    std::vector<std::uint32_t> masks((vocabulary_size + 31) / 32);
    state->fill_masks(masks, {});
    for (std::size_t token = 0; token < vocabulary_size; ++token) if (allowed(masks, static_cast<TokenId>(token))) {
        auto trial = state->fork();
        trial->accept(std::vector<TokenId>{static_cast<TokenId>(token)});
    }
}
}
int main() {
    // Raw byte vocabulary, composite tokens and EOS are protocol fixtures. There
    // is no model/tokenizer/GPU initialization in this test.
    std::vector<std::string> vocabulary(256);
    for (std::size_t byte = 0; byte < vocabulary.size(); ++byte) vocabulary[byte] = std::string(1, static_cast<char>(byte));
    const TokenId number = static_cast<TokenId>(vocabulary.size()); vocabulary.push_back("123");
    const TokenId boolean = static_cast<TokenId>(vocabulary.size()); vocabulary.push_back("true");
    const TokenId bad_key = static_cast<TokenId>(vocabulary.size()); vocabulary.push_back("\"z");
    const TokenId eos = static_cast<TokenId>(vocabulary.size()); vocabulary.push_back("");
    StructuredCompiler compiler(vocabulary, {eos});
    const std::string schema = R"({"anyOf":[{"type":"object","properties":{"kind":{"const":"a"},"x":{"type":"array","items":{"type":"string"},"uniqueItems":true}},"required":["kind"],"additionalProperties":false},{"type":"object","properties":{"kind":{"const":"b"}},"required":["kind"],"additionalProperties":true}]})";
    auto state = compiler.compile({StructuredOutputKind::JsonSchema, schema});
    feed(state, R"({"kind":"a",)" );
    consistency(state, vocabulary.size());
    std::vector<std::uint32_t> masks((vocabulary.size() + 31) / 32);
    state->fill_masks(masks, {});
    REQUIRE(!allowed(masks, bad_key));
    feed(state, R"("x":)" );
    consistency(state, vocabulary.size());
    state->fill_masks(masks, {});
    REQUIRE(!allowed(masks, number));
    REQUIRE(!allowed(masks, boolean));
    feed(state, R"(["v",)" );
    consistency(state, vocabulary.size());
    feed(state, R"("other"]})" );
    consistency(state, vocabulary.size());
    state->fill_masks(masks, {});
    REQUIRE(allowed(masks, eos));
    std::cout << "Unique semantic mask and committed matcher acceptance agree PASS\n";
}
