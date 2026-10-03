#include "text/structured_output.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ninfer;
using namespace ninfer::text;
void require(bool condition, const char* text) { if (!condition) throw std::runtime_error(text); }
std::vector<TokenId> tokens(std::string_view text) {
    std::vector<TokenId> result;
    for (unsigned char c : text) result.push_back(c);
    return result;
}
int main() {
    try {
        std::vector<std::string> vocab;
        for (int i=0;i<256;++i) vocab.push_back(std::string(1,static_cast<char>(i)));
        vocab.push_back("<EOS>");
        vocab.push_back("a\",\"a\"");
        vocab.push_back("a\",\"b\"");
        StructuredCompiler compiler(vocab,{256});
        const std::string schema=R"({"type":"array","items":{"type":"string"},"uniqueItems":true})";
        const std::size_t words=(vocab.size()+31)/32;
        auto grammar=compiler.compile({StructuredOutputKind::JsonSchema,schema});
        grammar->accept(tokens(R"(["a","a)"));
        std::vector<std::uint32_t> masks(words);
        grammar->fill_masks(masks,{});
        require(!(masks['"'/32] & (1U<<('"'%32))),"duplicate closing quote was allowed");
        require(masks['b'/32] & (1U<<('b'%32)),"free string lost a unique continuation");
        try { grammar->accept(tokens("\"")); require(false,"committed duplicate escaped validation"); }
        catch (const std::logic_error&) {}
        grammar->accept(tokens("b\"]")); grammar->accept(std::vector<TokenId>{256});

        auto speculative=compiler.compile({StructuredOutputKind::JsonSchema,schema});
        speculative->accept(tokens(R"(["a",)"));
        const auto draft=tokens(R"("a"])" );
        std::vector<std::uint32_t> draft_masks(words*(draft.size()+1));
        speculative->fill_masks(draft_masks,draft);
        require(!(draft_masks[2*words+'"'/32] & (1U<<('"'%32))),"MTP draft duplicate was allowed");
        auto sibling=speculative->fork();
        sibling->accept(tokens(R"("b"])")); sibling->accept(std::vector<TokenId>{256});
        speculative->accept(tokens(R"("c"])")); speculative->accept(std::vector<TokenId>{256});

        auto spelling=compiler.compile({StructuredOutputKind::JsonSchema,schema});
        spelling->accept(tokens(R"(["a","\u0061)")); spelling->fill_masks(masks,{});
        require(!(masks['"'/32] & (1U<<('"'%32))),"escaped duplicate bypassed semantic equality");

        auto compound=compiler.compile({StructuredOutputKind::JsonSchema,schema});
        compound->accept(tokens("[\"")); compound->fill_masks(masks,{});
        require(!(masks[257/32] & (1U<<(257%32))),"multi-item token bypassed uniqueness");
        require(masks[258/32] & (1U<<(258%32)),"valid multi-item token was masked");
        compound->accept(std::vector<TokenId>{258,']',256});

        const std::string finite=R"({"type":"array","uniqueItems":true,"items":{"type":"string","enum":["A","AB"]}})";
        auto prefix=compiler.compile({StructuredOutputKind::JsonSchema,finite});
        prefix->accept(tokens(R"(["AB","A)")); prefix->fill_masks(masks,{});
        require(!(masks['B'/32] & (1U<<('B'%32))),"finite duplicate prefix reached a dead end");
        prefix->accept(tokens(R"(")")); prefix->fill_masks(masks,{});
        require(!(masks[','/32] & (1U<<(','%32))),"exhausted finite domain still allowed a comma");
        prefix->accept(tokens("]")); prefix->accept(std::vector<TokenId>{256});

        auto reasoning=compiler.compile({StructuredOutputKind::JsonSchema,schema},{.reasoning_close="</think>"});
        reasoning->accept(tokens("thinking with </thi"));
        const auto crossing=tokens(R"(nk>["a","a"])" );
        std::vector<std::uint32_t> crossing_masks(words*(crossing.size()+1));
        reasoning->fill_masks(crossing_masks,crossing);
        const auto duplicate=crossing.size()-2;
        require(!(crossing_masks[duplicate*words+'"'/32] & (1U<<('"'%32))),"reasoning crossing lost uniqueness");
        reasoning->accept(tokens(R"(nk>["a","b"])")); reasoning->accept(std::vector<TokenId>{256});
        std::cout<<"OK native uniqueness: token masks, EOS, committed/fork/draft state, escapes and reasoning\n";
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
