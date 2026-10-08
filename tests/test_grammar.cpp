#include "text/grammar.h"

#include <algorithm>
#include <array>
#include <future>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<std::uint32_t> mask(ninfer::text::GrammarSession& session) {
    std::vector<std::uint32_t> result(session.mask_words());
    require(session.masks({}, result) == 0, "unexpected grammar dead end");
    return result;
}

bool allows(const std::vector<std::uint32_t>& words, int token) {
    return (words.at(token / 32) & (1u << (token % 32))) != 0;
}

auto compile(ninfer::text::GrammarCompiler& compiler, const std::string& source,
             std::string_view close, std::string_view continuation) {
    return compiler.compile(ninfer::OutputConstraint::grammar(source), close, continuation);
}

void consume(ninfer::text::GrammarSession& session, std::string_view text) {
    for (unsigned char byte : text) {
        require(allows(mask(session), byte), "grammar mask rejected valid bytes");
        session.accept(byte);
        session.confirm();
    }
}

void check_finite_language_masks() {
    const std::vector<std::string> vocabulary{"a",   "b",    "ab",   "ba",   "aaa",      "bb",
                                              "你",  "\xe4", "\xbd", "\xa0", "\xe4\xbd", "\xbd\xa0",
                                              "a你", "b你",  "a",    "x",    "",         ""};
    constexpr int eos = 16;
    ninfer::text::GrammarCompiler compiler(vocabulary, {eos}, 1024 * 1024);

    // Enumerate the complete language independently of the grammar parser and tokenizer.
    std::set<std::string> language;
    std::set<std::string> prefixes;
    for (int length = 1; length <= 3; ++length) {
        for (int bits = 0; bits < (1 << length); ++bits) {
            std::string word;
            for (int bit = 0; bit < length; ++bit) { word += bits & (1 << bit) ? 'b' : 'a'; }
            language.insert(word);
            language.insert(word + "你");
        }
    }
    for (const auto& word : language) {
        for (std::size_t size = 0; size <= word.size(); ++size) {
            prefixes.insert(word.substr(0, size));
        }
    }
    for (const auto& source :
         {"root ::= (\"a\" | \"b\"){1,3} \"你\"?", "root ::= letters suffix\n"
                                                   "letters ::= (\"a\" | \"b\"){1,3} (= suffix)\n"
                                                   "suffix ::= \"你\"?"}) {
        for (const auto& prefix : prefixes) {
            auto session     = compile(compiler, source, {}, prefix);
            const auto words = mask(*session);
            for (int token = 0; token < static_cast<int>(words.size() * 32); ++token) {
                bool expected = token == eos && language.contains(prefix);
                if (token < static_cast<int>(vocabulary.size()) && !vocabulary[token].empty()) {
                    const auto next = prefix + vocabulary[token];
                    expected = std::any_of(language.begin(), language.end(), [&](const auto& word) {
                        return word.starts_with(next);
                    });
                }
                require(allows(words, token) == expected, "mask differs from enumerated language");
                if (expected) {
                    session->accept(token);
                    session->discard();
                    require(mask(*session) == words, "rollback changed the legal token set");
                }
            }
        }
    }
}

void check_compiled_grammar_lifetime() {
    std::unique_ptr<ninfer::text::GrammarSession> session;
    {
        ninfer::text::GrammarCompiler compiler({"a", "b", ""}, {2}, 1);
        session    = compile(compiler, "root ::= \"ab\"", {}, {});
        auto other = compile(compiler, "root ::= \"ba\"", {}, {});
        require(allows(mask(*other), 1), "small-cache compilation failed");
    }
    session->accept(0);
    session->confirm();
    require(allows(mask(*session), 1), "compiler destruction invalidated active matcher");
    session->accept(1);
    session->confirm();
    require(allows(mask(*session), 2), "evicted grammar did not complete");
}

void check_draft_lookahead() {
    // IDs: a=0, b=1, c=2, EOS=3, forbidden special=4.
    ninfer::text::GrammarCompiler compiler({"a", "b", "c", "", ""}, {3}, 1024 * 1024);
    auto session       = compile(compiler, "root ::= \"ab\" | \"ac\"", {}, {});
    const auto initial = mask(*session);
    std::ostringstream diagnostics;
    auto* previous = std::cerr.rdbuf(diagnostics.rdbuf());
    try {
        for (int invalid : {0, 3, 4, -1, 5}) {
            const std::array<std::int32_t, 2> drafts{0, invalid};
            std::vector<std::uint32_t> words(3 * session->mask_words());
            require(session->masks(drafts, words) == 0, "invalid draft must not fail request");
            require(words[0] == (1u << 0) && words[1] == ((1u << 1) | (1u << 2)),
                    "invalid draft changed the target's legal correction tokens");
            require(mask(*session) == initial, "rejected draft did not restore matcher state");
        }
        const std::array<std::int32_t, 4> drafts{0, 1, 3, 4};
        std::vector<std::uint32_t> words(5 * session->mask_words());
        require(session->masks(drafts, words) == 0, "draft past EOS must not fail request");
        require(words[2] == (1u << 3), "complete draft must allow EOS");
        require(mask(*session) == initial, "lookahead through EOS did not restore matcher state");
    } catch (...) {
        std::cerr.rdbuf(previous);
        throw;
    }
    std::cerr.rdbuf(previous);
    require(diagnostics.str().empty(), "expected draft rejection emitted diagnostics");
    session->accept(0);
    session->confirm();
    session->accept(2);
    session->confirm();
    require(allows(mask(*session), 3), "generation failed after rejected draft lookahead");
}
} // namespace

int main() {
    try {
        check_finite_language_masks();
        check_compiled_grammar_lifetime();
        check_draft_lookahead();
        std::vector<std::string> vocab;
        for (int byte = 0; byte < 256; ++byte) { vocab.emplace_back(1, static_cast<char>(byte)); }
        vocab.emplace_back();                        // EOS
        vocab.emplace_back();                        // Forbidden special token
        vocab.emplace_back("\n</think>\n\n answer"); // One token crosses the channel boundary.
        ninfer::text::GrammarCompiler compiler(std::move(vocab), {256}, 4 * 1024 * 1024);

        auto observed = compiler.compile(ninfer::OutputConstraint::regex("a+"), {}, {});
        observed->observe(true, 0.25);
        require(observed->observation().cache == ninfer::ConstraintCacheAccess::Built &&
                    !observed->observation().complete,
                "cold constraint observation");
        const auto initial_positions = observed->observation().mask_positions;
        std::vector<std::uint32_t> lookahead(2 * observed->mask_words());
        observed->masks(std::array<std::int32_t, 1>{'a'}, lookahead);
        require(!observed->observation().complete &&
                    observed->observation().mask_positions == initial_positions + 2,
                "lookahead completion escaped rollback or lost real work");
        observed->accept('a');
        observed->confirm();
        require(observed->observation().complete && !observed->observation().terminated,
                "complete prefix was confused with EOS");
        observed->accept(256);
        observed->discard();
        require(observed->observation().complete && !observed->observation().terminated,
                "discarded EOS changed committed completion");
        observed->accept(256);
        observed->confirm();
        observed->uploaded(64);
        require(observed->observation().terminated &&
                    observed->observation().mask_upload_bytes == 64,
                "termination or transfer observation missing");
        auto reused = compiler.compile(ninfer::OutputConstraint::regex("a+"), {}, "a");
        require(reused->observation().cache == ninfer::ConstraintCacheAccess::Hit &&
                    reused->observation().complete && reused->observation().mask_positions == 0,
                "cache reused request state or lost continuation completion");

        auto literal = compile(compiler, "root ::= \"ab\" | \"ac\"", {}, {});
        require(allows(mask(*literal), 'a') && !allows(mask(*literal), 256),
                "initial literal mask");
        literal->accept('a');
        literal->discard();
        require(allows(mask(*literal), 'a'), "discard did not restore matcher");
        consume(*literal, "ac");
        require(allows(mask(*literal), 256) && !allows(mask(*literal), 'a') &&
                    !allows(mask(*literal), 257),
                "completion must admit only EOS");
        literal->accept(256);
        literal->discard();
        require(allows(mask(*literal), 256), "EOS rollback");
        auto token_rule = compile(compiler, "root ::= Token(97) \"b\"", {}, {});
        consume(*token_rule, "ab");
        require(allows(mask(*token_rule), 256), "ordinary token reference did not complete");
        auto empty = compile(compiler, "root ::= \"\"", {}, {});
        require(allows(mask(*empty), 256) && !allows(mask(*empty), 'a'),
                "empty grammar must allow EOS");

        auto plain_a = compile(compiler, "root ::= \"a\"", {}, {});
        consume(*plain_a, "a");
        require(allows(mask(*plain_a), 256), "plain literal did not complete");
        auto embedded_nul = compile(compiler, "root ::= \"a\\0b\"", {}, {});
        consume(*embedded_nul, "a");
        require(allows(mask(*embedded_nul), 0) && !allows(mask(*embedded_nul), 256),
                "embedded NUL was truncated during grammar compilation or cache lookup");
        consume(*embedded_nul, std::string_view("\0b", 2));
        require(allows(mask(*embedded_nul), 256), "embedded NUL literal did not complete");

        auto unicode = compile(compiler,
                               "# recursive Unicode lists\nroot ::= \"[\" (item (\",\" "
                               "item)*)? \"]\"\nitem ::= [你界] | root",
                               {}, {});
        consume(*unicode, "[你,[界,你],[]]");
        require(allows(mask(*unicode), 256), "recursive Unicode completion");
        auto repeat = compile(compiler, "root ::= [a-c]{2,4} \"!\"?", {}, "ab");
        consume(*repeat, "c!");
        require(allows(mask(*repeat), 256), "continuation and bounded repetition");

        auto thinking = compile(compiler, "root ::= \" answer\"", "\n</think>\n\n", {});
        consume(*thinking, "reason\n<other>\n");
        require(allows(mask(*thinking), 258), "cross-boundary token not admitted");
        thinking->accept(258);
        thinking->confirm();
        require(allows(mask(*thinking), 256), "thinking wrapper did not reach content completion");

        for (const auto& source : {"bad grammar", "root ::= missing", "root ::= Token(256)",
                                   "root ::= Regex(\".*\")", "root[temperature=0.5] ::= \"a\""}) {
            bool rejected = false;
            try {
                (void)compile(compiler, source, {}, {});
            } catch (const std::exception&) { rejected = true; }
            require(rejected, "invalid grammar or special token reference accepted");
        }
        bool rejected = false;
        try {
            (void)compile(compiler, "root ::= \"abc\"", {}, "ax");
        } catch (const std::exception&) { rejected = true; }
        require(rejected, "invalid continuation accepted");
        std::vector<std::future<void>> concurrent;
        for (int i = 0; i < 8; ++i) {
            concurrent.push_back(std::async(std::launch::async, [&compiler] {
                auto session = compile(compiler, "root ::= \"independent\"", {}, {});
                consume(*session, "independent");
                require(allows(mask(*session), 256),
                        "shared compilation shared mutable matcher state");
            }));
        }
        for (auto& task : concurrent) { task.get(); }
        ninfer::text::GrammarCompiler limited({"a", ""}, {1}, 1024 * 1024);
        auto dead = compile(limited, "root ::= \"ab\"", {}, {});
        std::vector<std::uint32_t> dead_words(2 * dead->mask_words());
        require(dead->masks(std::array<std::int32_t, 1>{0}, dead_words) == 2,
                "reachable dead end was not assigned to its prediction position");
        require(allows(mask(*dead), 0), "dead lookahead damaged accepted state");
        std::cout << "grammar: literals, recursion, Unicode, repetition, EOS, transactions, "
                     "framing passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
