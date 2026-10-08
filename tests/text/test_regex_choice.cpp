#include "text/grammar.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
using Constraint = ninfer::OutputConstraint;
using Compiler   = ninfer::text::GrammarCompiler;
using Json       = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Compiler compiler() {
    std::vector<std::string> vocabulary;
    for (int i = 0; i < 256; ++i) vocabulary.emplace_back(1, static_cast<char>(i));
    vocabulary.emplace_back();
    return {std::move(vocabulary), {256}, 16 * 1024 * 1024};
}

bool accepts(Compiler& compiler, const Constraint& constraint, std::string_view candidate) {
    auto session = compiler.compile(constraint, {}, {});
    std::vector<std::uint32_t> words(session->mask_words());
    const auto allows = [&](unsigned token) {
        return session->masks({}, words) == 0 && (words[token / 32] & (1u << (token % 32)));
    };
    for (unsigned char token : candidate) {
        if (!allows(token)) return false;
        session->accept(token);
        session->confirm();
    }
    return allows(256);
}

void choice_masks() {
    std::vector<std::string> vocabulary;
    for (int i = 0; i < 256; ++i) vocabulary.emplace_back(1, static_cast<char>(i));
    for (const auto token : {"ab", "b!", "a|b", "你", "好"}) vocabulary.emplace_back(token);
    const int eos = static_cast<int>(vocabulary.size());
    vocabulary.emplace_back();
    vocabulary.emplace_back();
    Compiler compiler(vocabulary, {eos}, 1024 * 1024);
    const std::set<std::string> language{
        "", "a", "ab", "a|b", "你好", "\"\\\n", std::string("a\0b", 3)};
    std::vector<std::string> choices(language.rbegin(), language.rend());
    choices.push_back("ab");
    std::set<std::string> prefixes;
    for (const auto& word : language)
        for (std::size_t n = 0; n <= word.size(); ++n) prefixes.insert(word.substr(0, n));
    for (const auto& prefix : prefixes) {
        auto session = compiler.compile(Constraint::choice(choices), {}, prefix);
        std::vector<std::uint32_t> words(session->mask_words());
        require(session->masks({}, words) == 0, "choice reached a dead end");
        for (unsigned token = 0; token < words.size() * 32; ++token) {
            bool expected = token == eos && language.contains(prefix);
            if (token < vocabulary.size() && !vocabulary[token].empty()) {
                const auto next = prefix + vocabulary[token];
                expected        = std::any_of(language.begin(), language.end(),
                                              [&](const auto& value) { return value.starts_with(next); });
            }
            require(bool(words[token / 32] & (1u << (token % 32))) == expected,
                    "choice mask differs from literal-set prefix membership");
            if (expected) {
                session->accept(token);
                session->discard();
                std::vector<std::uint32_t> restored(words.size());
                require(session->masks({}, restored) == 0 && restored == words,
                        "choice rollback changed the next-token set");
            }
        }
    }
}

void contracts(Compiler& compiler) {
    choice_masks();
    require(accepts(compiler, Constraint::regex(""), "") &&
                !accepts(compiler, Constraint::regex(""), "a"),
            "empty regex language changed");
    require(accepts(compiler, Constraint::regex("a(b|c){1,2}"), "abc") &&
                !accepts(compiler, Constraint::regex("a(b|c){1,2}"), "xabc"),
            "regex must match complete content");
    require(accepts(compiler, Constraint::regex("^a|b$"), "a") &&
                accepts(compiler, Constraint::regex("^a|b$"), "b"),
            "branch anchors changed");
    require(accepts(compiler, Constraint::regex("(|a)b?"), "") &&
                accepts(compiler, Constraint::regex("(a|)b"), "b"),
            "empty alternative changed");
    for (const auto& space : {" ", "\t", "\v", "\u2003", "\ufeff"}) {
        require(accepts(compiler, Constraint::regex(R"(\s)"), space),
                "Unicode whitespace rejected");
        require(!accepts(compiler, Constraint::regex(R"(\S)"), space),
                "negated whitespace changed");
    }
    for (const auto& line : {"\n", "\r", "\u2028", "\u2029"})
        require(!accepts(compiler, Constraint::regex("."), line), "dot admitted a line terminator");
    require(accepts(compiler, Constraint::regex("[你好😀]{1,2}"), "你😀"),
            "regex Unicode character count changed");
    require(!accepts(compiler, Constraint::regex(R"(\d+)"), "１２") &&
                !accepts(compiler, Constraint::regex(R"(\w+)"), "你"),
            "ASCII digit/word class changed");
    for (const auto pattern : {".", "[^a]", R"([\u0000-\uFFFF])", "[^]*"})
        require(!accepts(compiler, Constraint::regex(pattern), "\xed\xa0\x80"),
                "regex admitted surrogate UTF-8");
    const std::string nul("a\0b", 3);
    require(accepts(compiler, Constraint::regex(nul), nul) &&
                !accepts(compiler, Constraint::regex(nul), "a"),
            "NUL truncated regex source");
    require(accepts(compiler, Constraint::regex(R"(a\x00b)"), nul), "hexadecimal NUL rejected");
    require(accepts(compiler, Constraint::regex(R"(\x41B)"), "AB") &&
                !accepts(compiler, Constraint::regex(R"(\x41B)"), "Л"),
            "hex escape consumed a following hexadecimal literal");
    require(accepts(compiler, Constraint::choice({"a:b", "c"}), "a:b") &&
                !accepts(compiler, Constraint::choice({"a", "b:c"}), "a:b"),
            "choice cache confused literal boundaries");
    require(accepts(compiler, Constraint::regex("a|b"), "a") &&
                !accepts(compiler, Constraint::choice({"a|b"}), "a"),
            "choice interpreted literal regex syntax");
    require(accepts(compiler, Constraint::json_schema(R"({"type":"string","pattern":"p"})"),
                    "\"apple\"") &&
                !accepts(compiler, Constraint::regex("p"), "apple"),
            "regex changed JSON Schema pattern search semantics");

    for (const auto& constraint :
         {Constraint::choice({}), Constraint::choice({"\xff"}), Constraint::regex("\\q"),
          Constraint::regex("a^b"), Constraint::regex("(a$)"), Constraint::regex("["),
          Constraint::regex("(a"), Constraint::regex("a{2,1}"), Constraint::regex("(?=a)a"),
          Constraint::regex("(?<=a)b"), Constraint::regex("(a)\\1"), Constraint::regex("\\bword"),
          Constraint::regex("\\p{L}"), Constraint::regex("(?i)a"), Constraint::regex("\\uD800"),
          Constraint::regex("\xc0\xaf")}) {
        bool rejected = false;
        try {
            (void)compiler.compile(constraint, {}, {});
        } catch (const ninfer::RequestError& error) {
            rejected = error.kind() == ninfer::text::constraint_error_kind(constraint.kind);
        }
        require(rejected, "invalid choice/regex was accepted or misclassified");
    }
    for (const auto& constraint : {Constraint::choice({"abc"}), Constraint::regex("abc")}) {
        auto session = compiler.compile(constraint, {}, "ab");
        session->accept('c');
        session->confirm();
        session->accept(256);
        session->discard();
        bool rejected = false;
        try {
            (void)compiler.compile(constraint, {}, "ax");
        } catch (const ninfer::RequestError&) { rejected = true; }
        require(rejected, "invalid continuation accepted");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        auto compiled = compiler();
        if (argc == 2 && std::string_view(argv[1]) == "--probe") {
            std::string line;
            while (std::getline(std::cin, line)) {
                Json result;
                try {
                    const auto input = Json::parse(line);
                    const auto constraint =
                        input.contains("grammar")
                            ? Constraint::grammar(input["grammar"].get<std::string>())
                        : input.contains("regex")
                            ? Constraint::regex(input["regex"].get<std::string>())
                            : Constraint::choice(input["choice"].get<std::vector<std::string>>());
                    (void)compiled.compile(constraint, {}, {});
                    result["accepted"] = Json::array();
                    for (const auto& candidate : input["candidates"])
                        result["accepted"].push_back(
                            accepts(compiled, constraint, candidate.get<std::string>()));
                } catch (const ninfer::RequestError& error) { result = {{"error", error.what()}}; }
                std::cout << result.dump() << '\n';
            }
        } else {
            contracts(compiled);
            std::cout << "choice/regex contracts passed\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
