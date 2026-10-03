#include "text/unique_strings.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

using ninfer::text::UniqueStringState;
namespace {
void require(bool condition, const char* expression, int line) {
    if (!condition) throw std::runtime_error("line " + std::to_string(line) + ": " + expression);
}
#define REQUIRE(expression) require(static_cast<bool>(expression), #expression, __LINE__)
const std::string free_array = R"({"type":"array","items":{"type":"string"},"uniqueItems":true})";
void rejected_schema(const std::string& schema) {
    bool threw = false;
    try { UniqueStringState::AnalyzeSchema(schema); }
    catch (const std::exception&) { threw = true; }
    REQUIRE(threw);
}
void bytes(UniqueStringState& state, std::string_view text) {
    for (char c : text) REQUIRE(state.accept(std::string_view(&c, 1)));
}
}

int main() {
    {
        UniqueStringState s(free_array);
        REQUIRE(s.has_constraints());
        REQUIRE(s.accept(R"(["alpha",)"));
        REQUIRE(!s.accepts(R"("alpha"])"));
        REQUIRE(!s.accept(R"("alpha"])"));
        // Failure is transactional: a different value still completes this lane.
        REQUIRE(s.accept(R"("beta"])"));
        REQUIRE(s.accept(""));
    }
    {
        UniqueStringState s(free_array);
        REQUIRE(s.accept(R"(["a",)"));
        auto fork = s;
        REQUIRE(fork.accept(R"("b",)"));
        REQUIRE(s.accept(R"("b"])"));
        REQUIRE(!fork.accepts(R"("b"])"));
        REQUIRE(fork.accept(R"("c"])"));
    }
    {
        UniqueStringState s(free_array);
        bytes(s, R"(["a",)");
        REQUIRE(!s.accepts(R"("\u0061"])"));
        REQUIRE(s.accept(R"("\u0062"])"));
    }
    {
        UniqueStringState s(free_array);
        REQUIRE(s.accept(R"(["\n",)"));
        REQUIRE(!s.accepts(R"("\u000A"])"));
        REQUIRE(s.accept(R"("\t"])"));
    }
    {
        UniqueStringState s(free_array);
        REQUIRE(s.accept(R"(["\uD83D\uDE00",)"));
        REQUIRE(!s.accepts("\"\xf0\x9f\x98\x80\"]"));
        REQUIRE(s.accept(R"("\uD83D\uDE01"])"));
    }
    {
        UniqueStringState s(free_array);
        bytes(s, "[\"\xf0\x9f\x98\x80\",");
        REQUIRE(!s.accepts(R"("\ud83d\ude00"])"));
        REQUIRE(s.accept(R"("other"])"));
    }
    {
        UniqueStringState s(free_array);
        REQUIRE(s.accept(R"(["\u0000",)"));
        REQUIRE(!s.accepts(R"("\u0000"])"));
        REQUIRE(s.accept(R"("\u0001"])"));
    }
    {
        UniqueStringState s(free_array);
        REQUIRE(s.accept("[\""));
        REQUIRE(!s.accepts(std::string(1, static_cast<char>(0xc0))));
        REQUIRE(!s.accepts(R"(\uD800x)"));
        REQUIRE(s.accept(R"(valid"])"));
    }
    {
        UniqueStringState s(R"({"type":"array","items":{"type":"string","enum":["a","ab"]},"uniqueItems":true})");
        REQUIRE(s.accept(R"(["a",)"));
        REQUIRE(s.accepts(R"("a)"));
        REQUIRE(!s.accepts(R"("a")"));
        REQUIRE(!s.accepts(R"("ab",)"));
        REQUIRE(s.accept(R"("ab")"));
        REQUIRE(!s.accepts(","));
        REQUIRE(s.accept("]"));
    }
    {
        UniqueStringState s(R"({"type":"array","items":{"enum":["page","other"]},"uniqueItems":true})");
        REQUIRE(s.accept(R"(["page",)"));
        REQUIRE(!s.accepts(R"("page")"));
        REQUIRE(!s.accepts(R"("other",)"));
        REQUIRE(s.accept(R"("other"])"));
        UniqueStringState constant(R"({"type":"array","items":{"const":"page"},"uniqueItems":true})");
        REQUIRE(constant.accept(R"(["page")"));
        REQUIRE(!constant.accepts(","));
        REQUIRE(constant.accept("]"));
    }
    {
        UniqueStringState s(R"({"type":"array","items":{"type":"string","enum":["b","c"]},"uniqueItems":true})");
        REQUIRE(s.accept("[\""));
        REQUIRE(s.accepts(R"(\u006)"));
        REQUIRE(!s.accepts(R"(\u007)"));
        bytes(s, R"(\u0062",)");
        REQUIRE(!s.accepts(R"("b")"));
        REQUIRE(s.accept(R"("\u0063"])"));
    }
    {
        UniqueStringState s(R"({"type":"array","items":{"type":"string","enum":["😀","x"]},"uniqueItems":true})");
        REQUIRE(s.accept("[\""));
        REQUIRE(s.accepts(R"(\uD8)"));
        REQUIRE(!s.accepts(R"(\uD800)"));
        bytes(s, R"(\uD83D\uDE00",)");
        REQUIRE(s.accept(R"("x"])"));
    }
    {
        UniqueStringState s(R"({"type":"object","properties":{"ignored":true,"tokens":{"type":"array","items":{"type":"string"},"uniqueItems":true}},"additionalProperties":false})");
        REQUIRE(s.accept(R"({"ignored":["a","a"],"tokens":["a",)"));
        REQUIRE(!s.accepts(R"("a"]})"));
        REQUIRE(s.accept(R"("b"]})"));
    }
    {
        UniqueStringState s(R"({"type":"array","items":{"type":"array","items":{"type":"string"},"uniqueItems":true}})");
        REQUIRE(s.accept(R"([["a"],["a"]])"));
        UniqueStringState bad(R"({"type":"array","items":{"type":"array","items":{"type":"string"},"uniqueItems":true}})");
        REQUIRE(!bad.accepts(R"([["a","a"]])"));
    }
    {
        UniqueStringState s(R"({"$defs":{"S":{"type":"string"}},"type":"array","items":{"$ref":"#/$defs/S"},"uniqueItems":true})");
        REQUIRE(s.accept(R"(["a",)"));
        REQUIRE(!s.accepts(R"("\u0061"])"));
        REQUIRE(s.accept(R"("b"])"));
    }
    {
        UniqueStringState s(R"({"type":"array","prefixItems":[{"type":"array","items":{"type":"string"},"uniqueItems":true},{"type":"array","items":{"type":"string"},"uniqueItems":true}],"items":false})");
        REQUIRE(s.accept(R"([["a"],["a"]])"));
    }
    {
        // Duplicate references revisit one immutable schema interpretation. The
        // recursion has no artificial depth/time cutoff; context memoization
        // must reach a fixed point instead of growing 1 -> 2 -> 4 aliases.
        const std::string schema = R"({"type":"object","properties":{"values":{"type":"array","items":{"type":"string"},"uniqueItems":true},"children":{"type":"array","items":{"anyOf":[{"$ref":"#"},{"$ref":"#"}]}}},"additionalProperties":false})";
        const auto started = std::chrono::steady_clock::now();
        UniqueStringState s(schema);
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        std::cout << "Duplicate recursive references plan construction milliseconds: " << elapsed << '\n';
        REQUIRE(s.accept(R"({"values":["a"],"children":[{"values":["a"],"children":[{"values":["a"],"children":[]}]}]})"));
        UniqueStringState bad(schema);
        REQUIRE(!bad.accepts(R"({"values":["a"],"children":[{"values":["b","\u0062"]}]})"));
    }
    {
        UniqueStringState s(R"({"anyOf":[{"type":"null"},{"type":"array","items":{"type":"string"},"uniqueItems":true}]})");
        REQUIRE(s.accept("null"));
        UniqueStringState array(R"({"anyOf":[{"type":"null"},{"type":"array","items":{"type":"string"},"uniqueItems":true}]})");
        REQUIRE(!array.accepts(R"(["a","a"])"));
    }
    {
        UniqueStringState s(free_array, "END");
        REQUIRE(s.accept(R"(Reasoning may contain ["a","a"] and arbitrary text E)"));
        REQUIRE(s.needs_check("ND[\"a\",\"a\"]"));
        REQUIRE(!s.accepts(R"(ND["a","a"])"));
        REQUIRE(s.accept(R"(ND["a","b"])"));
    }
    {
        UniqueStringState s(free_array, "END");
        REQUIRE(s.accept(R"(thought END<tool>["a","a"]</tool>)"));
        REQUIRE(!s.needs_check(R"(["a","a"])"));
        REQUIRE(s.accept(R"(["a","a"])"));
    }
    {
        UniqueStringState s(R"({"anyOf":[{"type":"object","properties":{"x":{"type":"array","items":{"type":"string"},"uniqueItems":true}},"additionalProperties":false},{"type":"object","properties":{"other":{"type":"string"}},"additionalProperties":false}]})");
        REQUIRE(s.accept(R"({"x":["a",)"));
        REQUIRE(!s.accepts(R"("a"]})"));
        REQUIRE(s.accept(R"("b"]})"));
    }
    {
        const std::string schema = R"({"anyOf":[{"type":"object","properties":{"kind":{"const":"a"},"x":{"type":"array","items":{"type":"string"},"uniqueItems":true}},"required":["kind"],"additionalProperties":false},{"type":"object","properties":{"kind":{"const":"b"}},"required":["kind"],"additionalProperties":true}]})";
        UniqueStringState constrained(schema);
        REQUIRE(constrained.accept(R"({"kind":"a",)"));
        REQUIRE(constrained.needs_check(R"("z)"));
        REQUIRE(!constrained.accepts(R"("z)"));
        REQUIRE(constrained.accept(R"("x":)"));
        REQUIRE(constrained.needs_check("123"));
        REQUIRE(!constrained.accepts("123"));
        REQUIRE(constrained.needs_check("true"));
        REQUIRE(!constrained.accepts("true"));
        REQUIRE(constrained.accept(R"(["v",)"));
        REQUIRE(!constrained.accepts(R"("v"]})"));
        UniqueStringState open(schema);
        REQUIRE(open.accept(R"({"kind":"b","x":["v","v"]})"));
    }
    {
        UniqueStringState s(R"({"type":"array","items":{"type":"integer"},"uniqueItems":true,"maxItems":1})");
        REQUIRE(!s.has_constraints());
        UniqueStringState off(R"({"type":"array","items":{"type":"object"},"uniqueItems":false})");
        REQUIRE(!off.has_constraints());
        REQUIRE(!off.needs_check("anything"));
    }
    rejected_schema(R"({"type":"array","items":{"type":"integer"},"uniqueItems":true})");
    rejected_schema(R"({"type":"array","items":true,"uniqueItems":true})");
    rejected_schema(R"({"type":"array","items":{"enum":["page",1]},"uniqueItems":true})");
    rejected_schema(R"({"type":"array","items":{"type":"string","pattern":"a"},"uniqueItems":true})");
    rejected_schema(R"({"type":"array","items":{"type":"string","enum":["a"]},"minItems":2,"uniqueItems":true})");
    rejected_schema(R"({"anyOf":[{"type":"array","items":{"type":"string"},"uniqueItems":true},{"type":"array","items":{"type":"string"}}]})");
    rejected_schema(R"({"anyOf":[{"type":"object","properties":{"x":{"type":"array","items":{"type":"string"},"uniqueItems":true}},"additionalProperties":false},{"type":"object","required":["other"],"additionalProperties":true}]})");
    std::cout << "Unique string parser, masks, forks, escapes and schema contracts PASS\n";
}
