// capy-flash structured output limits (OutputConstraintLimits): input limits before anything
// recursive, compilation deadline, empty languages, matcher time budgets of a generating request,
// and the vendor cache that must not keep failed compilations. CPU only; a byte-level vocabulary
// makes every language tokenizable.

#include "text/constraint_limits.h"
#include "text/grammar.h"

#include "support/thread_safe_cache.h"
#include "support/work_budget.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Json  = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
using ninfer::OutputConstraint;
using ninfer::OutputConstraintLimits;
using ninfer::RequestError;
using ninfer::RequestErrorKind;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) return;
    std::cerr << "FAIL: " << label << '\n';
    ++failures;
}

// The error kind (and message) of a call, or nullopt when it does not throw RequestError.
std::optional<RequestError> request_error(const std::function<void()>& function) {
    try {
        function();
    } catch (const RequestError& error) { return error; }
    return std::nullopt;
}

bool rejected(const std::function<void()>& function, RequestErrorKind kind,
              const std::string& needle = {}) {
    const auto error = request_error(function);
    return error && error->kind() == kind &&
           std::string(error->what()).find(needle) != std::string::npos;
}

std::string nested_items_text(std::size_t levels) {
    std::string text;
    for (std::size_t i = 0; i < levels; ++i) text += R"({"type":"array","items":)";
    text += R"({"type":"integer"})";
    return text + std::string(levels, '}');
}

std::vector<std::string> byte_vocabulary() {
    std::vector<std::string> vocabulary;
    for (int byte = 0; byte < 256; ++byte) vocabulary.emplace_back(1, static_cast<char>(byte));
    vocabulary.emplace_back(); // EOS
    return vocabulary;
}

constexpr int kEos = 256;

void test_json_document_limits() {
    const OutputConstraintLimits limits;
    // 100 000 levels: iterative check, no stack overflow (the audit's crash was at 5 000).
    const Json deep  = Json::parse(nested_items_text(100000));
    const auto error = request_error([&] { ninfer::text::check_json_schema_limits(deep, limits); });
    check(error && error->kind() == RequestErrorKind::ConstraintLimitExceeded &&
              error->pointer().starts_with("/items/items") &&
              error->pointer().size() == 256 * std::string("/items").size(),
          "deep schema: pointer of the first container past the limit");
    check(!request_error([&] {
        ninfer::text::check_json_schema_limits(Json::parse(nested_items_text(255)), limits);
    }),
          "255 nested schemas (256 containers) are within the default depth");
    Json wide = {{"type", "object"}, {"properties", Json::object()}};
    for (int i = 0; i < 4097; ++i) wide["properties"]["p" + std::to_string(i)] = Json::object();
    const auto members =
        request_error([&] { ninfer::text::check_json_schema_limits(wide, limits); });
    check(members && members->pointer() == "/properties" &&
              std::string(members->what()).find("--constraint-max-items") != std::string::npos,
          "object members over the limit");
    Json values = Json::array();
    for (int i = 0; i < 300; ++i) values.push_back(Json::array({1, 2, 3, 4}));
    check(rejected([&] { ninfer::text::check_json_schema_limits(values, {.max_bytes = 1000}); },
                   RequestErrorKind::ConstraintLimitExceeded, "--constraint-max-kib"),
          "value count bounds the size of a parsed document");
}

void test_source_limits() {
    const OutputConstraintLimits limits;
    using ninfer::text::check_constraint_limits;
    const auto over = [&](const OutputConstraint& constraint, const char* flag) {
        return rejected([&] { check_constraint_limits(constraint, limits); },
                        RequestErrorKind::ConstraintLimitExceeded, flag);
    };
    const auto within = [&](const OutputConstraint& constraint) {
        return !request_error([&] { check_constraint_limits(constraint, limits); });
    };
    // JSON text: the depth scan does not parse, so even 200 000 levels are cheap.
    check(over(OutputConstraint::json_schema(std::string(200000, '[')), "--constraint-max-depth"),
          "JSON text depth");
    check(within(OutputConstraint::json_schema(R"({"pattern":"[[[[[[[[[[[[[[[[[[[["})")),
          "brackets inside JSON strings are not nesting");
    check(over(OutputConstraint::json_schema(std::string(300 * 1024, ' ')), "--constraint-max-kib"),
          "JSON text bytes");
    const std::string open(257, '('), close(257, ')');
    check(over(OutputConstraint::regex(open + "a" + close), "--constraint-max-depth"),
          "regex group depth");
    check(within(OutputConstraint::regex(std::string(256, '(') + "a" + std::string(256, ')'))),
          "regex at the depth limit");
    check(within(OutputConstraint::regex("[(" + std::string(500, '(') + "]\\(\\(\\(")),
          "bracketed and escaped parentheses are not groups");
    check(within(OutputConstraint::regex("[]((((]x")), "a leading ] is a class member");
    check(over(OutputConstraint::grammar("root ::= " + open + "\"a\"" + close),
               "--constraint-max-depth"),
          "GBNF group depth");
    check(within(OutputConstraint::grammar("root ::= \"" + open + "\" [(] # ((((((\n")),
          "parentheses in GBNF literals, classes and comments are not groups");
    std::string rules = "root ::= r0\n";
    for (int i = 0; i < 5000; ++i)
        rules += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) + "\n";
    rules += "r5000 ::= \"a\"\n";
    check(over(OutputConstraint::grammar(rules), "--constraint-max-items"), "GBNF rule count");
    std::vector<std::string> choices(4097, "x");
    check(over(OutputConstraint::choice(choices), "--constraint-max-items"), "choice entries");
    check(over(OutputConstraint::choice({std::string(300 * 1024, 'x')}), "--constraint-max-kib"),
          "choice bytes");
}

void test_compilation_limits() {
    using ninfer::text::GrammarCompiler;
    GrammarCompiler compiler(byte_vocabulary(), {kEos}, 64U << 20);
    // B2 through the compiler (direct Engine users): a 5 000-level schema is refused, not a crash.
    check(rejected(
              [&] {
                  (void)compiler.compile(OutputConstraint::json_schema(nested_items_text(5000)), {},
                                         {});
              },
              RequestErrorKind::ConstraintLimitExceeded, "--constraint-max-depth"),
          "deep schema refused by the compiler");
    // M4: languages without any first token are refused at compilation, with and without the
    // reasoning prefix, instead of a dead end after the prefill.
    for (const std::string close : {std::string(), std::string("</t>\n")}) {
        check(rejected(
                  [&] {
                      (void)compiler.compile(
                          OutputConstraint::grammar("root ::= [^\\x00-\\U0010FFFF]"), close, {});
                  },
                  RequestErrorKind::InvalidGrammar, "accepts no output"),
              "empty GBNF language");
        check(rejected(
                  [&] { (void)compiler.compile(OutputConstraint::regex("[^\\s\\S]"), close, {}); },
                  RequestErrorKind::InvalidRegex, "accepts no output"),
              "empty regex language");
        check(
            !request_error([&] { (void)compiler.compile(OutputConstraint::regex(""), close, {}); }),
            "the empty-string language is not empty");
        check(!request_error([&] {
            (void)compiler.compile(OutputConstraint::grammar("root ::= \"yes\" | \"no\""), close,
                                   {});
        }),
              "an ordinary grammar is accepted");
    }
    // Compilation deadline: a schema that takes longer than 1 ms is refused with the limit.
    GrammarCompiler hurried(byte_vocabulary(), {kEos}, 64U << 20, {.compile_timeout_ms = 1});
    Json properties = Json::object();
    for (int i = 0; i < 400; ++i) properties["p" + std::to_string(i)] = {{"type", "string"}};
    const std::string slow =
        Json{{"type", "object"}, {"properties", properties}, {"additionalProperties", false}}
            .dump();
    const auto started = Clock::now();
    check(rejected([&] { (void)hurried.compile(OutputConstraint::json_schema(slow), {}, {}); },
                   RequestErrorKind::ConstraintLimitExceeded, "--constraint-compile-timeout-ms"),
          "compilation over its deadline");
    std::cout << "compile deadline 1 ms: refused after "
              << std::chrono::duration<double, std::milli>(Clock::now() - started).count()
              << " ms\n";
    check(!request_error(
              [&] { (void)compiler.compile(OutputConstraint::json_schema(slow), {}, {}); }),
          "the same schema compiles without the deadline");
}

// Walks a session with one byte token per round (mask, accept, confirm) until it reports a dead
// end or `steps` tokens. Returns the steps taken.
int walk(ninfer::text::GrammarSession& session, int token, int steps) {
    std::vector<std::uint32_t> words(session.mask_words());
    for (int step = 0; step < steps; ++step) {
        if (session.masks({}, words) != 0) return step;
        try {
            session.accept(token);
        } catch (const std::logic_error&) {
            // A budget overrun inside the accept: the Engine closes the row the same way.
            session.discard();
            return step;
        }
        session.confirm();
    }
    return steps;
}

void test_matcher_budget() {
    using ninfer::text::GrammarCompiler;
    // B3: the audit's ambiguous grammar and regex. Without a budget each mask costs more as the
    // output grows (seconds per token after ~1000 tokens); with it only this session fails, fast.
    const OutputConstraintLimits tight{
        .round_budget_ms = 20, .request_budget_ms = 200, .round_allowance_ms = 1};
    GrammarCompiler compiler(byte_vocabulary(), {kEos}, 64U << 20, tight);
    for (const auto& constraint :
         {OutputConstraint::grammar("root ::= e\ne ::= e e | \"a\" | \"b\""),
          OutputConstraint::regex("(a*)*"), OutputConstraint::regex("((a|b)*)*")}) {
        auto session         = compiler.compile(constraint, {}, {});
        const auto started   = Clock::now();
        const int steps      = walk(*session, 'a', 100000);
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        std::cout << "budget: " << constraint.source << " failed after " << steps << " tokens, "
                  << seconds
                  << " s: " << (session->budget_failure() ? *session->budget_failure() : "-")
                  << '\n';
        check(steps < 100000 && session->budget_failure() != nullptr,
              "ambiguous constraint did not exhaust its budget: " + constraint.source);
        check(seconds < 5.0, "budget failure took too long: " + constraint.source);
        // Failed for good: every position dead with a one-token safe mask, accept refused,
        // discard and observation safe.
        std::vector<std::uint32_t> words(session->mask_words() * 4);
        const std::vector<std::int32_t> drafts{'a', 'a', 'a'};
        check(session->masks(drafts, words) == 0xFU && words[0] == 1U &&
                  words[session->mask_words()] == 1U,
              "a failed session must report every position dead");
        // accept() may still succeed when the matcher is consistent (a budget overrun measured
        // after the step); after an interrupted step it refuses. Either way nothing breaks.
        try {
            session->accept('a');
        } catch (const std::logic_error&) {}
        session->discard();
        (void)session->observation();
    }
    // Per-round interruption inside the parser: one round over 1 ms fails the session even with
    // no cumulative budget.
    GrammarCompiler strict(byte_vocabulary(), {kEos}, 64U << 20,
                           {.round_budget_ms = 1, .request_budget_ms = 0});
    auto session =
        strict.compile(OutputConstraint::grammar("root ::= e\ne ::= e e | \"a\""), {}, {});
    const int steps = walk(*session, 'a', 100000);
    check(session->budget_failure() &&
              session->budget_failure()->find("--constraint-round-budget-ms") != std::string::npos,
          "per-round budget did not fail the session");
    std::cout << "round budget 1 ms: failed after " << steps << " tokens\n";
    // Memory: the history of an ambiguous grammar grows quadratically; the bound fails the session
    // before it holds more than --constraint-matcher-mib.
    GrammarCompiler bounded(
        byte_vocabulary(), {kEos}, 64U << 20,
        {.round_budget_ms = 0, .request_budget_ms = 0, .matcher_bytes = 1U << 20});
    auto ambiguous    = bounded.compile(OutputConstraint::regex("((a|b)*)*"), {}, {});
    const int reached = walk(*ambiguous, 'a', 100000);
    check(ambiguous->budget_failure() &&
              ambiguous->budget_failure()->find("--constraint-matcher-mib") != std::string::npos,
          "matcher memory bound did not fail the session");
    std::cout << "matcher bound 1 MiB: failed after " << reached << " tokens\n";
    // Ordinary constraints stay far below the default budget.
    GrammarCompiler defaults(byte_vocabulary(), {kEos}, 64U << 20);
    auto json = defaults.compile(
        OutputConstraint::json_schema(R"({"type":"array","items":{"type":"string"}})"), {}, {});
    std::vector<std::uint32_t> words(json->mask_words());
    for (const char c : std::string("[\"")) {
        check(json->masks({}, words) == 0, "JSON prefix");
        json->accept(static_cast<unsigned char>(c));
        json->confirm();
    }
    check(walk(*json, 'x', 5000) == 5000 && json->budget_failure() == nullptr,
          "a long ordinary JSON string must not exhaust the default budget");
}

// The vendor cache must not keep failed computations: they held the key (the whole source) at size
// 0 forever, and a later success for the same key was never computed.
struct SizeOf {
    std::size_t operator()(const int&) const { return 1; }
};

struct NoComputer {
    int operator()(const std::string&) const { return 0; }
};

void test_cache_drops_failures() {
    xgrammar::ThreadSafeLRUCache<std::string, int, NoComputer, SizeOf> cache(1024);
    bool threw = false;
    try {
        (void)cache.Get("key", [](const std::string&) -> int { throw std::runtime_error("x"); });
    } catch (const std::runtime_error&) { threw = true; }
    check(threw, "the failing computation propagates");
    int computed    = 0;
    const int value = cache.Get("key", [&](const std::string&) { return ++computed, 7; });
    check(value == 7 && computed == 1, "a failed computation stayed in the cache");
    (void)cache.Get("key", [&](const std::string&) { return ++computed, 8; });
    check(computed == 1, "a successful computation is cached");
}

} // namespace

// Second audit N1: allOf/$ref diamonds expanded 2^depth parts, outside any deadline, and the vendor
// semaphore made later compilations wait for ever.
std::string diamond(const char* op, int levels, const std::string& leaf, const std::string& salt) {
    Json defs = Json::object();
    for (int i = 0; i < levels; ++i) {
        const std::string next = "#/$defs/d" + std::to_string(i + 1);
        defs["d" + std::to_string(i)] =
            i + 1 < levels ? Json{{op, Json::array({Json{{"$ref", next}}, Json{{"$ref", next}}})}}
                           : Json::parse(leaf);
    }
    defs["salt"] = Json{{"title", salt}};
    return Json{{"$defs", defs}, {"$ref", "#/$defs/d0"}}.dump();
}

// Finite-value validation (enum against an anyOf diamond) is exponential too; it must run under
// the compilation deadline.
std::string slow_schema(const std::string& salt) {
    Json schema = Json::parse(diamond("anyOf", 40, R"({"type":"integer"})", salt));
    schema.erase("$ref");
    schema["allOf"] = Json::array({Json{{"enum", Json::array({1})}}, Json{{"$ref", "#/$defs/d0"}}});
    return schema.dump();
}

void test_composition_dag() {
    using ninfer::text::GrammarCompiler;
    GrammarCompiler compiler(byte_vocabulary(), {kEos}, 64U << 20);
    // allOf of the same reference is that reference: the 40-level diamond is one object schema.
    auto started = Clock::now();
    check(!request_error([&] {
        (void)compiler.compile(
            OutputConstraint::json_schema(diamond("allOf", 40, R"({"type":"object"})", "a")), {},
            {});
    }) && Clock::now() - started < std::chrono::seconds(2),
          "allOf diamond of depth 40 must compile at once (it expanded 2^40 parts)");
    // Distinct parts beyond --constraint-max-items: 70 groups of 64 bounds.
    Json defs = Json::object(), all = Json::array();
    for (int g = 0; g < 70; ++g) {
        Json parts = Json::array();
        for (int k = 0; k < 64; ++k) parts.push_back(Json{{"minimum", g * 100 + k}});
        defs["g" + std::to_string(g)] = Json{{"allOf", parts}};
        all.push_back(Json{{"$ref", "#/$defs/g" + std::to_string(g)}});
    }
    check(rejected(
              [&] {
                  (void)compiler.compile(
                      OutputConstraint::json_schema(Json{{"$defs", defs}, {"allOf", all}}.dump()),
                      {}, {});
              },
              RequestErrorKind::ConstraintLimitExceeded, "expands to more than"),
          "a conjunction over the part limit must be a 400");
    // Exponential validation stops at the deadline.
    GrammarCompiler timed(byte_vocabulary(), {kEos}, 64U << 20, {.compile_timeout_ms = 500});
    started = Clock::now();
    check(rejected(
              [&] { (void)timed.compile(OutputConstraint::json_schema(slow_schema("x")), {}, {}); },
              RequestErrorKind::ConstraintLimitExceeded, "--constraint-compile-timeout-ms"),
          "exponential schema validation must stop at the compilation deadline");
    const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    std::cout << "exponential normalization refused after " << seconds << " s (deadline 0.5 s)\n";
    check(seconds < 2.0, "the compilation deadline was not respected in the normalization");
    // A legitimate schema queued behind slow ones: the wait for a slot does not count against its
    // compilation timeout (rel/v0.10 verification). Concurrency 1, two slow compilations that run
    // to their 0.8 s timeout one after the other: the third waits ~1.5 s, then compiles.
    const auto queued = [&](std::uint32_t wait_ms) {
        GrammarCompiler busy(
            byte_vocabulary(), {kEos}, 64U << 20,
            {.compile_timeout_ms = 800, .compile_concurrency = 1, .compile_wait_ms = wait_ms});
        std::vector<std::thread> slow;
        for (int i = 0; i < 2; ++i) {
            slow.emplace_back([&busy, i] {
                try {
                    (void)busy.compile(
                        OutputConstraint::json_schema(slow_schema(std::to_string(i))), {}, {});
                } catch (const RequestError&) {}
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // first come, first served
        }
        const auto begun    = Clock::now();
        const auto third    = request_error([&] {
            (void)busy.compile(
                OutputConstraint::json_schema(R"({"type":"object","title":"third"})"), {}, {});
        });
        const double waited = std::chrono::duration<double>(Clock::now() - begun).count();
        for (auto& thread : slow) thread.join();
        std::cout << "third compilation behind two slow ones (wait limit " << wait_ms
                  << " ms): " << waited << " s, " << (third ? third->what() : "compiled") << '\n';
        return std::pair{third, waited};
    };
    const auto [compiled, waited] = queued(30000);
    check(!compiled && waited > 1.0 && waited < 3.0,
          "a legitimate schema waiting for a slot was refused or waited too long");
    const auto [busy_error, gave_up] = queued(300);
    check(busy_error && busy_error->kind() == RequestErrorKind::Overloaded &&
              std::string(busy_error->what()).find("--constraint-compile-wait-ms") !=
                  std::string::npos &&
              gave_up < 1.0,
          "the wait for a slot was not bounded by --constraint-compile-wait-ms");
}

// Second audit N3: the matcher budget is CPU time of the thread, the wall clock only a wider net.
void test_cpu_time_budget() {
    using namespace std::chrono_literals;
    {
        xgrammar::ScopedWorkDeadline scope(Clock::now() + 10s, xgrammar::ThreadCpuTime() + 50ms);
        std::this_thread::sleep_for(200ms); // waiting (a busy machine) costs no CPU time
        bool threw = false;
        try {
            xgrammar::ChargeWork(4096);
        } catch (const xgrammar::WorkBudgetExceeded&) { threw = true; }
        check(!threw, "time without CPU work consumed the CPU budget");
        const auto spin_until = xgrammar::ThreadCpuTime() + 100ms;
        while (xgrammar::ThreadCpuTime() < spin_until) {}
        try {
            xgrammar::ChargeWork(4096);
        } catch (const xgrammar::WorkBudgetExceeded&) { threw = true; }
        check(threw, "CPU work beyond the budget was not interrupted");
    }
    {
        xgrammar::ScopedWorkDeadline scope(Clock::now() + 50ms, xgrammar::ThreadCpuTime() + 10s);
        std::this_thread::sleep_for(100ms);
        bool threw = false;
        try {
            scope.Check();
        } catch (const xgrammar::WorkBudgetExceeded&) { threw = true; }
        check(threw, "the wall-clock safety net did not fire");
    }
}

// Third audit N9: strict JSON Schema by default (XGrammar's and vLLM's default). An open object
// with declared properties let the model add keys whose masks cost 56-95 ms each, so the matcher
// budget closed most realistic requests.
bool generates(ninfer::text::GrammarCompiler& compiler, const OutputConstraint& constraint,
               std::string_view text) {
    auto session = compiler.compile(constraint, {}, {});
    std::vector<std::uint32_t> words(session->mask_words());
    for (const unsigned char byte : text) {
        if (session->masks({}, words) != 0 || !((words[byte / 32] >> (byte % 32)) & 1U))
            return false;
        session->accept(byte);
        session->confirm();
    }
    return session->masks({}, words) == 0 && ((words[kEos / 32] >> (kEos % 32)) & 1U);
}

void test_json_strict() {
    using ninfer::text::GrammarCompiler;
    GrammarCompiler strict(byte_vocabulary(), {kEos}, 64U << 20);
    GrammarCompiler literal(byte_vocabulary(), {kEos}, 64U << 20, {.json_strict = false});
    const auto schema = [](const char* text) { return OutputConstraint::json_schema(text); };
    const auto open   = schema(R"({"type":"object","properties":{"a":{"type":"integer"}}})");
    check(generates(strict, open, R"({"a":1})"), "strict: declared property");
    check(!generates(strict, open, R"({"a":1,"b":2})"), "strict: undeclared key must be refused");
    check(generates(literal, open, R"({"a":1,"b":2})"),
          "flag off: undeclared key admitted as before");
    const auto any_extra = schema(
        R"({"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":true})");
    check(generates(strict, any_extra, R"({"a":1,"b":[true]})"),
          "strict: explicit additionalProperties true still admits extra keys");
    const auto typed_extra = schema(
        R"({"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":{"type":"string"}})");
    check(generates(strict, typed_extra, R"({"a":1,"b":"x"})") &&
              !generates(strict, typed_extra, R"({"a":1,"b":2})"),
          "strict: additionalProperties schema applies to extra keys");
    const auto array = schema(R"({"type":"array"})");
    check(generates(strict, array, "[]") && generates(strict, array, "[1,\"x\"]") &&
              generates(literal, array, "[1]"),
          "an array without items or prefixItems admits any items in both modes");
    const auto tuple = schema(R"({"type":"array","prefixItems":[{"type":"integer"}]})");
    check(generates(strict, tuple, "[1]") && !generates(strict, tuple, "[1,2]"),
          "strict: no items after prefixItems unless items says so");
    check(generates(literal, tuple, "[1,2]"), "flag off: items after prefixItems admitted");
    check(generates(strict, schema(R"({"type":"array","items":{"type":"integer"}})"), "[1,2]"),
          "strict: items keep their meaning");
    check(generates(strict, schema(R"({"type":"object","description":"free-form"})"), R"({"x":1})"),
          "strict: an object without declared properties stays any object");
    check(generates(strict,
                    schema(R"({"allOf":[{"type":"object"},{"type":"object","title":"t"}]})"),
                    R"({"x":1})"),
          "strict: a merged object without declared properties stays any object");
    // Composition (allOf) closes to the union of the declared properties.
    const auto merged = schema(
        R"({"allOf":[{"type":"object","properties":{"a":{"type":"integer"}}},{"properties":{"b":{"type":"integer"}}}]})");
    check(generates(strict, merged, R"({"a":1,"b":2})") &&
              !generates(strict, merged, R"({"a":1,"c":2})"),
          "strict: allOf closes to the union of the declared properties");
    check(generates(literal, merged, R"({"a":1,"c":2})"), "flag off: allOf stays open");
    // json_object is any object in both modes.
    check(generates(strict, OutputConstraint::json_object(), R"({"x":{"y":[1]}})"),
          "json_object must stay any object");
}

int main() {
    test_json_document_limits();
    test_source_limits();
    test_compilation_limits();
    test_matcher_budget();
    test_cache_drops_failures();
    test_composition_dag();
    test_cpu_time_budget();
    test_json_strict();
    if (failures == 0) std::cout << "constraint limit tests passed\n";
    return failures == 0 ? 0 : 1;
}
