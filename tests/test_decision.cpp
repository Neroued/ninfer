#include "decision/decision.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ninfer::decision;

namespace {
int checks = 0;

void expect(bool condition, const char* name) {
    ++checks;
    if (!condition) { throw std::runtime_error(name); }
}

template <class F>
void rejects(F&& action, const char* name) {
    bool threw = false;
    try {
        action();
    } catch (const std::exception&) { threw = true; }
    expect(threw, name);
}

Json example() {
    return Json::parse(R"({"contexts":["hello"],"schema":{
        "yes":{"type":"boolean","description":"Is this a greeting?"},
        "priority":{"type":"integer","minimum":1,"maximum":3,"description":"Priority"}}})");
}

Tokens bytes(const std::string& s) { return Tokens(s.begin(), s.end()); }
} // namespace

int main() {
    try {
        auto request = parse_request(example());
        expect(request.fields.size() == 2 && request.fields[0].values[0] == true, "typed booleans");
        expect(request.fields[1].values == std::vector<Json>({1, 2, 3}), "integer grid");
        auto schema  = Json::parse(R"({"contexts":[""],"schema":{"type":"object","properties":{
            "n":{"type":"number","minimum":0.1,"maximum":0.3,"multipleOf":0.1,"x-aggregate":"mean"},
            "s":{"type":"string","enum":["a","ab","雪"]}},"required":["n","s"],"additionalProperties":false}})");
        auto numeric = parse_request(schema);
        expect(numeric.fields[0].encoded == std::vector<std::string>({"0.1", "0.2", "0.3"}),
               "fixed decimal grid");
        expect(numeric.fields[1].values[2] == "雪", "unicode enum");
        auto named        = Json::parse(R"({"contexts":[""],"schema":{
            "properties":{"type":"boolean","description":"Has properties?"},
            "type":{"type":"enum","choices":["a","b"],"description":"Which type?"}}})");
        auto named_fields = parse_request(named);
        expect(named_fields.fields.size() == 2 && named_fields.fields[0].name == "properties" &&
                   named_fields.fields[0].values == std::vector<Json>({true, false}),
               "properties is a compact field name, not a wrapper");
        named["schema"]["properties"] = {{"enum", {"x", "y"}}, {"description", "Which property?"}};
        expect(parse_request(named).fields[0].values == std::vector<Json>({"x", "y"}),
               "compact properties enum may omit type");
        named["schema"] = Json{{"properties", named["schema"]}};
        expect(parse_request(named).fields[0].name == "properties",
               "wrapper without type may contain properties and type fields");
        named["schema"]["type"] = "object";
        expect(parse_request(named).fields.size() == 2,
               "explicit object wrapper preserves reserved-looking property names");

        auto large_grid = Json::parse(R"({"contexts":[""],"schema":{
            "n":{"type":"number","description":"Large values","minimum":9007199254740992,
                 "maximum":9007199254740993,"step":1}}})");
        rejects([&] { (void)parse_request(large_grid); },
                "distinct integer bounds must not collapse");
        auto large_schema = large_grid;
        large_schema["schema"]["n"].erase("step");
        large_schema["schema"]["n"]["multipleOf"] = 1;
        large_schema["schema"]                    = Json{{"properties", large_schema["schema"]}};
        rejects([&] { (void)parse_request(large_schema); }, "JSON Schema bounds must not collapse");
        for (const auto& value :
             Json::parse("[9007199254740993,-9007199254740993,9223372036854775807,"
                         "18446744073709551615,-9223372036854775807]")) {
            large_grid["schema"]["n"]["minimum"] = value;
            large_grid["schema"]["n"]["maximum"] = value;
            rejects([&] { (void)parse_request(large_grid); }, "lossy singleton bound rejects");
        }
        large_grid["schema"]["n"]["minimum"] = 0;
        large_grid["schema"]["n"]["maximum"] = 9007199254740992ULL;
        large_grid["schema"]["n"]["step"]    = 9007199254740993ULL;
        rejects([&] { (void)parse_request(large_grid); }, "lossy integer step rejects");
        large_grid["schema"]["n"]["minimum"] = 9007199254740992ULL;
        large_grid["schema"]["n"]["maximum"] = 9007199254740996ULL;
        large_grid["schema"]["n"]["step"]    = 2;
        expect(parse_request(large_grid).fields[0].encoded ==
                   std::vector<std::string>(
                       {"9007199254740992", "9007199254740994", "9007199254740996"}),
               "exactly representable large number grids remain supported");
        for (const auto& value : Json::parse("[-9223372036854775808,9223372036854775808]")) {
            large_grid["schema"]["n"]["minimum"] = value;
            large_grid["schema"]["n"]["maximum"] = value;
            expect(parse_request(large_grid).fields[0].encoded ==
                       std::vector<std::string>({value.dump()}),
                   "exact signed and unsigned limits remain supported");
        }
        auto invalid                     = example();
        invalid["schema"]["yes"]["type"] = "array";
        rejects([&] { (void)parse_request(invalid); }, "unsupported schema type");
        invalid             = example();
        invalid["contexts"] = Json::array();
        rejects([&] { (void)parse_request(invalid); }, "empty contexts");
        invalid                             = example();
        invalid["schema"]["yes"]["minimum"] = "bad";
        // An unrelated numeric constraint must not silently masquerade as supported JSON Schema.
        rejects([&] { (void)parse_request(invalid); }, "irrelevant numeric constraint");
        invalid["schema"]["yes"]["anyOf"] = Json::array();
        rejects([&] { (void)parse_request(invalid); }, "unsupported keyword");
        invalid             = example();
        invalid["tree_max"] = 0;
        rejects([&] { (void)parse_request(invalid); }, "tree max bounds");
        invalid                 = example();
        invalid["cache_prompt"] = "yes";
        rejects([&] { (void)parse_request(invalid); }, "cache bool type");
        invalid                                  = example();
        invalid["schema"]["priority"]["minimum"] = INT64_MIN;
        invalid["schema"]["priority"]["maximum"] = INT64_MAX;
        rejects([&] { (void)parse_request(invalid); }, "integer subtraction overflow");
        invalid["schema"]["priority"]["minimum"] = INT64_MAX - 1;
        const auto edge                          = parse_request(invalid);
        expect(edge.fields[1].values.size() == 2 && edge.fields[1].values.back() == INT64_MAX,
               "int64 upper endpoint does not overflow iteration");
        invalid         = schema;
        invalid["mode"] = "greedy";
        rejects([&] { (void)parse_request(invalid); },
                "greedy mean rejects instead of silently using mode");
        invalid                                         = schema;
        invalid["schema"]["properties"]["n"]["minimum"] = 0.05;
        rejects([&] { (void)parse_request(invalid); }, "multipleOf grid alignment");
        invalid                                      = schema;
        invalid["schema"]["properties"]["s"]["enum"] = {"a", "a"};
        rejects([&] { (void)parse_request(invalid); }, "duplicate choices");
        invalid                                     = example();
        invalid["schema"]["priority"]["multipleOf"] = 2;
        rejects([&] { (void)parse_request(invalid); }, "unsupported integer stride");
        invalid                                         = schema;
        invalid["schema"]["properties"]["s"]["choices"] = {"b"};
        rejects([&] { (void)parse_request(invalid); }, "conflicting choice lists");
        auto small_grid = Json::parse(R"({"contexts":[""],"schema":{
            "n":{"type":"number","description":"Small values","minimum":0.000000001,
                 "maximum":0.000000003,"step":0.000000001}}})");
        expect(parse_request(small_grid).fields[0].encoded ==
                   std::vector<std::string>({"0.000000001", "0.000000002", "0.000000003"}),
               "nine decimal places remain distinct");
        small_grid["schema"]["n"]["minimum"] = .1;
        small_grid["schema"]["n"]["maximum"] = 25.5;
        small_grid["schema"]["n"]["step"]    = .1;
        expect(parse_request(small_grid).fields[0].values.size() == 255, "maximum decimal grid");

        for (const bool wrapped : {false, true}) {
            auto grid = Json::parse(R"({"contexts":[""],"schema":{
                "n":{"type":"number","description":"Grid","minimum":0,
                     "maximum":254000000001,"step":1000000000}}})");
            if (wrapped) {
                grid["schema"]["n"]["multipleOf"] = grid["schema"]["n"]["step"];
                grid["schema"]["n"].erase("step");
                grid["schema"] = Json{{"properties", grid["schema"]}};
            }
            auto& spec = wrapped ? grid["schema"]["properties"]["n"] : grid["schema"]["n"];
            rejects([&] { (void)parse_request(grid); }, "large off-grid maximum rejects");
            spec["maximum"] = 254000000000ULL;
            const auto exact = parse_request(grid).fields[0];
            expect(exact.values.size() == 255 && exact.values.back() == spec["maximum"],
                   "large aligned maximum remains included");
            spec["maximum"] = 1.000000001;
            spec[wrapped ? "multipleOf" : "step"] = 1;
            rejects([&] { (void)parse_request(grid); }, "small off-grid maximum rejects");
        }
        auto decimal_grid = Json::parse(R"({"contexts":[""],"schema":{
            "n":{"type":"number","description":"Decimal","minimum":1000000000000000.1,
                 "maximum":1000000000000001.1,"step":1}}})");
        for (const double sign : {1.0, -1.0}) {
            auto& spec = decimal_grid["schema"]["n"];
            spec["minimum"] = sign > 0 ? 1000000000000000.1 : -1000000000000001.1;
            spec["maximum"] = sign > 0 ? 1000000000000001.1 : -1000000000000000.1;
            const auto field = parse_request(decimal_grid).fields[0];
            expect(field.values.size() == 2 && field.values.front() == spec["minimum"] &&
                       field.values.back() == spec["maximum"],
                   "large fractional bounds round trip without becoming integers");
        }
        auto& decimal_spec = decimal_grid["schema"]["n"];
        decimal_spec["minimum"] = 10000000000.1;
        decimal_spec["maximum"] = 10000000000.3;
        decimal_spec["step"] = .1;
        expect(parse_request(decimal_grid).fields[0].encoded ==
                   std::vector<std::string>({"10000000000.1", "10000000000.2", "10000000000.3"}),
               "decimal endpoints tolerate subtraction cancellation without losing values");
        decimal_spec["minimum"] = -.3;
        decimal_spec["maximum"] = 0;
        expect(parse_request(decimal_grid).fields[0].values == std::vector<Json>({-.3, -.2, -.1, 0}),
               "decimal cancellation at zero preserves the inclusive grid");
        decimal_spec["minimum"] = 1.0000000001;
        decimal_spec["maximum"] = 1.0000000001;
        rejects([&] { (void)parse_request(decimal_grid); }, "tenth decimal place is not rounded away");
        auto aligned = schema;
        auto& aligned_spec = aligned["schema"]["properties"]["n"];
        aligned_spec["minimum"] = 254000000001ULL;
        aligned_spec["maximum"] = 255000000001ULL;
        aligned_spec["multipleOf"] = 1000000000;
        rejects([&] { (void)parse_request(aligned); }, "large off-grid multipleOf minimum rejects");
        aligned_spec["minimum"] = 1000000000000000.1;
        aligned_spec["maximum"] = 1000000000000001.1;
        aligned_spec["multipleOf"] = 1;
        rejects([&] { (void)parse_request(aligned); }, "fractional minimum is not an integer multiple");

        CandidateTrie trie;
        trie.add({90, 80, 10, 30, 99});
        trie.add({90, 80, 10, 40, 99});
        trie.add({90, 80, 20, 99});
        trie.build();
        expect(trie.prefix() == Tokens({90, 80}) && trie.nodes().size() == 2,
               "factor token prefix");
        expect(trie.prompt(1) == Tokens({90, 80, 10}), "second divergence prompt");
        // Independent probability oracle: root branches .75/.25, child .2/.8 => .15/.60/.25.
        auto distribution = trie.distribution(
            {{static_cast<float>(std::log(3.0)), 0}, {0, static_cast<float>(std::log(4.0))}});
        expect(std::abs(distribution[0] - .15) < 1e-7 && std::abs(distribution[1] - .60) < 1e-7 &&
                   std::abs(distribution[2] - .25) < 1e-7,
               "exact conditional trie probabilities");
        auto extreme = log_softmax(std::vector<float>{10000, -10000});
        expect(extreme[0] == 0 && extreme[1] == -20000, "stable log softmax");
        rejects(
            [&] { (void)log_softmax(std::vector<float>{std::numeric_limits<float>::quiet_NaN()}); },
            "non-finite scores fail loudly");
        CandidateTrie singleton;
        singleton.add({1, 2});
        singleton.build();
        expect(singleton.distribution({}) == std::vector<double>{1},
               "one choice requires no inference");
        CandidateTrie collision;
        collision.add({1, 2});
        collision.add({1, 2});
        rejects([&] { collision.build(); }, "tokenization collision");
        CandidateTrie prefix;
        prefix.add({1, 2});
        prefix.add({1, 2, 3});
        rejects([&] { prefix.build(); }, "unterminated prefix collision");

        FieldResult f;
        f.winner        = 0;
        f.probabilities = {.1, .2, .7};
        auto assembled  = assemble({numeric.fields[0]}, {f});
        expect(assembled["decision"]["n"] == .3 && assembled["fields"]["n"]["probability"] == .7,
               "mean projects to a valid grid value with its own probability");
        numeric.fields[0].aggregate = "median";
        expect(assemble({numeric.fields[0]}, {f})["decision"]["n"] == .3, "median aggregate");

        Backend backend;
        std::size_t calls       = 0;
        std::vector<bool> cache_flags;
        backend.check_cancelled = [] {};
        backend.tokenize = [](const auto& system, const auto& context, const auto& continuation) {
            // Simulated context-sensitive tokenizer: the space and first answer character merge.
            auto tokens = bytes(system + "<user>" + context + "<assistant>" + continuation);
            for (std::size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i - 1] == ' ' && (tokens[i] == 't' || tokens[i] == 'f')) {
                    tokens[i - 1] = 1000 + tokens[i];
                    tokens.erase(tokens.begin() + i);
                }
            }
            return tokens;
        };
        backend.score = [&](std::vector<ScoreRow> rows, bool cache) {
            ++calls;
            cache_flags.push_back(cache);
            ScoreBatch out;
            for (const auto& row : rows) {
                expect(row.prompt_size() > 0 && row.candidates.size() >= 2, "native score row");
                expect(!row.cache_frontiers.empty(), "cache opt-out preserves shared frontiers");
                for (auto marker : row.cache_frontiers) {
                    expect(marker <= row.prompt_size(),
                           "cache boundary stays within branch prompt");
                }
                out.logits.emplace_back(row.candidates.size(), 0);
                out.computed_tokens += row.prompt_size();
            }
            return out;
        };
        auto evaluated = evaluate(request, backend);
        expect(calls == 1 && evaluated["object"] == "decision", "all tree fields scored together");
        expect(evaluated["results"][0]["decision"]["yes"].is_boolean(),
               "JSON assembled from typed schema");
        expect(evaluated["results"][0]["fields"]["yes"]["probability"] == .5,
               "merged boundary logits");
        request.cache_prompt = false;
        (void)evaluate(request, backend);
        expect(cache_flags == std::vector<bool>({true, false}),
               "cache opt-out refreshes prior-call state without erasing sharing");
        request.mode = "greedy";
        auto greedy  = evaluate(request, backend);
        expect(!greedy["results"][0]["fields"]["yes"]["tree"].get<bool>(), "greedy mode exposed");
        auto branching   = parse_request(Json::parse(R"({"contexts":["A","B"],"schema":{
            "pick":{"type":"enum","description":"Choose","choices":["aa","ab","b"]}}})"));
        backend.tokenize = [](const auto& system, const auto& context, const auto& continuation) {
            return bytes(system + "<user>" + context + "<assistant>" + continuation);
        };
        backend.score = [&](std::vector<ScoreRow> rows, bool cache) {
            cache_flags.push_back(cache);
            ScoreBatch out;
            if (rows.size() == 4) {
                expect(rows[0].prefix.data() == rows[1].prefix.data(),
                       "divergence rows borrow one shared trunk");
                expect(rows[0].prefix.data() != rows[2].prefix.data(),
                       "one scoring call includes independent contexts");
            }
            for (const auto& row : rows) {
                // Independent oracle: branch a=.6, b=.4; within a, aa=.51, ab=.49.
                // Tree winner is b (.4); greedy takes aa (.306), requiring two rounds.
                const bool child = row.prompt().back() == 'a';
                out.logits.push_back({static_cast<float>(std::log(child ? .51 : .6)),
                                      static_cast<float>(std::log(child ? .49 : .4))});
            }
            return out;
        };
        auto exact = evaluate(branching, backend);
        expect(exact["results"].size() == 2 && exact["results"][0]["decision"]["pick"] == "b" &&
                   std::abs(exact["results"][1]["fields"]["pick"]["probability"].get<double>() -
                            .4) < 1e-7,
               "exact tree differs from greedy and preserves context order");
        branching.mode = "greedy";
        branching.cache_prompt = false;
        cache_flags.clear();
        auto walked    = evaluate(branching, backend);
        expect(cache_flags == std::vector<bool>({false, true}),
               "cache opt-out rebuilds once then shares later greedy rounds");
        expect(walked["results"][0]["decision"]["pick"] == "aa" &&
                   walked["timings"]["rounds"] == 2 &&
                   std::abs(walked["results"][0]["fields"]["pick"]["probability"].get<double>() -
                            .306) < 1e-7,
               "multi-round greedy path probability");
        branching.mode     = "tree";
        branching.contexts = {"A", "B", "C", "D", "E", "F", "G", "H", "I"};
        const auto oracle  = backend.score;
        std::vector<std::size_t> batch_sizes;
        backend.score = [&](std::vector<ScoreRow> rows, bool cache) {
            batch_sizes.push_back(rows.size());
            return oracle(std::move(rows), cache);
        };
        cache_flags.clear();
        const auto cohorts = evaluate(branching, backend);
        expect(cache_flags == std::vector<bool>({false, true}),
               "later context cohorts share the request's rebuilt prefix");
        expect(batch_sizes == std::vector<std::size_t>({16, 2}) && cohorts["results"].size() == 9 &&
                   cohorts["timings"]["rounds"] == 2,
               "nine contexts use bounded cohorts without serializing fields");
        rejects([&] { (void)evaluate(Request{}, backend); }, "empty native request");
        auto escaped = parse_request(Json::parse(R"({"contexts":[""],"schema":{
            "quoted\"key":{"type":"enum","description":"Escapes","choices":["a\"","a\\","a\n"]}}})"));
        expect(escaped.fields[0].encoded ==
                   std::vector<std::string>({"\"a\\\"\"", "\"a\\\\\"", "\"a\\n\""}),
               "JSON string escaping is lossless");
        backend.check_cancelled = [] { throw std::runtime_error("cancelled"); };
        rejects([&] { (void)evaluate(request, backend); }, "cancel before inference");
        std::cout << checks << " decision checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decision test failed: " << error.what() << '\n';
        return 1;
    }
}
