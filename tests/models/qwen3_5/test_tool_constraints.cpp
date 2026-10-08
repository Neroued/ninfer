#include "models/qwen3_5/frontend/tool_contract.h"
#include "models/qwen3_5/frontend/tool_grammar.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/grammar.h"

#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

namespace {
namespace frontend = ninfer::models::qwen3_5::frontend;
using Json         = nlohmann::ordered_json;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

ninfer::text::GrammarCompiler compiler() {
    std::vector<std::string> tokens;
    for (int i = 0; i < 256; ++i) tokens.emplace_back(1, static_cast<char>(i));
    tokens.emplace_back();
    return {std::move(tokens), {256}, 32 * 1024 * 1024};
}

auto contract(const Json& schema, bool strict, ninfer::ToolChoice choice = {}) {
    std::vector<std::string> definitions{
        Json{{"type", "function"},
             {"function", {{"name", "call"}, {"parameters", schema}, {"strict", strict}}}}
            .dump()};
    return frontend::select_tool_call_contract(
        frontend::build_tool_call_output_contract(definitions), choice);
}

bool accepts(ninfer::text::GrammarCompiler& compiler,
             const frontend::ToolCallOutputContract& contract, std::string_view value) {
    auto matcher = frontend::compile_tool_grammar(compiler, contract, {}, {});
    std::vector<std::uint32_t> words(matcher->mask_words());
    for (unsigned char c : value) {
        if (matcher->masks({}, words) || !(words[c / 32] & (1u << (c % 32)))) return false;
        matcher->accept(c);
        matcher->confirm();
    }
    return matcher->masks({}, words) == 0 && (words[256 / 32] & 1u);
}

std::string call(std::string_view value, std::string_view name = "call") {
    return "<tool_call>\n<function=" + std::string(name) + ">\n<parameter=x>\n" +
           std::string(value) + "\n</parameter>\n</function>\n</tool_call>";
}

Json schema(const Json& value) {
    return {{"type", "object"},
            {"properties", {{"x", value}}},
            {"required", {"x"}},
            {"additionalProperties", false}};
}

void basic_contracts(ninfer::text::GrammarCompiler& compiled) {
    const Json source{
        {"type", "object"},
        {"properties", {{"a", {{"type", "string"}}}, {"b", {{"type", "integer"}, {"minimum", 5}}}}},
        {"required", {"a", "b"}},
        {"additionalProperties", true}};
    const auto basic = contract(source, false);
    require(basic->constrained, "ordinary tools did not enable basic constraints");
    const std::string reordered =
        "<tool_call>\n<function=call>\n<parameter=b>\n1\n</parameter>\n"
        "<parameter=a>\n007\n</parameter>\n<parameter=extra-key.键>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>";
    require(accepts(compiled, *basic, reordered),
            "basic closed or reordered an open parameter list");
    frontend::ToolCallOutputDecoder decoded(basic, 64);
    (void)decoded.feed(reordered);
    const auto arguments = Json::parse(decoded.finish().tool_calls.at(0).arguments_json);
    require(arguments == Json{{"b", 1}, {"a", "007"}, {"extra-key.键", true}},
            "non-strict value normalization changed");
    require(accepts(compiled, *basic, "An ordinary answer."), "default auto forced a call");
    require(
        !accepts(compiled, *basic, "<tool_call>\n<function=missing>\n</function>\n</tool_call>"),
        "default auto admitted an undeclared function");

    for (const Json& complex :
         {Json{{"anyOf", {{{"type", "object"}}, {{"type", "object"}, {"required", {"x"}}}}}},
          Json{{"$ref", "https://example.test/schema"}},
          Json{{"type", "object"}, {"patternProperties", {{".*", {{"type", "string"}}}}}},
          schema({{"type", {"string", "null"}}, {"format", "uri"}})}) {
        const auto bound = contract(complex, false);
        require(accepts(compiled, *bound, call("hello")),
                "non-strict schema reached strict validation or closed the parameter list");
    }

    const auto integer         = contract(schema({{"type", "integer"}}), false);
    const std::string repeated = "<tool_call>\n<function=call>\n<parameter=x>\n1\n</parameter>\n"
                                 "<parameter=x>\n2\n</parameter>\n</function>\n</tool_call>";
    require(accepts(compiled, *integer, repeated),
            "generic argument syntax rejected repeated keys");
    frontend::ToolCallOutputDecoder duplicate(integer, 64);
    (void)duplicate.feed(repeated);
    require(duplicate.finish().tool_calls.at(0).arguments_json == R"({"x":2})",
            "non-strict duplicate was not resolved to one final value");

    // These requests share the same compiled envelope but have different value normalization.
    const auto string = contract(schema({{"type", "string"}}), false);
    for (const auto& bound : {integer, string}) {
        require(accepts(compiled, *bound, call("7")),
                "shared basic grammar changed with value types");
        frontend::ToolCallOutputDecoder decoder(bound, 64);
        (void)decoder.feed(call("7"));
        const auto args = Json::parse(decoder.finish().tool_calls.at(0).arguments_json);
        require(args["x"] == (bound == string ? Json("7") : Json(7)),
                "compiled grammar leaked another request's argument types");
    }

    const auto two = reordered + "\n" + call("later");
    for (std::size_t cut = 0; cut < two.size(); ++cut) {
        frontend::ToolCallOutputDecoder decoder(basic, 64);
        (void)decoder.feed(std::string_view(two).substr(0, cut));
        const auto result = decoder.finish(ninfer::FinishReason::OutputLimit);
        require(result.tool_calls.size() == (cut >= reordered.size() ? 1 : 0),
                "non-strict truncation lost a completed call");
    }
    ninfer::ToolChoice automatic;
    automatic.constraints = ninfer::ToolConstraintMode::Automatic;
    require(!contract(source, false, automatic)->constrained,
            "explicit auto ignored the free route");
    automatic.mode = ninfer::ToolChoiceMode::Required;
    require(contract(source, false, automatic)->constrained, "auto relaxed a required choice");
    require(contract(schema({{"type", "integer"}}), true, automatic)->constrained,
            "auto relaxed strict arguments");
}

void selected_contracts(ninfer::text::GrammarCompiler& compiled) {
    const auto parameters = schema({{"type", "integer"}, {"minimum", 2}, {"maximum", 3}});
    std::vector<std::string> definitions;
    for (const char* name : {"strict", "loose", "excluded"}) {
        definitions.push_back(
            Json{{"type", "function"},
                 {"function",
                  {{"name", name},
                   {"parameters",
                    std::string_view(name) == "excluded" ? Json{{"type", "object"}} : parameters},
                   {"strict", std::string_view(name) != "loose"}}}}
                .dump());
    }
    const auto declarations = frontend::build_tool_call_output_contract(definitions);
    ninfer::ToolChoice choice;
    choice.constraints   = ninfer::ToolConstraintMode::Automatic;
    choice.allowed_names = std::vector<std::string>{"strict", "loose"};
    const auto selected  = frontend::select_tool_call_contract(declarations, choice);
    require(accepts(compiled, *selected, "ordinary answer"), "selection forced an auto call");
    require(!accepts(compiled, *selected, call("wrong", "strict")),
            "non-strict sibling relaxed strict values");
    require(!accepts(compiled, *selected, call("2", "excluded")),
            "selection admitted an excluded function");
    const auto mixed = call("2", "strict") + "\n" + call("wrong", "loose");
    require(accepts(compiled, *selected, mixed), "mixed strict/non-strict calls rejected");
    frontend::ToolCallOutputDecoder decoder(selected, 64);
    (void)decoder.feed(mixed);
    const auto calls = decoder.finish().tool_calls;
    require(calls.size() == 2 && Json::parse(calls[0].arguments_json)["x"] == 2 &&
                Json::parse(calls[1].arguments_json)["x"] == "wrong",
            "selected tools used another function's argument codec");

    choice.allowed_names = std::vector<std::string>{"loose"};
    choice.mode          = ninfer::ToolChoiceMode::Required;
    choice.parallel      = false;
    const auto single    = frontend::select_tool_call_contract(declarations, choice);
    require(accepts(compiled, *single, call("wrong", "loose")) &&
                !accepts(compiled, *single, call("2", "strict")) &&
                !accepts(compiled, *single, "ordinary answer"),
            "reselecting declarations lost the required function policy");
    for (const auto& names : {std::vector<std::string>{}, std::vector<std::string>{"missing"}}) {
        choice.allowed_names = names;
        bool rejected        = false;
        try {
            (void)frontend::select_tool_call_contract(declarations, choice);
        } catch (const ninfer::RequestError&) { rejected = true; }
        require(rejected, "required selection accepted an empty or unknown function set");
    }
}

void run() {
    auto compiled = compiler();
    basic_contracts(compiled);
    selected_contracts(compiled);
    ninfer::ToolChoice required;
    required.mode = ninfer::ToolChoiceMode::Required;
    auto fixed =
        contract(schema({{"type", "integer"}, {"minimum", 2}, {"maximum", 3}}), true, required);
    require(accepts(compiled, *fixed, call("2")), "strict integer rejected");
    require(!accepts(compiled, *fixed, call("4")), "integer range ignored");
    require(!accepts(compiled, *fixed, call("\"2\"")), "integer coerced from string");
    require(!accepts(compiled, *fixed, "ordinary answer"), "required choice admitted text");
    require(accepts(compiled, *fixed, call("2") + "\n" + call("3")), "parallel calls rejected");
    required.parallel = false;
    auto single       = contract(schema({{"type", "integer"}}), true, required);
    require(!accepts(compiled, *single, call("2") + "\n" + call("3")),
            "single choice admitted second call");
    auto automatic = contract(schema({{"type", "string"}}), true);
    require(accepts(compiled, *automatic, "ordinary answer"), "auto rejected plain text");
    require(accepts(compiled, *automatic, "Calling.\n\n" + call("你好")),
            "auto rejected preface/call");
    require(!accepts(compiled, *automatic,
                     "<tool_call>\n<function=unknown>\n</function>\n</tool_call>"),
            "auto escaped through free text");
    auto disabled = required;
    disabled.mode = ninfer::ToolChoiceMode::None;
    auto no_tools = contract(schema({{"type", "string"}}), true, disabled);
    require(accepts(compiled, *no_tools, "ordinary answer") &&
                !accepts(compiled, *no_tools, call("a")),
            "none allowed a tool marker");
    ninfer::ToolChoice basic;
    basic.constraints = ninfer::ToolConstraintMode::Basic;
    auto structure    = contract(schema({{"type", "integer"}, {"minimum", 2}}), false, basic);
    require(accepts(compiled, *structure, call("not an integer")),
            "basic unexpectedly enforced argument types");
    auto bounded =
        contract(schema({{"type", "string"}, {"minLength", 1}, {"maxLength", 2}}), true, required);
    require(accepts(compiled, *bounded, call("😀")), "Unicode length mismatch");
    require(!accepts(compiled, *bounded, call("abc")), "length bound dropped");
    auto patterned = contract(schema({{"type", "string"}, {"pattern", "p"}}), true, required);
    require(accepts(compiled, *patterned, call("apple")), "raw pattern lost search semantics");
    require(!accepts(compiled, *patterned, call("none")), "raw pattern ignored");
    auto intersection = contract(
        schema({{"type", "string"}, {"pattern", "你|😀"}, {"minLength", 2}, {"maxLength", 3}}),
        true, required);
    require(accepts(compiled, *intersection, call("你好")) &&
                accepts(compiled, *intersection, call("😀ab")) &&
                !accepts(compiled, *intersection, call("你")) &&
                !accepts(compiled, *intersection, call("你abc")) &&
                !accepts(compiled, *intersection, call("abc")),
            "strict string conjunction lost pattern or Unicode length");
    auto conjunction = contract(schema({{"allOf",
                                         {Json{{"type", "string"}, {"pattern", "a"}},
                                          Json{{"pattern", "b"}, {"maxLength", 3}}}}}),
                                true, required);
    require(accepts(compiled, *conjunction, call("ab")) &&
                !accepts(compiled, *conjunction, call("aaaa")),
            "strict parameter domain disagrees with normalized conjunction");
    const auto value = std::string("  你好\n");
    frontend::ToolCallOutputDecoder decoder(automatic, 64);
    const auto text = call(value);
    for (char c : text) require(decoder.feed(std::string_view(&c, 1)).empty(), "tool bytes leaked");
    auto result = decoder.finish();
    require(result.tool_calls.size() == 1 &&
                Json::parse(result.tool_calls[0].arguments_json)["x"] == value,
            "raw string changed");
    auto nested           = contract(schema({{"type", "object"},
                                             {"properties", {{"s", {{"type", "string"}}}}},
                                             {"required", {"s"}},
                                             {"additionalProperties", false}}),
                                     true, required);
    const auto json_value = std::string(R"({"s": "</parameter>"})");
    require(accepts(compiled, *nested, call(json_value)), "nested marker rejected");
    frontend::ToolCallOutputDecoder json_decoder(nested, 64);
    (void)json_decoder.feed(call(json_value));
    require(Json::parse(json_decoder.finish().tool_calls[0].arguments_json)["x"]["s"] ==
                "</parameter>",
            "nested marker broke parser");
    const auto two = call("2") + "\n" + call("3");
    for (auto reason : {ninfer::FinishReason::OutputLimit, ninfer::FinishReason::Cancelled})
        for (std::size_t cut = 0; cut < two.size(); ++cut) {
            frontend::ToolCallOutputDecoder truncated(fixed, 64);
            (void)truncated.feed(std::string_view(two).substr(0, cut));
            const auto output = truncated.finish(reason);
            require(output.tool_calls.size() == (cut >= call("2").size() ? 1 : 0),
                    "interruption lost a complete call");
        }
    const auto seed = call("你好").substr(0, call("你好").size() - 20);
    frontend::ToolCallOutputDecoder continuation(automatic, 64);
    continuation.initialize_continuation(seed);
    auto match = frontend::compile_tool_grammar(compiled, *automatic, {}, seed);
    (void)continuation.feed(std::string_view(call("你好")).substr(seed.size()));
    require(continuation.finish().tool_calls.size() == 1, "continuation did not complete the call");
    bool rejected = false;
    try {
        (void)contract(schema({{"type", {"string", "null"}}}), true, required);
    } catch (const ninfer::RequestError&) { rejected = true; }
    require(rejected, "ambiguous top-level representation accepted");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--probe") {
            auto compiled = compiler();
            std::string line;
            while (std::getline(std::cin, line)) {
                Json response;
                try {
                    const auto input = Json::parse(line);
                    ninfer::ToolChoice choice;
                    choice.mode           = ninfer::ToolChoiceMode::Required;
                    auto bound            = contract(input.at("schema"), true, choice);
                    response["accepted"]  = Json::array();
                    response["arguments"] = Json::array();
                    for (const auto& text : input.at("candidates")) {
                        bool allowed = accepts(compiled, *bound, text.get<std::string>());
                        response["accepted"].push_back(allowed);
                        if (allowed) {
                            frontend::ToolCallOutputDecoder decode(bound, 64);
                            (void)decode.feed(text.get<std::string>());
                            response["arguments"].push_back(
                                Json::parse(decode.finish().tool_calls[0].arguments_json));
                        } else
                            response["arguments"].push_back(nullptr);
                    }
                } catch (const std::exception& error) { response = {{"error", error.what()}}; }
                std::cout << response.dump() << '\n';
            }
        } else {
            run();
            std::cout << "tool constraints passed\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
