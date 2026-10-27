#include "output_fixture.h"
#include "models/glm5next/frontend/tool_decoder.h"
#include "models/glm5next/frontend/tool_grammar.h"
#include "text/grammar.h"

#include <iostream>

namespace {
using namespace ninfer;
using namespace glm_fixture;
using Contract = glm::ToolCallOutputContract;

std::vector<std::string> vocabulary() {
    std::vector<std::string> pieces;
    for (int c = 0; c < 256; ++c) pieces.push_back(std::string(1, static_cast<char>(c)));
    pieces.emplace_back(); // EOS
    pieces.push_back(
        "<tool_call>call<arg_key>x</arg_key><arg_value>a</arg_value><arg_key>x</arg_key>");
    return pieces;
}

std::vector<TokenId> bytes(std::string_view input, bool eos = true) {
    std::vector<TokenId> ids;
    for (unsigned char c : input) ids.push_back(c);
    if (eos) ids.push_back(256);
    return ids;
}

std::shared_ptr<const Contract> contract(const Json& schema, bool strict = true,
                                         ToolChoice choice = {.mode = ToolChoiceMode::Required}) {
    const std::vector source{declaration(schema, strict)};
    return glm::select_tool_call_contract(glm::build_tool_call_output_contract(source), choice);
}

bool accepts(text::GrammarCompiler& compiler, const Contract& contract, std::string_view source,
             const std::optional<OutputConstraint>& body = {}) {
    auto session =
        glm::compile_tool_grammar(compiler, std::make_shared<Contract>(contract), {}, {}, body);
    try {
        session->accept(bytes(source));
    } catch (const std::logic_error&) { return false; }
    session->confirm();
    return session->observation().terminated;
}

std::string call(std::string value, std::string name = "x") {
    return "<tool_call>call<arg_key>" + name + "</arg_key><arg_value>" + value +
           "</arg_value></tool_call>";
}

Json decode(const Contract& contract, std::string_view source) {
    glm::ToolCallOutputDecoder decoder(std::make_shared<Contract>(contract), 128);
    Output output;
    for (auto c : source) output.append(decoder.feed(std::string_view(&c, 1)));
    output.append(decoder.finish(FinishReason::StopToken));
    if (auto error = decoder.error()) std::rethrow_exception(error->exception());
    require(output.calls == decoder.take_tool_calls(),
            "constraint decoder event aggregate differs");
    return output.calls.empty() ? Json{} : Json::parse(output.calls.front().arguments_json);
}

void basic(text::GrammarCompiler& compiler) {
    const auto schema = tool_schema(
        {{"x", {{"type", "string"}}}, {"n", {{"type", "integer"}, {"minimum", 10}}}}, {"x", "n"});
    const auto bound = contract(schema, false, {});
    for (const auto& source :
         {call("007"), call("1", "n"), call("unknown", "new"), call("\"unterminated", "new"),
          call("value", "a\"&键"), call("value", "<other>"), call("value", "")}) {
        require(accepts(compiler, *bound, source), "basic grammar rejected: " + source);
        (void)decode(*bound, source);
    }
    require(decode(*bound, call("007"))["x"] == "007", "string value was inferred from spelling");
    require(decode(*bound, call("1", "n"))["n"] == 1, "basic imposed strict value range");
    require(accepts(compiler, *bound, "ordinary text"), "auto required a tool call");
    require(!accepts(compiler, *bound, call("bad", "n")),
            "known JSON parameter escaped through open-name rule");
    require(!accepts(compiler, *bound, call("1e999", "n")), "unpublishable number admitted");
    const auto duplicate = "<tool_call>call<arg_key>x</arg_key><arg_value>a</arg_value><arg_key>x</"
                           "arg_key><arg_value>b</arg_value></tool_call>";
    require(!accepts(compiler, *bound, duplicate), "duplicate names admitted");
    require(!accepts(compiler, *bound, "<tool_call>other</tool_call>"),
            "unknown function admitted");
    auto single = contract(schema, false, {.parallel = false});
    require(!accepts(compiler, *single, call("a") + call("b")),
            "single-call policy admitted multiple calls");
    auto none = contract(schema, false, {.mode = ToolChoiceMode::None});
    require(accepts(compiler, *none, "text") && !accepts(compiler, *none, call("a")),
            "none selection is ineffective");
    auto required = contract(schema, false);
    require(!accepts(compiler, *required, "text") && accepts(compiler, *required, call("a")),
            "required selection is ineffective");
    const auto json = OutputConstraint::json_object();
    require(accepts(compiler, *bound, "{\"value\":1}", json) &&
                accepts(compiler, *bound, call("a"), json) &&
                !accepts(compiler, *required, "{}", json),
            "combined branch policy differs");

    for (const Json& property :
         {Json{{"$ref", "#/$defs/text"}}, Json{{"enum", {"123", "456"}}}, Json{{"const", "123"}},
          Json{{"allOf", {{{"$ref", "#/$defs/text"}}, {{"minLength", 5}}}}}}) {
        auto hinted             = tool_schema({{"x", property}});
        hinted["$defs"]["text"] = {{"type", "string"}};
        const auto basic        = contract(hinted, false);
        require(accepts(compiler, *basic, call("123")) && decode(*basic, call("123"))["x"] == "123",
                "basic string type was lost through a reference or assertion");
        require(accepts(compiler, *basic, call("9")) && decode(*basic, call("9"))["x"] == "9",
                "basic encoding hints enforced strict value constraints");
    }
    for (const Json& property : {Json{{"$ref", "#/$defs/missing"}},
                                 Json{{"anyOf", {{{"type", "string"}}, Json::object()}}}}) {
        const auto unknown = contract(tool_schema({{"x", property}}), false);
        require(accepts(compiler, *unknown, call("123")) &&
                    decode(*unknown, call("123"))["x"] == 123,
                "unknown type acquired a speculative encoding");
    }
}

void parameter_names(text::GrammarCompiler& compiler) {
    const auto bound = contract(tool_schema({{"", {{"type", "integer"}}},
                                             {"n", {{"type", "integer"}}},
                                             {"name", {{"type", "integer"}}},
                                             {"names", {{"type", "string"}}},
                                             {"键", {{"type", "integer"}}}}),
                                false);
    for (const auto* name : {"", "n", "name", "键"}) {
        require(accepts(compiler, *bound, call("1", name)), "known JSON name rejected");
        require(!accepts(compiler, *bound, call("raw", name)), "known name bypassed JSON encoding");
    }
    for (const auto* name : {"na", "names", "named", "键名", "a</arg_", "</arg_kex>", "\n\""})
        require(accepts(compiler, *bound, call("raw", name)), "open name prefix rejected");

    const std::vector<std::string> pieces{"name</arg_key><arg_value>raw",
                                          "names</arg_key><arg_value>raw",
                                          "键</arg_key><arg_value>{",
                                          "键名</arg_key><arg_value>raw",
                                          "</arg_key><arg_value>raw",
                                          "n</arg_key><arg_value>1",
                                          ""};
    text::GrammarCompiler tokens(pieces, {6}, 1 << 20);
    auto session = glm::compile_tool_grammar(tokens, bound, {}, "<tool_call>call<arg_key>");
    std::vector<std::uint32_t> masks(session->mask_words() * 2);
    const TokenId proposed = 2;
    session->masks(std::span(&proposed, 1), masks);
    require(masks[0] == ((1u << 1) | (1u << 2) | (1u << 3) | (1u << 5)),
            "name/value boundary token mask differs from encoding contract");
    session->accept(1);
    session->discard();
    session->masks({}, std::span(masks).first(session->mask_words()));
    require(masks[0] == ((1u << 1) | (1u << 2) | (1u << 3) | (1u << 5)),
            "name dispatch rollback changed token masks");
}

void strict(text::GrammarCompiler& compiler) {
    const auto string = contract(tool_schema(
        {{"x", {{"type", "string"}, {"pattern", "a"}, {"minLength", 2}, {"maxLength", 3}}}},
        {"x"}));
    require(accepts(compiler, *string, call("ab")) && !accepts(compiler, *string, call("a")) &&
                !accepts(compiler, *string, call("bb")) &&
                !accepts(compiler, *string, call("abcd")),
            "strict string constraints were relaxed");
    const auto spaces =
        contract(tool_schema({{"x", {{"type", "string"}, {"const", " a "}}}}, {"x"}));
    require(accepts(compiler, *spaces, call(" a ")) && !accepts(compiler, *spaces, call("  a  ")) &&
                decode(*spaces, call(" a "))["x"] == " a ",
            "format whitespace changed strict value");
    const auto key = contract(tool_schema({{"a\"&键", {{"type", "string"}}}}, {"a\"&键"}));
    require(accepts(compiler, *key, call("ok", "a\"&键")), "GLM key was XML-attribute escaped");
    const auto nested = contract(tool_schema({{"x",
                                               {{"type", "object"},
                                                {"properties", {{"tag", {{"type", "string"}}}}},
                                                {"required", {"tag"}},
                                                {"additionalProperties", false}}}},
                                             {"x"}));
    const auto source = call("{\"tag\": \"</arg_value>\"}");
    require(accepts(compiler, *nested, source) &&
                decode(*nested, source)["x"]["tag"] == "</arg_value>",
            "nested JSON delimiter parsed as framing");
    rejects([&] { (void)contract(tool_schema({{"x", {{"type", {"string", "null"}}}}})); },
            "mixes raw string");
}

void transactions(text::GrammarCompiler& compiler) {
    const auto bound = contract(tool_schema({{"x", {{"type", "string"}}}}), false);
    auto session     = glm::compile_tool_grammar(compiler, bound, {}, {});
    std::vector<std::uint32_t> masks(session->mask_words() * 2);
    const TokenId duplicate = 257;
    (void)session->masks(std::span(&duplicate, 1), masks);
    require(!(masks[duplicate / 32] & (1u << (duplicate % 32))),
            "one token duplicate escaped name mask");
    const auto prefix = bytes("<tool_call>call<arg_key>x</arg_key><arg_value>a</arg_value>", false);
    session->accept(prefix);
    session->discard();
    session->accept(prefix);
    session->confirm();
    const auto next = bytes("<arg_key>x</arg_key>", false);
    session->accept(std::span(next).first(next.size() - 1));
    session->confirm();
    masks.resize(session->mask_words());
    (void)session->masks({}, masks);
    require(!(masks['>' / 32] & (1u << ('>' % 32))),
            "duplicate argument completion was not masked");
    auto restored       = glm::compile_tool_grammar(compiler, bound, {}, {});
    const auto proposal = bytes("<tool", false);
    masks.resize(restored->mask_words() * (proposal.size() + 1));
    (void)restored->masks(proposal, masks);
    restored->accept(bytes(call("valid")));
    restored->confirm();
    require(restored->observation().terminated, "lookahead changed committed grammar");
    auto continuation = glm::compile_tool_grammar(compiler, bound, {},
                                                  "<tool_call>call<arg_key>x</arg_key><arg_value>");
    continuation->accept(bytes("value</arg_value></tool_call>"));
    continuation->confirm();
}

void integration(const glm::Frontend& f, const text::Tokenizer& tokenizer) {
    auto p = f.prepare(input());
    for (const auto& [constraint, answer] : std::vector<std::pair<OutputConstraint, std::string>>{
             {OutputConstraint::json_object(), "{}"},
             {OutputConstraint::json_schema(R"({"type":"integer","minimum":1})"), "2"},
             {OutputConstraint::choice({"yes", "no"}), "yes"},
             {OutputConstraint::regex("[a-z]{2}"), "ab"},
             {OutputConstraint::grammar("root ::= \"ok\""), "ok"}}) {
        auto session = f.make_output_session(p, {}, {}, {.budget = 1}, constraint, {});
        Output output;
        commit(session, tokenizer.encode("x"), output);
        (void)session.preview_control(session.pending_control_tokens(), 100);
        output.append(session.commit_preview());
        commit(session, tokenizer.encode(answer + "<|endoftext|>"), output);
        require(output.content == answer && output.reasoning == "x",
                "text constraint control composition failed");
    }
    const auto tool =
        f.prepare(input({declaration(tool_schema({{"x", {{"type", "string"}}}}), false)}));
    rejects(
        [&] {
            (void)f.make_output_session(tool, {}, {}, {}, OutputConstraint::choice({"yes"}), {});
        },
        "JSON output");
    rejects([&] { (void)f.make_output_session(tool, {}, {.raw = true}, {}, {}, {}); },
            "default EOS");
    rejects(
        [&] { (void)f.make_output_session(p, {}, {}, {}, {}, {.mode = ToolChoiceMode::Required}); },
        "no callable tools");
    auto session = f.make_output_session(tool, {}, {}, {}, OutputConstraint::json_object(), {});
    Output output;
    commit(session, tokenizer.encode("r</think>  {\"s\": \"<tool_call>literal\"}<|endoftext|>"),
           output);
    require(output.calls.empty() && Json::parse(output.content)["s"] == "<tool_call>literal",
            "JSON content entered tool parser");

    auto required       = f.make_output_session(tool, {}, {}, {}, {},
                                                {.mode          = ToolChoiceMode::Required,
                                                 .allowed_names = std::vector<std::string>{"call"},
                                                 .parallel      = false});
    const auto wire     = tokenizer.encode("r</think>" + call("hello") + "<|observation|>");
    const auto proposed = std::span(wire).first(std::min<std::size_t>(8, wire.size()));
    std::vector<std::uint32_t> masks((proposed.size() + 1) * ((tokenizer.vocab_size() + 31) / 32));
    require(required.grammar_masks(proposed, masks) == 0,
            "valid real-token draft has a dead grammar position");
    (void)required.preview_model(wire, 1000, FinishReason::OutputLimit);
    required.discard_preview();
    Output tool_output;
    commit(required, wire, tool_output);
    require(tool_output.calls.size() == 1 &&
                Json::parse(tool_output.calls[0].arguments_json) == Json{{"x", "hello"}},
            "tool grammar preview/discard lost the selected call");

    auto owned = [&] {
        std::shared_ptr<const text::Tokenizer> local_tokenizer;
        auto frontend = output_frontend(fixtures / "prompt-tokenizer.json", local_tokenizer);
        auto prepared = frontend.prepare(input());
        auto result   = frontend.make_output_session(prepared, {}, {}, {},
                                                     OutputConstraint::choice({"yes"}), {});
        return std::pair{std::move(result), local_tokenizer->encode("r</think>yes<|endoftext|>")};
    }();
    Output retained;
    commit(owned.first, owned.second, retained);
    require(retained.content == "yes", "output session borrowed destroyed frontend resources");
}

void probe(text::GrammarCompiler& compiler) {
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto input = Json::parse(line);
            const auto bound = contract(input.at("schema"));
            Json result      = Json::array();
            for (const auto& item : input.at("candidates")) {
                const auto source   = item.get<std::string>();
                const bool accepted = accepts(compiler, *bound, source);
                result.push_back({{"accepted", accepted},
                                  {"arguments", accepted ? decode(*bound, source) : Json{}}});
            }
            std::cout << Json{{"results", result}}.dump() << '\n';
        } catch (const std::exception& error) {
            std::cout << Json{{"error", error.what()}}.dump() << '\n';
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        text::GrammarCompiler compiler(vocabulary(), {256}, 64 * 1024 * 1024);
        if (argc > 1 && std::string_view(argv[1]) == "--probe") {
            probe(compiler);
            return 0;
        }
        basic(compiler);
        parameter_names(compiler);
        strict(compiler);
        transactions(compiler);
        std::shared_ptr<const text::Tokenizer> tokenizer;
        const auto f = output_frontend(argc > 1 ? std::filesystem::path(argv[1])
                                                : fixtures / "prompt-tokenizer.json",
                                       tokenizer);
        integration(f, *tokenizer);
        std::cout << "GLM grammar, decoder and output transactions passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
