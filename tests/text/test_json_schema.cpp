#include "text/grammar.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using Json       = nlohmann::ordered_json;
using Constraint = ninfer::OutputConstraint;

ninfer::text::GrammarCompiler make_compiler() {
    std::vector<std::string> vocabulary;
    for (int i = 0; i < 256; ++i) vocabulary.emplace_back(1, static_cast<char>(i));
    vocabulary.emplace_back();
    return {std::move(vocabulary), {256}, 4 * 1024 * 1024};
}

bool accepts(ninfer::text::GrammarCompiler& compiler, const Constraint& constraint,
             const std::string& candidate) {
    auto matcher = compiler.compile(constraint, {}, {});
    std::vector<std::uint32_t> words(matcher->mask_words());
    const auto allows = [&](int token) {
        return matcher->masks({}, words) == 0 && (words[token / 32] & (1u << (token % 32)));
    };
    for (unsigned char byte : candidate) {
        if (!allows(byte)) return false;
        matcher->accept(byte);
        matcher->confirm();
    }
    return allows(256);
}

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void contracts(ninfer::text::GrammarCompiler& compiler) {
    auto object = Constraint::json_object();
    for (const std::string text : {"{}", R"({"x":[1,true,null,{"y":"你好"}]})"})
        check(accepts(compiler, object, text), "JSON object rejected valid content");
    for (const std::string text : {"[]", "1", "{}{}", R"({"x":})"})
        check(!accepts(compiler, object, text), "JSON object admitted invalid content");
    auto schema = Constraint::json_schema(
        R"({"type":"object","properties":{"description":{"const":"first"},"second":{"const":{"description":"second"}}},"required":["description","second"],"additionalProperties":false})");
    check(accepts(compiler, schema, R"({"description":"first","second":{"description":"second"}})"),
          "annotation-named properties were lost");
    check(!accepts(compiler, schema, R"({"description":"first","second":{"description":"first"}})"),
          "literal data was aliased in schema cache");
    auto length = Constraint::json_schema(R"({"type":"string","minLength":1,"maxLength":1})");
    for (const std::string value :
         std::vector<std::string>{"你", "😀", "\n", "\t", std::string(1, '\0'), "\"", "\\"})
        check(accepts(compiler, length, Json(value).dump()),
              "JSON character length/encoding mismatch");
    check(!accepts(compiler, length, Json("ab").dump()), "string length was ignored");
    auto pattern = Constraint::json_schema(R"({"type":"string","pattern":"p"})");
    check(accepts(compiler, pattern, "\"apple\""), "pattern must use search semantics");
    check(!accepts(compiler, pattern, "\"absent\""), "pattern admitted a nonmatching string");
    const auto dot = Constraint::json_schema(R"({"type":"string","pattern":"^a.b$"})");
    for (const std::string separator : {"\n", "\r", "\u2028", "\u2029"})
        check(!accepts(compiler, dot, Json("a" + separator + "b").dump()),
              "ECMAScript dot admitted a line terminator");
}
} // namespace

int main(int argc, char** argv) {
    try {
        auto compiler = make_compiler();
        if (argc == 2 && std::string_view(argv[1]) == "--probe") {
            std::string line;
            while (std::getline(std::cin, line)) {
                Json response;
                try {
                    const auto input = Json::parse(line);
                    const auto constraint =
                        input.value("json_object", false)
                            ? Constraint::json_object()
                            : Constraint::json_schema(
                                  input.contains("schema_source")
                                      ? input["schema_source"].get<std::string>()
                                      : input.at("schema").dump());
                    auto session         = compiler.compile(constraint, {}, {});
                    response["accepted"] = Json::array();
                    for (const auto& text : input.at("candidates"))
                        response["accepted"].push_back(
                            accepts(compiler, constraint, text.get<std::string>()));
                } catch (const ninfer::RequestError& error) {
                    response = {{"error", error.what()},
                                {"kind", static_cast<int>(error.kind())},
                                {"pointer", error.pointer()}};
                } catch (const std::exception& error) { response = {{"error", error.what()}}; }
                std::cout << response.dump() << std::endl;
            }
        } else {
            contracts(compiler);
            std::cout << "JSON/schema CPU contracts passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
