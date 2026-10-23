#include "models/glm5next/frontend/chat_template.h"
#include "chat_fixture.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace {
using namespace ninfer;
using Json = nlohmann::ordered_json;
using models::glm5next::CompiledChatTemplate;
using models::glm5next::RenderedChat;

using namespace glm_fixture;

std::size_t boundary(const RenderedChat& rendered, models::glm5next::TextSource source,
                     std::size_t offset) {
    const auto result = models::frontend::source_boundary(rendered.output, source.tag, offset);
    require(result.has_value(), "missing source boundary");
    return *result;
}

void provenance(const CompiledChatTemplate& compiled) {
    auto user = message(ChatRole::User, "é ");
    user.parts.push_back({.text = "<|assistant|><think>literal"});
    auto rendered     = compiled.render(std::vector{user});
    const auto first  = *rendered.messages[0].parts[0];
    const auto second = *rendered.messages[0].parts[1];
    const auto begin  = rendered.output.text.find("é ");
    require(boundary(rendered, first, first.bytes.begin) == begin &&
                boundary(rendered, first, first.bytes.end) == begin + user.parts[0].text.size() &&
                boundary(rendered, second, second.bytes.begin) == begin + user.parts[0].text.size(),
            "text part byte mapping changed");
    require(text::overlaps(rendered.output.literal_spans, begin, begin + 1) &&
                text::overlaps(rendered.output.literal_spans, begin + 3, begin + 16) &&
                !text::overlaps(rendered.output.literal_spans, rendered.output.text.size() - 7,
                                rendered.output.text.size()),
            "literal content and generation control markers were mixed");

    auto assistant         = message(ChatRole::Assistant, "");
    assistant.tool_calls   = {{"a", "first", R"({"v":"<think>"})"}, {"b", "second", "{}"}};
    auto b                 = message(ChatRole::Tool, "same");
    b.tool_call_id         = "b";
    auto a                 = message(ChatRole::Tool, "same");
    a.tool_call_id         = "a";
    a.tool_result_is_error = true;
    PromptOptions opts;
    opts.tool_jsons = {
        R"({"type":"function","function":{"name":"first","parameters":{"type":"object"}}})"};
    rendered      = compiled.render(std::vector{user, assistant, b, a}, opts);
    const auto pa = boundary(rendered, *rendered.messages[3].parts[0],
                             rendered.messages[3].parts[0]->bytes.begin);
    const auto pb = boundary(rendered, *rendered.messages[2].parts[0], 0);
    require(pa < pb && rendered.output.text.substr(pa, 4) == "same" &&
                rendered.output.text.substr(pb, 4) == "same" &&
                rendered.output.text.substr(pa - 13, 13) == "[tool_error]\n",
            "reordered tool results lost original identities or source offsets");
    for (auto tag : {rendered.messages[1].tool_calls[0], rendered.tools[0]})
        require(std::any_of(rendered.output.regions.begin(), rendered.output.regions.end(),
                            [&](const auto& region) { return region.tag == tag; }),
                "tool call or declaration lost its source");

    auto tail              = message(ChatRole::Assistant, "　é <|assistant|></think>　");
    tail.reasoning_content = "r </think> literal";
    opts                   = {};
    opts.continuation      = PromptContinuationMode::ContinueFinalAssistant;
    rendered               = compiled.render(std::vector{user, tail}, opts);
    require(!rendered.starts_in_reasoning &&
                rendered.continuation_content == "é <|assistant|></think>" &&
                rendered.output.text.ends_with(rendered.continuation_content),
            "continuation was affected by literal structural markers");
    const auto source = *rendered.messages.back().content;
    require(boundary(rendered, source, 0) ==
                    rendered.output.text.size() - rendered.continuation_content.size() &&
                boundary(rendered, source, source.bytes.end) == rendered.output.text.size(),
            "trimmed continuation boundaries were lost");
    tail.parts[0].text = "\x1c\x1f";
    rendered           = compiled.render(std::vector{user, tail}, opts);
    require(rendered.continuation_content.empty() && rendered.output.text.ends_with("</think>"),
            "continuation trim disagrees with Python whitespace semantics");
}

void options_and_failures(const CompiledChatTemplate& compiled, const std::string& source) {
    const std::vector messages{message(ChatRole::User, "question")};
    for (auto effort : {ReasoningEffort::None, ReasoningEffort::Minimal, ReasoningEffort::Medium,
                        ReasoningEffort::XHigh}) {
        PromptOptions options;
        options.reasoning_effort = effort;
        rejects([&] { (void)compiled.render(messages, options); }, "reasoning_effort");
    }
    for (const auto* kwargs :
         {R"({"enable_thinking":false})", R"({"enable_thinking":0})", R"({"clear_thinking":1})",
          R"({"preserve_thinking":"true"})", R"({"reasoning_effort":"medium"})",
          R"({"add_vision_id":true})", R"({"messages":[]})", R"({"tools":[]})",
          R"({"add_generation_prompt":false})", R"({"continue_final_message":true})",
          R"({"eos_token":"x"})"}) {
        PromptOptions options;
        options.chat_template_kwargs_json = kwargs;
        rejects([&] { (void)compiled.render(messages, options); }, "");
    }
    PromptOptions options;
    options.preserve_thinking         = true;
    options.chat_template_kwargs_json = R"({"clear_thinking":true})";
    rejects([&] { (void)compiled.render(messages, options); }, "conflicting");
    options                 = {};
    options.enable_thinking = false;
    rejects([&] { (void)compiled.render(messages, options); }, "disabled thinking");
    options                          = {};
    options.defaults.enable_thinking = false;
    rejects([&] { (void)compiled.render(messages, options); }, "disabled thinking");
    options                           = {};
    options.reasoning_effort          = ReasoningEffort::Low;
    options.chat_template_kwargs_json = R"({"reasoning_effort":"high"})";
    rejects([&] { (void)compiled.render(messages, options); }, "conflicting");
    options              = {};
    options.continuation = PromptContinuationMode::ContinueFinalAssistant;
    rejects([&] { (void)compiled.render(messages, options); }, "final text-only assistant");
    auto tail = message(ChatRole::Assistant, "prefix");
    tail.tool_calls.push_back({"a", "f", "{}"});
    rejects([&] { (void)compiled.render(std::vector{messages[0], tail}, options); },
            "final text-only assistant");
    tail.tool_calls.clear();
    tail.parts.push_back({.kind = MessagePartKind::Media});
    rejects([&] { (void)compiled.render(std::vector{messages[0], tail}, options); },
            "final text-only assistant");
    tail.parts.pop_back();

    CompiledChatTemplate suffix({text::JinjaTemplate(source + "<eop>", "suffix"), Json::object()});
    const auto rendered = suffix.render(std::vector{messages[0], tail}, options);
    require(rendered.output.text.ends_with("prefix"), "continuation retained a template suffix");
    const std::string repeated = "[gMASK]<sop><|assistant|><think></think>{{ messages[-1].content "
                                 "}}{{ messages[-1].content }}";
    CompiledChatTemplate ambiguous({text::JinjaTemplate(repeated, "ambiguous"), Json::object()});
    rejects([&] { (void)ambiguous.render(std::vector{tail}, options); },
            "expose the final assistant text once");
    const PreparationControl expired{.deadline = std::chrono::steady_clock::now() -
                                                 std::chrono::seconds(1)};
    try {
        (void)compiled.render(messages, {}, expired);
        throw std::runtime_error("expired preparation was accepted");
    } catch (const RequestError& error) {
        require(error.kind() == RequestErrorKind::QueueTimeout,
                "preparation error identity changed");
    }
}

void golden(const std::string& source, const CompiledChatTemplate& compiled,
            const text::Tokenizer* tokenizer) {
    const auto path = std::filesystem::path(NINFER_SOURCE_DIR) /
                      "tests/models/glm5next/fixtures/chat-golden.json";
    const auto data = Json::parse(read(path));
    text::JinjaTemplate raw(source, "GLM source template");
    for (const auto& item : data.at("cases")) {
        const auto name = item.at("name").get<std::string>();
        std::string rendered;
        if (item.contains("input")) {
            const auto input  = prompt(item["input"]);
            const auto result = compiled.render(input.messages, input.options);
            require(result.starts_in_reasoning ==
                        (input.options.continuation == PromptContinuationMode::NewAssistantTurn),
                    name + ": wrong initial output channel");
            rendered = result.output.text;
        } else {
            auto context        = item["reference"]["options"];
            context["messages"] = item["reference"]["messages"];
            rendered            = raw.render(context).text;
        }
        require(rendered == item["expected"]["text"].get<std::string>(),
                name + ": HF rendered bytes differ");
        // Raw-template tokenization follows HF's added-token behavior. Public literal spans
        // are checked independently by provenance(), not flattened into HF control tokens.
        if (tokenizer && !item.contains("input"))
            require(tokenizer->encode(rendered) == item["expected"]["ids"].get<std::vector<int>>(),
                    name + ": HF token IDs differ");
    }
    std::cout << data["cases"].size() << " GLM chat golden cases passed\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto fixture = std::filesystem::path(NINFER_SOURCE_DIR) /
                             "tests/models/glm5next/fixtures/chat_template.jinja";
        const auto source = read(fixture);
        const models::frontend::TextResources resources{.tokenizer_config_json = "{}",
                                                        .chat_template_jinja   = source};
        CompiledChatTemplate compiled(models::frontend::load_chat_template(resources, {}));
        std::unique_ptr<text::Tokenizer> tokenizer;
        if (argc == 3 && std::string_view(argv[1]) == "--resources") {
            const auto root       = std::filesystem::path(argv[2]);
            const auto vocabulary = read(root / "tokenizer.json");
            const auto config     = read(root / "tokenizer_config.json");
            tokenizer =
                std::make_unique<text::Tokenizer>(text::TokenizerResources{vocabulary, config});
        } else if (argc != 1) {
            throw std::invalid_argument("usage: chat_template_test [--resources DIR]");
        }
        golden(source, compiled, tokenizer.get());
        provenance(compiled);
        options_and_failures(compiled, source);
        std::cout << "GLM source mappings, options and continuation passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
