#include "output_fixture.h"
#include "models/glm5next/frontend/tool_decoder.h"
#include "models/glm5next/frontend/prepared_prompt.h"

#include <iostream>

namespace {
using namespace ninfer;
using namespace glm_fixture;
using Session = models::frontend::OutputSession;

std::vector<std::string> declarations() {
    const auto properties = Json{{"s", {{"type", "string"}}},      {"n", {{"type", "integer"}}},
                                 {"b", {{"type", "boolean"}}},     {"z", {{"type", "null"}}},
                                 {"o", {{"type", "object"}}},      {"a", {{"type", "array"}}},
                                 {"a\"&键", {{"type", "string"}}}, {"", {{"type", "string"}}}};
    return {declaration(tool_schema(properties)),
            declaration(tool_schema(Json::object()), false, "noop")};
}

void golden(const glm::Frontend& f, const text::Tokenizer& tokenizer,
            const std::filesystem::path& path) {
    const auto cases = Json::parse(read(path))["cases"];
    const auto p     = f.prepare(input(declarations()));
    for (const auto& test : cases) {
        const auto wire = test["wire"].get<std::string>();
        const auto ids  = test["ids"].get<std::vector<TokenId>>();
        require(tokenizer.encode(wire) == ids, "HF output token IDs differ");
        for (auto width : {ids.size(), std::size_t(1), std::size_t(3)}) {
            auto session = f.make_output_session(p, {}, {}, {}, {},
                                                 {.constraints = ToolConstraintMode::Automatic});
            Output result;
            for (std::size_t i = 0; i < ids.size(); i += width)
                commit(session, std::span(ids).subspan(i, std::min(width, ids.size() - i)), result);
            require(result.reasoning == test["reasoning"].get<std::string>() &&
                        result.content == test["content"].get<std::string>(),
                    test["name"].get<std::string>() + ": output channels differ");
            require(result.calls == session.take_tool_calls(),
                    "tool aggregate differs from published events");
            require(result.calls.size() == test["calls"].size(), "tool count differs");
            for (std::size_t i = 0; i < result.calls.size(); ++i)
                require(result.calls[i].name == test["calls"][i]["name"].get<std::string>() &&
                            Json::parse(result.calls[i].arguments_json) ==
                                test["calls"][i]["arguments"] &&
                            result.calls[i].status == ToolCallStatus::Complete,
                        "GLM tool types or values differ from source arguments");
            auto history                = input();
            auto assistant              = message(ChatRole::Assistant, result.content);
            assistant.reasoning_content = result.reasoning;
            for (std::size_t i = 0; i < result.calls.size(); ++i)
                assistant.tool_calls.push_back(
                    {std::to_string(i), result.calls[i].name, result.calls[i].arguments_json});
            history.messages.push_back(std::move(assistant));
            for (std::size_t i = 0; i < result.calls.size(); ++i) {
                auto response         = message(ChatRole::Tool, "ok");
                response.tool_call_id = std::to_string(i);
                history.messages.push_back(std::move(response));
            }
            const auto prepared = f.prepare(std::move(history));
            const auto expected = test["history_ids"].get<std::vector<TokenId>>();
            const auto& actual  = glm::PreparedPromptAccess::view(prepared).token_ids;
            require(tokenizer.decode(actual) == test["history_text"].get<std::string>(),
                    test["name"].get<std::string>() + ": tool history template bytes differ");
            const auto literals = test["literal_value_closers"].get<std::size_t>();
            if (!literals)
                require(actual == expected, "tool history token IDs differ from HF");
            else {
                // HF raw encoding recognizes added tokens even inside JSON strings. The public
                // input contract protects these source literals while retaining structural tags.
                const auto close = tokenizer.encode("</arg_value>").at(0);
                require(std::count(actual.begin(), actual.end(), close) + literals ==
                            std::count(expected.begin(), expected.end(), close),
                        "literal tool delimiters became input structure");
            }
        }
    }
    std::cout << cases.size() << " HF output cases passed\n";
}

void controls(const glm::Frontend& f, const text::Tokenizer& tokenizer) {
    const auto prompt = f.prepare(input());
    auto session      = f.make_output_session(prompt, {}, {}, {.budget = 2}, {}, {});
    Output output;
    auto word = tokenizer.encode("x");
    require(word.size() == 1, "fixture must contain x");
    for (const bool raw : {false, true}) {
        auto unlimited = f.make_output_session(prompt, {}, {.raw = raw}, {}, {}, {});
        (void)unlimited.preview_model(word, 100, FinishReason::OutputLimit);
        unlimited.discard_preview();
        require(unlimited.thinking_stats().model_thinking_tokens == 0,
                "discard advanced unbudgeted thinking");
        Output visible;
        commit(unlimited, tokenizer.encode("x</think>answer"), visible);
        const auto stats = unlimited.thinking_stats();
        require(stats.model_thinking_tokens == tokenizer.encode("x</think>").size() &&
                    !stats.configured_budget && !stats.applied && stats.injected_tokens == 0 &&
                    unlimited.model_token_budget_remaining(100) == 100,
                "unbudgeted thinking depends on presentation or includes post-close content");
    }
    commit(session, word, output);
    const auto decision = session.preview_model(word, 100, FinishReason::OutputLimit);
    require(decision.continuation == runtime::ContinuationAction::ApplyTargetControl,
            "budget did not request control");
    session.discard_preview();
    require(session.pending_control_tokens().empty() &&
                session.thinking_stats().model_thinking_tokens == 1,
            "discard advanced thinking budget");
    commit(session, word, output);
    const auto control = std::vector<TokenId>(session.pending_control_tokens().begin(),
                                              session.pending_control_tokens().end());
    require(tokenizer.decode(control) == "</think>", "control contains foreign framing");
    (void)session.preview_control(control, 100);
    session.discard_preview();
    require(!session.pending_control_tokens().empty(), "discard consumed pending control");
    const auto closed = session.preview_control(control, 100);
    require(closed.prefix_execution_split_after == control.size(), "control lost execution split");
    output.append(session.commit_preview());
    commit(session, tokenizer.encode("  answer<|endoftext|>"), output);
    require(output.reasoning == "xx" && output.content == "  answer" &&
                session.thinking_stats().applied,
            "forced close changed content bytes or reasoning");
    require(session.thinking_stats().injected_tokens == control.size(),
            "control token count differs");

    for (auto eos : f.default_stop_policy().token_ids) {
        auto terminal = f.make_output_session(prompt, {}, {}, {}, {}, {});
        auto ids      = tokenizer.encode("r</think>answer");
        ids.push_back(eos);
        ids.push_back(word.front());
        const auto stopped = terminal.preview_model(ids, 100, FinishReason::OutputLimit);
        require(stopped.finish_reason == FinishReason::StopToken &&
                    stopped.accepted_tokens == ids.size() - 1,
                "default EOS did not truncate accepted prefix");
        (void)terminal.commit_preview();
    }
    auto raw = f.make_output_session(prompt, {}, {.raw = true}, {}, {}, {});
    Output raw_out;
    commit(raw, tokenizer.encode("r</think>answer<|endoftext|>"), raw_out);
    require(raw_out.reasoning.empty() && raw_out.content == "r</think>answer<|endoftext|>",
            "raw output was interpreted");

    auto stopped = f.make_output_session(
        prompt, {.strings = {{.text = "END", .channel = OutputChannel::Content}}}, {}, {}, {}, {});
    Output visible;
    commit(stopped, tokenizer.encode("r</think>aE"), visible);
    const auto stop =
        stopped.preview_model(tokenizer.encode("NDtail"), 100, FinishReason::OutputLimit);
    visible.append(stopped.commit_preview());
    require(stop.finish_reason == FinishReason::StopString && visible.content == "a" &&
                stopped.matched_stop_string() == "END",
            "split stop string leaked");

    const auto tools = f.prepare(input(declarations()));
    auto implicit    = f.make_output_session(tools, {}, {}, {.budget = 100}, {},
                                             {.constraints = ToolConstraintMode::Automatic});
    Output tool_out;
    const auto ids = tokenizer.encode("reason<tool_call>noop</tool_call><|observation|>");
    for (auto id : ids) commit(implicit, std::span(&id, 1), tool_out);
    require(tool_out.reasoning == "reason" && tool_out.calls.size() == 1 &&
                !implicit.thinking_stats().applied,
            "implicit tool opener did not end reasoning");
    auto unlimited_tools = f.make_output_session(tools, {}, {}, {}, {},
                                                 {.constraints = ToolConstraintMode::Automatic});
    Output unlimited_tool_out;
    commit(unlimited_tools, ids, unlimited_tool_out);
    require(unlimited_tools.thinking_stats().model_thinking_tokens ==
                tokenizer.encode("reason<tool_call>").size(),
            "implicit tool opener did not close unbudgeted thinking accounting");

    auto continued_input = input();
    continued_input.messages.push_back(message(ChatRole::Assistant, "prefix"));
    continued_input.options.continuation = PromptContinuationMode::ContinueFinalAssistant;
    const auto continued                 = f.prepare(continued_input);
    auto continuation = f.make_output_session(continued, {}, {}, {.budget = 1}, {}, {});
    Output continued_out;
    commit(continuation, tokenizer.encode(" suffix<|endoftext|>"), continued_out);
    require(continued_out.content == " suffix" &&
                !continuation.thinking_stats().configured_budget &&
                continuation.thinking_stats().model_thinking_tokens == 0,
            "continuation restarted reasoning or repeated its prefix");

    auto split = f.make_output_session(prompt, {}, {}, {}, {}, {});
    Output split_out;
    for (char c : std::string("reason</think> content"))
        commit(split, tokenizer.encode(std::string(1, c)), split_out);
    commit(split, tokenizer.encode("<|endoftext|>"), split_out);
    require(split_out.reasoning == "reason" && split_out.content == " content",
            "reasoning delimiter split across ordinary tokens changed output");
}

void decoder_boundaries() {
    const auto bound =
        glm::select_tool_call_contract(glm::build_tool_call_output_contract(declarations()), {});
    const std::string wire =
        "pre<tool_call>call<arg_key>s</arg_key><arg_value>你\\\" \n</arg_value>"
        "<arg_key>o</arg_key><arg_value>{\"tag\":\"</arg_value>\"}</arg_value></tool_call>";
    for (std::size_t split = 0; split <= wire.size(); ++split) {
        glm::ToolCallOutputDecoder decoder(bound, 128);
        Output out;
        out.append(decoder.feed(std::string_view(wire).substr(0, split)));
        out.append(decoder.feed(std::string_view(wire).substr(split)));
        out.append(decoder.finish(FinishReason::StopToken));
        require(!decoder.error() && out.content == "pre" && out.calls.size() == 1,
                "split tool framing failed");
        require(Json::parse(out.calls[0].arguments_json) ==
                    Json{{"s", "你\\\" \n"}, {"o", {{"tag", "</arg_value>"}}}},
                "split tool value changed");
    }
    const auto raw = glm::build_tool_call_output_contract(declarations());
    for (auto reason : {FinishReason::OutputLimit, FinishReason::ContextCapacity,
                        FinishReason::Cancelled, FinishReason::StopString}) {
        glm::ToolCallOutputDecoder decoder(raw, 128);
        Output out;
        out.append(decoder.feed("<tool_call>call<arg_key>s</arg_key><arg_value>abc"));
        out.append(decoder.finish(reason));
        require(!decoder.error() && out.calls[0].status == ToolCallStatus::Incomplete &&
                    out.calls[0].arguments_json == "{\"s\":\"abc",
                "interruption repaired an incomplete call");
    }
    for (bool constrained : {false, true}) {
        glm::ToolCallOutputDecoder decoder(constrained ? bound : raw, 128);
        (void)decoder.feed(
            "<tool_call>call<arg_key>n</arg_key><arg_value>not-json</arg_value></tool_call>");
        require(decoder.error() &&
                    decoder.error()->kind ==
                        (constrained ? models::frontend::ToolOutputErrorKind::ContractViolation
                                     : models::frontend::ToolOutputErrorKind::InvalidModelOutput),
                "tool failure has wrong ownership");
    }
    glm::ToolCallOutputDecoder continuation(bound, 128);
    continuation.initialize_continuation("<tool_call>call<arg_key>s</arg_key><arg_value>begin");
    Output continued;
    continued.append(continuation.feed("end</arg_value></tool_call>"));
    continued.append(continuation.finish(FinishReason::StopToken));
    require(continued.calls.size() == 1 &&
                Json::parse(continued.calls[0].arguments_json) == Json{{"s", "beginend"}},
            "tool continuation lost its structured prefix");
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto path =
            argc > 1 ? std::filesystem::path(argv[1]) : fixtures / "prompt-tokenizer.json";
        std::shared_ptr<const text::Tokenizer> tokenizer;
        const auto frontend = output_frontend(path, tokenizer);
        golden(frontend, *tokenizer,
               argc > 2 ? std::filesystem::path(argv[2]) : fixtures / "output-golden.json");
        controls(frontend, *tokenizer);
        decoder_boundaries();
        std::cout << "GLM output transactions, controls and streaming passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
