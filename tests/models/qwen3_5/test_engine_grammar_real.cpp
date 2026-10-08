#include "ninfer/engine.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::PromptInput prompt(bool thinking = false) {
    ninfer::PromptInput input;
    input.options.enable_thinking = thinking;
    input.messages.push_back({.role  = ninfer::ChatRole::User,
                              .parts = {{.kind = ninfer::MessagePartKind::Text,
                                         .text = "Return the requested answer."}}});
    input.context_cache.session_key = "grammar-test";
    return input;
}

ninfer::RequestOptions literal(const std::string& answer, float temperature = 0.0f) {
    ninfer::RequestOptions request;
    request.constraint =
        ninfer::OutputConstraint::grammar("root ::= " + nlohmann::json(answer).dump());
    request.execution.requested_output_tokens = 160;
    request.execution.sampling.temperature    = temperature;
    request.execution.sampling.top_k          = 20;
    request.execution.sampling.seed           = 78123;
    return request;
}

struct Sink final : ninfer::OutputSink {
    std::string content;

    void start(ninfer::GenerationStart) override {}

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta delta) override {
        if (delta.channel == ninfer::OutputChannel::Content) { content += delta.text; }
    }
};

void choice_and_regex(ninfer::Engine& engine, unsigned concurrency, bool speculative) {
    const std::vector<std::string> choices{"route_search", "route_calculate", "route_answer"};
    const std::regex pattern("(BUG|TASK)-[0-9]{4}");
    auto choice             = literal("unused", 0.8f);
    choice.constraint       = ninfer::OutputConstraint::choice(choices);
    auto regex              = choice;
    regex.constraint        = ninfer::OutputConstraint::regex("(BUG|TASK)-[0-9]{4}");
    const auto valid_choice = [&](const std::string& value) {
        return std::find(choices.begin(), choices.end(), value) != choices.end();
    };
    for (const auto& request : {choice, regex}) {
        Sink sink;
        const auto result = engine.generate(engine.prepare(prompt()), request, &sink);
        require(result.finish_reason == ninfer::FinishReason::StopToken &&
                    sink.content == result.content,
                "choice/regex streaming or completion changed");
        require(request.constraint->kind == ninfer::OutputConstraintKind::Choice
                    ? valid_choice(result.content)
                    : std::regex_match(result.content, pattern),
                "choice/regex produced content outside its language");
        if (speculative)
            require(result.speculative.rounds > 0, "choice/regex bypassed speculation");
    }
    const std::string literal_bytes = " 你好 \"a|b\\c\"\n";
    auto exact                      = choice;
    exact.constraint                = ninfer::OutputConstraint::choice({literal_bytes});
    require(engine.generate(engine.prepare(prompt()), exact).content == literal_bytes,
            "choice changed Unicode, whitespace or regex metacharacters");
    auto thinking                      = choice;
    thinking.execution.thinking.budget = 2;
    const auto thought                 = engine.generate(engine.prepare(prompt(true)), thinking);
    require(valid_choice(thought.content) &&
                thought.finish_reason == ninfer::FinishReason::StopToken,
            "choice constrained the thinking channel or lost the content boundary");
    for (const auto& constraint : {ninfer::OutputConstraint::choice({"TASK-12", "TASK-123"}),
                                   ninfer::OutputConstraint::regex("(BUG|TASK)-[0-9]{4}")}) {
        auto input                 = prompt();
        input.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
        input.context_cache.session_key.reset();
        input.messages.push_back(
            {.role  = ninfer::ChatRole::Assistant,
             .parts = {{.kind = ninfer::MessagePartKind::Text, .text = "TASK-"}}});
        auto request       = choice;
        request.constraint = constraint;
        const auto result  = engine.generate(engine.prepare(input), request);
        const auto full    = "TASK-" + result.content;
        require(result.finish_reason == ninfer::FinishReason::StopToken &&
                    (constraint.kind == ninfer::OutputConstraintKind::Choice
                         ? full == "TASK-12" || full == "TASK-123"
                         : std::regex_match(full, pattern)),
                "choice/regex continuation did not consume its existing prefix");
    }
    for (const auto& constraint :
         {ninfer::OutputConstraint::choice({""}), ninfer::OutputConstraint::regex("")}) {
        auto request       = choice;
        request.constraint = constraint;
        const auto result  = engine.generate(engine.prepare(prompt()), request);
        require(result.content.empty() && result.finish_reason == ninfer::FinishReason::StopToken,
                "empty choice/regex did not finish with empty content");
    }
    auto limited                              = regex;
    limited.constraint                        = ninfer::OutputConstraint::regex("[ab]{1000}");
    limited.execution.requested_output_tokens = 2;
    const auto partial                        = engine.generate(engine.prepare(prompt()), limited);
    require(partial.finish_reason == ninfer::FinishReason::OutputLimit &&
                !partial.content.empty() && partial.content.size() < 1000 &&
                partial.content.find_first_not_of("ab") == std::string::npos,
            "regex truncation was reported as a completed match");
    ninfer::RequestOptions free;
    free.execution.requested_output_tokens = 8;
    std::vector<ninfer::GenerationHandle> handles;
    for (unsigned row = 0; row < concurrency; ++row)
        handles.push_back(engine.submit(engine.prepare(prompt()), row % 3 == 0   ? choice
                                                                  : row % 3 == 1 ? regex
                                                                                 : free));
    for (unsigned row = 0; row < concurrency; ++row) {
        const auto result = handles[row].wait();
        if (row % 3 != 2)
            require(result.finish_reason == ninfer::FinishReason::StopToken &&
                        (row % 3 == 0 ? valid_choice(result.content)
                                      : std::regex_match(result.content, pattern)),
                    "mixed batch used another row's choice/regex");
    }
    std::cout << "choice/regex: literals, fullmatch, thinking, continuation, EOS, truncation and "
                 "mixed rows passed\n";
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const std::string backend = argc > 1 ? argv[1] : "none";
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 1024;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        options.max_concurrency = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3])) : 2;
        options.enable_vision   = argc > 4 && std::string_view(argv[4]) == "vision";
        options.prefill_chunk   = 128;
        options.kv_cache        = ninfer::KvCacheStorage::Fp8E4M3Row256;
        options.context_cache.device_state_slots  = 8;
        options.context_cache.host_capacity_bytes = 128ULL << 20;
        options.use_cuda_graph = argc < 3 || std::string_view(argv[2]) != "eager";
        if (backend == "mtp")
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        else if (backend == "dflash")
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        else if (backend == "dflash2")
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        else
            require(backend == "none", "unknown backend");
        if (backend != "none") {
            const auto draft_tokens           = std::getenv("NINFER_TEST_DRAFT_TOKENS");
            options.speculative.draft_tokens  = draft_tokens ? std::stoul(draft_tokens) : 3;
            options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        }
        ninfer::Engine engine(options);
        choice_and_regex(engine, options.max_concurrency, backend != "none");
        const std::string answer =
            "  {\"value\":\"你好\",\"literal\":\"<tool_call>x</tool_call>\"}";
        for (float temperature : {0.0f, 0.8f}) {
            Sink sink;
            auto request                                 = literal(answer, temperature);
            request.execution.sampling.presence_penalty  = 0.5f;
            request.execution.sampling.frequency_penalty = 0.125f;
            const auto result = engine.generate(engine.prepare(prompt()), request, &sink);
            require(result.content == answer && sink.content == answer && result.tool_calls.empty(),
                    "grammar content bytes or streaming publication changed");
            require(result.finish_reason == ninfer::FinishReason::StopToken,
                    "grammar did not finish through EOS");
            if (backend != "none")
                require(result.speculative.rounds > 0, "constraint bypassed speculative backend");
            std::cout << "literal temp=" << temperature << " reuse=" << result.reused_prompt_tokens
                      << " prepare_ms=" << result.timings.prepare_seconds * 1000
                      << " decode_ms=" << result.timings.decode_seconds * 1000 << '\n';
        }

        auto thinking                      = literal(answer);
        thinking.execution.thinking.budget = 2;
        auto thought = engine.generate(engine.prepare(prompt(true)), thinking);
        require(thought.content == answer && thought.thinking.applied &&
                    thought.thinking.injected_tokens > 0,
                "thinking control broke constrained content");

        auto continued = prompt();
        continued.context_cache.session_key.reset();
        continued.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
        const std::string prefix       = "{\"value\":";
        const std::string whole        = prefix + "\"你好\"}";
        continued.messages.push_back(
            {.role  = ninfer::ChatRole::Assistant,
             .parts = {{.kind = ninfer::MessagePartKind::Text, .text = prefix}}});
        const auto suffix = engine.generate(engine.prepare(continued), literal(whole));
        require(prefix + suffix.content == whole,
                "continuation matcher did not start at rendered content prefix");

        ninfer::RequestOptions free;
        free.execution.requested_output_tokens = 12;
        free.execution.sampling.temperature    = 0.0f;
        std::vector<ninfer::GenerationHandle> mixed;
        for (unsigned row = 0; row < options.max_concurrency; ++row) {
            const std::string row_answer = answer + std::to_string(row);
            mixed.push_back(engine.submit(engine.prepare(prompt()),
                                          row % 2 ? free : literal(row_answer, 0.8f)));
        }
        for (unsigned row = 0; row < mixed.size(); ++row) {
            const auto result = mixed[row].wait();
            if (row % 2 == 0)
                require(result.content == answer + std::to_string(row),
                        "mixed batch used another row's grammar");
        }
        if (options.enable_vision) {
            auto image_prompt = prompt();
            ninfer::MessagePart image;
            image.kind               = ninfer::MessagePartKind::Media;
            image.media.kind         = ninfer::MediaKind::Image;
            image.media.media_type   = "image/x-portable-pixmap";
            image.media.source_name  = "pattern.ppm";
            const std::string header = "P6\n64 64\n255\n";
            image.media.bytes.assign(header.begin(), header.end());
            image.media.bytes.resize(header.size() + 64 * 64 * 3, 42);
            image_prompt.messages[0].parts.insert(image_prompt.messages[0].parts.begin(),
                                                  std::move(image));
            require(engine.generate(engine.prepare(image_prompt), literal(answer)).content ==
                        answer,
                    "Vision prefill lost first-token grammar binding");
        }

        auto truncated       = literal(std::string(1024, 'a'));
        truncated.constraint = ninfer::OutputConstraint::grammar("root ::= \"a\"{1024}");
        truncated.execution.requested_output_tokens = 2;
        auto partial = engine.generate(engine.prepare(prompt()), truncated);
        require(partial.finish_reason == ninfer::FinishReason::OutputLimit &&
                    !partial.content.empty() &&
                    partial.content.find_first_not_of('a') == std::string::npos,
                "length-limited output is not a valid grammar prefix");
        const auto raw_prompt = engine.tokenize_text("A raw prompt checkpoint. Answer: ");
        auto seed             = literal("yes");
        seed.execution.requested_output_tokens = 1;
        (void)engine.generate(engine.prepare_tokens(raw_prompt), seed);
        const auto hit = engine.generate(engine.prepare_tokens(raw_prompt), literal("no"));
        require(hit.content == "no" && hit.reused_prompt_tokens == raw_prompt.size(),
                "exact prefix hit reused grammar state or bypassed first-token mask");
        auto raw = engine.generate(engine.prepare_tokens(engine.tokenize_text("Answer: ")),
                                   literal(answer));
        require(raw.content == answer, "raw-token prompt did not constrain newly generated bytes");

        // Cross a context profile during generation, then return to short, differently masked
        // requests. This exercises repeated handoffs and profile changes on the same Program.
        std::vector<ninfer::TokenId> long_prompt(480, 198);
        std::string long_answer;
        for (unsigned i = 0; i < 80; ++i) { long_answer += std::to_string(i) + ":a;"; }
        auto long_request                              = literal(long_answer);
        long_request.execution.allow_prefix_reuse      = false;
        long_request.execution.requested_output_tokens = 500;
        const auto long_result = engine.generate(engine.prepare_tokens(long_prompt), long_request);
        require(long_result.content == long_answer, "context profile change lost grammar position");
        for (const std::string answer_after : {"after-long", "different mask"}) {
            auto next                         = literal(answer_after);
            next.execution.allow_prefix_reuse = false;
            require(engine.generate(engine.prepare(prompt()), next).content == answer_after,
                    "short request reused a stale draft handoff or mask");
        }

        // JSON entry points use the same tokenizer, row mapping and transaction path as GBNF.
        const nlohmann::ordered_json schema = {
            {"type", "object"},
            {"properties",
             {{"description", {{"type", "string"}, {"enum", {"你好", "code"}}}},
              {"values",
               {{"type", "array"},
                {"prefixItems",
                 {{{"type", "number"}, {"minimum", 1e-8}, {"maximum", 2e-8}},
                  {{"type", "number"}, {"exclusiveMinimum", 0.1}, {"maximum", 0.2}}}},
                {"items", false},
                {"minItems", 2},
                {"maxItems", 2}}}}},
            {"required", {"description", "values"}},
            {"additionalProperties", false}};
        auto json_request       = literal("unused", 0.8f);
        json_request.constraint = ninfer::OutputConstraint::json_schema(schema.dump());
        const auto record       = [&](const nlohmann::ordered_json& spec,
                                const ninfer::GenerationResult& value) {
            require(value.finish_reason == ninfer::FinishReason::StopToken,
                          "JSON generation was truncated");
            require(nlohmann::json::accept(value.content), "JSON output is not parseable");
            if (const char* report = std::getenv("NINFER_TEST_SCHEMA_REPORT")) {
                std::ofstream file(report, std::ios::app);
                file << nlohmann::ordered_json{{"schema", spec}, {"content", value.content}}.dump()
                     << '\n';
                require(bool(file), "could not record schema validation output");
            }
        };
        const auto structured = engine.generate(engine.prepare(prompt()), json_request);
        record(schema, structured);
        const auto parsed = nlohmann::json::parse(structured.content);
        require(parsed.size() == 2 && parsed.contains("description") &&
                    parsed["values"].size() == 2 && parsed["values"][0] >= 1e-8 &&
                    parsed["values"][0] <= 2e-8 && parsed["values"][1] > 0.1 &&
                    parsed["values"][1] <= 0.2,
                "schema fields missing");
        auto json_thinking                      = json_request;
        json_thinking.execution.thinking.budget = 2;
        record(schema, engine.generate(engine.prepare(prompt(true)), json_thinking));
        auto numeric_prefix = prompt();
        numeric_prefix.context_cache.session_key.reset();
        numeric_prefix.options.continuation =
            ninfer::PromptContinuationMode::ContinueFinalAssistant;
        const std::string partial_number = "{\"description\":\"你好\",\"values\":[1.5e-";
        numeric_prefix.messages.push_back(
            {.role  = ninfer::ChatRole::Assistant,
             .parts = {{.kind = ninfer::MessagePartKind::Text, .text = partial_number}}});
        auto numeric_suffix    = engine.generate(engine.prepare(numeric_prefix), json_request);
        numeric_suffix.content = partial_number + numeric_suffix.content;
        record(schema, numeric_suffix);
        auto json_prompt                      = prompt();
        json_prompt.messages[0].parts[0].text = "Return exactly the JSON object {\"ok\":true}.";
        auto object_request                   = json_request;
        object_request.constraint             = ninfer::OutputConstraint::json_object();
        object_request.execution.sampling.temperature = 0;
        record({{"type", "object"}}, engine.generate(engine.prepare(json_prompt), object_request));

        const auto fixed  = nlohmann::ordered_json{{"const", {{"text", "你好\n"}, {"n", 2}}}};
        auto continuation = prompt();
        continuation.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
        continuation.context_cache.session_key.reset();
        const std::string json_prefix = "{\"text\":";
        continuation.messages.push_back(
            {.role  = ninfer::ChatRole::Assistant,
             .parts = {{.kind = ninfer::MessagePartKind::Text, .text = json_prefix}}});
        auto fixed_request       = json_request;
        fixed_request.constraint = ninfer::OutputConstraint::json_schema(fixed.dump());
        auto continued_json      = engine.generate(engine.prepare(continuation), fixed_request);
        continued_json.content   = json_prefix + continued_json.content;
        record(fixed, continued_json);
        auto changed       = json_request;
        changed.constraint = ninfer::OutputConstraint::json_schema(R"({"const":{"changed":true}})");
        const auto changed_result = engine.generate(engine.prepare(prompt()), changed);
        record({{"const", {{"changed", true}}}}, changed_result);
        require(changed_result.reused_prompt_tokens > 0,
                "schema switch did not exercise prefix reuse");

        std::vector<ninfer::GenerationHandle> json_batch;
        for (unsigned row = 0; row < options.max_concurrency; ++row) {
            auto selected = row % 3 == 0 ? json_request : row % 3 == 1 ? object_request : free;
            json_batch.push_back(engine.submit(engine.prepare(json_prompt), selected));
        }
        for (unsigned row = 0; row < json_batch.size(); ++row) {
            const auto value = json_batch[row].wait();
            if (row % 3 == 0)
                record(schema, value);
            else if (row % 3 == 1)
                record({{"type", "object"}}, value);
        }
        auto limited       = json_request;
        limited.constraint = ninfer::OutputConstraint::json_schema(
            R"({"type":"array","items":{"const":"word"},"minItems":128,"maxItems":128})");
        limited.execution.requested_output_tokens = 2;
        require(engine.generate(engine.prepare(prompt()), limited).finish_reason ==
                    ninfer::FinishReason::OutputLimit,
                "JSON truncation was reported as normal completion");

        auto invalid       = literal("yes");
        invalid.constraint = ninfer::OutputConstraint::grammar("root ::= missing");
        bool rejected      = false;
        try {
            (void)engine.submit(engine.prepare(prompt()), invalid);
        } catch (const ninfer::RequestError& error) {
            rejected = error.kind() == ninfer::RequestErrorKind::InvalidGrammar;
        }
        require(rejected && engine.is_available(),
                "invalid grammar was not isolated before admission");
        require(engine.generate(engine.prepare(prompt()), literal("yes")).content == "yes",
                "Engine did not remain usable after rejected request");
        std::cout << "constraints " << backend << (options.use_cuda_graph ? " graph" : " eager")
                  << ": content, sampling, thinking, continuation, mixed batch, truncation, raw "
                     "input, JSON/schema passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GBNF integration: " << error.what() << '\n';
        return 1;
    }
}
