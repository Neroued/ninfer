#include "ninfer/engine.h"

#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using Json = nlohmann::ordered_json;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Json parameters() {
    return {{"type", "object"},
            {"properties",
             {{"message", {{"type", "string"}, {"const", " 你好\n"}}},
              {"payload",
               {{"type", "object"},
                {"properties",
                 {{"text", {{"type", "string"}, {"const", "</parameter>"}}},
                  {"number", {{"type", "integer"}, {"minimum", 2}, {"maximum", 4}}}}},
                {"required", {"text", "number"}},
                {"additionalProperties", false}}}}},
            {"required", {"message", "payload"}},
            {"additionalProperties", false}};
}

ninfer::PromptInput prompt(const Json& schema, bool strict = true) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.options.tool_jsons.push_back(Json{
        {"type", "function"},
        {"function",
         {{"name", "record"},
          {"description", "Record the requested values"},
          {"parameters", schema},
          {"strict", strict}}}}.dump());
    input.messages.push_back(
        {.role  = ninfer::ChatRole::User,
         .parts = {{.kind = ninfer::MessagePartKind::Text,
                    .text = "Call record once with the values in its schema. Set number to 3."}}});
    input.context_cache.session_key = "tools-real";
    return input;
}

ninfer::RequestOptions request() {
    ninfer::RequestOptions value;
    value.tool_choice.mode                  = ninfer::ToolChoiceMode::Required;
    value.tool_choice.parallel              = false;
    value.execution.requested_output_tokens = 256;
    value.execution.sampling.temperature    = 0.0f;
    value.execution.sampling.seed           = 718;
    return value;
}

void record(std::string_view name, const ninfer::GenerationResult& value, const Json& schema) {
    Json result{{"case", name},
                {"schema", schema},
                {"calls", Json::array()},
                {"tokens", value.generated_token_ids.size()},
                {"prepare_ms", value.timings.prepare_seconds * 1000},
                {"prefill_ms", value.timings.prefill_seconds * 1000},
                {"decode_ms", value.timings.decode_seconds * 1000},
                {"rounds", value.speculative.rounds},
                {"drafted", value.speculative.drafted_tokens},
                {"accepted", value.speculative.accepted_tokens},
                {"reused", value.reused_prompt_tokens},
                {"prompt", value.prompt.prompt_tokens},
                {"preemptions", value.scheduling.preemptions},
                {"snapshot_restores", value.scheduling.snapshot_restores},
                {"replay_restores", value.scheduling.replay_restores}};
    for (const auto& call : value.tool_calls)
        result["calls"].push_back(
            {{"name", call.name}, {"arguments", Json::parse(call.arguments_json)}});
    std::cout << result.dump() << '\n';
    if (const char* path = std::getenv("NINFER_TEST_TOOL_REPORT"))
        std::ofstream(path, std::ios::app) << result.dump() << '\n';
}

void validate(const ninfer::GenerationResult& result, std::string_view name = "record") {
    require(result.finish_reason == ninfer::FinishReason::StopToken, "strict call truncated");
    require(result.tool_calls.size() == 1 && result.tool_calls[0].name == name,
            "selection/cardinality failed");
    const auto args = Json::parse(result.tool_calls[0].arguments_json);
    require(args.size() == 2 && args["message"] == " 你好\n" &&
                args["payload"]["text"] == "</parameter>" &&
                args["payload"]["number"].is_number_integer() && args["payload"]["number"] >= 2 &&
                args["payload"]["number"] <= 4,
            "strict arguments changed");
}

struct Sink final : ninfer::OutputSink {
    std::string content;

    void start(ninfer::GenerationStart) override {}

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta delta) override {
        if (delta.channel == ninfer::OutputChannel::Content) content += delta.text;
    }
};

void exercise_basic(ninfer::Engine& engine, unsigned concurrency, bool speculative) {
    const Json open_schema{
        {"type", "object"},
        {"properties", {{"message", {{"type", "string"}}}}},
        {"additionalProperties", true},
        {"patternProperties", {{"^extra", {{"type", "integer"}, {"multipleOf", 2}}}}}};
    auto input = prompt(open_schema, false);
    input.messages[0].parts[0].text =
        "Call record once with message=ready and extra_count=8. Only make the tool call.";
    auto options                 = request();
    options.tool_choice.mode     = ninfer::ToolChoiceMode::Auto;
    options.tool_choice.parallel = true;
    Sink sink;
    const auto result = engine.generate(engine.prepare(input), options, &sink);
    require(result.finish_reason == ninfer::FinishReason::StopToken && !result.tool_calls.empty(),
            "default tool fixture did not complete a call");
    require(sink.content == result.content &&
                result.content.find("<tool_call>") == std::string::npos,
            "default tool framing leaked into the text stream");
    for (const auto& call : result.tool_calls)
        require(call.name == "record" && Json::parse(call.arguments_json).is_object(),
                "default tool output is not a callable argument object");
    if (speculative) require(result.speculative.rounds > 0, "default tools bypassed speculation");

    // A root union also stays out of strict schema compilation. Preseed the first call's
    // open parameter to check that continuation shares the same broad argument language.
    const Json complex_schema{{"anyOf", {open_schema, Json{{"type", "object"}}}}};
    auto continued                 = prompt(complex_schema, false);
    continued.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    continued.context_cache.session_key.reset();
    continued.messages.push_back(
        {.role  = ninfer::ChatRole::Assistant,
         .parts = {{.kind = ninfer::MessagePartKind::Text,
                    .text = "<tool_call>\n<function=record>\n<parameter=custom-key>\nready"}}});
    const auto completed = engine.generate(engine.prepare(continued), options);
    require(completed.finish_reason == ninfer::FinishReason::StopToken &&
                !completed.tool_calls.empty() &&
                Json::parse(completed.tool_calls[0].arguments_json).contains("custom-key"),
            "open-parameter continuation was not completed");

    std::vector<ninfer::GenerationHandle> handles;
    for (unsigned row = 0; row < concurrency; ++row) {
        auto mixed  = row % 3 == 0 ? prompt(parameters()) : input;
        auto choice = row % 3 == 0 ? request() : options;
        if (row % 3 == 2) choice.tool_choice.constraints = ninfer::ToolConstraintMode::Automatic;
        handles.push_back(engine.submit(engine.prepare(mixed), choice));
    }
    for (unsigned row = 0; row < handles.size(); ++row) {
        const auto output = handles[row].wait();
        if (row % 3 == 0)
            validate(output);
        else
            require(output.finish_reason == ninfer::FinishReason::StopToken,
                    "mixed basic/free request did not complete");
    }
    std::cout << "default tools: open schema, complex schema, continuation, stream and mixed rows "
                 "passed\n";
}

void composed_output(ninfer::Engine& engine) {
    const auto schema  = parameters();
    auto input         = prompt(schema);
    auto options       = request();
    options.constraint = ninfer::OutputConstraint::json_schema(
        R"({"type":"object","properties":{"answer":{"type":"string","enum":["ok <tool_call>","wrong"],"pattern":"^ok","maxLength":30}},"required":["answer"],"additionalProperties":false})");
    auto first = engine
                     .submit(engine.prepare(input), options, ninfer::OutputConsumerMode::Aggregate,
                             {.phase_timings = true})
                     .wait();
    validate(first);
    require(first.constraint && first.constraint->complete && first.constraint->terminated &&
                first.constraint->branch == ninfer::ConstraintOutputBranch::Tools &&
                first.constraint->timings_collected && first.constraint->mask_positions > 0 &&
                first.constraint->mask_upload_bytes > 0,
            "tool branch lost constraint state or work observations");
    input.messages.push_back(
        {.role       = ninfer::ChatRole::Assistant,
         .tool_calls = {{"composed_call", "record", first.tool_calls[0].arguments_json}}});
    input.messages.push_back({.role         = ninfer::ChatRole::Tool,
                              .parts        = {{.kind = ninfer::MessagePartKind::Text,
                                                .text = "Recorded. Give the final JSON answer."}},
                              .tool_call_id = "composed_call"});
    // Prefix fixes the legal output branch; correctness does not depend on a probabilistic
    // decision to use another tool when both branches are available.
    input.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    input.messages.push_back(
        {.role  = ninfer::ChatRole::Assistant,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = "{\"answer\":"}}});
    options.tool_choice.mode        = ninfer::ToolChoiceMode::Auto;
    options.tool_choice.constraints = ninfer::ToolConstraintMode::Automatic;
    Sink sink;
    auto second = engine.generate(engine.prepare(input), options, &sink);
    require(second.finish_reason == ninfer::FinishReason::StopToken && second.tool_calls.empty() &&
                sink.content == second.content &&
                Json::parse("{\"answer\":" + second.content)["answer"] == "ok <tool_call>" &&
                second.constraint && second.constraint->terminated &&
                second.constraint->branch == ninfer::ConstraintOutputBranch::Content,
            "tool-result JSON continuation lost schema, literal marker or stream bytes");
    std::cout << "tools + JSON: required call, result continuation, literal marker and "
                 "observations passed\n";
}

void exercise(ninfer::Engine& engine, unsigned concurrency, bool speculative) {
    composed_output(engine);
    const auto coordinates = Json::parse(R"({"type":"object","properties":{
        "position":{"type":"array","prefixItems":[
            {"type":"number","minimum":30,"maximum":31},
            {"type":"number","minimum":-121,"maximum":-120}],"minItems":2,"items":false},
        "confidence":{"type":"number","exclusiveMinimum":0.1,"maximum":0.2}},
        "required":["position","confidence"],"additionalProperties":false})");
    for (float temperature : {0.0f, 0.8f}) {
        auto input = prompt(coordinates);
        input.messages[0].parts[0].text =
            "Call record once with position [30.5, -120.25] and confidence 0.15.";
        auto options                           = request();
        options.execution.sampling.temperature = temperature;
        const auto result                      = engine.generate(engine.prepare(input), options);
        require(result.finish_reason == ninfer::FinishReason::StopToken &&
                    result.tool_calls.size() == 1,
                "positional numeric tool call did not complete");
        const auto value = Json::parse(result.tool_calls[0].arguments_json);
        require(value["position"].size() == 2 && value["position"][0] >= 30 &&
                    value["position"][0] <= 31 && value["position"][1] >= -121 &&
                    value["position"][1] <= -120 && value["confidence"] > 0.1 &&
                    value["confidence"] <= 0.2,
                "published tuple/number arguments violate their schema");
        if (speculative)
            require(result.speculative.rounds > 0, "numeric tools bypassed speculation");
        record("positional_numbers", result, coordinates);
    }
    const auto schema = parameters();
    auto options      = request();
    ninfer::GenerationResult first;
    for (int i = 0; i < 4; ++i) {
        options.execution.sampling.temperature = i < 2 ? 0.0f : 0.8f;
        Sink sink;
        auto result = engine.generate(engine.prepare(prompt(schema)), options, &sink);
        validate(result);
        require(sink.content == result.content && sink.content.empty(),
                "tool framing leaked into text stream");
        if (speculative)
            require(result.speculative.rounds > 0, "tools bypassed speculative backend");
        record(i == 0 ? "cold" : "warm", result, schema);
        first = std::move(result);
    }
    auto roundtrip   = prompt(schema);
    const auto& call = first.tool_calls[0];
    roundtrip.messages.push_back(
        {.role       = ninfer::ChatRole::Assistant,
         .tool_calls = {{"call_roundtrip", call.name, call.arguments_json}}});
    roundtrip.messages.push_back(
        {.role         = ninfer::ChatRole::Tool,
         .parts        = {{.kind = ninfer::MessagePartKind::Text, .text = "Recorded."}},
         .tool_call_id = "call_roundtrip"});
    auto next                              = request();
    next.tool_choice.mode                  = ninfer::ToolChoiceMode::None;
    next.execution.requested_output_tokens = 16;
    const auto reuse                       = engine.generate(engine.prepare(roundtrip), next);
    require(reuse.tool_calls.empty(), "none emitted a tool call");
    record("tool_result_reuse", reuse, schema);
    // Consecutive newlines in this value can be generated as two tokens and re-encoded as one.
    // Exercise full endpoint reuse separately with a canonical integer parameter.
    const Json cache_schema{{"type", "object"},
                            {"properties", {{"number", {{"type", "integer"}, {"const", 3}}}}},
                            {"required", {"number"}},
                            {"additionalProperties", false}};
    auto cache_input      = prompt(cache_schema);
    const auto cache_call = engine.generate(engine.prepare(cache_input), request());
    require(cache_call.tool_calls.size() == 1, "cache fixture did not emit a call");
    cache_input.messages.push_back(
        {.role       = ninfer::ChatRole::Assistant,
         .tool_calls = {{"cache_call", "record", cache_call.tool_calls[0].arguments_json}}});
    cache_input.messages.push_back(
        {.role         = ninfer::ChatRole::Tool,
         .parts        = {{.kind = ninfer::MessagePartKind::Text, .text = "Recorded."}},
         .tool_call_id = "cache_call"});
    const auto cache_hit = engine.generate(engine.prepare(cache_input), next);
    record("canonical_tool_result_reuse", cache_hit, cache_schema);
    require(cache_hit.reused_prompt_tokens >=
                cache_call.prompt.prompt_tokens + cache_call.generated_token_ids.size() - 3,
            "canonical tool history failed to reuse the generated call prefix");

    auto continued = prompt(schema);
    continued.context_cache.session_key.reset();
    continued.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    continued.messages.push_back(
        {.role  = ninfer::ChatRole::Assistant,
         .parts = {{.kind = ninfer::MessagePartKind::Text,
                    .text = "<tool_call>\n<function=record>\n<parameter=message>\n"}}});
    validate(engine.generate(engine.prepare(continued), request()));
    auto thinking                     = prompt(schema);
    thinking.options.enable_thinking  = true;
    auto thought                      = request();
    thought.execution.thinking.budget = 2;
    const auto result                 = engine.generate(engine.prepare(thinking), thought);
    validate(result);
    require(result.thinking.injected_tokens > 0, "thinking fixture did not force control tokens");
    record("thinking", result, schema);

    auto limited                              = request();
    limited.execution.requested_output_tokens = 2;
    auto partial = engine.generate(engine.prepare(prompt(schema)), limited);
    require(partial.finish_reason == ninfer::FinishReason::OutputLimit &&
                partial.tool_calls.empty() && partial.content.empty(),
            "partial first call became a tool or visible framing");

    std::vector<ninfer::GenerationHandle> handles;
    for (unsigned i = 0; i < concurrency; ++i) {
        auto row                        = request();
        auto input                      = prompt(schema, i % 2 == 0);
        auto declaration                = Json::parse(input.options.tool_jsons[0]);
        declaration["function"]["name"] = "record_" + std::to_string(i);
        input.options.tool_jsons[0]     = declaration.dump();
        if (i % 2) {
            row.tool_choice.mode                  = ninfer::ToolChoiceMode::Auto;
            row.tool_choice.parallel              = true;
            row.execution.requested_output_tokens = 8;
        }
        handles.push_back(engine.submit(engine.prepare(input), row));
    }
    for (unsigned i = 0; i < concurrency; ++i) {
        const auto result = handles[i].wait();
        if (i % 2 == 0) validate(result, "record_" + std::to_string(i));
    }
}

void pressure(ninfer::Engine& engine, bool snapshot, bool cancel) {
    const Json schema{{"type", "object"},
                      {"properties",
                       {{"message", {{"type", "string"}, {"pattern", "^(ab cd ){180}$"}}},
                        {"position",
                         {{"type", "array"},
                          {"prefixItems",
                           {{{"type", "integer"}, {"minimum", 1}, {"maximum", 3}},
                            {{"type", "number"}, {"minimum", 0.1}, {"maximum", 0.2}}}},
                          {"minItems", 2},
                          {"items", false}}}}},
                      {"required", {"message", "position"}},
                      {"additionalProperties", false}};
    auto options                              = request();
    options.execution.requested_output_tokens = 650;
    options.execution.allow_prefix_reuse      = false;
    auto first = prompt(schema), second = prompt(schema);
    first.messages[0].parts[0].text  = "Call record. First request.";
    second.messages[0].parts[0].text = "Call record. Second request.";
    auto a                           = engine.submit(engine.prepare(first), options);
    auto b                           = engine.submit(engine.prepare(second), options);
    std::string expected;
    for (int i = 0; i < 180; ++i) expected += "ab cd ";
    if (cancel) {
        bool observed        = false;
        const auto stopped   = b.wait(nullptr, ninfer::CancellationView([&] {
                                        observed |= engine.runtime_stats().paused_requests > 0;
                                        return observed;
                                    }));
        const auto completed = a.wait();
        require(observed && stopped.finish_reason == ninfer::FinishReason::Cancelled &&
                    stopped.tool_calls.empty() && stopped.content.empty(),
                "paused cancellation published incomplete tool framing");
        require(completed.tool_calls.size() == 1 &&
                    Json::parse(completed.tool_calls[0].arguments_json)["message"] == expected,
                "cancellation changed another request's tool state");
        record("cancel_paused", stopped, schema);
        require(engine.is_available(), "tool cancellation made Engine unavailable");
        return;
    }
    std::uint64_t restores = 0;
    for (auto* handle : {&a, &b}) {
        auto result = handle->wait();
        require(result.finish_reason == ninfer::FinishReason::StopToken &&
                    result.tool_calls.size() == 1 &&
                    Json::parse(result.tool_calls[0].arguments_json)["message"] == expected,
                "recovery changed tool parser/matcher progress");
        restores +=
            snapshot ? result.scheduling.snapshot_restores : result.scheduling.replay_restores;
        record(snapshot ? "snapshot" : "replay", result, schema);
    }
    require(restores > 0, "pressure fixture did not restore a preempted request");
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
        const std::string mode    = argc > 2 ? argv[2] : "graph";
        const bool pressure_mode  = mode == "snapshot" || mode == "replay" || mode == "cancel";
        require(pressure_mode || mode == "graph" || mode == "eager" || mode == "basic",
                "unknown test mode");
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = pressure_mode ? 1024 : 1536;
        options.max_concurrency = argc > 3 ? std::stoul(argv[3]) : 2;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(
            pressure_mode ? 1024 : options.max_concurrency * 1024);
        options.prefill_chunk                     = 128;
        options.kv_cache                          = ninfer::KvCacheStorage::Fp8E4M3Row256;
        options.use_cuda_graph                    = mode != "eager";
        options.context_cache.enabled             = !pressure_mode;
        options.context_cache.device_state_slots  = pressure_mode ? 0 : 8;
        options.context_cache.host_capacity_bytes = mode == "replay" ? 0 : 512ULL << 20;
        if (backend == "mtp")
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        else if (backend == "dflash")
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        else if (backend == "dflash2")
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        else
            require(backend == "none", "unknown backend");
        if (backend != "none") {
            options.speculative.draft_tokens  = 3;
            options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        }
        ninfer::Engine engine(options);
        if (pressure_mode)
            pressure(engine, mode != "replay", mode == "cancel");
        else if (mode == "basic")
            exercise_basic(engine, options.max_concurrency, backend != "none");
        else
            exercise(engine, options.max_concurrency, backend != "none");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
