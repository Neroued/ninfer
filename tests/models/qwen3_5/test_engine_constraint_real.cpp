// Structured output on a real artifact through the public Engine (P5/P6). The Engine is built from
// ninfer-serve flags (after "--"), so the production profiles run unchanged:
//
//   ninfer_engine_constraint_real_test <phase> -- <artifact> --tp 2 --spec mtp ...
//
// Phases:
//   main    rejection before admission; >= 100 generations under varied JSON Schemas (thinking off
//           and on with a thinking budget, greedy and T=0.7), each ending with EOS in parseable
//           JSON; every result goes to NINFER_TEST_SCHEMA_REPORT (JSONL) for the external
//           validator (validate_constraint_report.py, `jsonschema`). Mixed batch: with C >= 2,
//           C free greedy rows run twice, then row 0 is replaced by a constrained row whose
//           language is row 0's free answer (a literal): the constrained row must reproduce it
//           and the free rows must stay identical token for token. DFlash2: constrained rows draft
//           nothing (fallback steps only, D3) while free rows still draft.
//   deadend NINFER_CONSTRAINT_DEAD_END_AFTER must be set (read by this program, which injects
//           the dead end through runtime::testing; the engine reads no environment hook): a
//           constrained request fails with ConstraintDeadEnd, a concurrent free one completes,
//           and the Engine stays usable.
//   contain every matcher failure path (dead end and mask-fill exception at the first token and
//           in decode, thinking-control span refused) fails only the constrained request; C-1
//           free requests submitted with it finish normally; the Engine stays usable.
//   budget  the audit's B1/B4 case: emoji reasoning cut by thinking budgets 1024..1031 under a
//           JSON Schema; every answer is one valid JSON value ending on EOS, no second close.
//   suite   rejection, prefix hit, contain, budget and a reduced validity run (what ctest runs
//           when NINFER_TEST_CONSTRAINT_*_ARGS are configured).
//   equiv   greedy token ids of free and constrained cases to NINFER_TEST_TOKENS_OUT (JSONL), to
//           compare an MTP Engine with an ordinary one.
//   bench   ms/round of one constrained row against the same tokens free (C=1), and of a batch
//           of C rows with one constrained row against the all-free batch, alternated ABBA.
//
// Returns 77 without arguments (ctest runs without an artifact).

#include "ninfer/engine.h"
#include "runtime/engine/constraint_masks.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::FinishReason;
using ninfer::OutputConstraint;
using Json = nlohmann::ordered_json;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

ninfer::EngineOptions engine_options(const ninfer::serve::ServeOptions& serve) {
    // The same mapping as serve::GenerationService.
    ninfer::EngineOptions options;
    options.artifact_path          = serve.artifact_path;
    options.chat_template_path     = serve.chat_template_path;
    options.device                 = serve.device;
    options.tp                     = serve.tp;
    options.devices                = serve.devices;
    options.max_context            = serve.max_context;
    options.kv_capacity            = serve.kv_capacity;
    options.max_concurrency        = serve.max_concurrency;
    options.max_pending_requests   = std::max<std::uint32_t>(serve.max_pending_requests, 64);
    options.pending_timeout_ms     = std::max<std::uint32_t>(serve.pending_timeout_ms, 600'000);
    options.decode_share           = serve.decode_share;
    options.prefill_chunk          = serve.prefill_chunk;
    options.kv_lease_window_tokens = serve.kv_lease_window_tokens;
    options.kv_cache               = serve.kv_cache;
    options.elastic_kv             = serve.elastic_kv;
    options.kv_ram_active          = serve.kv_ram_active;
    if (serve.miss_split) {
        options.routed_experts.miss_split.mode      = *serve.miss_split;
        options.routed_experts.miss_split.calibrate = serve.miss_split_calibrate;
    }
    if (serve.expert_cpu_threads) {
        options.routed_experts.cpu_threads      = *serve.expert_cpu_threads;
        options.routed_experts.gguf_cpu_threads = *serve.expert_cpu_threads;
    }
    if (serve.stream_min_tokens) {
        options.routed_experts.stream_min_tokens = *serve.stream_min_tokens;
    }
    options.enable_vision            = serve.enable_vision;
    options.vision_device            = serve.vision_device;
    options.max_vision_tokens        = serve.max_vision_tokens;
    options.use_cuda_graph           = serve.use_cuda_graph;
    options.tp_mailbox               = serve.tp_mailbox;
    options.embedding_host           = serve.embedding_host;
    options.host_pinned_max_bytes    = serve.host_pinned_max_bytes;
    options.speculative              = serve.speculative;
    options.context_cache            = serve.context_cache;
    options.media_cache_bytes        = serve.media_cache_bytes;
    options.media_live_bytes         = serve.media_live_bytes;
    options.media_preprocess_threads = serve.media_preprocess_threads;
    return options;
}

ninfer::PromptInput prompt(const std::string& text, bool thinking) {
    ninfer::PromptInput input;
    input.options.enable_thinking = thinking;
    input.messages.push_back({.role  = ninfer::ChatRole::User,
                              .parts = {{.kind = ninfer::MessagePartKind::Text, .text = text}}});
    return input;
}

ninfer::RequestOptions request(std::uint32_t tokens, float temperature,
                               std::optional<OutputConstraint> constraint) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens    = tokens;
    options.execution.sampling.temperature       = temperature;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.execution.sampling.seed              = 4242;
    // Root prefill for every request: a reused prefix takes another numerical route, and the
    // comparisons below need the same route on both sides.
    options.execution.allow_prefix_reuse = false;
    options.constraint                   = std::move(constraint);
    return options;
}

std::string finish_name(FinishReason reason) {
    switch (reason) {
    case FinishReason::StopToken:
        return "stop_token";
    case FinishReason::OutputLimit:
        return "output_limit";
    case FinishReason::ContextCapacity:
        return "context_capacity";
    default:
        return "other";
    }
}

struct SchemaCase {
    std::string name;
    std::optional<Json> schema; // nullopt = json_object
    std::vector<std::string> topics;
};

std::vector<SchemaCase> schema_cases() {
    const auto string_max = [](int n) { return Json{{"type", "string"}, {"maxLength", n}}; };
    const auto integer    = [](int lo, int hi) {
        return Json{{"type", "integer"}, {"minimum", lo}, {"maximum", hi}};
    };
    std::vector<SchemaCase> cases;
    cases.push_back(
        {"person",
         Json{{"type", "object"},
              {"properties",
               {{"name", string_max(40)}, {"age", integer(0, 120)}, {"city", string_max(30)}}},
              {"required", {"name", "age", "city"}},
              {"additionalProperties", false}},
         {"a fictional engineer", "a famous painter", "a child who loves trains"}});
    cases.push_back(
        {"sentiment",
         Json{{"type", "object"},
              {"properties",
               {{"sentiment", {{"enum", {"positive", "negative", "neutral"}}}},
                {"confidence", {{"type", "number"}, {"minimum", 0}, {"maximum", 1}}}}},
              {"required", {"sentiment", "confidence"}},
              {"additionalProperties", false}},
         {"'I love this product, it changed my life'",
          "'The delivery was late and the box was broken'", "'The meeting is at 3 pm'"}});
    cases.push_back(
        {"order",
         Json{{"type", "object"},
              {"properties",
               {{"order",
                 {{"type", "object"},
                  {"properties",
                   {{"id", integer(1, 99999)},
                    {"items",
                     {{"type", "array"},
                      {"maxItems", 4},
                      {"items",
                       {{"type", "object"},
                        {"properties", {{"sku", string_max(12)}, {"qty", integer(1, 50)}}},
                        {"required", {"sku", "qty"}},
                        {"additionalProperties", false}}}}}}},
                  {"required", {"id", "items"}},
                  {"additionalProperties", false}}},
                {"paid", {{"type", "boolean"}}}}},
              {"required", {"order", "paid"}},
              {"additionalProperties", false}},
         {"a bakery order", "an order of office supplies", "a hardware store order"}});
    cases.push_back(
        {"colors",
         Json{{"type", "array"}, {"items", string_max(20)}, {"minItems", 1}, {"maxItems", 5}},
         {"colors of the rainbow", "warm colors", "colors of the Italian flag"}});
    cases.push_back(
        {"review",
         Json{{"type", "object"},
              {"properties",
               {{"title", string_max(60)},
                {"tags", {{"type", "array"}, {"items", string_max(15)}, {"maxItems", 3}}},
                {"rating", integer(1, 5)}}},
              {"required", {"title"}},
              {"additionalProperties", false}},
         {"a review of a pizzeria", "a review of a laptop", "a review of a novel"}});
    cases.push_back(
        {"result",
         Json{{"type", "object"},
              {"properties", {{"result", {{"anyOf", {string_max(30), integer(0, 1000)}}}}}},
              {"required", {"result"}},
              {"additionalProperties", false}},
         {"what is 17*23", "the capital of France", "how many legs a spider has"}});
    cases.push_back({"summary",
                     Json{{"type", "object"},
                          {"properties",
                           {{"kind", {{"const", "summary"}}},
                            {"ok", {{"type", "boolean"}}},
                            {"count", integer(0, 10)},
                            {"text", string_max(80)}}},
                          {"required", {"kind", "ok", "count", "text"}},
                          {"additionalProperties", false}},
                     {"the plot of Hamlet", "the water cycle", "how a bicycle works"}});
    cases.push_back({"measure",
                     Json{{"type", "object"},
                          {"properties",
                           {{"value", {{"type", {"string", "null"}}, {"maxLength", 20}}},
                            {"unit", {{"enum", {"m", "km", "cm"}}}}}},
                          {"required", {"value", "unit"}},
                          {"additionalProperties", false}},
                     {"the height of the Eiffel tower", "the length of a pencil",
                      "the distance from Rome to Milan"}});
    cases.push_back({"object",
                     std::nullopt,
                     {"a JSON object describing a cat", "a JSON object with three fruits",
                      "a JSON object describing today's weather"}});
    cases.push_back({"parola",
                     Json{{"type", "object"},
                          {"properties",
                           {{"parola", string_max(30)},
                            {"traduzione", string_max(30)},
                            {"lingua", {{"enum", {"it", "en", "de"}}}}}},
                          {"required", {"parola", "traduzione", "lingua"}},
                          {"additionalProperties", false}},
                     {"la parola 'farfalla'", "la parola 'Zürich' (città)", "the word 'snow'"}});
    return cases;
}

OutputConstraint constraint_of(const SchemaCase& entry) {
    return entry.schema ? OutputConstraint::json_schema(entry.schema->dump())
                        : OutputConstraint::json_object();
}

std::string schema_prompt(const SchemaCase& entry, const std::string& topic) {
    std::string text = "Return JSON about " + topic + ".";
    if (entry.schema) { text += " It must follow this JSON Schema: " + entry.schema->dump(); }
    return text;
}

struct Variant {
    bool thinking;
    float temperature;
};

// `constrained` false: the same jobs free (the "planning" phase, A/B of admission behaviour).
void validity(ninfer::Engine& engine, std::uint32_t concurrency, bool constrained = true,
              bool reuse = false, bool quick = false) {
    const char* report_path = std::getenv("NINFER_TEST_SCHEMA_REPORT");
    std::ofstream report;
    if (report_path != nullptr) { report.open(report_path, std::ios::app); }
    const std::vector<Variant> variants{{false, 0.0F}, {false, 0.7F}, {true, 0.0F}, {true, 0.7F}};

    struct Job {
        const SchemaCase* entry;
        std::string topic;
        Variant variant;
    };

    std::vector<Job> jobs;
    const auto cases = schema_cases();
    for (const auto& entry : cases) {
        for (const auto& topic : entry.topics) {
            for (const auto& variant : variants) { jobs.push_back({&entry, topic, variant}); }
            // quick (ctest suite): first topic of each case, thinking off T=0 and on T=0.7.
            if (quick) {
                jobs.erase(jobs.end() - 4, jobs.end());
                jobs.push_back({&entry, topic, variants[0]});
                jobs.push_back({&entry, topic, variants[3]});
                break;
            }
        }
    }
    std::size_t valid = 0, eos = 0, rejected = 0;
    std::uint64_t rounds = 0, fallback = 0, drafted = 0, accepted = 0;
    for (std::size_t begin = 0; begin < jobs.size(); begin += concurrency) {
        const std::size_t end = std::min(jobs.size(), begin + concurrency);
        std::vector<ninfer::GenerationHandle> handles;
        for (std::size_t index = begin; index < end; ++index) {
            const Job& job = jobs[index];
            auto options =
                request(1024, job.variant.temperature,
                        constrained ? std::optional(constraint_of(*job.entry)) : std::nullopt);
            options.execution.sampling.seed      = 1000 + index;
            options.execution.allow_prefix_reuse = reuse;
            if (job.variant.thinking) { options.execution.thinking.budget = 128; }
            handles.push_back(engine.submit(
                engine.prepare(prompt(schema_prompt(*job.entry, job.topic), job.variant.thinking)),
                options));
        }
        for (std::size_t index = begin; index < end; ++index) {
            const Job& job = jobs[index];
            ninfer::GenerationResult result;
            try {
                result = handles[index - begin].wait();
            } catch (const ninfer::RequestError& error) {
                ++rejected;
                check(false, "case " + job.entry->name + " / " + job.topic +
                                 ": request error: " + error.what());
                continue;
            }
            if (!constrained) { continue; }
            const bool parsed   = nlohmann::json::accept(result.content);
            const bool stop_eos = result.finish_reason == FinishReason::StopToken;
            valid += parsed ? 1U : 0U;
            eos += stop_eos ? 1U : 0U;
            rounds += result.speculative.rounds;
            fallback += result.speculative.fallback_steps;
            drafted += result.speculative.drafted_tokens;
            accepted += result.speculative.accepted_tokens;
            check(parsed && stop_eos, "case " + job.entry->name + " / " + job.topic +
                                          (job.variant.thinking ? " thinking" : "") +
                                          " T=" + std::to_string(job.variant.temperature) +
                                          ": finish " + finish_name(result.finish_reason) +
                                          ", content [" + result.content + "]");
            if (report) {
                report << Json{{"case", job.entry->name},
                               {"schema",
                                job.entry->schema ? *job.entry->schema : Json{{"type", "object"}}},
                               {"thinking", job.variant.thinking},
                               {"temperature", job.variant.temperature},
                               {"finish", finish_name(result.finish_reason)},
                               {"reasoning_tokens", result.reasoning_tokens},
                               {"tokens", result.generated_token_ids.size()},
                               {"content", result.content}}
                              .dump(-1, ' ', false, Json::error_handler_t::replace)
                       << '\n';
            }
        }
    }
    std::cout << "validity" << (constrained ? "" : " (free)") << (reuse ? " reuse" : "") << ": "
              << jobs.size() << " generations, " << rejected << " request errors, " << valid
              << " parseable, " << eos << " ended on EOS; speculative rounds " << rounds
              << " fallback " << fallback << " accepted " << accepted << "/" << drafted << "\n";
}

// C free greedy rows submitted together. Returns their results.
std::vector<ninfer::GenerationResult>
run_batch(ninfer::Engine& engine, const std::vector<std::string>& texts,
          const std::vector<ninfer::RequestOptions>& options) {
    std::vector<ninfer::PreparedPrompt> prompts;
    for (const auto& text : texts) { prompts.push_back(engine.prepare(prompt(text, false))); }
    std::vector<ninfer::GenerationHandle> handles;
    for (std::size_t row = 0; row < texts.size(); ++row) {
        handles.push_back(engine.submit(std::move(prompts[row]), options[row]));
    }
    std::vector<ninfer::GenerationResult> results;
    for (auto& handle : handles) { results.push_back(handle.wait()); }
    return results;
}

const std::vector<std::string> kFreeTexts{
    "Write a short paragraph about the history of Rome.",
    "Explain in a few sentences how a refrigerator works.",
    "List five tips for learning a new language, one per line.",
    "Describe the taste of coffee to someone who never tried it.",
    "Write a haiku about autumn and explain it briefly.",
    "Give three reasons why the sky is blue.",
    "What are the main differences between C and Python?",
    "Tell a very short story about a lost key.",
};

void mixed_batch(ninfer::Engine& engine, std::uint32_t concurrency, bool dflash2) {
    if (concurrency < 2) {
        std::cout << "mixed batch: skipped (max_concurrency 1)\n";
        return;
    }
    const std::size_t rows = concurrency;
    std::vector<std::string> texts(kFreeTexts.begin(), kFreeTexts.begin() + rows);
    std::vector<ninfer::RequestOptions> free(rows, request(96, 0.0F, std::nullopt));
    const auto reference = run_batch(engine, texts, free);
    const auto repeat    = run_batch(engine, texts, free);
    std::size_t stable   = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        stable += reference[row].generated_token_ids == repeat[row].generated_token_ids ? 1U : 0U;
    }
    // Row 0 constrained to its own free answer: greedy over the legal set picks the same argmax.
    auto mixed_options          = free;
    mixed_options[0].constraint = OutputConstraint::choice({reference[0].content});
    const auto mixed            = run_batch(engine, texts, mixed_options);
    std::size_t identical       = 0;
    for (std::size_t row = 1; row < rows; ++row) {
        identical += mixed[row].generated_token_ids == reference[row].generated_token_ids ? 1U : 0U;
    }
    const bool constrained_same = mixed[0].content == reference[0].content &&
                                  mixed[0].generated_token_ids == reference[0].generated_token_ids;
    std::cout << "mixed batch C=" << rows << ": free reference stable " << stable << "/" << rows
              << ", free rows identical next to a constrained row " << identical << "/" << rows - 1
              << ", constrained row reproduces its free answer "
              << (constrained_same ? "yes" : "no") << " (" << mixed[0].generated_token_ids.size()
              << " tokens, finish " << finish_name(mixed[0].finish_reason) << ")\n";
    check(stable == rows, "free reference batch is not reproducible");
    check(identical == rows - 1, "a constrained row changed the free rows of its batch");
    check(constrained_same, "constrained greedy row did not reproduce its free answer");
    if (dflash2) {
        check(mixed[0].speculative.rounds == 0 && mixed[0].speculative.fallback_steps > 0,
              "DFlash2 constrained row drafted (D3 requires extent 0)");
        check(mixed[1].speculative.rounds > 0, "DFlash2 free row next to a constrained one did "
                                               "not draft");
        std::cout << "dflash2: constrained row fallback steps "
                  << mixed[0].speculative.fallback_steps << ", free row rounds "
                  << mixed[1].speculative.rounds << "\n";
    }
}

void rejection(ninfer::Engine& engine) {
    auto invalid  = request(16, 0.0F, OutputConstraint::grammar("root ::= missing"));
    bool rejected = false;
    try {
        (void)engine.submit(engine.prepare(prompt("hi", false)), invalid);
    } catch (const ninfer::RequestError& error) {
        rejected = error.kind() == ninfer::RequestErrorKind::InvalidGrammar;
    }
    check(rejected && engine.is_available(), "invalid grammar not rejected before admission");
    const auto yes = engine.generate(engine.prepare(prompt("Say yes.", false)),
                                     request(16, 0.0F, OutputConstraint::choice({"yes", "no"})));
    check(yes.content == "yes" || yes.content == "no", "choice output outside its language");
    check(yes.finish_reason == FinishReason::StopToken, "choice did not end on EOS");
}

// A zero-suffix prefix hit samples the first token from the retained hidden with the lane's
// sampling config: the mask must be bound there too (upstream 08902f73 test).
void exact_hit(ninfer::Engine& engine) {
    const auto raw = engine.tokenize_text("A raw prompt checkpoint. Answer yes or no: ");
    auto seed      = request(1, 0.0F, std::nullopt);
    seed.execution.allow_prefix_reuse = true;
    (void)engine.generate(engine.prepare_tokens(raw), seed);
    for (const char* only : {"no", "yes"}) {
        auto hit                         = request(16, 0.0F, OutputConstraint::choice({only}));
        hit.execution.allow_prefix_reuse = true;
        const auto result                = engine.generate(engine.prepare_tokens(raw), hit);
        std::cout << "exact hit: reused " << result.reused_prompt_tokens << "/" << raw.size()
                  << " content [" << result.content << "]\n";
        check(result.content == only && result.finish_reason == FinishReason::StopToken,
              "prefix-hit request bypassed its first-token mask");
    }
}

void dead_end(ninfer::Engine& engine, std::uint32_t concurrency) {
    for (int repeat = 0; repeat < 2; ++repeat) {
        auto constrained =
            engine.submit(engine.prepare(prompt("Return a JSON object about cats.", repeat == 1)),
                          request(256, 0.0F, OutputConstraint::json_object()));
        std::optional<ninfer::GenerationHandle> free;
        if (concurrency >= 2) {
            free = engine.submit(engine.prepare(prompt(kFreeTexts[repeat], false)),
                                 request(64, 0.0F, std::nullopt));
        }
        bool dead = false;
        try {
            (void)constrained.wait();
        } catch (const ninfer::RequestError& error) {
            dead = error.kind() == ninfer::RequestErrorKind::ConstraintDeadEnd;
            std::cout << "dead end: " << error.what() << "\n";
        }
        check(dead, "constrained request did not fail with ConstraintDeadEnd");
        if (free) {
            const auto result = free->wait();
            check(!result.content.empty(), "free request next to a dead end lost its output");
            std::cout << "dead end: concurrent free request finished with "
                      << result.generated_token_ids.size() << " tokens\n";
        }
        check(engine.is_available(), "Engine unavailable after a dead end");
    }
    const auto after = engine.generate(engine.prepare(prompt("What is 17*23?", false)),
                                       request(64, 0.0F, std::nullopt));
    std::cout << "dead end: next request [" << after.content << "]\n";
    check(!after.generated_token_ids.empty() && (after.finish_reason == FinishReason::StopToken ||
                                                 after.finish_reason == FinishReason::OutputLimit),
          "Engine unusable after dead ends");
}

const char* kDogSchema =
    R"({"type":"object","properties":{"name":{"type":"string","maxLength":40},)"
    R"("legs":{"type":"integer","minimum":0,"maximum":10}},"required":["name","legs"],)"
    R"("additionalProperties":false})";

// Containment (post-audit principle): whatever path the matcher fails in, only the constrained
// request fails (ConstraintDeadEnd); the C-1 free requests submitted with it finish normally and
// the Engine stays usable. Paths: dead end at the first token (prefill) and in decode, an
// exception out of the mask fill at the first token and in decode, and the thinking-control span
// refused by the matcher (audit B1: the Engine used to stop for everyone).
void containment(ninfer::Engine& engine, std::uint32_t concurrency) {
    namespace hooks = ninfer::runtime::testing;

    struct Path {
        const char* name;
        std::size_t dead_end_after;
        std::size_t mask_failure_after;
        bool refuse_control;
    };

    const std::vector<Path> paths{
        {"dead end, prefill", 1, 0, false},        {"dead end, decode", 6, 0, false},
        {"mask fill error, prefill", 0, 1, false}, {"mask fill error, decode", 0, 6, false},
        {"thinking control refused", 0, 0, true},
    };
    const std::size_t free_rows = std::max<std::uint32_t>(concurrency, 2) - 1;
    for (const auto& path : paths) {
        hooks::constraint_dead_end_after.store(path.dead_end_after);
        hooks::constraint_mask_failure_after.store(path.mask_failure_after);
        hooks::constraint_refuse_control.store(path.refuse_control);
        auto options = request(256, 0.0F, OutputConstraint::json_object());
        if (path.refuse_control) { options.execution.thinking.budget = 16; }
        auto constrained = engine.submit(
            engine.prepare(prompt("Return a JSON object about cats.", path.refuse_control)),
            options);
        std::vector<ninfer::GenerationHandle> free;
        for (std::size_t row = 0; row < free_rows; ++row) {
            free.push_back(engine.submit(
                engine.prepare(prompt(kFreeTexts[row % kFreeTexts.size()], row % 2 == 1)),
                request(96, 0.0F, std::nullopt)));
        }
        bool failed_alone = false;
        std::string error_text;
        try {
            (void)constrained.wait();
        } catch (const ninfer::RequestError& error) {
            failed_alone = error.kind() == ninfer::RequestErrorKind::ConstraintDeadEnd;
            error_text   = error.what();
        }
        std::size_t normal = 0;
        for (auto& handle : free) {
            try {
                const auto result = handle.wait();
                normal += !result.generated_token_ids.empty() &&
                                  (result.finish_reason == FinishReason::StopToken ||
                                   result.finish_reason == FinishReason::OutputLimit)
                              ? 1U
                              : 0U;
            } catch (const std::exception& error) {
                std::cout << "containment: free request failed: " << error.what() << "\n";
            }
        }
        hooks::constraint_dead_end_after.store(0);
        hooks::constraint_mask_failure_after.store(0);
        hooks::constraint_refuse_control.store(false);
        const bool available = engine.is_available();
        std::cout << "containment [" << path.name << "]: constrained "
                  << (failed_alone ? "failed alone" : "DID NOT FAIL AS EXPECTED") << " ("
                  << error_text << "), free " << normal << "/" << free.size()
                  << " finished normally, engine " << (available ? "available" : "UNAVAILABLE")
                  << "\n";
        check(failed_alone, std::string(path.name) + ": constrained request did not fail alone");
        check(normal == free.size(), std::string(path.name) + ": a free request did not finish");
        check(available, std::string(path.name) + ": Engine unavailable afterwards");
    }
    const auto after =
        engine.generate(engine.prepare(prompt("Return a JSON object with the name and the number "
                                              "of legs of a dog.",
                                              false)),
                        request(256, 0.0F, OutputConstraint::json_schema(kDogSchema)));
    check(nlohmann::json::accept(after.content) && after.finish_reason == FinishReason::StopToken,
          "constrained request after the containment paths is not valid JSON: [" + after.content +
              "]");
}

// Audit B1/B4/I3 on the real model: a prompt that fills the reasoning with emoji (cut characters
// at the budget), JSON Schema, thinking budgets around 1024 (the audit's failing values on both
// models), submitted together. Every request must end on EOS with parseable JSON whose content
// does not start with a second close; the Engine stays available.
void budget_cuts(ninfer::Engine& engine, std::uint32_t concurrency) {
    const Json schema =
        Json{{"type", "object"},
             {"properties", {{"answer", {{"type", "string"}}}, {"count", {{"type", "integer"}}}}},
             {"required", {"answer", "count"}},
             {"additionalProperties", false}};
    std::string text = "Count the flamingos. Here is the list: ";
    for (int index = 0; index < 300; ++index) { text += "🦩🦚"; }
    text += "\nIn your thinking, go through the list emoji by emoji, writing each emoji you see "
            "followed by nothing else, before answering. Answer with a JSON object with the "
            "fields answer (one sentence) and count (the number of flamingos).";
    std::vector<std::uint32_t> budgets;
    for (std::uint32_t budget = 1024; budget <= 1031; ++budget) { budgets.push_back(budget); }
    std::size_t valid = 0, cut = 0;
    for (std::size_t begin = 0; begin < budgets.size(); begin += concurrency) {
        const std::size_t end = std::min(budgets.size(), begin + concurrency);
        std::vector<ninfer::GenerationHandle> handles;
        for (std::size_t index = begin; index < end; ++index) {
            auto options =
                request(budgets[index] + 400, 0.0F, OutputConstraint::json_schema(schema.dump()));
            options.execution.thinking.budget = budgets[index];
            handles.push_back(engine.submit(engine.prepare(prompt(text, true)), options));
        }
        for (std::size_t index = begin; index < end; ++index) {
            const std::uint32_t budget = budgets[index];
            try {
                const auto result = handles[index - begin].wait();
                const auto marker = result.reasoning.find("Considering the limited time");
                const bool split =
                    marker != std::string::npos && marker >= 5 &&
                    result.reasoning.substr(0, marker).find("\xef\xbf\xbd") != std::string::npos;
                cut += split ? 1U : 0U;
                const bool ok = nlohmann::json::accept(result.content) &&
                                result.finish_reason == FinishReason::StopToken &&
                                result.content.find("</think>") == std::string::npos;
                valid += ok ? 1U : 0U;
                std::cout << "budget " << budget << ": finish " << finish_name(result.finish_reason)
                          << ", reasoning tokens " << result.reasoning_tokens
                          << (split ? ", cut character before the control" : "") << ", content ["
                          << result.content.substr(0, 120) << "]\n";
                check(ok, "budget " + std::to_string(budget) +
                              ": content is not one valid JSON "
                              "value ending on EOS");
            } catch (const std::exception& error) {
                check(false,
                      "budget " + std::to_string(budget) + ": request failed: " + error.what());
            }
        }
        check(engine.is_available(), "Engine unavailable after budget cuts");
    }
    std::cout << "budget cuts: " << valid << "/" << budgets.size() << " valid, " << cut
              << " with a cut character before the control suffix\n";
}

void equivalence(ninfer::Engine& engine) {
    const char* path = std::getenv("NINFER_TEST_TOKENS_OUT");
    if (path == nullptr) { throw std::invalid_argument("equiv needs NINFER_TEST_TOKENS_OUT"); }
    std::ofstream out(path);
    const auto cases = schema_cases();
    for (const auto& entry : cases) {
        for (const auto& topic : entry.topics) {
            const auto text = schema_prompt(entry, topic);
            for (const bool constrained : {false, true}) {
                const auto result = engine.generate(
                    engine.prepare(prompt(text, false)),
                    request(192, 0.0F,
                            constrained ? std::optional(constraint_of(entry)) : std::nullopt));
                out << Json{{"case", entry.name + "/" + topic},
                            {"constrained", constrained},
                            {"finish", finish_name(result.finish_reason)},
                            {"tokens", result.generated_token_ids}}
                           .dump(-1, ' ', false, Json::error_handler_t::replace)
                    << '\n';
            }
        }
    }
    std::cout << "equiv: wrote " << cases.size() * 6 << " cases\n";
}

double ms_per_round(const ninfer::GenerationResult& result) {
    const double steps =
        static_cast<double>(result.speculative.rounds + result.speculative.fallback_steps);
    const double count = steps > 0 ? steps : static_cast<double>(result.generated_token_ids.size());
    return 1e3 * result.timings.decode_seconds / std::max(1.0, count - 1.0);
}

void bench(ninfer::Engine& engine, std::uint32_t concurrency) {
    const std::uint32_t tokens = 256;
    const std::string text =
        "Write a detailed technical explanation of how TCP congestion control works.";
    // C=1: the same tokens free and constrained to a literal of the free answer (identical work,
    // so the difference is the mask cost), then a schema-constrained answer.
    const auto reference =
        engine.generate(engine.prepare(prompt(text, false)), request(tokens, 0.0F, std::nullopt));
    const auto literal = OutputConstraint::choice({reference.content});
    const Json schema  = Json{{"type", "array"},
                              {"minItems", 30},
                              {"maxItems", 30},
                              {"items",
                               {{"type", "object"},
                                {"properties",
                                 {{"step", {{"type", "integer"}, {"minimum", 1}, {"maximum", 99}}},
                                  {"note", {{"type", "string"}, {"maxLength", 40}}}}},
                                {"required", {"step", "note"}},
                                {"additionalProperties", false}}}};
    const std::string schema_text =
        "Explain TCP congestion control as a JSON array of 30 steps with a short note each.";
    for (int cycle = 0; cycle < 2; ++cycle) {
        for (const bool constrained : {false, true, true, false}) {
            const auto result = engine.generate(
                engine.prepare(prompt(text, false)),
                request(tokens, 0.0F, constrained ? std::optional(literal) : std::nullopt));
            // Informational: a model whose free greedy output is not reproducible across
            // requests (Flash-Next with offloaded experts) changes the free side.
            if (result.generated_token_ids != reference.generated_token_ids) {
                std::cout << "bench note: tokens differ from the reference run\n";
            }
            std::cout << "bench c1 " << (constrained ? "literal" : "free   ") << " ms/round "
                      << ms_per_round(result) << " rounds " << result.speculative.rounds
                      << " fallback " << result.speculative.fallback_steps << " tokens "
                      << result.generated_token_ids.size() << " decode_s "
                      << result.timings.decode_seconds << "\n";
        }
        for (const bool constrained : {false, true, true, false}) {
            const auto result = engine.generate(
                engine.prepare(prompt(schema_text, false)),
                request(1024, 0.0F,
                        constrained ? std::optional(OutputConstraint::json_schema(schema.dump()))
                                    : std::nullopt));
            const double accepted = result.speculative.drafted_tokens
                                        ? static_cast<double>(result.speculative.accepted_tokens) /
                                              static_cast<double>(result.speculative.drafted_tokens)
                                        : 0.0;
            std::cout << "bench c1 schema " << (constrained ? "constrained" : "free       ")
                      << " ms/round " << ms_per_round(result) << " tokens "
                      << result.generated_token_ids.size() << " acceptance " << accepted
                      << " tok/s "
                      << static_cast<double>(result.generated_token_ids.size()) /
                             std::max(1e-9, result.timings.decode_seconds)
                      << "\n";
        }
    }
    if (concurrency < 2) { return; }
    std::vector<std::string> texts(kFreeTexts.begin(), kFreeTexts.begin() + concurrency);
    std::vector<ninfer::RequestOptions> free(concurrency, request(128, 0.0F, std::nullopt));
    const auto batch_reference = run_batch(engine, texts, free);
    auto mixed                 = free;
    mixed[0].constraint        = OutputConstraint::choice({batch_reference[0].content});
    for (int cycle = 0; cycle < 2; ++cycle) {
        for (const bool constrained : {false, true, true, false}) {
            const auto results = run_batch(engine, texts, constrained ? mixed : free);
            double worst       = 0;
            for (const auto& result : results) { worst = std::max(worst, ms_per_round(result)); }
            std::cout << "bench c" << concurrency << " "
                      << (constrained ? "1-constrained" : "all-free") << " ms/round row0 "
                      << ms_per_round(results[0]) << " row1 " << ms_per_round(results[1]) << " max "
                      << worst << "\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    // ctest: without arguments, NINFER_TEST_CONSTRAINT_SERVE_ARGS (space-separated ninfer-serve
    // flags, set by tests/cmake/ConstraintTests.cmake) selects the suite phase.
    std::vector<std::string> from_environment;
    std::vector<char*> forwarded;
    if (const char* flags = std::getenv("NINFER_TEST_CONSTRAINT_SERVE_ARGS");
        argc == 1 && flags != nullptr && *flags != '\0') {
        from_environment = {argv[0], "suite", "--"};
        std::istringstream stream(flags);
        for (std::string flag; stream >> flag;) { from_environment.push_back(flag); }
        for (auto& value : from_environment) { forwarded.push_back(value.data()); }
        argc = static_cast<int>(forwarded.size());
        argv = forwarded.data();
    }
    if (argc < 4 || std::string_view(argv[2]) != "--") {
        std::cout << "skip: usage <main|suite|contain|budget|deadend|equiv|bench> -- <ninfer-serve "
                     "flags>\n";
        return 77;
    }
    try {
        const std::string phase = argv[1];
        std::vector<char*> serve_argv{argv[0]};
        for (int index = 3; index < argc; ++index) { serve_argv.push_back(argv[index]); }
        const auto serve = ninfer::serve::parse_serve_options(static_cast<int>(serve_argv.size()),
                                                              serve_argv.data());
        ninfer::Engine engine(engine_options(serve));
        // ninfer-serve's default: the reasoning EOS guard is on (constrained rows skip it).
        engine.configure_reasoning_eos_guard(serve.reasoning_eos_guard);
        const std::uint32_t concurrency = serve.max_concurrency;
        const bool dflash2 = serve.speculative.backend == ninfer::SpeculativeBackend::DFlash2;
        const auto started = std::chrono::steady_clock::now();
        if (phase == "main") {
            rejection(engine);
            exact_hit(engine);
            validity(engine, concurrency);
            mixed_batch(engine, concurrency, dflash2);
        } else if (phase == "planningc") {
            validity(engine, concurrency, true, true);
        } else if (phase == "planning") {
            validity(engine, concurrency, false, true);
            validity(engine, concurrency, false, false);
        } else if (phase == "deadend") {
            // The test injects the dead end; the engine itself reads no environment hook.
            const char* after = std::getenv("NINFER_CONSTRAINT_DEAD_END_AFTER");
            if (after == nullptr) {
                throw std::invalid_argument("deadend needs NINFER_CONSTRAINT_DEAD_END_AFTER");
            }
            ninfer::runtime::testing::constraint_dead_end_after.store(
                static_cast<std::size_t>(std::strtoull(after, nullptr, 10)));
            dead_end(engine, concurrency);
            ninfer::runtime::testing::constraint_dead_end_after.store(0);
        } else if (phase == "contain") {
            containment(engine, concurrency);
        } else if (phase == "budget") {
            budget_cuts(engine, concurrency);
        } else if (phase == "suite") {
            // ctest with an artifact: rejection, prefix hit, containment of every failure path,
            // the audit's budget cuts, and a reduced validity run.
            rejection(engine);
            exact_hit(engine);
            containment(engine, concurrency);
            budget_cuts(engine, concurrency);
            validity(engine, concurrency, true, false, true);
        } else if (phase == "equiv") {
            equivalence(engine);
        } else if (phase == "bench") {
            bench(engine, concurrency);
        } else {
            throw std::invalid_argument("unknown phase " + phase);
        }
        std::cout
            << "phase " << phase << " took "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            << " s\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
