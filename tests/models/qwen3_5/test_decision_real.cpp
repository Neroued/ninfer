#include "decision/decision.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::PromptInput chat() {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.messages                = {
        {.role = ninfer::ChatRole::User, .parts = {{.text = "Count from one to five."}}}};
    return input;
}

struct Case {
    std::vector<ninfer::TokenId> prefix, suffix, candidates;
};

std::vector<ninfer::BranchScoreRow> rows(const std::vector<Case>& cases, bool cache) {
    std::vector<ninfer::BranchScoreRow> out;
    for (const auto& c : cases) {
        out.push_back({c.prefix, c.suffix, c.candidates,
                       cache ? static_cast<std::uint32_t>(c.prefix.size()) : 0, cache ? 63U : 0U});
    }
    return out;
}

// BF16 GEMM shapes and prefill/ReplaySSM arithmetic differ. Compare conditional probabilities,
// not raw unnormalized logits. This threshold is fixed before the artifact comparison runs.
constexpr double kConditionalProbabilityTolerance = 0.02;

double compare(const std::vector<float>& a, const std::vector<float>& b) {
    expect(a.size() == b.size(), "candidate score shape changed");
    const auto pa = ninfer::decision::log_softmax(a);
    const auto pb = ninfer::decision::log_softmax(b);
    double worst  = 0;
    for (std::size_t i = 0; i < pa.size(); ++i) {
        worst = std::max(worst, std::abs(std::exp(pa[i]) - std::exp(pb[i])));
    }
    expect(worst <= kConditionalProbabilityTolerance,
           "branch conditional probabilities disagree with the reference");
    return worst;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        ninfer::EngineOptions options;
        options.artifact_path    = artifact;
        options.max_context      = 4096;
        options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.max_concurrency  = 2;
        options.enable_decisions = true;
        options.context_cache.device_state_slots  = 4;
        options.context_cache.host_capacity_bytes = 0;
        const auto* spec_env                      = std::getenv("NINFER_DECISION_TEST_SPEC");
        const std::string spec                    = spec_env ? spec_env : "none";
        if (spec == "mtp") {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        } else if (spec == "dflash") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        } else if (spec == "dflash2") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        } else {
            expect(spec == "none", "unknown NINFER_DECISION_TEST_SPEC");
        }
        if (spec != "none") { options.speculative.draft_tokens = 3; }
        if (const char* head = std::getenv("NINFER_DECISION_TEST_PROPOSAL_HEAD");
            head && std::string(head) == "1") {
            options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        }
        std::vector<Case> cases;
        ninfer::BranchScoreResult scored;
        std::size_t decision_kv_bytes = 0;
        double worst                  = 0;
        ninfer::MemorySummary native_memory;
        ninfer::RequestOptions generation;
        generation.execution.requested_output_tokens    = 8;
        generation.execution.allow_prefix_reuse         = false;
        generation.execution.sampling.temperature       = 0;
        generation.execution.sampling.presence_penalty  = 0;
        generation.execution.sampling.frequency_penalty = 0;
        std::vector<ninfer::TokenId> native_output;
        {
            auto native_options             = options;
            native_options.enable_decisions = false;
            ninfer::Engine native(native_options);
            native_memory = native.memory_summary();
            native_output = native.generate(native.prepare(chat()), generation).generated_token_ids;
        }
        {
            ninfer::Engine engine(options);
            const auto enabled_memory = engine.memory_summary();
            decision_kv_bytes         = enabled_memory.kv_payload_bytes;
            expect(enabled_memory.minimum_runtime_reservation_bytes ==
                           native_memory.minimum_runtime_reservation_bytes &&
                       enabled_memory.runtime_reservation_bytes ==
                           native_memory.runtime_reservation_bytes &&
                       enabled_memory.kv_capacity_increment_bytes ==
                           native_memory.kv_capacity_increment_bytes &&
                       enabled_memory.kv_payload_bytes == native_memory.kv_payload_bytes &&
                       enabled_memory.sequence.capacity_bytes ==
                           native_memory.sequence.capacity_bytes &&
                       enabled_memory.workspace.capacity_bytes ==
                           native_memory.workspace.capacity_bytes &&
                       enabled_memory.cuda_graph_allowance_bytes ==
                           native_memory.cuda_graph_allowance_bytes,
                   "enabling decisions changed the native startup resource plan");
            const auto baseline = engine.generate(engine.prepare(chat()), generation);
            expect(!baseline.generated_token_ids.empty(), "empty ordinary generation");
            expect(baseline.generated_token_ids == native_output,
                   "enabling decisions changed ordinary generation");

            auto candidates = engine.tokenize_text("true false");
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
            expect(candidates.size() >= 2, "need two distinct candidate tokens");
            const auto source = engine.tokenize_text("The customer asks about their account. ");
            expect(!source.empty(), "empty source tokens");
            std::vector<ninfer::TokenId> common;
            // Common prefix ends inside a 64-token KV page; each context crosses that page.
            for (std::size_t i = 0; i < 63; ++i) { common.push_back(source[i % source.size()]); }
            for (int context = 0; context < 2; ++context) {
                for (std::size_t length : {0, 1, 15, 16, 17, 33}) {
                    Case c{common, {}, candidates};
                    c.prefix.push_back(candidates[context]);
                    c.prefix.insert(c.prefix.end(), source.begin(),
                                    source.begin() + std::min<std::size_t>(3, source.size()));
                    for (std::size_t i = 0; i < length; ++i) {
                        c.suffix.push_back(source[i % source.size()]);
                    }
                    cases.push_back(std::move(c));
                }
            }
            const auto batch = rows(cases, true);
            scored           = engine.score_branches(batch);
            expect(scored.largest_batch >= 1 && scored.largest_batch <= 8 &&
                       scored.reused_tokens > 0,
                   "branches did not share a prefix and native batch");
            const auto cached = engine.score_branches(batch);
            expect(cached.computed_tokens < scored.computed_tokens, "exact root was not reused");
            const auto uncached = engine.score_branches(rows(cases, false));
            expect(uncached.reused_tokens == 0 && uncached.largest_batch <= 8 &&
                       uncached.largest_batch >= scored.largest_batch,
                   "cache opt-out or full batch width was not honored");
            for (std::size_t i = 0; i < cases.size(); ++i) {
                worst = std::max({worst, compare(scored.logits[i], cached.logits[i]),
                                  compare(scored.logits[i], uncached.logits[i])});
            }
            auto root_only = rows(cases, true);
            root_only.resize(1);
            root_only[0].root_tokens = root_only[0].trunk_tokens;
            const auto no_suffix     = engine.score_branches(root_only);
            worst = std::max(worst, compare(scored.logits[0], no_suffix.logits[0]));

            // Cancel after GPU work begins; cleanup must finish before borrowed inputs are
            // released.
            std::atomic<unsigned> polls = 0;
            ninfer::PreparationControl mid_batch{
                .cancellation = ninfer::CancellationView([&] { return ++polls >= 4; })};
            bool mid_cancel = false;
            try {
                (void)engine.score_branches(batch, mid_batch);
            } catch (const ninfer::RequestError& error) {
                mid_cancel = error.kind() == ninfer::RequestErrorKind::Cancelled;
            }
            expect(mid_cancel && engine.is_available(), "cancelled branch poisoned the Engine");
            const auto recovered = engine.score_branches(batch);
            for (std::size_t i = 0; i < cases.size(); ++i) {
                worst = std::max(worst, compare(scored.logits[i], recovered.logits[i]));
            }

            auto request = ninfer::decision::parse_request(ninfer::decision::Json::parse(R"({
            "contexts":["The customer says hello.","The customer asks about a late invoice."],
            "schema":{
                "greeting":{"type":"boolean","description":"Is this a greeting?"},
                "topic":{"type":"enum","choices":["billing","bill payment","other"],"description":"Topic"},
                "priority":{"type":"integer","minimum":1,"maximum":12,"description":"Priority"}
            },"mode":"tree"
        })"));
            ninfer::decision::DecisionEngine decisions(engine);
            for (bool cache : {true, true, false}) {
                request.cache_prompt = cache;
                const auto response  = decisions.run(request);
                expect(response.at("results").size() == 2, "lost decision context");
                for (const auto& result : response.at("results")) {
                    expect(result.at("decision").at("greeting").is_boolean(), "lost boolean type");
                    const auto priority = result.at("decision").at("priority").get<int>();
                    expect(priority >= 1 && priority <= 12, "integer outside its finite domain");
                    for (const auto& detail : result.at("fields")) {
                        const double probability = detail.at("probability").get<double>();
                        expect(std::isfinite(probability) && probability >= 0 && probability <= 1,
                               "invalid decision probability");
                    }
                }
                if (!cache) {
                    expect(response.at("usage").at("cached_tokens") == 0, "cache opt-out ignored");
                }
            }
            (void)engine.score_branches(batch); // Retain a known root before chat reclaims it.
            const auto after = engine.generate(engine.prepare(chat()), generation);
            expect(after.generated_token_ids == baseline.generated_token_ids,
                   "decision branches leaked state into ordinary generation");
            const auto after_chat = engine.score_branches(batch);
            expect(after_chat.computed_tokens == scored.computed_tokens,
                   "chat did not release the optional decision root");
            for (std::size_t i = 0; i < cases.size(); ++i) {
                worst = std::max(worst, compare(scored.logits[i], after_chat.logits[i]));
            }
            ninfer::PreparationControl cancelled{.cancellation =
                                                     ninfer::CancellationView([] { return true; })};
            bool did_cancel = false;
            try {
                (void)decisions.run(request, cancelled);
            } catch (const ninfer::RequestError& error) {
                did_cancel = error.kind() == ninfer::RequestErrorKind::Cancelled;
            }
            expect(did_cancel && engine.is_available(),
                   "cancellation did not preserve Engine availability");
        }
        // Independent execution route: ordinary causal prefill + full-vocabulary log-softmax.
        // Destroy generation first so the model is never resident twice.
        options.enable_decisions      = false;
        options.purpose               = ninfer::EnginePurpose::CausalScoring;
        options.speculative           = {};
        options.max_concurrency       = 1;
        options.context_cache.enabled = false;
        ninfer::Engine reference(options);
        if (spec == "none") {
            expect(reference.memory_summary().kv_payload_bytes == decision_kv_bytes,
                   "decision mode reserved a duplicate KV payload");
        }
        for (std::size_t i = 0; i < cases.size(); ++i) {
            std::vector<float> expected;
            for (auto candidate : cases[i].candidates) {
                auto tokens = cases[i].prefix;
                tokens.insert(tokens.end(), cases[i].suffix.begin(), cases[i].suffix.end());
                const auto first_target = static_cast<std::uint32_t>(tokens.size());
                tokens.push_back(candidate);
                const auto logp = reference.score_tokens(std::move(tokens), first_target);
                expect(logp.size() == 1, "causal reference returned the wrong score count");
                expected.push_back(logp.front());
            }
            worst = std::max(worst, compare(scored.logits[i], expected));
        }
        std::cout << "OK decision_real (spec=" << spec << ", max_probability_error=" << worst
                  << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decision_real failed: " << error.what() << '\n';
        return 1;
    }
}
