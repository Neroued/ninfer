// Qwen4Exp (Qwen3.8-Flash-Next) MTP rounds through the public Engine on a synthetic artifact
// with an MTP layer (NINFER_TEST_QWEN4_ARTIFACT; the 4-layer synthetic artifact of
// tools/capy_flash/make_test_artifact.sh is the intended one). Skips (77) without it or without a
// CUDA device.
//
// Oracle: the no-draft route. NINFER_QWEN4_MTP_NO_DRAFTS=1 runs every round with extent 0 (the
// anchor alone, the same verify / fold / alignment kernels), so with greedy sampling an MTP run
// must produce exactly its tokens whatever the drafts were; a difference is a rollback bug (GDN
// replay fold, PLE fold of fold_qwen4_ple and its [count,3] control rows, QSA re-append, the
// carried wide hidden, the PLE n-gram rows staged for the K+1 columns). On the near-uniform
// synthetic model the MTP layer's own drafts are almost never accepted, so a third run takes its
// drafts from the no-draft output (NINFER_QWEN4_MTP_ORACLE): every round is fully accepted.
//   1. mapped experts: MTP == no-draft == oracle drafts; the oracle run accepts every draft and
//      its last round has extent 0 (one token of budget left: a forced fallback round). The
//      prompt (33 tokens) leaves the windows across QSA block boundaries (position % 4 == 0).
//   2. offload experts (24 slots per layer, NINFER_QWEN4_EXPERT_FREEZE=1 so both runs place
//      every expert on the same side): MTP == no-draft == oracle drafts. Exercises the offload
//      backend in MTP windows (Text layers once, the MTP layer per alignment and draft call).
//   3. a cancelled row: two concurrent MTP requests, one cancelled while the other decodes; the
//      cancelled lane commits nothing (commit count 0 in the GDN and PLE folds). The survivor must
//      finish its budget, and a request decoded afterwards on the same Engine must equal the same
//      request on a fresh Engine (no state of the cancelled lane leaks into a later one).
//   6. prefix reuse through the MTP prefix-reuse bridge (T66b), at tp 1 (mapped and offload
//      experts) and, with two devices, at tp 2 (offload), residency fixed at initialization
//      (NINFER_QWEN4_EXPERT_FREEZE=all: placement must not depend on what an Engine served
//      before). (a) Shared prefix: chat request A, then request B sharing A's system message
//      resumes from the capture at its end (F) through the bridge (the MTP column F-1 from the
//      retained target hidden and B's token F; at tp 2 both ranks, each from its own retained
//      copy). Oracle: B alone on a fresh Engine, which splits its prefill at the same capture
//      frontier, so its Text state is computed identically: equal greedy tokens. (b)
//      Continuation: a raw-token turn, then a second turn extending it (answer and new tokens),
//      and one repeating exactly the turn's ledger, resume the private endpoint through the
//      bridge; each must equal its no-draft control. (Resumed and fresh runs of (b) are not
//      compared: the resumed Text state of the answer comes from verification windows, the fresh
//      one from a prompt chunk, and the near-uniform synthetic logits turn such last-bit
//      differences into different greedy tokens. For the same reason the draft proposals are not
//      compared: on this model they flip between near-equal tokens from run to run of one
//      configuration. The bridge's MTP state is checked on the real model, through acceptance.)
//   7. repeatable MTP head: the teacher-forced MTP pass (NINFER_QWEN4_TFDUMP with
//      NINFER_QWEN4_TFDUMP_MTP=1: every prompt column runs the MTP layer at T = 1, on the
//      hyper-connection small-T route) of one prompt on two fresh Engines writes the same MTP
//      logits bit for bit: a buffer that two launches of the pass read and write concurrently
//      (as the MTP layer's FFN mixer gates, which have a buffer of their own) would change them.
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <iostream>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// 160 pg19 tokens (Qwen tokenizer), the first 64 of them.
const std::vector<ninfer::TokenId> kIds = {
    987,   17956, 1931,   539,   81249, 28442,  321,   279,   7849,   42500, 198,   30058, 5961,
    7659,  506,   1696,   1074,  2068,  83846,  9504,  4909,  318,    1919,  999,   557,   198,
    8293,  1931,  494,    5167,  76409, 1801,   2420,  539,   561,    198,   34499, 28732, 10052,
    15256, 52682, 5946,   26467, 669,   206717, 4616,  357,   177414, 80281, 15810, 13,    14200,
    262,   1897,  206717, 4616,  357,   177414, 80281, 15810, 271,    262,   7466,  198,
};
constexpr std::uint32_t kDrafts = 3;
// 1 token from the prefill, then rounds of K+1 = 4 under oracle drafts: 1 + 4 * 10 = 41, so the
// 42nd token comes from a round with one token of budget left (extent 0).
constexpr std::uint32_t kOutput = 42;

// A per-process path `stem` + pid under NINFER_TEST_SCRATCH_DIR (the system temporary directory
// when unset).
std::string scratch_path(const std::string& stem) {
    const char* scratch              = std::getenv("NINFER_TEST_SCRATCH_DIR");
    const std::filesystem::path root = scratch != nullptr && *scratch
                                           ? std::filesystem::path(scratch)
                                           : std::filesystem::temp_directory_path();
    return (root / (stem + std::to_string(::getpid()))).string();
}

class ScopedEnv {
public:
    ScopedEnv(const char* name, const std::string& value) : name_(name) {
        ::setenv(name, value.c_str(), 1);
    }

    ~ScopedEnv() { ::unsetenv(name_); }

    ScopedEnv(const ScopedEnv&)            = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
};

enum class Experts { Mapped, Offload };

ninfer::EngineOptions options(const char* artifact, Experts experts, std::uint32_t lanes) {
    ninfer::EngineOptions o;
    o.artifact_path            = artifact;
    o.purpose                  = ninfer::EnginePurpose::Generation;
    o.kv_cache                 = ninfer::KvCacheStorage::BFloat16;
    o.prefill_chunk            = 128;
    o.max_context              = 256;
    o.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
    o.max_concurrency          = lanes;
    o.context_cache.enabled    = false;
    o.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    o.speculative.draft_tokens = kDrafts;
    if (experts == Experts::Mapped) {
        o.routed_experts.backend = ninfer::RoutedExpertBackend::Mapped;
    } else {
        o.routed_experts.backend         = ninfer::RoutedExpertBackend::Offload;
        o.routed_experts.slots_per_layer = 24;
    }
    return o;
}

ninfer::RequestOptions greedy(std::uint32_t tokens) {
    ninfer::RequestOptions r;
    r.execution.requested_output_tokens = tokens;
    r.execution.allow_prefix_reuse      = false;
    r.execution.sampling.temperature    = 0.0F;
    r.stop.include_model_defaults       = false;
    r.output.raw                        = true;
    return r;
}

std::vector<ninfer::TokenId> prompt(std::size_t begin, std::size_t count) {
    return {kIds.begin() + static_cast<std::ptrdiff_t>(begin),
            kIds.begin() + static_cast<std::ptrdiff_t>(begin + count)};
}

ninfer::GenerationResult generate(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& ids,
                                  std::uint32_t tokens) {
    return engine.generate(engine.prepare_tokens(ids, false), greedy(tokens));
}

// T43: a draft token shortlist file for the proposal heads (empty = the full vocabulary).
std::string g_shortlist;
// T43: tensor-parallel width of generate_once (2 = both GPUs; MTP at tp 2 needs offload experts).
int g_tp = 1;

ninfer::GenerationResult generate_once(const char* artifact, Experts experts,
                                       const std::vector<ninfer::TokenId>& ids) {
    auto o                        = options(artifact, experts, 1);
    o.speculative.draft_shortlist = g_shortlist;
    if (g_tp == 2) {
        o.tp      = 2;
        o.devices = {0, 1};
    }
    ninfer::Engine engine(o);
    return generate(engine, ids, kOutput);
}

void print(const char* label, const ninfer::GenerationResult& r) {
    const auto& s = r.speculative;
    std::printf("  %-28s %zu tokens, rounds %llu, drafted %llu, accepted %llu, fallback %llu\n",
                label, r.generated_token_ids.size(), static_cast<unsigned long long>(s.rounds),
                static_cast<unsigned long long>(s.drafted_tokens),
                static_cast<unsigned long long>(s.accepted_tokens),
                static_cast<unsigned long long>(s.fallback_steps));
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// The MTP logits of every column of the 64-token prompt, by dump file name, from a fresh Engine.
std::map<std::string, std::string> teacher_forced_mtp_logits(const char* artifact,
                                                             const std::string& directory) {
    std::filesystem::remove_all(directory);
    {
        const ScopedEnv dump("NINFER_QWEN4_TFDUMP", directory);
        const ScopedEnv mtp("NINFER_QWEN4_TFDUMP_MTP", "1");
        const ScopedEnv layers("NINFER_QWEN4_TFDUMP_LAYERS", "none");
        const ScopedEnv logits("NINFER_QWEN4_TFDUMP_LOGITS", "0");
        auto o           = options(artifact, Experts::Mapped, 1);
        o.use_cuda_graph = false;
        ninfer::Engine engine(o);
        (void)generate(engine, kIds, 1);
    }
    std::map<std::string, std::string> columns;
    const std::filesystem::path mtp = std::filesystem::path(directory) / "mtp_logits";
    if (std::filesystem::exists(mtp)) {
        for (const auto& entry : std::filesystem::directory_iterator(mtp)) {
            std::ifstream in(entry.path(), std::ios::binary);
            columns[entry.path().filename().string()] =
                std::string(std::istreambuf_iterator<char>(in), {});
        }
    }
    std::filesystem::remove_all(directory);
    return columns;
}

void repeatable_mtp_head(const char* artifact) {
    const std::string base = scratch_path("ninfer_qwen4_mtp_tfdump_");
    const auto first       = teacher_forced_mtp_logits(artifact, base + "_a");
    const auto second      = teacher_forced_mtp_logits(artifact, base + "_b");
    std::size_t differing  = 0;
    for (const auto& [name, bytes] : first) {
        const auto other = second.find(name);
        if (other == second.end() || other->second != bytes) { ++differing; }
    }
    std::printf("repeatable MTP head: %zu columns, %zu differ between two Engines\n", first.size(),
                differing);
    check(first.size() + 1 >= kIds.size() && first.size() == second.size(),
          "repeatable MTP head: the dump does not cover the prompt columns");
    check(differing == 0, "repeatable MTP head: two runs of one prompt gave different MTP logits");
}

// MTP, no-draft and oracle-draft runs of one prompt with `experts`.
void exactness(const char* artifact, Experts experts, const char* label) {
    const std::vector<ninfer::TokenId> ids = prompt(0, 33);
    const auto mtp                         = generate_once(artifact, experts, ids);
    ninfer::GenerationResult no_draft;
    {
        const ScopedEnv none("NINFER_QWEN4_MTP_NO_DRAFTS", "1");
        no_draft = generate_once(artifact, experts, ids);
    }
    const std::string oracle_path = scratch_path("ninfer_qwen4_mtp_oracle_") + ".txt";
    {
        std::ofstream out(oracle_path);
        for (const auto id : ids) { out << id << ' '; }
        for (const auto id : no_draft.generated_token_ids) { out << id << ' '; }
        out << '\n';
    }
    ninfer::GenerationResult oracle;
    {
        const ScopedEnv drafts("NINFER_QWEN4_MTP_ORACLE", oracle_path);
        oracle = generate_once(artifact, experts, ids);
    }
    std::remove(oracle_path.c_str());
    std::printf("%s:\n", label);
    print("MTP", mtp);
    print("no drafts", no_draft);
    print("oracle drafts", oracle);
    check(no_draft.generated_token_ids.size() == kOutput, std::string(label) + ": full budget");
    check(no_draft.speculative.drafted_tokens == 0,
          std::string(label) + ": the control drafts nothing");
    check(mtp.generated_token_ids == no_draft.generated_token_ids,
          std::string(label) + ": MTP tokens equal the no-draft tokens");
    check(oracle.generated_token_ids == no_draft.generated_token_ids,
          std::string(label) + ": oracle-draft tokens equal the no-draft tokens");
    check(oracle.speculative.drafted_tokens > 0 &&
              oracle.speculative.accepted_tokens == oracle.speculative.drafted_tokens,
          std::string(label) + ": every oracle draft is accepted");
    check(oracle.speculative.fallback_steps >= 1,
          std::string(label) + ": the last round (one token of budget) has extent 0");
}

// Counts the streamed output deltas of one request.
struct CountingSink final : ninfer::OutputSink {
    std::atomic<int> deltas{0};

    void start(ninfer::GenerationStart) override {}

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta) override { ++deltas; }
};

void cancelled_row(const char* artifact) {
    const std::vector<ninfer::TokenId> long_prompt  = prompt(0, 33);
    const std::vector<ninfer::TokenId> short_prompt = prompt(20, 17);
    const std::vector<ninfer::TokenId> later_prompt = prompt(40, 21);
    ninfer::GenerationResult survivor, cancelled, later;
    {
        ninfer::Engine engine(options(artifact, Experts::Mapped, 2));
        ninfer::GenerationHandle a =
            engine.submit(engine.prepare_tokens(long_prompt, false), greedy(200),
                          ninfer::OutputConsumerMode::Streaming);
        ninfer::GenerationHandle b =
            engine.submit(engine.prepare_tokens(short_prompt, false), greedy(64));
        // Cancelled once it has streamed output for a while, i.e. while both lanes decode (B is
        // admitted right after A; rounds take ~20 ms on the synthetic model).
        CountingSink sink;
        cancelled = a.wait(&sink, ninfer::CancellationView([&] { return sink.deltas >= 6; }));
        survivor  = b.wait();
        later     = generate(engine, later_prompt, kOutput);
    }
    ninfer::GenerationResult fresh;
    {
        ninfer::Engine engine(options(artifact, Experts::Mapped, 2));
        fresh = generate(engine, later_prompt, kOutput);
    }
    std::printf("cancelled row:\n");
    print("cancelled", cancelled);
    print("survivor", survivor);
    print("later request", later);
    print("same request, fresh engine", fresh);
    check(cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
              cancelled.generated_token_ids.size() >= 6 &&
              cancelled.generated_token_ids.size() < 200,
          "cancelled row: the request is cancelled mid-decode");
    check(survivor.generated_token_ids.size() == 64,
          "cancelled row: the survivor finishes its budget");
    check(later.generated_token_ids == fresh.generated_token_ids,
          "cancelled row: a later request equals the same request on a fresh engine");
}

ninfer::RequestOptions greedy_reuse(std::uint32_t tokens) {
    ninfer::RequestOptions r       = greedy(tokens);
    r.execution.allow_prefix_reuse = true;
    return r;
}

ninfer::EngineOptions reuse_options(const char* artifact, Experts experts, int tp) {
    auto o                  = options(artifact, experts, 1);
    o.context_cache.enabled = true;
    if (tp == 2) {
        o.tp                                   = 2;
        o.devices                              = {0, 1};
        o.context_cache.host_state_slots       = 0; // rank 1 has no Host tier
        o.context_cache.host_kv_capacity_bytes = 0;
    }
    return o;
}

// A chat prompt: a shared system message and one question, with an explicit shared-prefix
// boundary after the system message (the capture frontier F).
ninfer::PromptInput chat_prompt(const std::string& question) {
    const auto message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage m;
        m.role = role;
        m.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return m;
    };
    ninfer::PromptInput input;
    input.messages.push_back(message(
        ninfer::ChatRole::System,
        "You are the warehouse assistant of the Verona depot. Every morning the depot receives "
        "crates of apples, pears and plums from the farms of the Adige valley; each crate is "
        "logged with its farm, its weight and the day of arrival. Answer questions about the "
        "log briefly and precisely, and say so when the log does not contain the answer."));
    input.messages.push_back(message(ninfer::ChatRole::User, question));
    input.options.enable_thinking = false;
    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count = 1,
        .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence            = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
    });
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    return input;
}

// One Engine runs `prompts` in order.
std::vector<ninfer::GenerationResult> run_engine(const ninfer::EngineOptions& o,
                                                 const std::vector<ninfer::PromptInput>& prompts) {
    std::vector<ninfer::GenerationResult> results;
    ninfer::Engine engine(o);
    for (const auto& p : prompts) {
        results.push_back(engine.generate(engine.prepare(p), greedy_reuse(kOutput)));
    }
    return results;
}

// Section 6 (see the header).
void prefix_reuse(const char* artifact, Experts experts, int tp, const char* label) {
    const ninfer::EngineOptions o = reuse_options(artifact, experts, tp);
    const std::string l(label);
    std::printf("%s:\n", label);
    {
        const ninfer::PromptInput a =
            chat_prompt("How many crates of pears arrived on the third day?");
        const ninfer::PromptInput b = chat_prompt("Which farm sent the heaviest crate of plums?");
        const auto both             = run_engine(o, {a, b});
        const auto first            = run_engine(o, {a});
        const auto fresh            = run_engine(o, {b});
        const ninfer::GenerationResult& resumed = both[1];
        const ninfer::GenerationResult& cold    = fresh[0];
        print("shared prefix: A", both[0]);
        print("shared prefix: B resumed", resumed);
        print("shared prefix: B fresh", cold);
        std::printf("  B resumed: reused %u of %u prompt tokens (path %d); fresh reused %u\n",
                    resumed.reused_prompt_tokens, resumed.prompt.prompt_tokens,
                    static_cast<int>(resumed.prefix_reuse_path), cold.reused_prompt_tokens);
        check(both[0].generated_token_ids == first[0].generated_token_ids,
              l + ": request A is deterministic");
        check(resumed.prefix_reuse_path == ninfer::PrefixReusePath::SharedStablePrefix &&
                  resumed.reused_prompt_tokens > 0,
              l + ": B resumes A's shared prefix");
        check(cold.reused_prompt_tokens == 0, l + ": the fresh Engine has nothing to resume");
        check(resumed.speculative.drafted_tokens > 0, l + ": B decodes in MTP rounds");
        check(resumed.generated_token_ids == cold.generated_token_ids,
              l + ": resumed B tokens equal the fresh Engine's");
    }
    {
        const std::vector<ninfer::TokenId> first_prompt = prompt(0, 33);
        const auto continuation                         = [&](bool drafts, bool extend) {
            std::optional<ScopedEnv> none;
            if (!drafts) { none.emplace("NINFER_QWEN4_MTP_NO_DRAFTS", "1"); }
            ninfer::Engine engine(o);
            const auto first =
                engine.generate(engine.prepare_tokens(first_prompt), greedy_reuse(24));
            std::vector<ninfer::TokenId> second = first_prompt;
            second.insert(second.end(), first.generated_token_ids.begin(),
                                                  first.generated_token_ids.end());
            if (extend) {
                const std::vector<ninfer::TokenId> more = prompt(40, 17);
                second.insert(second.end(), more.begin(), more.end());
            }
            return engine.generate(engine.prepare_tokens(second), greedy_reuse(kOutput));
        };
        const ninfer::GenerationResult mtp      = continuation(true, true);
        const ninfer::GenerationResult no_draft = continuation(false, true);
        // The first turn's whole ledger as the next prompt (the endpoint resumes all but its
        // last token).
        const ninfer::GenerationResult exact          = continuation(true, false);
        const ninfer::GenerationResult exact_no_draft = continuation(false, false);
        print("exact continuation: MTP", exact);
        print("exact continuation: no drafts", exact_no_draft);
        std::printf("  exact continuation: reused %u of %u prompt tokens (path %d)\n",
                    exact.reused_prompt_tokens, exact.prompt.prompt_tokens,
                    static_cast<int>(exact.prefix_reuse_path));
        check(exact.prefix_reuse_path == ninfer::PrefixReusePath::PrivateEndpoint,
              l + ": the exact continuation resumes the first turn's endpoint");
        check(exact.generated_token_ids == exact_no_draft.generated_token_ids,
              l + ": exact continuation MTP tokens equal the no-draft tokens");
        print("continuation: MTP", mtp);
        print("continuation: no drafts", no_draft);
        std::printf("  continuation: reused %u of %u prompt tokens (path %d)\n",
                    mtp.reused_prompt_tokens, mtp.prompt.prompt_tokens,
                    static_cast<int>(mtp.prefix_reuse_path));
        check(mtp.prefix_reuse_path == ninfer::PrefixReusePath::PrivateEndpoint &&
                  mtp.reused_prompt_tokens > first_prompt.size(),
              l + ": the second turn resumes the first turn's endpoint");
        check(mtp.speculative.drafted_tokens > 0, l + ": the second turn decodes in MTP rounds");
        check(mtp.generated_token_ids == no_draft.generated_token_ids,
              l + ": resumed MTP tokens equal the resumed no-draft tokens");
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_QWEN4_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::cout << "skip: no CUDA device\n";
        return 77;
    }
    try {
        // 6. prefix reuse through the MTP bridge (T66b), first: NINFER_TEST_QWEN4_REUSE_ONLY=1
        //    runs it alone. Residency fixed at initialization: the offload backend's GPU (W4A4)
        //    and CPU (W4A8) experts are not bit-identical.
        const bool reuse_only = std::getenv("NINFER_TEST_QWEN4_REUSE_ONLY") != nullptr;
        {
            const ScopedEnv validation("NINFER_QWEN4_VALIDATION", "1");
            const ScopedEnv freeze("NINFER_QWEN4_EXPERT_FREEZE", "all");
            prefix_reuse(artifact, Experts::Mapped, 1, "prefix reuse, tp 1, mapped experts");
            prefix_reuse(artifact, Experts::Offload, 1, "prefix reuse, tp 1, offload experts");
            if (devices >= 2) {
                prefix_reuse(artifact, Experts::Offload, 2, "prefix reuse, tp 2, offload experts");
            } else {
                std::cout << "skip: tp 2 prefix reuse needs two devices\n";
            }
        }
        if (reuse_only) {
            if (failures != 0) {
                std::cerr << failures << " failure(s)\n";
                return 1;
            }
            std::cout << "OK qwen4 mtp engine (prefix reuse only)\n";
            return 0;
        }
        repeatable_mtp_head(artifact);
        exactness(artifact, Experts::Mapped, "mapped experts");
        {
            const ScopedEnv validation("NINFER_QWEN4_VALIDATION", "1");
            const ScopedEnv freeze("NINFER_QWEN4_EXPERT_FREEZE", "1");
            exactness(artifact, Experts::Offload, "offload experts (frozen residency)");
        }
        cancelled_row(artifact);
        // 4. T43 draft shortlist: proposals over the ID prefix [0, 3000) plus three high IDs (the
        //    gathered extra rows; 3003 IDs, completed to 3008 rows): MTP == no-draft == oracle
        //    drafts (the verification head is the full one, whatever the proposals).
        g_shortlist = scratch_path("ninfer_qwen4_mtp_shortlist_") + ".txt";
        {
            std::ofstream out(g_shortlist);
            out << "# test shortlist\n";
            for (int id = 0; id < 3000; ++id) { out << id << "\n"; }
            out << "100000\n200000\n248046\n";
        }
        exactness(artifact, Experts::Mapped, "draft shortlist, mapped experts");
        {
            const ScopedEnv validation("NINFER_QWEN4_VALIDATION", "1");
            const ScopedEnv freeze("NINFER_QWEN4_EXPERT_FREEZE", "1");
            exactness(artifact, Experts::Offload, "draft shortlist, offload experts");
        }
        // 5. The same shortlist at tp 2 (two devices): 3008 rows split 1504 / 1504 over the ranks,
        //    rank 1's half gathered from both vocabulary halves (IDs 100000 from rank 0's,
        //    200000 and 248046 from its own).
        if (devices >= 2) {
            g_tp = 2;
            const ScopedEnv validation("NINFER_QWEN4_VALIDATION", "1");
            const ScopedEnv freeze("NINFER_QWEN4_EXPERT_FREEZE", "1");
            exactness(artifact, Experts::Offload, "draft shortlist, tp 2 (split), offload experts");
            g_tp = 1;
        } else {
            std::cout << "skip: tp 2 shortlist case needs two devices\n";
        }
        std::remove(g_shortlist.c_str());
        g_shortlist.clear();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "OK qwen4 mtp engine\n";
    return 0;
}
