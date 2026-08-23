#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// The headline configuration: one lane, MTP. Session A's parked entry holds
// the budget while session B is resident, so A's return must restore A rather
// than destroy its own entry to park B. The budget is the total pinned bytes;
// each parked entry takes only its real size (its GDN state alone is ~308 MB,
// so a small session still costs ~310 MB), and the budget decides how many fit.
ninfer::EngineOptions host_kv_engine_options(const char* artifact, std::uint64_t budget_bytes) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk             = 1024;
    options.max_concurrency           = 1;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = true;
    options.host_kv_cache_bytes       = budget_bytes;
    return options;
}

ninfer::RequestOptions greedy(int tokens, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

// The evicting-restore configuration: three lanes and a 12-page pool (3 x 4).
// The entitlement is the full growth ceiling (prompt + output budget), reserved
// up front per sequence, so a 192-token budget gives a 4-page ceiling and a
// 384-token budget gives a 7-page ceiling.
//
// The sequence: A runs on lane 0 with a 384-token budget (7 pages) and completes
// (its state is retained). D replaces A on lane 0, parking A into the host cache
// (entry 0, 7 pages). B, C and E are then submitted concurrently with 192-token
// budgets (4 pages each), so they take lanes 0, 1, 2 (B replaces D, parking it);
// when they complete, lanes 0, 1, 2 are retained (B, C, E states) and the pool
// is saturated (3 x 4 pages). A's continuation (a 192-token budget, a 10-page
// entitlement) matches A's host entry (7 pages), but the pool is saturated by B,
// C and E, so the probe's can_restore_lane check fails on every lane (parking any
// one resident frees 4 pages, but the other two hold 8, leaving 4 free < the
// entry's 7). The probe defers (keeps the pool saturated), and the admission's
// evict_retained loop parks the retained sessions, freeing the pages the
// evicting-restore step needs to bring A's entry back instead of re-prefilling.
ninfer::EngineOptions evicting_restore_engine_options(const char* artifact,
                                                      std::uint64_t budget_bytes) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 768;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(768);
    options.prefill_chunk             = 1024;
    options.max_concurrency           = 3;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = true;
    options.host_kv_cache_bytes       = budget_bytes;
    return options;
}

int exercise_host_kv_park_restore(ninfer::Engine& engine) {
    // Session A: a prompt that generates four tokens.
    const std::vector<ninfer::TokenId> prompt_a{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult a_first =
        engine.generate(engine.prepare_tokens(prompt_a), greedy(4, false));
    if (a_first.generated_token_ids.size() != 4) {
        std::cerr << "session A did not generate four tokens\n";
        return 1;
    }

    // Session B: an unrelated prompt. At --max-concurrency 1 it takes A's lane,
    // and A's continuation state is parked into the single slab.
    const std::vector<ninfer::TokenId> prompt_b{248045, 846, 198, 999, 248046, 198};
    const ninfer::GenerationResult b_first =
        engine.generate(engine.prepare_tokens(prompt_b), greedy(4, false));
    if (b_first.generated_token_ids.size() != 4) {
        std::cerr << "session B did not generate four tokens\n";
        return 1;
    }

    // A returns while B is resident: A's parked entry holds the only slab, so
    // the restore must not sacrifice it to park B.
    std::vector<ninfer::TokenId> a_continuation = prompt_a;
    a_continuation.insert(a_continuation.end(), a_first.generated_token_ids.begin(),
                          a_first.generated_token_ids.end());
    a_continuation.push_back(198);
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt_a.size() + a_first.generated_token_ids.size() - 1);
    const ninfer::GenerationResult a_reused =
        engine.generate(engine.prepare_tokens(a_continuation), greedy(2, true));
    if (a_reused.reused_prompt_tokens != expected_reuse) {
        std::cerr << "parked session A was not restored: reused=" << a_reused.reused_prompt_tokens
                  << ", expected " << expected_reuse << '\n';
        return 1;
    }
    if (a_reused.generated_token_ids.size() != 2) {
        std::cerr << "restored session A did not generate two tokens\n";
        return 1;
    }

    // Greedy determinism through the restored state: a cold re-prefill of the
    // same continuation must produce the same next token. A corrupted restore
    // (for example a zeroed GDN recurrent state) diverges here.
    const ninfer::GenerationResult a_cold =
        engine.generate(engine.prepare_tokens(a_continuation), greedy(2, false));
    if (a_cold.generated_token_ids.size() != 2 ||
        a_cold.generated_token_ids[0] != a_reused.generated_token_ids[0]) {
        std::cerr << "restored session A diverged from the cold baseline: restored="
                  << a_reused.generated_token_ids[0] << ", cold=" << a_cold.generated_token_ids[0]
                  << '\n';
        return 1;
    }
    return 0;
}

// The evicting-restore path. A runs on lane 0 with a 384-token budget (a 7-page
// entitlement) and completes (its state is retained). D replaces A on lane 0,
// parking A into the host cache (entry 0, a 7-page entitlement). B, C and E are
// submitted concurrently with 192-token budgets (4 pages each), so they take
// lanes 0, 1, 2 (B replaces D, parking it); when they complete, lanes 0, 1, 2
// are retained (B, C, E states) and the 12-page pool is saturated (3 x 4 pages).
//
// A's continuation (a 192-token budget, a 10-page entitlement) matches A's host
// entry (7 pages). The pool is saturated by B, C and E, so even after parking
// any one lane's resident (4 pages freed), the other two lanes (8 pages) leave
// only 4 free, which is less than the entry's 7 pages: the probe's
// can_restore_lane check fails on every lane, so it defers (keeps the pool
// saturated) instead of parking cumulatively. The admission's evict_retained
// loop parks all three retained sessions, freeing the pages, and the
// evicting-restore step restores A's entry: A reuses its cached prefix instead
// of re-prefilling. Without the evicting restore, A would re-prefill cold
// (reused == 0) even though its entry is in the host cache.
int exercise_evicting_restore(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt_a{248045, 846, 198, 5834, 248046, 198};
    const std::vector<ninfer::TokenId> prompt_b{248045, 846, 198, 999, 248046, 198};
    const std::vector<ninfer::TokenId> prompt_c{248045, 846, 198, 1234, 248046, 198};
    const std::vector<ninfer::TokenId> prompt_d{248045, 846, 198, 5678, 248046, 198};
    const std::vector<ninfer::TokenId> prompt_e{248045, 846, 198, 7777, 248046, 198};

    // A runs on lane 0 with a 384-token budget (a 7-page entitlement) and
    // completes; its state is retained.
    const ninfer::GenerationResult a_first =
        engine.generate(engine.prepare_tokens(prompt_a), greedy(384, false));
    if (a_first.generated_token_ids.empty()) {
        std::cerr << "evicting-restore: A generated no tokens\n";
        return 1;
    }

    // D replaces A on lane 0, parking A into the host cache (entry 0, 7 pages).
    const ninfer::GenerationResult d_first =
        engine.generate(engine.prepare_tokens(prompt_d), greedy(192, false));
    if (d_first.generated_token_ids.empty()) {
        std::cerr << "evicting-restore: D generated no tokens\n";
        return 1;
    }

    // B, C and E are submitted concurrently with 192-token budgets (4 pages
    // each), so they take lanes 0, 1, 2 (B replaces D, parking it). When they
    // complete, lanes 0, 1, 2 are retained (B, C, E states) and the 12-page
    // pool is saturated (3 x 4 pages).
    auto b_handle = engine.submit(engine.prepare_tokens(prompt_b), greedy(192, false));
    auto c_handle = engine.submit(engine.prepare_tokens(prompt_c), greedy(192, false));
    auto e_handle = engine.submit(engine.prepare_tokens(prompt_e), greedy(192, false));
    const ninfer::GenerationResult b_first = b_handle.wait();
    const ninfer::GenerationResult c_first = c_handle.wait();
    const ninfer::GenerationResult e_first = e_handle.wait();
    (void)b_first;
    (void)c_first;
    (void)e_first;

    // A's continuation matches A's host entry (7 pages). The pool is saturated
    // by B, C and E, so the probe's can_restore_lane check fails on every lane
    // (parking any one resident frees 4 pages, but the other two hold 8, leaving
    // 4 free < the entry's 7). The probe defers (keeps the pool saturated), and
    // the admission's evicting-restore step restores A's entry after parking all
    // three retained sessions. A must reuse its cached prefix.
    std::vector<ninfer::TokenId> a_continuation = prompt_a;
    a_continuation.insert(a_continuation.end(), a_first.generated_token_ids.begin(),
                          a_first.generated_token_ids.end());
    a_continuation.push_back(198);
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt_a.size() + a_first.generated_token_ids.size() - 1);
    const ninfer::GenerationResult a_reused =
        engine.generate(engine.prepare_tokens(a_continuation), greedy(192, true));
    if (a_reused.reused_prompt_tokens != expected_reuse) {
        std::cerr << "evicting-restore: A was not restored after B, C and E saturated the pool: "
                  << "reused=" << a_reused.reused_prompt_tokens << ", expected " << expected_reuse
                  << '\n';
        return 1;
    }
    if (a_reused.generated_token_ids.empty()) {
        std::cerr << "evicting-restore: restored session A generated no tokens\n";
        return 1;
    }

    // Greedy determinism through the evicting restore: a cold re-prefill of the
    // same continuation must produce the same next token. A corrupted restore
    // (for example a zeroed GDN recurrent state) diverges here.
    const ninfer::GenerationResult a_cold =
        engine.generate(engine.prepare_tokens(a_continuation), greedy(192, false));
    if (a_cold.generated_token_ids.empty() ||
        a_cold.generated_token_ids[0] != a_reused.generated_token_ids[0]) {
        std::cerr << "evicting-restore: restored session A diverged from the cold baseline: "
                  << "restored=" << a_reused.generated_token_ids[0]
                  << ", cold=" << a_cold.generated_token_ids[0] << '\n';
        return 1;
    }
    return 0;
}

// The entry > plan variant of the evicting restore (the iteration-2 review's
// finding-1 trigger). A is a chat prompt (a single user message) that captures
// a TurnClosure rewrite checkpoint at the generation-prompt frontier. A runs
// with a 512-token budget (a 9-page entitlement) and completes; D replaces A on
// lane 0, parking A into the host cache (entry 0, a 9-page entitlement, a
// checkpoint at the prompt frontier). B, C and E are then submitted concurrently
// with 128-token budgets (3 pages each) and saturate the 12-page pool (3 x 3
// pages, 9 entitled).
//
// The incoming request is the SAME chat prompt as A (a 64-token budget, a
// 2-page entitlement). It matches A's host entry at the checkpoint (a short
// reuse) better than any resident, but the entry's 9-page stored entitlement is
// LARGER than the incoming plan's 2-page entitlement - the opposite direction
// from exercise_evicting_restore above. The pool is saturated, so the probe's
// can_restore_lane check fails on every lane and it defers. The admission must
// park the other retained lanes until BOTH the entry (9 pages) and the plan
// (2 pages) fit, then empty the selected resident and restore A - WITHOUT
// discarding the resident before proving the (larger) entry restores. A buggy
// transaction that stops parking when only the (smaller) plan fits turns the
// checkpoint reuse into a cold prefill (reused == 0).
int exercise_evicting_restore_large_entry(ninfer::Engine& engine) {
    auto chat_prompt = [](const std::string& text) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        return input;
    };
    const std::vector<ninfer::TokenId> prompt_b{248045, 846, 198, 999, 248046, 198};
    const std::vector<ninfer::TokenId> prompt_c{248045, 846, 198, 1234, 248046, 198};
    const std::vector<ninfer::TokenId> prompt_d{248045, 846, 198, 5678, 248046, 198};
    const std::vector<ninfer::TokenId> prompt_e{248045, 846, 198, 7777, 248046, 198};

    // A runs on lane 0 with a 512-token budget (a 9-page entitlement) and
    // completes; its state is retained. A's chat prompt captures a TurnClosure
    // checkpoint at the generation-prompt frontier.
    const ninfer::GenerationResult a_first =
        engine.generate(engine.prepare(chat_prompt("alpha")), greedy(512, false));
    if (a_first.generated_token_ids.empty()) {
        std::cerr << "evicting-restore-large-entry: A generated no tokens\n";
        return 1;
    }

    // D replaces A on lane 0, parking A into the host cache (entry 0, 9 pages,
    // a checkpoint at the prompt frontier).
    const ninfer::GenerationResult d_first =
        engine.generate(engine.prepare_tokens(prompt_d), greedy(128, false));
    if (d_first.generated_token_ids.empty()) {
        std::cerr << "evicting-restore-large-entry: D generated no tokens\n";
        return 1;
    }

    // B, C and E are submitted concurrently with 128-token budgets (3 pages
    // each), so they take lanes 0, 1, 2 (B replaces D, parking it). When they
    // complete, lanes 0, 1, 2 are retained (B, C, E states) and the 12-page
    // pool is saturated (3 x 3 pages).
    auto b_handle = engine.submit(engine.prepare_tokens(prompt_b), greedy(128, false));
    auto c_handle = engine.submit(engine.prepare_tokens(prompt_c), greedy(128, false));
    auto e_handle = engine.submit(engine.prepare_tokens(prompt_e), greedy(128, false));
    const ninfer::GenerationResult b_first = b_handle.wait();
    const ninfer::GenerationResult c_first = c_handle.wait();
    const ninfer::GenerationResult e_first = e_handle.wait();
    (void)b_first;
    (void)c_first;
    (void)e_first;

    // The incoming request is the SAME chat prompt as A (a 64-token budget, a
    // 2-page entitlement). It matches A's host entry at the checkpoint (a short
    // reuse), but the entry's 9-page entitlement exceeds the plan's 2 pages. The
    // pool is saturated, so the probe defers and the admission's evicting-restore
    // step must park the other retained lanes until both the entry and the plan
    // fit, then restore A. A must reuse its cached checkpoint, not re-prefill
    // cold (reused == 0). The residents (B, C, E) have raw-token prompts that do
    // not share A's chat prefix, so any reuse here comes from A's entry.
    const ninfer::GenerationResult a_reused =
        engine.generate(engine.prepare(chat_prompt("alpha")), greedy(64, true));
    if (a_reused.reused_prompt_tokens == 0) {
        std::cerr << "evicting-restore-large-entry: A was not restored (entry > plan): "
                     "reused=0, expected a checkpoint reuse\n";
        return 1;
    }
    if (a_reused.generated_token_ids.empty()) {
        std::cerr << "evicting-restore-large-entry: restored session A generated no tokens\n";
        return 1;
    }

    // Greedy determinism through the evicting restore: a cold re-prefill of the
    // same incoming prompt must produce the same next token. A corrupted restore
    // (for example a zeroed GDN recurrent state) diverges here.
    const ninfer::GenerationResult a_cold =
        engine.generate(engine.prepare(chat_prompt("alpha")), greedy(64, false));
    if (a_cold.generated_token_ids.empty() ||
        a_cold.generated_token_ids[0] != a_reused.generated_token_ids[0]) {
        std::cerr << "evicting-restore-large-entry: restored session A diverged from the cold "
                     "baseline: restored="
                  << a_reused.generated_token_ids[0] << ", cold=" << a_cold.generated_token_ids[0]
                  << '\n';
        return 1;
    }
    return 0;
}

// A request that disabled prefix reuse must not trigger a host restore or
// evict the resident to make room. Run on a FRESH small-budget engine so it is
// mutation-adequate: at a budget that holds only one entry, without the gate,
// A's restore would evict B's entry to make room, so B's later restore would
// fail (reused != expected). With the gate, A re-prefills cold (reused == 0)
// and B's entry survives.
int exercise_no_reuse_skips_host_cache(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt_a{248045, 846, 198, 5834, 248046, 198};
    // Park A (generate, so its continuation lands in the only slab).
    const ninfer::GenerationResult a_first =
        engine.generate(engine.prepare_tokens(prompt_a), greedy(4, false));
    if (a_first.generated_token_ids.size() != 4) {
        std::cerr << "no-reuse: A did not generate four tokens\n";
        return 1;
    }
    // B takes the lane, parking A into the only slab.
    const std::vector<ninfer::TokenId> prompt_b{248045, 846, 198, 999, 248046, 198};
    const ninfer::GenerationResult b_first =
        engine.generate(engine.prepare_tokens(prompt_b), greedy(4, false));
    if (b_first.generated_token_ids.size() != 4) {
        std::cerr << "no-reuse: B did not generate four tokens\n";
        return 1;
    }
    // A returns but disables reuse. The host cache must NOT restore A: it
    // re-prefills cold and B's slab must survive untouched.
    std::vector<ninfer::TokenId> a_continuation = prompt_a;
    a_continuation.insert(a_continuation.end(), a_first.generated_token_ids.begin(),
                          a_first.generated_token_ids.end());
    a_continuation.push_back(198);
    ninfer::RequestOptions no_reuse = greedy(2, false);  // allow_prefix_reuse = false
    const ninfer::GenerationResult a_no_reuse =
        engine.generate(engine.prepare_tokens(a_continuation), no_reuse);
    if (a_no_reuse.reused_prompt_tokens != 0) {
        std::cerr << "no-reuse: A restored from host despite allow_prefix_reuse=false: reused="
                  << a_no_reuse.reused_prompt_tokens << '\n';
        return 1;
    }
    if (a_no_reuse.generated_token_ids.size() != 2) {
        std::cerr << "no-reuse: A did not generate two tokens\n";
        return 1;
    }
    // B returns with reuse enabled and must still restore - the no-reuse A
    // request did not evict B's parked continuation to make room for nothing.
    std::vector<ninfer::TokenId> b_continuation = prompt_b;
    b_continuation.insert(b_continuation.end(), b_first.generated_token_ids.begin(),
                          b_first.generated_token_ids.end());
    b_continuation.push_back(198);
    const ninfer::GenerationResult b_reused =
        engine.generate(engine.prepare_tokens(b_continuation), greedy(2, true));
    const std::uint32_t b_expected =
        static_cast<std::uint32_t>(prompt_b.size() + b_first.generated_token_ids.size() - 1);
    if (b_reused.reused_prompt_tokens != b_expected) {
        std::cerr << "no-reuse: B was not restored after the no-reuse A request: reused="
                  << b_reused.reused_prompt_tokens << ", expected " << b_expected << '\n';
        return 1;
    }
    return 0;
}

// A request whose prepared identity is not reusable must not trigger host
// restore either, even if allow_prefix_reuse is true. prepare_tokens(..., false)
// yields a non-reusable identity; the planner gates on prompt.identity.reusable
// (request_plan_impl.h), and the executor host-cache block must match or it
// restores, evicts the resident, then the planner cold-resets anyway.
int exercise_non_reusable_identity_skips_host_cache(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt_a{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult a_first =
        engine.generate(engine.prepare_tokens(prompt_a), greedy(4, false));
    if (a_first.generated_token_ids.size() != 4) {
        std::cerr << "non-reusable: A did not generate four tokens\n";
        return 1;
    }
    const std::vector<ninfer::TokenId> prompt_b{248045, 846, 198, 999, 248046, 198};
    const ninfer::GenerationResult b_first =
        engine.generate(engine.prepare_tokens(prompt_b), greedy(4, false));
    if (b_first.generated_token_ids.size() != 4) {
        std::cerr << "non-reusable: B did not generate four tokens\n";
        return 1;
    }
    // A returns with reuse ENABLED but a non-reusable identity. The host cache
    // must be bypassed (identity.reusable is false): A re-prefills cold and
    // B's slab survives.
    std::vector<ninfer::TokenId> a_continuation = prompt_a;
    a_continuation.insert(a_continuation.end(), a_first.generated_token_ids.begin(),
                          a_first.generated_token_ids.end());
    a_continuation.push_back(198);
    const ninfer::GenerationResult a_non_reuse =
        engine.generate(engine.prepare_tokens(a_continuation, false), greedy(2, true));
    if (a_non_reuse.reused_prompt_tokens != 0) {
        std::cerr << "non-reusable: A restored from host despite identity.reusable=false: reused="
                  << a_non_reuse.reused_prompt_tokens << '\n';
        return 1;
    }
    if (a_non_reuse.generated_token_ids.size() != 2) {
        std::cerr << "non-reusable: A did not generate two tokens\n";
        return 1;
    }
    // B must still restore.
    std::vector<ninfer::TokenId> b_continuation = prompt_b;
    b_continuation.insert(b_continuation.end(), b_first.generated_token_ids.begin(),
                          b_first.generated_token_ids.end());
    b_continuation.push_back(198);
    const ninfer::GenerationResult b_reused =
        engine.generate(engine.prepare_tokens(b_continuation), greedy(2, true));
    const std::uint32_t b_expected =
        static_cast<std::uint32_t>(prompt_b.size() + b_first.generated_token_ids.size() - 1);
    if (b_reused.reused_prompt_tokens != b_expected) {
        std::cerr << "non-reusable: B was not restored after the non-reusable A request: reused="
                  << b_reused.reused_prompt_tokens << ", expected " << b_expected << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* nvfp4 = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    if (nvfp4 == nullptr || *nvfp4 == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_NVFP4_WEIGHTS is not set\n";
        return 77;
    }

    // Each parked entry costs ~310 MB (its GDN state alone is ~308 MB), so a
    // 512 MiB budget holds one entry but not two, and 2048 MiB holds two. The
    // MiB constant is uint64_t so the multiply does not wrap 32-bit int.
    constexpr std::uint64_t kMiB          = 1024 * 1024;
    constexpr std::uint64_t kOneEntryBudget = 512 * kMiB;     // fits one, not two
    constexpr std::uint64_t kTwoEntryBudget = 2048 * kMiB;    // fits two
    // Holds three of the evicting test's ~311 MB parked entries but not four,
    // so parking the fourth (in the admission's evict_retained loop) forces an
    // LRU eviction. The deferred target is the oldest entry, so without the
    // per-park protection it is the LRU victim and the restore fails.
    constexpr std::uint64_t kEvictingLruBudget = 1024 * kMiB;

    // Small budget: the failure branch - the protected park of the resident
    // fails (no free budget) and the resident is discarded so the restore can
    // run.
    {
        ninfer::Engine engine(host_kv_engine_options(nvfp4, kOneEntryBudget));
        if (const int result = exercise_host_kv_park_restore(engine); result != 0) {
            return result;
        }
    }

    // Large budget: the success branch - the resident park succeeds (free
    // budget is available), and the restore must run on the just-parked lane.
    // This is the general case; the small-budget test above only exercises the
    // emergency discard.
    {
        ninfer::Engine engine(host_kv_engine_options(nvfp4, kTwoEntryBudget));
        if (const int result = exercise_host_kv_park_restore(engine); result != 0) {
            return result;
        }
    }

    // Fresh small-budget engines for the bypass tests, so they are
    // mutation-adequate: at a one-entry budget, removing the gate evicts B's
    // entry to make room for A, so B's later restore fails (reused != expected).
    // Each gets its own engine so a prior test's lane state cannot mask the gate.
    {
        ninfer::Engine engine(host_kv_engine_options(nvfp4, kOneEntryBudget));
        if (const int result = exercise_no_reuse_skips_host_cache(engine); result != 0) {
            return result;
        }
    }
    {
        ninfer::Engine engine(host_kv_engine_options(nvfp4, kOneEntryBudget));
        if (const int result = exercise_non_reusable_identity_skips_host_cache(engine); result != 0) {
            return result;
        }
    }

    // Evicting restore: three lanes, a pool that holds three sessions'
    // entitlements (12 pages). A is parked when D is admitted; B, C and E
    // (concurrent) saturate the pool, so A's continuation can only be restored
    // by parking the retained sessions. A large budget keeps the parked entries
    // resident.
    {
        ninfer::Engine engine(evicting_restore_engine_options(nvfp4, kTwoEntryBudget));
        if (const int result = exercise_evicting_restore(engine); result != 0) {
            return result;
        }
    }

    // Evicting restore under LRU pressure: the same sequence, but a budget that
    // holds only three of the parked entries, so parking the fourth in the
    // evict_retained loop forces an LRU eviction. The deferred target (A, the
    // oldest entry) must survive because every park in the transaction protects
    // it; without that protection it is the LRU victim and the restore fails
    // (the request cold-prefills). This is the regression guard for the
    // "other-lane parking evicts the entry being restored" bug.
    {
        ninfer::Engine engine(evicting_restore_engine_options(nvfp4, kEvictingLruBudget));
        if (const int result = exercise_evicting_restore(engine); result != 0) {
            return result;
        }
    }

    // Evicting restore with a LARGE entry (entry > plan): A's parked entry has a
    // 9-page entitlement but the incoming plan is only 2 pages, and the incoming
    // matches A's entry at a rewrite checkpoint (a short reuse). The admission
    // must park the other lanes until both the entry and the plan fit, then
    // restore A without discarding the resident before proving the (larger)
    // entry restores. A large budget keeps the parked entries resident (no LRU
    // pressure).
    {
        ninfer::Engine engine(evicting_restore_engine_options(nvfp4, kTwoEntryBudget));
        if (const int result = exercise_evicting_restore_large_entry(engine); result != 0) {
            return result;
        }
    }

    std::cout << "ok\n";
    return 0;
}
