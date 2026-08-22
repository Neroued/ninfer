#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
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

    std::cout << "ok\n";
    return 0;
}
