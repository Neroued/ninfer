#include "targets/qwen3_6/impl/runtime/host_kv_cache.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

using ninfer::targets::qwen3_6::detail::HostKvCache;
using ninfer::targets::qwen3_6::detail::HostKvEntry;
using ninfer::targets::qwen3_6::detail::HostKvProvider;
using ninfer::targets::qwen3_6::PreparedPromptData;
using ninfer::targets::qwen3_6::RewriteCheckpointKind;

static_assert(std::is_base_of_v<HostKvProvider, HostKvCache>);

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%-64s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) { ++failures; }
}

// A prompt of `n` sequential tokens. positions is a 3-axis rope layout stored
// axis-major, so it holds 3*n entries; assign() rejects any other shape.
PreparedPromptData make_prompt(int n, int base = 0) {
    PreparedPromptData p;
    for (int i = 0; i < n; ++i) {
        p.token_ids.push_back(static_cast<ninfer::TokenId>(base + i));
        p.token_types.push_back(0);
    }
    p.positions.resize(static_cast<std::size_t>(3 * n));
    for (int axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < n; ++i) {
            p.positions[static_cast<std::size_t>(axis * n + i)] = i;
        }
    }
    p.identity.reusable = true;
    return p;
}

// An entry whose ledger is the first `frontier` tokens of `prompt`.
std::unique_ptr<HostKvEntry> make_entry(const PreparedPromptData& prompt, std::uint32_t frontier,
                                        ninfer::HostKvSlab* slab) {
    auto e = std::make_unique<HostKvEntry>();
    e->slab = slab;
    e->ledger.assign(prompt.token_ids.begin(),
                     prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(frontier));
    e->prefix_identity.assign(prompt);
    e->prefix_identity.truncate(frontier);
    e->execution_frontier = frontier;
    return e;
}

}  // namespace

int main() {
    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }

    // The budget is irrelevant to matching; it just needs to hold the entries.
    HostKvCache cache(2 * 4096);
    check(cache.used_bytes() == 0 && cache.budget_bytes() == 2 * 4096,
          "cache starts with the whole budget free");
    check(cache.size() == 0, "cache starts empty");
    check(!cache.find(make_prompt(10)).has_value(), "empty cache matches nothing");

    const PreparedPromptData session_a = make_prompt(100, 0);
    const PreparedPromptData session_b = make_prompt(100, 5000);

    auto* slab_a = cache.acquire_slab(0, 4096);
    check(slab_a != nullptr, "acquire_slab hands out a free region");
    cache.insert(make_entry(session_a, 80, slab_a));
    check(cache.size() == 1, "insert stores the entry");

    // Continuing session A: the prompt extends the parked ledger.
    const PreparedPromptData a_continued = make_prompt(120, 0);
    auto hit = cache.find(a_continued);
    check(hit.has_value(), "a continuation of the parked session matches");
    check(hit && hit->reuse_tokens == 80, "match reports the parked frontier as reusable");

    // An unrelated session must not match, or we would restore the wrong KV.
    check(!cache.find(session_b).has_value(), "an unrelated session does not match");

    // A prompt that diverges before the frontier cannot append. With no
    // checkpoint recorded there is nothing to fall back to.
    PreparedPromptData a_diverged = make_prompt(120, 0);
    a_diverged.token_ids[40]      = 9999;
    check(!cache.find(a_diverged).has_value(),
          "divergence before the frontier does not match without a checkpoint");

    // Same divergence, but the entry carries a checkpoint below the divergence
    // point. 21% of real requests take this path, so it must resolve.
    auto* slab_b = cache.acquire_slab(0, 4096);
    auto ckpt    = make_entry(session_a, 80, slab_b);
    ckpt->checkpoint_valid    = true;
    ckpt->checkpoint_kind     = RewriteCheckpointKind::TurnClosure;
    ckpt->checkpoint_frontier = 30;
    cache.insert(std::move(ckpt));

    hit = cache.find(a_diverged);
    check(hit.has_value(), "divergence falls back to the rewrite checkpoint");
    check(hit && hit->reuse_tokens == 30, "checkpoint match reports the checkpoint frontier");

    // With both entries present, an exact continuation must prefer the one that
    // reuses more, matching find_admission_lane's rule.
    hit = cache.find(a_continued);
    check(hit && hit->reuse_tokens == 80, "the entry reusing the most tokens wins");

    check(cache.used_bytes() == cache.budget_bytes(), "the whole budget is now in use");

    // No free budget: acquiring must evict the least recently used entry.
    cache.touch(1);  // entry 1 is now newer than entry 0
    auto* slab_c = cache.acquire_slab(0, 4096);
    check(slab_c != nullptr, "acquire_slab evicts to make room");
    check(cache.size() == 1, "eviction removed exactly one entry");
    check(cache.entry(0).checkpoint_frontier == 30, "the least recently used entry was evicted");
    cache.release_slab(slab_c);  // return the held slab so the cache can park again

    // Eviction protection: the entry a restore is about to consume must never
    // be sacrificed to the park that precedes it. Park a second entry so the
    // cache holds two (one protected, one to evict) and no budget is free; the
    // protected acquire must evict the other entry, never the protected one.
    auto* slab_d = cache.acquire_slab(0, 4096);
    cache.insert(make_entry(session_b, 40, slab_d));
    check(cache.size() == 2, "a second entry fills the cache");
    const std::uint64_t protected_id = cache.entry(0).id;
    auto* slab_p = cache.acquire_slab(protected_id, 4096);
    check(slab_p != nullptr, "acquire_slab(protected) evicts the unprotected entry");
    check(cache.size() == 1, "the protected entry survived the eviction");
    check(cache.entry(0).id == protected_id, "the surviving entry is the protected one");
    // slab_p is still held; with only the protected entry left and no budget free,
    // a protected acquire has no victim and returns nullptr rather than
    // destroying the entry the restore needs.
    auto* slab_q = cache.acquire_slab(protected_id, 4096);
    check(slab_q == nullptr, "acquire_slab(protected) with no victim returns nullptr");
    check(cache.size() == 1, "the protected entry is still intact");
    cache.release_slab(slab_p);

    // Oversized park: a request larger than the entire budget can never fit, so
    // the satisfiability check must bail out before evicting anything. Without
    // it the eviction loop would evict every entry one at a time and still
    // fail, wiping the whole cache for a park that cannot happen.
    auto* slab_e = cache.acquire_slab(0, 4096);
    check(slab_e != nullptr, "the last free region is handed out");
    cache.insert(make_entry(session_b, 50, slab_e));
    check(cache.size() == 2 && cache.used_bytes() == cache.budget_bytes(),
          "the budget is now full");
    auto* slab_oversized = cache.acquire_slab(0, cache.budget_bytes() + 1);
    check(slab_oversized == nullptr, "an oversized park (budget+1) fails");
    check(cache.size() == 2, "the oversized park evicted nothing (cache intact)");
    check(cache.used_bytes() == cache.budget_bytes(), "the oversized park left the budget full");

    // Middle-protected park: a protected entry in the middle of the budget
    // splits the address space, so a SUM of free bytes overstates what one
    // contiguous range can hold. The satisfiability check must answer the
    // contiguity question (the largest span between protected boundaries), not
    // the total-bytes question, or it would evict the unprotected entries and
    // still fail. Layout after filling: [e0][e1][e2][e3], each 4096; protecting
    // e1 leaves a 4096-byte front and an 8192-byte back, so a 12288-byte park
    // cannot fit even after evicting e0, e2 and e3.
    {
        HostKvCache cache4(4 * 4096);
        auto* s0 = cache4.acquire_slab(0, 4096);
        cache4.insert(make_entry(session_a, 10, s0));
        auto* s1 = cache4.acquire_slab(0, 4096);
        cache4.insert(make_entry(session_b, 20, s1));
        auto* s2 = cache4.acquire_slab(0, 4096);
        cache4.insert(make_entry(session_a, 30, s2));
        auto* s3 = cache4.acquire_slab(0, 4096);
        cache4.insert(make_entry(session_b, 40, s3));
        check(cache4.size() == 4 && cache4.used_bytes() == cache4.budget_bytes(),
              "the 4-entry budget is full");
        const std::uint64_t mid_id = cache4.entry(1).id;  // the middle entry
        auto* slab_mid = cache4.acquire_slab(mid_id, 3 * 4096);
        check(slab_mid == nullptr, "a middle-protected park that cannot fit fails");
        check(cache4.size() == 4, "the middle-protected park evicted nothing");
        check(cache4.used_bytes() == cache4.budget_bytes(),
              "the middle-protected park left the budget full");
    }

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
