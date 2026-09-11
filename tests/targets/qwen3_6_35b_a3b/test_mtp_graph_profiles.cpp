// Contract of Variant::mtp_graph_profiles: profiles that resolve to different attention routes
// must not share a topology class, because instantiate_graph_family instantiates one
// cudaGraphExec_t per class and installs every other profile of that class through
// cudaGraphExecUpdate, which cannot cross a change of node count.

#include "ninfer/targets/qwen3_6/round_state.h"
#include "ninfer/types.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

namespace {

using ninfer::KvCacheStorage;
using ninfer::ops::CausalAttentionExecutionEnvelope;
using ninfer::ops::detail::causal_attention_resolve_route;
using ninfer::ops::detail::causal_attention_route_name;
using ninfer::ops::detail::CausalAttentionRoute;
using Variant = ninfer::targets::qwen3_6_35b_a3b::detail::Variant;

constexpr std::int32_t kQueryHeads =
    ninfer::targets::qwen3_6_35b_a3b::detail::TextConfig::query_heads;

// Every KV storage the engine can be configured with. The route table branches on storage, so a
// planner that is right for one of them is not thereby right for the rest.
constexpr KvCacheStorage kStorages[] = {
    KvCacheStorage::BFloat16,     KvCacheStorage::Int8Group64,      KvCacheStorage::Fp8E4M3Row256,
    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value,
};

CausalAttentionRoute route_at(std::uint32_t capacity, std::uint32_t draft_window,
                              std::uint32_t frontier, KvCacheStorage storage) {
    const std::uint64_t target =
        std::min<std::uint64_t>(capacity, static_cast<std::uint64_t>(frontier) + draft_window + 1);
    return causal_attention_resolve_route(
        kQueryHeads, static_cast<std::int32_t>(draft_window) + 1, 1, storage,
        CausalAttentionExecutionEnvelope{1U, static_cast<std::uint32_t>(target)});
}

int failures = 0;

void check(std::uint32_t capacity, std::uint32_t draft_window, KvCacheStorage storage) {
    const int kv        = static_cast<int>(storage);
    const auto profiles = Variant::mtp_graph_profiles(capacity, draft_window);
    if (profiles.empty()) {
        std::cerr << "capacity=" << capacity << " k=" << draft_window << ": no profiles\n";
        ++failures;
        return;
    }

    std::uint32_t expected_min = 0;
    for (const auto& profile : profiles) {
        if (profile.min != expected_min || profile.max < profile.min) {
            std::cerr << "capacity=" << capacity << " k=" << draft_window
                      << ": frontier coverage has a hole at [" << profile.min << "," << profile.max
                      << "]\n";
            ++failures;
        }
        expected_min = profile.max + 1;
    }
    if (profiles.back().max != capacity - 1) {
        std::cerr << "capacity=" << capacity << " k=" << draft_window << ": coverage stops at "
                  << profiles.back().max << "\n";
        ++failures;
    }

    // 1. one route per profile: the executable installed for a profile is replayed across the
    //    whole frontier range of that profile.
    for (const auto& profile : profiles) {
        const CausalAttentionRoute lo = route_at(capacity, draft_window, profile.min, storage);
        const CausalAttentionRoute hi = route_at(capacity, draft_window, profile.max, storage);
        if (lo != hi) {
            std::cerr << "capacity=" << capacity << " k=" << draft_window << " kv=" << kv
                      << ": profile [" << profile.min << "," << profile.max << "] class "
                      << profile.topology_class << " spans a route flip "
                      << causal_attention_route_name(lo) << " -> "
                      << causal_attention_route_name(hi) << "\n";
            ++failures;
        }
    }

    // 2. one class per route: profiles of different routes must not share an executable.
    std::map<std::uint32_t, CausalAttentionRoute> route_of_class;
    for (const auto& profile : profiles) {
        const CausalAttentionRoute route = route_at(capacity, draft_window, profile.max, storage);
        const auto [it, inserted]        = route_of_class.emplace(profile.topology_class, route);
        if (!inserted && it->second != route) {
            std::cerr << "capacity=" << capacity << " k=" << draft_window << " kv=" << kv
                      << ": class " << profile.topology_class << " carries both "
                      << causal_attention_route_name(it->second) << " and "
                      << causal_attention_route_name(route) << " (profile [" << profile.min << ","
                      << profile.max << "])\n";
            ++failures;
        }
    }
}

} // namespace

int main() {
    // One past the widest window any registered verify path can request today
    // (kMtpDecodeMaximumDrafts is 5, kDFlashDecodeMaximumDrafts is 15), so the guard that adds
    // the route-flip boundary is itself covered rather than trusted.
    constexpr std::uint32_t kSweptDraftWindows =
        ninfer::targets::qwen3_6::kDFlashDecodeMaximumDrafts + 1U;
    for (const std::uint32_t capacity : {2048U, 16384U, 65536U, 262144U}) {
        for (std::uint32_t draft_window = 1; draft_window <= kSweptDraftWindows; ++draft_window) {
            for (const KvCacheStorage storage : kStorages) {
                check(capacity, draft_window, storage);
            }
        }
    }
    if (failures != 0) {
        std::cerr << failures << " MTP graph-profile contract violation(s)\n";
        return 1;
    }
    std::cout << "MTP graph profiles keep one attention route per topology class\n";
    return 0;
}
