#pragma once
// qus::runtime — Engine: owns the DeviceContext, arenas, WeightStore, KVCache, GdnState; drives
// load -> prefill -> decode loop. token-ids in -> token-ids out (greedy, v1).
// See docs/design.md §6 (runtime) and docs/l0-infrastructure-design.md §6 (lifecycle).
// NOTE: structure stub — no implementation yet.

namespace qus {

// TODO(impl): class Engine { void load(path); int prefill(span<int> ids); int decode_step(int prev); };

}  // namespace qus
