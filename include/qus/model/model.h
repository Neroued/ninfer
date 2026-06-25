#pragma once
// qus::model — the model card interface for Qwen3.6-27B: the hand-written static forward
// schedule (embed -> 64-layer 3:1 dispatch -> final norm -> lm_head) plus prefill/decode
// drivers. This is the L2 "compute graph"; implemented in src/model/qwen3_6_27b.cpp.
// See docs/qwen3.6-27b-architecture.md §9 (computation order) and docs/design.md §5/§6.
// NOTE: structure stub — no implementation yet.

namespace qus::model {

// TODO(impl): forward(prefill/decode) entry points that issue L1 kernel calls in order.

}  // namespace qus::model
