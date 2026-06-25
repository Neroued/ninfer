#pragma once

// qus::kernels — full GQA attention (24 Q / 4 KV heads, head_dim 256, scale 1/sqrt(256)):
// prefill (causal flash-style) and decode (single-query against KV cache) paths. Gating
// (* sigmoid(gate)) and o_proj are applied around it. See docs/qwen3.6-27b-architecture.md
// §6.4/§10.3. NOTE: structure stub — no implementation yet.

namespace qus::kernels {

// TODO(impl): attention_prefill(q,k,v,...) ; attention_decode(q, KVCache, layer, ...).

} // namespace qus::kernels
