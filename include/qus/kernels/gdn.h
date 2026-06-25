#pragma once
// qus::kernels — Gated-DeltaNet linear attention: causal_conv1d (depthwise k=4 +SiLU on
// [q;k;v]), gdn_gating (g=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b)), l2norm(q,k),
// and the gated delta rule: recurrent (decode) + chunked WY/UT (prefill). q-scale 1/sqrt(128).
// See docs/qwen3.6-27b-architecture.md §6.5/§7/§10.4.
// NOTE: structure stub — no implementation yet.

namespace qus::kernels {

// TODO(impl): causal_conv1d_{prefill,decode}; gdn_gating; l2norm; gated_delta_{recurrent,chunked}.

}  // namespace qus::kernels
