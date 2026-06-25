#pragma once
// qus::core — GdnState: fixed-size, context-independent Gated-DeltaNet state for the linear-
// attention layers — conv state [conv_dim, W-1] and SSM state [Hv, dv, dk] (fp32) per layer,
// in the Cache region, updated in place.
// L0 infrastructure. See docs/l0-infrastructure-design.md §5.4.
// NOTE: structure stub — no implementation yet.

namespace qus {

// TODO(impl): struct GdnState { Tensor conv[N_GDN]; Tensor ssm[N_GDN]; };

}  // namespace qus
