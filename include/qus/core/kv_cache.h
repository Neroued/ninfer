#pragma once
// qus::core — KVCache: contiguous per-(full-attn-layer) K/V buffers in the Cache region,
// position-indexed and append-only (no paging). Sized from (max_context, dims).
// L0 infrastructure. See docs/l0-infrastructure-design.md §5.3.
// NOTE: structure stub — no implementation yet.

namespace qus {

// TODO(impl): struct KVCache { Tensor k[N_FULL], v[N_FULL]; uint32_t pos; append(...); };

}  // namespace qus
