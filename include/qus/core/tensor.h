#pragma once
// qus::core — Tensor: a non-owning device-memory view {data, dtype, ne[4], nb[4]} with
// metadata-only view/slice/permute/reshape; plus QuantWeight {qdata, scales, n, k, group}
// describing a packed-4bit linear weight. The common currency passed to kernels.
// L0 infrastructure. See docs/l0-infrastructure-design.md §4.
// NOTE: structure stub — no implementation yet.

namespace qus {

// TODO(impl): struct Tensor { void* data; DType dtype; int32_t ne[4]; int64_t nb[4]; ... };
// TODO(impl): struct QuantWeight { const void* qdata; const void* scales; int32_t n,k,group; };

}  // namespace qus
