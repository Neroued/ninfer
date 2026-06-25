#pragma once
// qus::core — DeviceArena: a bump allocator over one cudaMalloc slab. Three lifetime regions
// (Weights / Cache persistent, Workspace transient with reset()). Plus pinned-host staging.
// No general allocator, no paging, no per-step cudaMalloc.
// L0 infrastructure. See docs/l0-infrastructure-design.md §3.
// NOTE: structure stub — no implementation yet.

namespace qus {

// TODO(impl): class DeviceArena { Tensor alloc(DType, shape, align=256); void reset(); ... };
// TODO(impl): pinned-host staging buffer for async H2D during weight load.

}  // namespace qus
