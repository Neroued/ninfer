#pragma once
// qus::core — WeightStore / loader: read the one fixed weight file, validate its header
// against the model config, upload tensors into the Weights region, and expose typed
// Tensor / QuantWeight views by role+layer.
// L0 infrastructure. See docs/l0-infrastructure-design.md §5.2 and docs/design.md §10.
// NOTE: structure stub — no implementation yet.

namespace qus {

// TODO(impl): class WeightStore { void load(path, DeviceArena& weights, DeviceContext&);
//                                 QuantWeight qweight(role, layer); Tensor weight(role, layer); };

}  // namespace qus
