#pragma once

// qus::model — ModelConfig: the constexpr frozen dimensions for Qwen3.6-27B (the 固化 truth).
// This is the L2 source of truth; L0/L1 take dims as parameters/template args.
// Fill from docs/qwen3.6-27b-architecture.md §2 (hyperparameters) — e.g. hidden 5120,
// 64 layers (48 GDN + 16 full), head_dim 256, GQA 24/4, GDN Hk16/Hv48/dk128, conv_kernel 4,
// vocab 248320, the 3:1 layer schedule, partial rope 0.25, mrope [11,11,10], etc.
// NOTE: structure stub — no values filled yet (structure only).

namespace qus::model {

// TODO(impl): struct ModelConfig { ... }; inline constexpr ModelConfig kQwen3_6_27B{ ... };
// TODO(impl): constexpr layer-type schedule: is_full(i) = ((i+1) % 4 == 0).

} // namespace qus::model
