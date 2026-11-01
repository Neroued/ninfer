#pragma once

// Two-device (tp 2) stream-split form of the hyper-connection mix.
//
// With models::LoadOptions::hc_split_streams, rank r holds streams [2r, 2r+2) of the wide residual
// R (S = 4 streams of H = 2560) and the matching shards of every mixer (models/qwen3_5/load/
// sharding.h): norm rows, Down|inject columns, Up rows. The mixer then needs two exchanges, the
// low-rank projection p (its rank partials contract disjoint streams) and the collapsed block input
// (each rank collapses its own streams); the combine that commits a block output into R stays
// local (hyper_connection_combine over the rank's streams and its inject gates).
//
// Exchange accounting per mixer: two allreduce_sum_f32 calls of (R + I) * T and H * T FP32
// elements (1.3 KiB and 10 KiB per column), on the pinned-host mailbox inside a captured round.

#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/hyper_connection.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Per-rank transient bytes (from a 256-byte aligned cursor) sufficient for every
 * hyper_connection_mix_split call with T in [min_tokens, max_tokens] and the given inject form.
 */
[[nodiscard]] std::size_t hyper_connection_mix_split_workspace_bytes(std::int32_t min_tokens,
                                                                     std::int32_t max_tokens,
                                                                     bool inject);

/**
 * Op: hyper-connection mix split by residual streams over two ranks
 *
 * Math: hyper_connection_mix of the concatenated residual R = [residual[0]; residual[1]] with the
 *   concatenated weights. Rank r evaluates, for its streams s in [2r, 2r+2):
 *     n_s     = grouped_offset_rmsnorm(residual[r], norm[r], 2 groups, eps)
 *     p_r     = down[r] . n_r                                       (BF16, a Linear output)
 *     p       = fl32(p_0 + p_1)                                     (exchange 1)
 *     low     = silu(p[0,R) / S)  (BF16),  inject_s = 2 sigmoid(p[R+s] / S)  (FP32)
 *     q_r     = (1/S) sum_{s in r} sigmoid((up[r] . low)_s) * n_s    (FP32 partial)
 *     mixed   = RNE_bf16(fl32(q_0 + q_1))                           (exchange 2)
 * Logical shapes, per rank: residual BF16 [2H, T] (stream-major, the rank's streams); weights[r]
 *   norm BF16 [2H], down Q8 row-split [R + I, 2H], up Q8 row-split [2H, R]; mixed BF16 [H, T]
 *   (identical on both ranks); inject FP32 [2, T] (the rank's streams), non-null exactly when
 *   down has the R + S = 324 rows of a block mixer.
 * Supported domain: S = 4, H = 2560, R = 320; any T >= 1; eps > 0 and finite.
 * Numeric: the general route of hyper_connection_mix with the projection's two rank partials
 *   staged in BF16 and summed in FP32, and the collapse summed in FP32 across the ranks and rounded
 *   once; judged against the same FP64 oracle and criteria as hyper_connection_mix.
 * Effects: writes mixed[r] and inject[r]; residual is read only. Rank r's work runs on
 *   ec.dev[r]->stream; consecutive calls need no host synchronization (allreduce_sum_f32).
 * Workspace: hyper_connection_mix_split_workspace_bytes per rank.
 * Execution: two routes, same contract and criteria.
 *   - fused (T <= 8, inside a two-device capture whose PeerEvents carry a pipelined PeerMailbox
 *     whose slots fit the H * T * 4-byte exchange, on devices an earlier eager call prepared, and
 *     unless NINFER_HC_SPLIT_ROUTE=composed): one cooperative launch per rank (one CTA per SM pair,
 *     at least 56) that exchanges both sums mid-kernel through the mailbox slots
 *     (PeerMailbox::exchange_slot), staging the activations in FP16 like hyper_connection_mix's
 *     small-T route; the projection's rank partials are FP32. Like that route, it starts only when
 *     every SM can host its CTA, so nothing on the rank's device may wait for work queued after it.
 *   - composed (every other call): grouped norm, Q8 Linear, element kernels and two
 *     allreduce_sum_f32 calls (mailbox when captured, staged copies otherwise). An eager call
 *     prepares the fused route's kernels on both devices: make one before capturing.
 *   The call preserves the caller's current device.
 */
void hyper_connection_mix_split(const std::array<Tensor, 2>& residual,
                                const std::array<HyperConnectionWeights, 2>& weights, float eps,
                                const std::array<Tensor, 2>& mixed,
                                const std::array<Tensor*, 2>& inject,
                                const std::array<WorkspaceArena*, 2>& workspace,
                                const ExecutionContext& ec, const PeerEvents& events);

/**
 * Op: hyper-connection combine followed by the split mix (the pending block output committed into
 * the rank's streams, then the next mixer), as hyper_connection_combine_mix over the split.
 *
 * previous_y[r] BF16 [H, T] (the block output, identical on both ranks); previous_inject[r] FP32
 * [2, T] (the rank's streams' gates of the previous mixer, not overlapping inject[r], as in
 * hyper_connection_combine_mix). Updates residual[r] in place to its
 * committed values (one BF16 rounding each) and writes mixed and inject as
 * hyper_connection_mix_split over the committed residual. Same routes and workspace.
 */
void hyper_connection_combine_mix_split(const std::array<Tensor, 2>& residual,
                                        const std::array<Tensor, 2>& previous_y,
                                        const std::array<Tensor, 2>& previous_inject,
                                        const std::array<HyperConnectionWeights, 2>& weights,
                                        float eps, const std::array<Tensor, 2>& mixed,
                                        const std::array<Tensor*, 2>& inject,
                                        const std::array<WorkspaceArena*, 2>& workspace,
                                        const ExecutionContext& ec, const PeerEvents& events);

} // namespace ninfer::ops
