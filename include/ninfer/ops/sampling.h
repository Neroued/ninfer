#pragma once

#include "core/device.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Counter-based RNG subkey. Distinct purposes keep draws at the same logical position separate.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill               = 0,
    kSamplePurposeDecode                = 1,
    kSamplePurposeSpeculativeAccept     = 2,
    kSamplePurposeSpeculativeCorrection = 3,
    kSamplePurposeSpeculativeBonus      = 4,
    kSamplePurposeDFlash2Proposal       = 5,
};

// Borrowed round operand. Positions are contiguous packed vocabulary masks; null disables it.
// Empty support is permitted only where the consuming Op defines it. sample() and
// speculative_accept_* require a legal token at each consumed position; their caller carries
// dead-end status separately and supplies safe filler for dead/unreachable positions.
struct SamplingMask {
    const std::uint32_t* words = nullptr;
    std::int32_t stride        = 0;
};

// Device-resident sampling parameters and operands. token_counts is an optional device I32
// [token_domain] committed generated-token occurrence-count array used by both penalties.
struct SamplingConfig {
    float temperature       = 0.0f; // <= 0 => greedy argmax over allowed tokens
    std::int32_t top_k      = 20;   // runtime contract is [1,20]; Op defensively caps otherwise
    float top_p             = 1.0f; // >= 1 => disabled
    float min_p             = 0.0f; // <= 0 => disabled
    float presence_penalty  = 0.0f;
    float frequency_penalty = 0.0f;
    unsigned long long seed = 0;
    const std::int32_t* token_counts = nullptr; // device [token_domain] i32, or null
    SamplingMask mask;
};

// Caller-owned transient capacity for every parallel sampling-lane count in the inclusive
// interval. For sample(), one lane is one batch row; speculative acceptance uses the same
// workspace substrate for its verification columns. token_domain is the fixed route profile.
// Invalid profiles or intervals throw; a legal single-block route returns zero.
[[nodiscard]] std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain,
                                                            std::int32_t min_lanes,
                                                            std::int32_t max_lanes);

/**
 * Produces one token id per independent request row. `logits` is contiguous BF16
 * [physical_rows,B], `out` and `logical_positions` are contiguous I32 [B], and only vocabulary
 * rows v in [0,token_domain) participate. `configs` is a device-resident contiguous
 * SamplingConfig[B] array. Greedy and stochastic rows may coexist in one invocation.
 * A non-null config.mask excludes vocabulary rows before argmax/top-k and normalization. sample()
 * consumes mask position zero; speculative acceptance consumes its corresponding verify position.
 *
 * For row b with configs[b].temperature<=0:
 *
 * With either greedy or positive-temperature sampling, let
 * c_v=configs[b].token_counts[v] (or zero when the pointer is null):
 *
 *   adjusted_v = float(logits[v,b])
 *                - configs[b].presence_penalty * (c_v > 0)
 *                - configs[b].frequency_penalty * c_v.
 *
 * A greedy row selects min argmax_v adjusted_v and skips filters and RNG. Candidates for a
 * positive-temperature row are sorted by adjusted_v descending with lower token id breaking
 * ties. Per-row top_k
 * in [1,19] keeps that many candidates; top_k<=0 or top_k>=20 keeps min(20,token_domain).
 * Candidate weights are exp(adjusted_v/temperature-max). min_p removes the suffix below
 * min_p*max_weight; top_p keeps the shortest remaining prefix whose cumulative weight reaches
 * top_p times the pre-truncation candidate weight. At least the best candidate remains, the
 * support is renormalized, and one id is drawn for that row.
 *
 * Row b uses counter-based RNG key
 * (configs[b].seed,logical_positions[b],purpose), without mutable RNG state or dependence on the
 * compact row index. Counts, masks and all inputs remain read-only. `out` must not overlap
 * logits, configs, logical_positions or any token-count array. The Op writes all of out, uses
 * caller-owned transient storage reported by sampling_workspace_capacity_bytes(), and has no
 * persistent-state side effect. The caller commits its final output with the count-update Op.
 */
void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            MemoryArena& workspace, CudaExecutionView execution);

// Adds every id in the contiguous non-empty I32 token_ids vector to the contiguous I32
// [token_domain] committed count array. IDs must be in [0,token_domain).
void increment_token_counts(const Tensor& token_ids, Tensor& token_counts,
                            CudaExecutionView execution);

// Adds the first valid_columns[b] IDs from I32 token_ids [T,B] to count slot slots[b] in
// I32 token_counts [V,S]. T,V,S>0, B=1..8; valid_columns/slots are contiguous I32 [B].
// 0<=valid_columns[b]<=T; live rows have slots[b] in [0,S). IDs are in [0,V), and counts
// remain representable in I32. A zero-length row reads neither IDs
// nor its slot. Repeated IDs or slots add all occurrences atomically. Inputs are contiguous;
// counts has contiguous vocabulary entries and a non-overlapping, naturally aligned slot
// stride. All storage belongs to execution.device(); counts overlaps no input. Only the
// addressed counts change. Device metadata is replayable; no workspace or allocation.
void increment_token_counts_batch(const Tensor& token_ids, const Tensor& valid_columns,
                                  const Tensor& slots, Tensor& token_counts,
                                  CudaExecutionView execution);

} // namespace ninfer::ops
