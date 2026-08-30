#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Counter-based RNG subkey. Distinct purposes keep draws at the same logical position separate.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill               = 0,
    kSamplePurposeDecode                = 1,
    kSamplePurposeSpeculativeAccept     = 2,
    kSamplePurposeSpeculativeCorrection = 3,
    kSamplePurposeSpeculativeBonus      = 4,
};

// Device-resident sampling parameters. token_counts is an optional device I32
// [token_domain] committed generated-token occurrence-count array used by both penalties.
struct SamplingConfig {
    float temperature          = 0.0f; // <= 0 => greedy argmax (bit-identical to argmax())
    std::int32_t top_k         = 20;   // runtime contract is [1,20]; Op defensively caps otherwise
    float top_p                = 1.0f; // >= 1 => disabled
    float min_p                = 0.0f; // <= 0 => disabled
    float presence_penalty     = 0.0f;
    float frequency_penalty    = 0.0f;
    unsigned long long seed    = 0;
    std::int32_t* token_counts = nullptr; // device [token_domain] i32, or null
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
 * compact row index. In every mode the selected token atomically increments
 * configs[b].token_counts when it is non-null. Non-null token-count arrays belonging to distinct
 * active requests must not alias. `out` must not overlap logits, configs, logical_positions, or any
 * token-count array. The Op writes all of out, uses caller-owned transient storage reported by
 * sampling_workspace_capacity_bytes(), and has no other persistent-state side effect.
 */
void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream);

// Adds every id in the contiguous non-empty I32 token_ids vector to the contiguous I32
// [token_domain] committed count array. IDs must be in [0,token_domain).
void increment_token_counts(const Tensor& token_ids, Tensor& token_counts, cudaStream_t stream);

} // namespace ninfer::ops
