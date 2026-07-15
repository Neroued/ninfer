#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Counter-based RNG subkey. Distinct purposes keep draws at the same logical position separate.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill     = 0,
    kSamplePurposeDecode      = 1,
    kSamplePurposeMtpAccept   = 2,
    kSamplePurposeMtpResample = 3,
    kSamplePurposeMtpBonus    = 4,
};

// Device-resident sampling parameters. token_counts is an optional device I32
// [token_domain] occurrence-count array used by both penalties.
struct SamplingConfig {
    float temperature          = 0.0f; // <= 0 => greedy argmax (bit-identical to argmax())
    std::int32_t top_k         = 0;    // clamped to 20: top_k <= 0 or top_k > 20 => 20
    float top_p                = 1.0f; // >= 1 => disabled
    float min_p                = 0.0f; // <= 0 => disabled
    float presence_penalty     = 0.0f;
    float frequency_penalty    = 0.0f;
    unsigned long long seed    = 0;
    std::int32_t* token_counts = nullptr; // device [token_domain] i32, or null
};

// Caller-owned transient storage for the registered multi-block candidate path.
// Returns zero for shapes that use the single-block fallback.
[[nodiscard]] std::size_t sampling_workspace_bytes(std::int32_t token_domain, std::int32_t columns);

/**
 * Produces one token id per logits column. `logits` is contiguous BF16 [physical_rows,T], `out`
 * is contiguous I32 [T], and only rows v in [0,token_domain) participate.
 *
 * With temperature<=0:
 *
 *   out[t] = min argmax_v float(logits[v,t]).
 *
 * Penalties, filters, RNG, and token_counts updates are skipped in this branch. With positive
 * temperature, let c_v=token_counts[v] (or zero when the pointer is null):
 *
 *   adjusted_v = float(logits[v,t])
 *                - presence_penalty * (c_v > 0)
 *                - frequency_penalty * c_v.
 *
 * Candidates are sorted by adjusted_v descending with lower token id breaking ties. top_k in
 * [1,19] keeps that many candidates; top_k<=0 or top_k>=20 keeps min(20,token_domain). Candidate
 * weights are exp(adjusted_v/temperature-max). min_p removes the suffix below
 * min_p*max_weight; top_p keeps the shortest remaining prefix whose cumulative weight reaches
 * top_p times the pre-truncation candidate weight. At least the best candidate remains, the
 * support is renormalized, and one id is drawn.
 *
 * `config` and `pos_base` are device pointers, with pos_base addressing one I32 scalar. Column t
 * uses counter-based RNG key (config->seed,*pos_base+t,purpose), without mutable RNG state. In the
 * positive-temperature branch the selected token atomically increments token_counts when it is
 * non-null. `out` must not overlap logits, config, pos_base, or token_counts. The Op writes all of
 * out, uses caller-owned transient storage reported by sampling_workspace_bytes(), and has no
 * other persistent state side effect.
 */
void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* config, const std::int32_t* pos_base, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
