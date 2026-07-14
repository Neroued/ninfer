#pragma once

// ninfer::kernels - token sampling from a logits column.
//
// sample() picks one token id per column of a BF16 [physical_rows, T] logits
// tensor, considering only [0, token_domain). It is the sampling counterpart
// of argmax(): with a greedy config
// (temperature <= 0) it reproduces argmax() exactly (lowest-index tie-break),
// so the greedy/parity path is unchanged. With temperature > 0 it applies
// presence/frequency penalties, temperature, top-k, top-p and min-p, then draws
// a token with a counter-based RNG keyed by (seed, position, purpose) so the
// output is reproducible and independent of any speculative-decode path.
//
// top-k is clamped to an internal candidate cap of 20 (the Qwen3.6 thinking
// default): a top_k <= 0 or a top_k > 20 both select the top 20 logits, and
// top-p / min-p then operate within that top-20 set. This matches the specialized
// engine scope; larger top-k semantics are intentionally not supported.
//
// The config is read from device memory (not passed by value) so a CUDA-graph
// capture stays valid across requests: only the buffer contents change.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::kernels {

// Counter-based RNG subkey identifying the call site. Distinct purposes keep
// two draws at the same absolute position (e.g. the prefill bonus token and the
// first decode token) from sharing randomness.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill     = 0,
    kSamplePurposeDecode      = 1,
    kSamplePurposeMtpAccept   = 2,
    kSamplePurposeMtpResample = 3,
    kSamplePurposeMtpBonus    = 4,
};

// Device-resident sampling parameters. `token_counts` is an optional [token_domain]
// int32 device buffer of per-token occurrence counts used by the presence and
// frequency penalties; when null, penalties are skipped. All other fields are
// plain scalars. temperature <= 0 selects exact greedy argmax.
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

// Samples one token id per column of `logits` (BF16 [physical_rows, T]) into
// `out` (I32 [T]). Columns remain strided by physical_rows while rows at or
// above token_domain are ignored. `config` and `pos_base` are device pointers;
// column t is RNG-keyed by (*pos_base + t). `purpose` is a SamplePurpose subkey.
void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* config,
            const std::int32_t* pos_base, std::int32_t purpose, cudaStream_t stream);

} // namespace ninfer::kernels
