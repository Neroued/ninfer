#pragma once

// qus::kernels - token sampling from a logits column.
//
// sample_column() picks one token id per column of a BF16 [vocab, T] logits
// tensor. It is the sampling counterpart of argmax(): with a greedy config
// (temperature <= 0) it reproduces argmax() exactly (lowest-index tie-break),
// so the greedy/parity path is unchanged. With temperature > 0 it applies
// presence/frequency penalties, temperature, top-k, top-p and min-p, then draws
// a token with a counter-based RNG keyed by (seed, position, purpose) so the
// output is reproducible and independent of any speculative-decode path.
//
// The config is read from device memory (not passed by value) so a CUDA-graph
// capture stays valid across requests: only the buffer contents change.

#include "qus/core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>  // cudaStream_t

namespace qus::kernels {

// Counter-based RNG subkey identifying the call site. Distinct purposes keep
// two draws at the same absolute position (e.g. the prefill bonus token and the
// first decode token) from sharing randomness.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill    = 0,
    kSamplePurposeDecode     = 1,
    kSamplePurposeMtpAccept  = 2,
    kSamplePurposeMtpResample = 3,
    kSamplePurposeMtpBonus   = 4,
};

// Device-resident sampling parameters. `token_counts` is an optional [vocab]
// int32 device buffer of per-token occurrence counts used by the presence and
// frequency penalties; when null, penalties are skipped. All other fields are
// plain scalars. temperature <= 0 selects exact greedy argmax.
struct SamplingConfig {
    float temperature       = 0.0f;  // <= 0 => greedy argmax (bit-identical to argmax())
    std::int32_t top_k      = 0;     // <= 0 => no explicit top-k limit (internal cap only)
    float top_p             = 1.0f;  // >= 1 => disabled
    float min_p             = 0.0f;  // <= 0 => disabled
    float presence_penalty  = 0.0f;
    float frequency_penalty = 0.0f;
    unsigned long long seed = 0;
    std::int32_t* token_counts = nullptr;  // device [vocab] i32, or null
};

// Samples one token id per column of `logits` (BF16 [vocab, T]) into `out`
// (I32 [T]). `config` and `pos_base` are device pointers; column t is RNG-keyed
// by (*pos_base + t). `purpose` is a constant SamplePurpose subkey.
void sample_column(const Tensor& logits, Tensor& out, const SamplingConfig* config,
                   const std::int32_t* pos_base, std::int32_t purpose, cudaStream_t stream);

} // namespace qus::kernels
