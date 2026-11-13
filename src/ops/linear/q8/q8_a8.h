#pragma once

// Q8_G32 prefill on int8 tensor cores (NINFER_Q8_A8, opt-in).
//
// The A16 tiles dequantize the int8 codes to BF16 and run BF16 MMA with FP32 accumulation (~89
// TFLOPS of the RTX 5070 Ti's ~100 BF16 ceiling at the Qwen3.8 Flash shapes). This route quantizes
// the activations to int8 once per call (per token and G-value group, G = 32 or 128) into a
// workspace and runs mma m16n8k32 s8 on the codes as they are: one MMA per 32-value weight group,
// its int32 partial exact; G = 32 accumulates acc = fma(sw * sx, float(part), acc); G = 128
// accumulates tmp = fma(float(part), sw, tmp) over the four weight groups of an activation group,
// then acc = fma(tmp, sx, acc). Activation rule: q = rint(x * (127 / amax)) in
// [-127, 127], s = amax / 127; a zero or subnormal amax gives a zero group, NaN or Inf gives s =
// NaN. This changes the arithmetic (activation quantization). Default off.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8A8Mode : int { Off = 0, G32 = 32, G128 = 128 };

// NINFER_Q8_A8: unset or 0 = Off; 1 or 32 = G32; 128 = G128. Read once (or the test override).
Q8A8Mode q8_a8_mode();
void q8_a8_force_mode_for_test(Q8A8Mode mode);

// The shapes the route takes (one rank's shard of the Qwen3.8 Flash dense projections at tp 2):
// 8192 x 2560, 6656 x 2560, 1280 x 2560 and 2560 x 3072.
bool q8_a8_admitted(std::int32_t n, std::int32_t k);

inline constexpr std::int32_t kQ8A8MinTokens   = 64;
inline constexpr std::int32_t kQ8A8PieceTokens = 4096;

// Workspace of one call: the quantized activations of one piece of at most kQ8A8PieceTokens tokens.
std::size_t q8_a8_workspace_bytes(std::int32_t k, std::int32_t tokens, Q8A8Mode mode);

// The route itself, without the switch, admission and T checks: n % 128 == 0, k % 128 == 0, padded
// K == K, out BF16 or FP32 [N, T]; `workspace` holds q8_a8_workspace_bytes(k, T, mode) bytes,
// 256-byte aligned.
void q8_a8_run(const Tensor& x, const Weight& w, Tensor& out, Q8A8Mode mode, void* workspace,
               std::size_t bytes, cudaStream_t stream);

// The route when switched on, admitted, T >= kQ8A8MinTokens and the arena has room; returns false,
// having enqueued nothing, otherwise (the caller runs the A16 tiles). Warns once per reason on
// stderr.
bool q8_a8_try_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena* arena,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
