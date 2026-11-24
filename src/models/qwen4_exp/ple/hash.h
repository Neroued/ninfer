#pragma once

#include "models/qwen4_exp/ple/parameters.h"

namespace ninfer::models::qwen4_exp::ple {

// Produces 16 global rows per raw ID, in head order. Multiplication wraps modulo 2^64;
// the XOR word is interpreted as signed before positive modulo. Tail is oldest first.
// Missing history is supplied as eos. A preceding eos masks only this lookup's older ID.
// Inputs are nonnegative IDs; output has exactly 16*ids.size() words and is disjoint.
void hash_tokens(std::span<const std::int32_t> ids, std::array<std::int32_t, 2> tail,
                 std::int32_t eos, const HashControls& controls, std::span<std::uint32_t> output);

} // namespace ninfer::models::qwen4_exp::ple
