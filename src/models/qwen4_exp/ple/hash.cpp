#include "models/qwen4_exp/ple/hash.h"

#include <bit>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::ple {

void hash_tokens(std::span<const std::int32_t> ids, std::array<std::int32_t, 2> tail,
                 std::int32_t eos, const HashControls& controls, std::span<std::uint32_t> output) {
    if (output.size() / kHeads != ids.size() || output.size() % kHeads || eos < 0 || tail[0] < 0 ||
        tail[1] < 0)
        throw std::invalid_argument("invalid PLE hash operands");
    for (auto id : ids)
        if (id < 0) throw std::invalid_argument("negative PLE token ID");
    for (std::size_t h = 0; h < kHeads; ++h)
        if (controls.vocab_sizes[h] <= 0 || controls.offsets[h] < 0 ||
            std::uint64_t(controls.offsets[h]) > std::numeric_limits<std::uint32_t>::max() ||
            std::uint64_t(controls.vocab_sizes[h] - 1) >
                std::numeric_limits<std::uint32_t>::max() - std::uint64_t(controls.offsets[h]))
            throw std::invalid_argument("PLE hash head exceeds U32 rows");
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const auto older = tail[1] == eos ? eos : tail[0];
        const auto pair  = std::uint64_t(ids[t]) * std::uint64_t(controls.multipliers[0]) ^
                          std::uint64_t(tail[1]) * std::uint64_t(controls.multipliers[1]);
        const auto triple = pair ^ std::uint64_t(older) * std::uint64_t(controls.multipliers[2]);
        for (std::size_t h = 0; h < kHeads; ++h) {
            const auto word = std::bit_cast<std::int64_t>(h < 8 ? pair : triple);
            auto index      = word % controls.vocab_sizes[h];
            if (index < 0) index += controls.vocab_sizes[h];
            output[t * kHeads + h] = static_cast<std::uint32_t>(index + controls.offsets[h]);
        }
        tail = {tail[1], ids[t]};
    }
}

} // namespace ninfer::models::qwen4_exp::ple
