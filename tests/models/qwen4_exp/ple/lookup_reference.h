#pragma once

#include "models/qwen4_exp/definition.h"
#include "models/qwen4_exp/ple/disk/table.h"

#include <algorithm>
#include <bit>

namespace ninfer::test::flash_next {

// Read the artifact row selected by the documented n-gram formula independently of the
// producer's row addressing, local range and collective. The embedding is a BF16 word lookup.
inline void check_ple_column(const artifact::Reader& reader,
                             const models::qwen4_exp::ModelDefinition& def, TokenId older,
                             TokenId previous, TokenId current,
                             std::span<const std::uint16_t> words) {
    namespace ple      = models::qwen4_exp::ple;
    const auto& params = def.ple_parameters();
    const auto& hash   = params.hash;
    if (previous == def.config().text.ple.eos) older = previous;
    if (words.size() != ple::kHeads * ple::kRowElements)
        throw std::runtime_error("wrong PLE feature width");
    for (std::uint32_t head = 0; head < ple::kHeads; ++head) {
        auto bits = std::uint64_t(current) * std::uint64_t(hash.multipliers[0]) ^
                    std::uint64_t(previous) * std::uint64_t(hash.multipliers[1]);
        if (head >= ple::kHeads / 2)
            bits ^= std::uint64_t(older) * std::uint64_t(hash.multipliers[2]);
        auto index = std::bit_cast<std::int64_t>(bits) % hash.vocab_sizes[head];
        if (index < 0) index += hash.vocab_sizes[head];
        const auto row = static_cast<std::uint64_t>(hash.offsets[head] + index);
        const auto shard =
            std::ranges::upper_bound(params.shards, row, {}, &ple::TableShard::row_end);
        if (shard == params.shards.end())
            throw std::runtime_error("PLE reference row exceeds table");
        const auto& object = reader.directory().tensor(shard->object);
        const auto local   = row - (shard->row_end - shard->geometry.shape.front());
        const auto offset  = (local / shard->geometry.rows_per_page) * 4096 +
                            (local % shard->geometry.rows_per_page) * ple::disk::kRowBytes;
        const auto encoded = reader.read_range(object.offset + offset, ple::disk::kRowBytes);
        std::array<std::uint16_t, ple::kRowElements> expected;
        decode_nf4_row(encoded, make_nf4_bf16_lut(shard->divisor), expected);
        if (!std::ranges::equal(expected, words.subspan(head * expected.size(), expected.size())))
            throw std::runtime_error("PLE lookup differs at global row " + std::to_string(row) +
                                     ", head " + std::to_string(head));
    }
}
} // namespace ninfer::test::flash_next
