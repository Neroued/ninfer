#pragma once

#include "artifact/schema.h"
#include "core/weight_view.h"

#include <array>
#include <span>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::qwen4_exp::ple {

enum class Placement { Disk, Device };

inline constexpr std::uint32_t kHeads             = 16;
inline constexpr std::uint32_t kRowElements       = 160;
inline constexpr std::uint32_t kEmbeddingElements = kHeads * kRowElements;
inline constexpr std::size_t kEmbeddingBytes      = kEmbeddingElements * sizeof(std::uint16_t);

struct HashGeometry {
    std::uint32_t ngram_size, heads_per_ngram;
};

struct HashControls {
    std::array<std::int64_t, 3> multipliers;
    std::array<std::int64_t, kHeads> vocab_sizes;
    std::array<std::int64_t, kHeads> offsets;
};

struct TableShard {
    artifact::ObjectHandle object;
    WeightGeometry geometry;
    std::uint64_t row_end;
    float divisor;
};

// Immutable lookup parameters in binding order. Representation and logical rows do not
// prescribe residency. Object handles belong to the Reader used to load these parameters.
struct LookupParameters {
    HashControls hash;
    std::vector<TableShard> shards;
    std::uint64_t rows;
};

// Parses controls and reads divisors, without materializing table rows or opening I/O backing.
// The supported representation is complete NF4_G16_E4M3FN/RowPagedG16_4K shards of width 160.
[[nodiscard]] LookupParameters load_lookup_parameters(const artifact::Reader&,
                                                      const artifact::Binding& table,
                                                      std::span<const std::byte> controls,
                                                      HashGeometry);

} // namespace ninfer::models::qwen4_exp::ple
