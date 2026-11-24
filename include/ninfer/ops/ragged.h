#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <limits>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::ops {

/**
 * 1D packed ragged sequence layout over N = total_tokens() columns partitioned into
 * S = num_seqs() non-empty segments [host_cu_seqlens[s], host_cu_seqlens[s+1]).
 *
 * Contract:
 * - S >= 1, N >= S; host_cu_seqlens has length S + 1, starts at 0, ends at N, and is
 *   strictly increasing (every segment has positive length L_s >= 1).
 * - cu_seqlens is contiguous Device I32 [S + 1] holding the identical prefix offsets.
 * - seq_lengths is contiguous Device I32 [S] holding L_s = cu_seqlens[s+1] - cu_seqlens[s].
 * - token_to_seq is contiguous Device I32 [N] mapping each flat column t in [cu[s], cu[s+1])
 *   to segment index s. The caller guarantees that Device metadata matches the Host layout.
 *   Device reads follow the execution stream and no metadata is read back.
 */
struct RaggedSequenceView {
    Tensor cu_seqlens;                             // Device I32 [S + 1]
    Tensor seq_lengths;                            // Device I32 [S]
    Tensor token_to_seq;                           // Device I32 [N]
    std::span<const std::int32_t> host_cu_seqlens; // Host [S + 1], strictly increasing

    [[nodiscard]] std::int32_t num_seqs() const noexcept {
        return host_cu_seqlens.size() >= 2 ? static_cast<std::int32_t>(host_cu_seqlens.size() - 1)
                                           : 0;
    }

    [[nodiscard]] std::int32_t total_tokens() const noexcept {
        return host_cu_seqlens.size() >= 2 ? host_cu_seqlens.back() : 0;
    }

    [[nodiscard]] std::int32_t segment_begin(std::int32_t s) const noexcept {
        return host_cu_seqlens[static_cast<std::size_t>(s)];
    }

    [[nodiscard]] std::int32_t segment_length(std::int32_t s) const noexcept {
        const auto idx = static_cast<std::size_t>(s);
        return host_cu_seqlens[idx + 1] - host_cu_seqlens[idx];
    }
};

inline void populate_ragged_host_metadata(std::span<const std::int32_t> lengths,
                                          std::span<std::int32_t> out_cu_seqlens,
                                          std::span<std::int32_t> out_seq_lengths,
                                          std::span<std::int32_t> out_token_to_seq) {
    if (lengths.empty() || lengths.size() >= std::numeric_limits<std::int32_t>::max() ||
        out_cu_seqlens.size() <= lengths.size() || out_seq_lengths.size() < lengths.size()) {
        throw std::invalid_argument("ragged metadata: invalid segment buffer capacity");
    }
    std::int64_t cursor = 0;
    out_cu_seqlens[0]   = 0;
    for (std::size_t s = 0; s < lengths.size(); ++s) {
        const auto len = lengths[s];
        if (len <= 0) {
            throw std::invalid_argument("ragged metadata: segment length must be positive");
        }
        out_seq_lengths[s] = len;
        cursor += len;
        if (cursor > std::numeric_limits<std::int32_t>::max() ||
            static_cast<std::size_t>(cursor) > out_token_to_seq.size()) {
            throw std::invalid_argument("ragged metadata: token_to_seq capacity exceeded");
        }
        out_cu_seqlens[s + 1] = static_cast<std::int32_t>(cursor);
        for (std::int64_t t = cursor - len; t < cursor; ++t) {
            out_token_to_seq[static_cast<std::size_t>(t)] = static_cast<std::int32_t>(s);
        }
    }
}

} // namespace ninfer::ops
