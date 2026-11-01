#pragma once

// ninfer::models::qwen3_5::execution - L2 prefetch hints for the decode exchanges (q27/dexch).
//
// A captured TP2 decode exchange waits ~5 us on PCIe with its device's DRAM idle, and the GEMM
// after it (behind an RMSNorm) streams weights that depend on nothing the exchange produces. The
// dense text forward therefore names, before each row-parallel projection whose all-reduce ends a
// sub-block, the leading rows of the weight that comes next; the exchange prefetches them into L2
// from blocks of its own (ops::PeerMailbox::set_l2_prefetch). The next GEMM's first CTAs read
// those rows first (their row blocks are the lowest), so their loads hit L2 instead of DRAM.
// Hints change no result bit: they move no data the forward reads.

#include "core/device.h"
#include "core/weight.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// One rank's hint: at most PeerMailbox::kL2PrefetchRanges ranges.
using ExchangePrefetch = std::vector<ops::PeerL2Range>;

// Bytes of `w` one row occupies (codes plus scales) in the formats a hint covers; 0 otherwise.
[[nodiscard]] std::size_t prefetch_row_bytes(const Weight& w) noexcept;

// Appends the ranges holding rows [row, row + rows) of `w` (clipped to its rows): codes, then
// scales (NVFP4: whole 128-row scale tiles). Appends nothing for a format without a row geometry
// here or for an empty row interval.
void append_prefetch_rows(const Weight& w, std::int64_t row, std::int64_t rows,
                          ExchangePrefetch& out);

// The leading rows of `w` that fit `bytes`, as ranges.
void append_prefetch_prefix(const Weight& w, std::size_t bytes, ExchangePrefetch& out);

// A paired gate|up parent (gate rows [0, n/2), up rows [n/2, n)): the leading rows of both halves,
// `bytes` in all, as a paired GEMM's first CTAs read them.
void append_prefetch_paired_prefix(const Weight& w, std::size_t bytes, ExchangePrefetch& out);

// The weight of a projection with a single parent; null for a paired one.
[[nodiscard]] const Weight* single_projection(const ops::ProjectionWeights& projection) noexcept;

// The hints of the dense block schedule (the plain residual, qwen4_forward/plain.h). A mixer's
// all-reduce is followed by the post-attention norm and the paired gate|up GEMM of the same layer;
// an FFN's by the next layer's input norm and input projection: Q|K|G|V for attention, for GDN the
// small BF16 A|B gating projection (whole, when it fits half the budget) and then Q|K|V|Z.
void append_ffn_input_prefetch(const DenseParameters& ffn, std::size_t bytes,
                               ExchangePrefetch& out);
void append_mixer_input_prefetch(const AttentionParameters& attention, std::size_t bytes,
                                 ExchangePrefetch& out);
void append_mixer_input_prefetch(const GdnParameters& gdn, std::size_t bytes,
                                 ExchangePrefetch& out);

// The prefetch budget of the mailbox a captured exchange of `exchange_bytes` would take, 0 when
// there is none (eager, staged, prefetch off).
[[nodiscard]] std::size_t exchange_prefetch_budget(const ops::PeerEvents& events,
                                                   const ExecutionContext& execution,
                                                   std::size_t exchange_bytes);

// Hands `hint[r]` to that mailbox for rank r's next exchange. No-op without one.
void stage_exchange_prefetch(const ops::PeerEvents& events, const ExecutionContext& execution,
                             std::size_t exchange_bytes,
                             const std::array<ExchangePrefetch, 2>& hint);

} // namespace ninfer::models::qwen3_5::execution
