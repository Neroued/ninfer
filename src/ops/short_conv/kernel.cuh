#pragma once

#include "launch.h"
#include "ops/block_attn_res/mix.cuh"

namespace ninfer::ops::short_conv_detail {

using attn_res_detail::Values;
using attn_res_detail::load_bf16;
using attn_res_detail::store_bf16;
inline constexpr float kConvScale = 0.44721359549995794f;

struct Column {
    int first, row, pos, length;
};

template <bool Ragged>
__device__ __forceinline__ Column locate(const Conv& a, int m) {
    if constexpr (Ragged) {
        const int row = a.token_to_seq[m], first = a.prefix[row];
        return {first, row, m - first, a.prefix[row + 1] - first};
    } else {
        const int row = m / a.width, first = row * a.width;
        return {first, row, m - first, a.valid ? a.valid[row] : a.width};
    }
}

__device__ __forceinline__ const Word* history(const Conv& a, int row) {
    return reinterpret_cast<const Word*>(reinterpret_cast<const char*>(a.history) +
                                         a.source_offsets[row]);
}

__device__ __forceinline__ Word* destination(const Conv& a, int row) {
    return reinterpret_cast<Word*>(reinterpret_cast<char*>(a.destination) +
                                   a.destination_offsets[row]);
}

__device__ __forceinline__ bool aliases_history(const Conv& a, const Column& col) {
    // A one-token row completes all its history reads before its own write.
    if (!a.overlapping_state || col.length == 1 || (col.pos >= 3 && col.pos != col.length - 1))
        return false;
    const auto src      = reinterpret_cast<std::uintptr_t>(history(a, col.row));
    const auto dst      = reinterpret_cast<std::uintptr_t>(destination(a, col.row));
    const auto distance = src < dst ? dst - src : src - dst;
    return distance < std::size_t(a.d) * 3 * sizeof(Word);
}

template <int V>
__device__ __forceinline__ Values<V> load_history(const Word* p) {
    if constexpr (V == 8) {
        if (reinterpret_cast<std::uintptr_t>(p) % 16 == 0) return load_bf16<V>(p);
    }
    Values<V> result;
#pragma unroll
    for (int v = 0; v < V; ++v) result.x[v] = __uint_as_float(unsigned(p[v]) << 16);
    return result;
}

__device__ __forceinline__ Word* snapshot(const Conv& a, const Column& col, int pos) {
    return a.destination + (std::int64_t(a.snapshot_slots[col.row]) + pos) * 3 * a.d;
}

__device__ __forceinline__ int aliased_snapshot(const Conv& a, const Column& col) {
    if (!a.overlapping_state) return -1;
    const auto src        = reinterpret_cast<std::uintptr_t>(history(a, col.row));
    const auto start      = reinterpret_cast<std::uintptr_t>(snapshot(a, col, 0));
    const auto slot_bytes = std::uint64_t(a.d) * 3 * sizeof(Word);
    if (src < start) return -1;
    const auto delta = src - start;
    if (delta >= slot_bytes * col.length) return -1;
    if (delta < slot_bytes) return 0;
    if (delta <= UINT32_MAX) return unsigned(delta) / unsigned(slot_bytes);
    // The reservation fits in the validated snapshot storage. Find a distant
    // alias with exact integer comparisons without a device 64-bit divide.
    int lo = 1, hi = col.length;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (mid * slot_bytes <= delta)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// A single owner handles all history-dependent columns whenever another CTA
// could overwrite their source. The rest of the segment has no state dependency.
template <bool Snapshot>
struct Work {
    Column col;
    int end, alias_slot;
    bool owns_history;

    __device__ __forceinline__ Work(const Conv& a, Column column) : col(column) {
        alias_slot = -1;
        if constexpr (Snapshot) alias_slot = aliased_snapshot(a, col);
        if constexpr (Snapshot)
            owns_history = alias_slot >= 0;
        else
            owns_history = aliases_history(a, col);
        if (!owns_history || col.pos >= 3)
            end = col.pos + 1;
        else if (col.pos == 0)
            end = min(3, col.length);
        else
            end = col.pos;
    }

    __device__ __forceinline__ int final_write() const {
        if (owns_history && col.pos == 0) {
            if constexpr (Snapshot)
                return alias_slot;
            else
                return col.length - 1;
        }
        if constexpr (!Snapshot) {
            if (!owns_history && col.pos == col.length - 1) return col.pos;
        }
        return -1;
    }

    __device__ __forceinline__ bool writes_snapshot(int pos) const { return pos != alias_slot; }
};

template <int V>
__device__ __forceinline__ Values<V> conv_sum(const Conv& a, const Column& col, int pos, int c) {
    const auto d        = std::int64_t(a.d);
    const auto* current = a.x + (std::int64_t(col.first) + pos) * d + c;
    const auto* h       = pos < 3 ? history(a, col.row) + c : nullptr;
    const auto input    = load_bf16<V>(current);
    Values<V> sum{};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const auto w   = load_bf16<V>(a.weight + k * d + c);
        const auto tap = pos - 3 + k;
        const auto x   = k == 3     ? input
                         : tap >= 0 ? load_bf16<V>(current + (k - 3) * d)
                                    : load_history<V>(h + (tap + 3) * d);
#pragma unroll
        for (int v = 0; v < V; ++v) sum.x[v] = fmaf(w.x[v], x.x[v], sum.x[v]);
    }
    Values<V> result;
#pragma unroll
    for (int v = 0; v < V; ++v) { result.x[v] = (sum.x[v] + input.x[v]) * kConvScale; }
    return result;
}

template <int V>
__device__ __forceinline__ void add_residual(const Conv& a, const Word* current,
                                             Values<V>& result) {
    if (a.addend) {
        const auto at  = current - a.x;
        const auto add = load_bf16<V>(a.addend + at);
        const auto res = a.residual ? load_bf16<V>(a.residual + at) : Values<V>{};
#pragma unroll
        for (int v = 0; v < V; ++v) {
            const float base = a.residual ? res.x[v] + add.x[v] : add.x[v];
            result.x[v]      = base + result.x[v];
        }
    }
}

template <int V>
__device__ __forceinline__ Values<V> convolve(const Conv& a, const Column& col, int pos, int c) {
    auto result = conv_sum<V>(a, col, pos, c);
#pragma unroll
    for (int v = 0; v < V; ++v) result.x[v] = __bfloat162float(__float2bfloat16_rn(result.x[v]));
    add_residual(a, a.x + (std::int64_t(col.first) + pos) * a.d + c, result);
#pragma unroll
    for (int v = 0; v < V; ++v) result.x[v] = __bfloat162float(__float2bfloat16_rn(result.x[v]));
    return result;
}

__device__ __forceinline__ uint4 round_conv_pack(const Conv& a, const Word* current,
                                                 Values<8> result) {
    const auto rounded = attn_res_detail::encode_bf16(result);
    if (!a.addend) return rounded;
    result = attn_res_detail::decode_bf16(rounded);
    add_residual(a, current, result);
    return attn_res_detail::encode_bf16(result);
}

__device__ __forceinline__ uint4 convolve_pack(const Conv& a, const Column& col, int pos, int c) {
    const auto* current = a.x + (std::int64_t(col.first) + pos) * a.d + c;
    return round_conv_pack(a, current, conv_sum<8>(a, col, pos, c));
}

template <int V, bool Snapshot>
__device__ __forceinline__ void update_state(const Conv& a, const Column& col, int pos, int c) {
    const auto d        = std::int64_t(a.d);
    auto* dst           = (Snapshot ? snapshot(a, col, pos) : destination(a, col.row)) + c;
    const auto* h       = pos < 2 ? history(a, col.row) + c : nullptr;
    const auto* current = a.x + (std::int64_t(col.first) + pos) * d + c;
    // Each channel shifts towards older slots. Forward copy keeps all loads
    // ahead of in-place overwrites, including for one- and two-token segments.
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const auto tap   = pos - 2 + k;
        const auto* from = tap >= 0 ? current + (k - 2) * d : h + (tap + 3) * d;
        auto* to         = dst + k * d;
        if constexpr (V == 8) {
            if (reinterpret_cast<std::uintptr_t>(from) % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(to) % 16 == 0) {
                *reinterpret_cast<uint4*>(to) = *reinterpret_cast<const uint4*>(from);
                continue;
            }
        }
#pragma unroll
        for (int v = 0; v < V; ++v) to[v] = from[v];
    }
}

} // namespace ninfer::ops::short_conv_detail
