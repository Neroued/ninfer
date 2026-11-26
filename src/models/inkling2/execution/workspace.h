#pragma once

#include "core/arena.h"
#include "models/common/vision/resources.h"
#include "models/inkling2/execution/profile.h"
#include "models/inkling2/execution_shape.h"
#include "ninfer/ops/relative_attention.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::models::inkling2::execution {

struct WorkspacePlan {
    std::size_t prefill       = 0;
    std::size_t ordinary      = 0;
    std::size_t finish        = 0;
    std::size_t general_bytes = 0;
    std::size_t bytes         = 0;
    std::optional<models::vision::Capacity> vision;
};

[[nodiscard]] WorkspacePlan plan_workspace(const ModelProfile&, const ExecutionShape&,
                                           std::int32_t sm_count, std::size_t vision_memory_budget);

namespace workspace {

struct MainRoots {
    Tensor stack;           // BF16 [H, T, max_blocks + 1]
    Tensor prelude_partial; // BF16 [H, T] (loop only)
    Tensor bank;            // BF16 [H, T, bank_blocks] (loop only)
    Tensor prelude_dests;   // FP32 [H, T, Q_core_start] (loop only)
    Tensor messages;        // FP32 [H, T, 4] (loop only)
    Tensor attn_in_bf16;    // BF16 [H, T, 3]
    Tensor attn_in_fp32;    // FP32 [H, T, 3] (loop only)
    Tensor attn_out;        // BF16 [H, T]
    Tensor attn_proj;       // BF16 [H, T]
};

struct AttentionBuffers {
    Tensor q;     // BF16 [q_dim, T]
    Tensor r;     // BF16 [r_dim, T]
    Tensor gate;  // BF16 [q_dim, T]
    Tensor k;     // BF16 [kv_dim, T]
    Tensor v;     // BF16 [kv_dim, T]
    Tensor q_out; // BF16 [q_dim, T]
    Tensor r_out; // BF16 [r_dim, T]
    Tensor sink;  // BF16 [hq, T]
    Tensor k_dyn; // BF16 [taps * hkv, T] (Full only)
    Tensor v_dyn; // BF16 [taps * hkv, T] (Full only)
    Tensor out;   // BF16 [q_dim, T]
    ops::RelativeAttentionChunkKV chunk_kv;
};

struct MoeBuffers {
    Tensor route_ids;     // I32 [K, T]
    Tensor route_weights; // FP32 [K, T]
};

template <class Allocator>
MainRoots main_roots(Allocator& a, const ModelProfile& p, std::int32_t columns) {
    const auto& c   = p.config.text;
    const int h     = dimension(c.hidden);
    const int slots = dimension(p.text.max_blocks + 1U);
    MainRoots out;
    out.stack = a.alloc(DType::BF16, {h, columns, slots});
    if (c.loop) {
        const int core_start_q = c.physical_layers.at(c.loop->core_start).is_swa ? 1 : 3;
        out.prelude_partial    = a.alloc(DType::BF16, {h, columns});
        out.bank               = a.alloc(DType::BF16, {h, columns, dimension(p.text.bank_blocks)});
        out.prelude_dests      = a.alloc(DType::FP32, {h, columns, core_start_q});
        out.messages           = a.alloc(DType::FP32, {h, columns, 4});
        out.attn_in_fp32       = a.alloc(DType::FP32, {h, columns, 3});
    }
    out.attn_in_bf16 = a.alloc(DType::BF16, {h, columns, 3});
    out.attn_out     = a.alloc(DType::BF16, {h, columns});
    out.attn_proj    = a.alloc(DType::BF16, {h, columns});
    return out;
}

template <class Allocator>
ops::RelativeAttentionChunkKV chunk_kv(Allocator& a, KvCacheStorage storage, std::int32_t head_dim,
                                       std::int32_t kv_heads, std::int32_t width,
                                       std::int32_t batch, bool ragged) {
    ops::RelativeAttentionChunkKV out;
    out.storage         = storage;
    const auto alloc_kv = [&](DType dtype, std::int32_t d0) {
        if (ragged) return a.alloc(dtype, {d0, kv_heads, width});
        return a.alloc(dtype, {d0, kv_heads, width, batch});
    };
    if (storage == KvCacheStorage::BFloat16) {
        out.k = alloc_kv(DType::BF16, head_dim);
        out.v = alloc_kv(DType::FP16, head_dim);
    } else {
        out.k        = alloc_kv(DType::FP8_E4M3FN, head_dim);
        out.v        = alloc_kv(DType::FP8_E4M3FN, head_dim);
        out.k_scales = alloc_kv(DType::U8, head_dim / 128);
        out.v_scales = alloc_kv(DType::U8, head_dim / 128);
    }
    return out;
}

template <class Allocator>
AttentionBuffers attention(Allocator& a, const TextConfig& c, bool is_swa,
                           KvCacheStorage swa_storage, std::int32_t width, std::int32_t batch,
                           bool ragged, bool need_chunk_kv) {
    const auto& attn  = c.attention(is_swa);
    const int columns = dimension(std::uint64_t(width) * batch);
    const int q_dim   = dimension(attn.query_width());
    const int r_dim   = dimension(c.rel_width(is_swa));
    const int kv_dim  = dimension(attn.kv_width());
    const int hq      = dimension(attn.heads);
    AttentionBuffers out;
    out.q     = a.alloc(DType::BF16, {q_dim, columns});
    out.r     = a.alloc(DType::BF16, {r_dim, columns});
    out.gate  = a.alloc(DType::BF16, {q_dim, columns});
    out.k     = a.alloc(DType::BF16, {kv_dim, columns});
    out.v     = a.alloc(DType::BF16, {kv_dim, columns});
    out.q_out = a.alloc(DType::BF16, {q_dim, columns});
    out.r_out = a.alloc(DType::BF16, {r_dim, columns});
    out.sink  = a.alloc(DType::BF16, {hq, columns});
    if (!is_swa) {
        const int dyn = dimension(std::uint64_t(c.sconv_kernel_size) * attn.kv_heads);
        out.k_dyn     = a.alloc(DType::BF16, {dyn, columns});
        out.v_dyn     = a.alloc(DType::BF16, {dyn, columns});
    }
    out.out = a.alloc(DType::BF16, {q_dim, columns});
    if (is_swa && need_chunk_kv) {
        out.chunk_kv = chunk_kv(a, swa_storage, dimension(attn.head_dim), dimension(attn.kv_heads),
                                width, batch, ragged);
    }
    return out;
}

template <class Allocator>
MoeBuffers moe(Allocator& a, const TextConfig& c, std::int32_t columns) {
    const int k = dimension(c.moe.top_k);
    return {a.alloc(DType::I32, {k, columns}), a.alloc(DType::FP32, {k, columns})};
}

} // namespace workspace
} // namespace ninfer::models::inkling2::execution
