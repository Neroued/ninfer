#pragma once

#include "models/common/vision/resources.h"
#include "models/glm5next/execution/profile.h"
#include "models/glm5next/execution_shape.h"

namespace ninfer::models::glm5next::execution {

struct WorkspacePlan {
    std::size_t prefill = 0, ordinary = 0, verify = 0, draft = 0, history = 0, finish = 0;
    std::size_t general_bytes = 0, bytes = 0;
    std::optional<models::vision::Capacity> vision;
};

[[nodiscard]] WorkspacePlan plan_workspace(const ModelProfile&, const ExecutionShape&,
                                           std::int32_t sm_count, std::size_t vision_budget);

namespace workspace {

struct Roots {
    Tensor residual, input, output;
    std::array<ops::MhcCoefficients, 2> coefficients;
};

struct AttentionBuffers {
    Tensor input, output, gathered;
};

template <class Arena>
AttentionBuffers attention(Arena& a, int hidden, int columns, int ranks) {
    return {a.alloc(DType::BF16, {hidden, columns}), a.alloc(DType::BF16, {hidden, columns}),
            a.alloc(DType::BF16, {hidden, columns * ranks})};
}

template <class Arena>
Roots roots(Arena& a, const TextConfig& c, int columns) {
    const auto h = dimension(c.hidden), s = dimension(c.mhc.streams);
    return {a.alloc(DType::BF16, {h, s, columns}),
            a.alloc(DType::BF16, {h, columns}),
            a.alloc(DType::BF16, {h, columns}),
            {{{a.alloc(DType::FP32, {s, columns}), a.alloc(DType::FP32, {s, s, columns})},
              {a.alloc(DType::FP32, {s, columns}), a.alloc(DType::FP32, {s, s, columns})}}}};
}

struct KdaBuffers {
    Tensor query, key, value, beta, forget_lowrank, gate_lowrank, forget, gate, output, normalized;
};

template <class Arena>
KdaBuffers kda(Arena& a, const ModelProfile& p, int columns, bool records) {
    const auto& c   = p.config.text.kda;
    const int width = dimension(std::uint64_t(p.local.kda_heads.size) * c.head_dim);
    const int low   = dimension(c.gate_lowrank);
    return {a.alloc(DType::BF16, {width, columns}),
            records ? Tensor{} : a.alloc(DType::BF16, {width, columns}),
            records ? Tensor{} : a.alloc(DType::BF16, {width, columns}),
            a.alloc(DType::BF16, {dimension(p.local.kda_heads.size), columns}),
            a.alloc(DType::BF16, {low, columns}),
            a.alloc(DType::BF16, {low, columns}),
            a.alloc(DType::BF16, {width, columns}),
            a.alloc(DType::BF16, {width, columns}),
            a.alloc(DType::BF16, {width, columns}),
            a.alloc(DType::BF16, {width, columns})};
}

struct MlaBuffers {
    Tensor query_lowrank, query, index_query, coefficients, output;
    ops::KpoolBlockSelection selection;
};

template <class Arena>
MlaBuffers mla(Arena& a, const TextConfig& c, int columns, bool frozen_selection,
               bool own_selection = true) {
    const auto& m = c.mla;
    MlaBuffers out{
        a.alloc(DType::BF16, {dimension(m.query_lowrank), columns}),
        a.alloc(DType::BF16, {dimension(m.query_width()), columns}),
        frozen_selection ? Tensor{}
                         : a.alloc(DType::BF16, {dimension(m.index_query_width()), columns}),
        frozen_selection ? Tensor{} : a.alloc(DType::FP32, {dimension(m.index_heads), columns}),
        a.alloc(DType::BF16, {dimension(m.output_width()), columns}),
        {}};
    if (own_selection)
        out.selection = {a.alloc(DType::I32, {dimension(m.pool_budget), columns}),
                         a.alloc(DType::I32, {columns}), a.alloc(DType::I32, {columns})};
    return out;
}

struct MtpStemBuffers {
    Tensor embedding, normalized;
};

struct MtpRoots {
    Tensor residual, input, output;
};

template <class Arena>
MtpRoots mtp_roots(Arena& a, const TextConfig& c, int columns) {
    const int h = dimension(c.hidden);
    return {a.alloc(DType::BF16, {h, columns}), a.alloc(DType::BF16, {h, columns}),
            a.alloc(DType::BF16, {h, columns})};
}

template <class Arena>
MtpStemBuffers mtp_stem(Arena& a, const TextConfig& c, int columns) {
    return {a.alloc(DType::BF16, {dimension(c.hidden), columns}),
            a.alloc(DType::BF16, {dimension(2ULL * c.hidden), columns})};
}

struct DraftBranch {
    Tensor prepared, delta, partial;
};

template <class Arena>
DraftBranch draft_branch(Arena& a, const ModelProfile& p, int width, int batch) {
    const int h      = dimension(p.config.text.hidden);
    const int groups = dimension(p.config.text.hidden / p.config.draft->dflash2->conv_group_size);
    const int taps   = dimension(p.config.draft->dflash2->conv_kernel_size);
    return {a.alloc(DType::BF16, {h, width, batch}),
            a.alloc(DType::BF16, {groups, taps, width, batch}),
            a.alloc(DType::BF16, {h, width, batch})};
}

struct DraftAttentionBuffers {
    DraftBranch branch;
    Tensor key, value, query, output;
};

template <class Arena>
DraftAttentionBuffers draft_attention(Arena& a, const ModelProfile& p, int width, int batch) {
    const auto& g = p.draft->attention;
    return {draft_branch(a, p, width, batch),
            a.alloc(DType::BF16, {g.head_dim, g.kv_heads, width, batch}),
            a.alloc(DType::BF16, {g.head_dim, g.kv_heads, width, batch}),
            a.alloc(DType::BF16, {g.head_dim, g.query_heads, width, batch}),
            a.alloc(DType::BF16, {g.head_dim, g.query_heads, width, batch})};
}

struct DraftFinishBuffers {
    Tensor hidden, projected, keys, gathered;
};

template <class Arena>
DraftFinishBuffers draft_finish(Arena& a, const ModelProfile& p, int steps, int batch) {
    const auto& d = *p.config.draft->dflash2;
    return {
        a.alloc(DType::BF16, {dimension(p.config.text.hidden), steps * batch}),
        a.alloc(DType::BF16, {dimension(d.selector_rank), steps, batch}),
        a.alloc(DType::I64, {dimension(d.selector_top_k), steps, batch}),
        a.alloc(DType::I64, {dimension(d.selector_top_k), steps, batch, dimension(p.local.size)})};
}

struct DraftContextBuffers {
    Tensor projected, gathered, normalized;
};

template <class Arena>
DraftContextBuffers draft_context(Arena& a, const ModelProfile& p, int columns) {
    return {a.alloc(DType::BF16, {p.draft->feature_projection.n, columns}),
            a.alloc(DType::BF16, {p.draft->feature_projection.n, columns, dimension(p.local.size)}),
            a.alloc(DType::BF16, {dimension(p.config.text.hidden), columns})};
}

} // namespace workspace
} // namespace ninfer::models::glm5next::execution
