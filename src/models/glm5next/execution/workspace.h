#pragma once

#include "models/glm5next/execution_shape.h"
#include "models/glm5next/execution/profile.h"
#include "models/common/vision/resources.h"

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

} // namespace workspace
} // namespace ninfer::models::glm5next::execution
