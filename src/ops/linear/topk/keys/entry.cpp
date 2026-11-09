#include "ops/linear/topk/keys/dispatch.h"
#include "ops/linear/common/operands.h"
#include "ops/common/host/operands.h"
#include <algorithm>
#include <array>
#include <climits>

namespace ninfer::ops {
namespace {
namespace tk             = detail::topk_keys;
constexpr const char* op = "linear_topk keys";

void validate(const Tensor& hidden, const Weight& head, int begin, int domain, LinearPolicy policy,
              const Tensor& keys, const MemoryArena& arena, CudaExecutionView execution) {
    (void)tk::select(head.qtype, head.n, head.k, policy, 1);
    if (begin < 0 || domain <= 0 || std::int64_t(begin) + head.n > INT_MAX)
        throw std::invalid_argument("linear_topk keys: invalid token range");
    namespace operand = common::operand;
    operand::device(arena.device(), execution.device(), op, "workspace");
    if (keys.ne[1] <= 0 || keys.ne[2] < 1 || keys.ne[2] > 8 ||
        std::int64_t(keys.ne[1]) * keys.ne[2] != hidden.ne[1])
        throw std::invalid_argument("linear_topk keys: expected [16,D,B] for [K,D*B]");
    operand::tensor(keys, DType::I64, {16, keys.ne[1], keys.ne[2]}, 16, op, "keys",
                    execution.device());
    if (head.qtype == QType::BF16)
        detail::validate_bf16_projection(hidden, head, {}, nullptr, execution, op);
    else {
        detail::validate_projection_operands(hidden, head, {}, nullptr, execution, op);
        if (head.qtype == QType::MXFP8)
            (void)common::validate_mxfp8_weight(head, op);
        else if (head.ndim != 2 || head.shape[0] != head.n || head.shape[1] != head.k ||
                 head.padded_shape[0] != head.n || head.padded_shape[1] != head.k ||
                 head.layout != QuantLayout::RowSplit || head.scale_dtype != DType::FP16 ||
                 head.group != 64 || head.group_size != 64 || head.qhigh || head.high_plane_bytes ||
                 head.payload_bytes < std::uint64_t(head.n) * (head.k / 2 + head.k / 64 * 2) ||
                 !operand::aligned(head.qdata, 16) || !operand::aligned(head.scales, 4))
            throw std::invalid_argument("linear_topk keys: invalid native Q4 weight");
    }
}

std::array<common::operand::Range, 4> live_ranges(const Tensor& x, const Weight& w,
                                                  const Tensor& keys) {
    const auto codes = std::size_t(w.n) * w.k * (w.qtype == QType::BF16 ? 2 : 1) /
                       (w.qtype == QType::Q4_G64_FP16 ? 2 : 1);
    const auto scales = w.qtype == QType::BF16
                            ? 0
                            : std::size_t(w.n) * (w.k / (w.qtype == QType::MXFP8 ? 32 : 64)) *
                                  (w.qtype == QType::MXFP8 ? 1 : 2);
    return {common::operand::range(x, op), common::operand::range(keys, op),
            common::operand::range(w.qdata, codes, op),
            common::operand::range(w.scales, scales, op)};
}
} // namespace

std::size_t linear_topk_workspace_capacity_bytes(QType type, int rows, int k, LinearPolicy policy,
                                                 int first, int last) {
    if (first < 1 || last < first)
        throw std::invalid_argument("linear_topk keys: invalid interval");
    (void)tk::select(type, rows, k, policy, 1);
    WorkspaceLayoutBuilder layout;
    for (int columns = 1; columns <= detail::kLinearTopKMaxChunkColumns; ++columns) {
        if (!detail::linear_topk_chunk_reachable(columns, first, last)) continue;
        auto scope = layout.scope();
        (void)tk::allocate(layout, tk::select(type, rows, k, policy, columns), rows, k, columns);
    }
    return layout.peak_bytes();
}

void linear_topk(const Tensor& hidden, const Weight& head, int begin, int domain,
                 LinearPolicy policy, Tensor& keys, MemoryArena& arena,
                 CudaExecutionView execution) {
    validate(hidden, head, begin, domain, policy, keys, arena, execution);
    const auto live = live_ranges(hidden, head, keys);
    common::operand::disjoint(live, op);
    if (begin >= domain) {
        CUDA_CHECK(cudaMemsetAsync(keys.data, 0, keys.bytes(), execution.stream()));
        return;
    }
    auto flat_keys = keys.view({16, hidden.ne[1]});
    for (int first = 0; first < hidden.ne[1];) {
        const int columns = std::min(detail::kLinearTopKMaxChunkColumns, hidden.ne[1] - first);
        const auto x      = hidden.slice(1, first, columns);
        auto out          = flat_keys.slice(1, first, columns);
        const auto plan   = tk::select(head.qtype, head.n, head.k, policy, columns);
        auto scope        = arena.scope();
        const auto work   = tk::allocate(arena, plan, head.n, head.k, columns);
        const auto check  = [&](const void* p, std::size_t bytes) {
            const auto range = common::operand::range(p, bytes, op);
            for (auto operand : live) common::operand::disjoint(range, operand, op);
        };
        for (const Tensor* t : {&work.candidates.partial_keys, &work.candidates.group_keys,
                                &work.candidates.secondary_keys, &work.candidates.group_done})
            check(t->data, t->bytes());
        if (plan.a8) {
            check(work.activation.codes, std::size_t(columns) * head.k);
            check(work.activation.scales, std::size_t(columns) * head.k / 32);
        }
        plan.producer(x, head, begin, domain, work.candidates, work.activation, execution.stream());
        tk::merge(work.candidates, out, execution.stream());
        first += columns;
    }
}
} // namespace ninfer::ops
