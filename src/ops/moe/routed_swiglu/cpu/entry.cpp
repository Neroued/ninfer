#include "ninfer/ops/moe.h"
#include "ops/common/host/operands.h"
#include "ops/moe/routed_swiglu/cpu/dispatch.h"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace impl            = detail::moe_routed_cpu;
namespace kernel          = detail::moe_cpu;
namespace operand         = common::operand;
constexpr int max_experts = 512;
constexpr auto op         = "routed_swiglu CPU";

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

class Operands {
public:
    Operands(const Tensor& x, Tensor& out, ConstMemorySpan scratch, int hidden) {
        operand::tensor(x, DType::BF16, {hidden, x.ne[1]}, 2, op, "x", Device::cpu());
        operand::tensor(out, DType::FP32, {hidden, x.ne[1]}, 4, op, "destination", Device::cpu());
        operand::device(scratch.device, Device::cpu(), op, "workspace");
        output_ = operand::range(out, op);
        if (scratch.bytes) {
            scratch_ = operand::range(scratch.data, scratch.bytes, op);
            operand::disjoint(output_, scratch_, op);
        }
        read(x.data, x.bytes());
    }

    void tensor(const Tensor& value, DType dtype, std::initializer_list<int> shape,
                const char* name) const {
        operand::tensor(value, dtype, shape, dtype_size(dtype), op, name, Device::cpu());
        read(value.data, value.bytes());
    }

    void bank(const WeightBankView& bank, QType type, QuantLayout format_layout, int rows,
              int columns, int count, const std::array<bool, max_experts>& selected) const {
        require(bank.parent && bank.parent->data, "routed_swiglu CPU: missing weight bank");
        operand::device(bank.parent->device, Device::cpu(), op, "weight bank");
        const auto& layout       = bank.layout;
        const auto& geometry     = bank.parent->geometry;
        const auto elements      = std::size_t(rows) * columns;
        const bool nvfp4         = type == QType::NVFP4;
        const auto native_layout = layout.layout == QuantLayout::BatchedBlockScaleK16M128x4
                                       ? QuantLayout::BlockScaleK16M128x4
                                       : layout.layout;
        const auto high_bytes    = type == QType::Q5_G64_FP16   ? elements / 8
                                   : type == QType::Q6_G64_FP16 ? elements / 4
                                                                : 0;
        require(layout.format == type && native_layout == format_layout &&
                    layout.region.count == unsigned(count) &&
                    layout.region.rows == unsigned(rows) && layout.columns == unsigned(columns),
                "routed_swiglu CPU: weight bank differs from the profile");
        require(geometry.group_size == (nvfp4 ? 16 : 64) &&
                    geometry.padded_columns == unsigned(columns) &&
                    geometry.scale_block_rows == 1 && layout.has_divisors == nvfp4,
                "routed_swiglu CPU: invalid weight geometry");
        require(layout.codes.expert_bytes == elements / 2 &&
                    layout.high.expert_bytes == high_bytes &&
                    layout.scales.expert_bytes == elements / (nvfp4 ? 16 : 32) &&
                    layout == describe_weight_bank(geometry, layout.region),
                "routed_swiglu CPU: invalid native weight planes");
        require(bank.parent->weight_divisors.size() == geometry.divisor_count,
                "routed_swiglu CPU: invalid divisor storage");
        for (int expert = 0; expert < count; ++expert) {
            if (!selected[expert]) { continue; }
            for (const auto& plane : {layout.codes, layout.high, layout.scales}) {
                if (!plane.expert_bytes) { continue; }
                const auto memory = bank.plane(plane, expert);
                require(operand::aligned(memory.data, 16),
                        "routed_swiglu CPU: weight plane must be 16-byte aligned");
                read(memory.data, memory.bytes);
            }
            if (nvfp4) {
                const auto* divisor = bank.parent->weight_divisors.data() + layout.divisor_begin +
                                      expert * layout.divisor_stride;
                require(std::isfinite(*divisor) && *divisor > 0,
                        "routed_swiglu CPU: invalid expert divisor");
                read(divisor, sizeof(float));
            }
        }
    }

private:
    void read(const void* pointer, std::size_t bytes) const {
        if (!bytes) { return; }
        const auto range = operand::range(pointer, bytes, op);
        operand::disjoint(output_, range, op);
        if (scratch_.begin != scratch_.end) { operand::disjoint(scratch_, range, op); }
    }

    operand::Range output_{}, scratch_{};
};

} // namespace

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   ExpertSubsetKind subset, std::int32_t first,
                                                   std::int32_t last) {
    const auto& shape = impl::shape_dispatch(profile);
    impl::validate_capacity(subset, first, last);
    return subset == ExpertSubsetKind::None ? 0 : shape.capacity(last);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CpuSwiGLUExperts& experts, const ExpertRoutes& routes, ExpertSubset subset,
                   Tensor& destination, MemoryArena& workspace, CpuExecutionView execution) {
    const auto& shape = impl::shape_dispatch(profile);
    const int tokens  = x.ne[1];
    impl::validate_capacity(subset.kind, tokens, tokens);
    const auto capacity = subset.kind == ExpertSubsetKind::None ? 0 : shape.capacity(tokens);
    auto scope          = workspace.scope();
    MemoryArena scratch(capacity ? workspace.alloc_bytes(capacity)
                                 : MemorySpan{nullptr, 0, workspace.device()});
    const Operands operands(x, destination, scratch.storage(), profile.ffn.hidden);
    if (subset.kind == ExpertSubsetKind::None) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }

    operands.tensor(routes.ids, DType::I32, {profile.top_k, tokens}, "route IDs");
    operands.tensor(routes.coefficients, DType::FP32, {profile.top_k, tokens}, "coefficients");
    const std::int32_t* mask = nullptr;
    if (subset.kind == ExpertSubsetKind::MaskOnes || subset.kind == ExpertSubsetKind::MaskZeros) {
        operands.tensor(subset.mask, DType::I32, {profile.experts}, "mask");
        mask = static_cast<const std::int32_t*>(subset.mask.data);
        for (int e = 0; e < profile.experts; ++e) {
            require(mask[e] == 0 || mask[e] == 1, "routed_swiglu CPU: mask must be binary");
        }
    }
    const auto* ids          = static_cast<const std::int32_t*>(routes.ids.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    std::array<bool, max_experts> selected{};
    bool has_work = false;
    for (int t = 0; t < tokens; ++t) {
        const auto offset = std::size_t(t) * profile.top_k;
        for (int j = 0; j < profile.top_k; ++j) {
            const int expert = ids[offset + j];
            require(expert >= 0 && expert < profile.experts &&
                        std::isfinite(coefficients[offset + j]),
                    "routed_swiglu CPU: invalid route ID or coefficient");
            for (int previous = 0; previous < j; ++previous) {
                require(expert != ids[offset + previous], "routed_swiglu CPU: duplicate route ID");
            }
            if (kernel::participates(subset.kind, mask, expert)) {
                selected[expert] = true;
                has_work         = true;
            }
        }
    }
    if (!has_work) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }
    operands.bank(experts.gate_up, profile.ffn.gate_up_type, profile.ffn.gate_up_layout,
                  2 * profile.ffn.intermediate, profile.ffn.hidden, profile.experts, selected);
    operands.bank(experts.down, profile.ffn.down_type, profile.ffn.down_layout, profile.ffn.hidden,
                  profile.ffn.intermediate, profile.experts, selected);
    const kernel::Invocation args{.input        = static_cast<const std::uint16_t*>(x.data),
                                  .ids          = ids,
                                  .coefficients = coefficients,
                                  .experts      = experts,
                                  .subset       = subset.kind,
                                  .mask         = mask,
                                  .output       = static_cast<float*>(destination.data),
                                  .tokens       = tokens};
    shape.launch(args, scratch, execution);
}

} // namespace ninfer::ops
