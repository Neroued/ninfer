#include "ops/moe/kernels/cpu/nvfp4/work.h"
#include "core/layout.h"
#include "core/nvtx.h"

#include <algorithm>
#include <array>

namespace ninfer::ops::detail::moe_cpu::nvfp4 {
namespace {
template <class Arena>
Scratch allocate(Arena& arena, int tokens, int token_block) {
    const int assignments = top_k * std::min(tokens, token_block);
    return {
        static_cast<Group*>(arena.alloc_bytes(sizeof(Group) * std::min(experts, assignments)).data),
        static_cast<Task*>(arena.alloc_bytes(sizeof(Task) * assignments).data),
        static_cast<Member*>(arena.alloc_bytes(sizeof(Member) * assignments).data),
        static_cast<float*>(arena.alloc(DType::FP32, {intermediate, assignments}).data)};
}

Matrix matrix(const WeightBankView& bank, int expert) noexcept {
    const auto& layout = bank.layout;
    const auto* base   = reinterpret_cast<const std::uint8_t*>(bank.parent->data);
    return {base + layout.codes.offset + expert * layout.codes.expert_stride,
            base + layout.scales.offset + expert * layout.scales.expert_stride,
            bank.parent->weight_divisors[layout.divisor_begin + expert * layout.divisor_stride]};
}
} // namespace

std::size_t capacity(int tokens, int token_block) {
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, tokens, token_block);
    return layout.peak_bytes(1);
}

void prepare_chunk(Work& work, int begin, int columns) noexcept {
    std::array<int, experts> indices;
    indices.fill(-1);
    work.group_count = work.task_count = 0;
    const auto& args                   = work.args;
    for (int t = 0; t < columns; ++t) {
        for (int j = 0; j < top_k; ++j) {
            const int expert = args.ids[std::size_t(begin + t) * top_k + j];
            if (!participates(args.subset, args.mask, expert)) { continue; }
            auto& index = indices[expert];
            if (index < 0) {
                index                      = work.group_count++;
                work.scratch.groups[index] = {matrix(args.experts.gate_up, expert),
                                              matrix(args.experts.down, expert), 0, 0};
            }
            ++work.scratch.groups[index].count;
        }
    }
    int offset = 0;
    for (int g = 0; g < work.group_count; ++g) {
        auto& group        = work.scratch.groups[g];
        group.first_member = offset;
        offset += group.count;
        group.count = 0;
    }
    for (int t = 0; t < columns; ++t) {
        for (int j = 0; j < top_k; ++j) {
            const auto slot  = std::size_t(begin + t) * top_k + j;
            const int expert = args.ids[slot];
            if (!participates(args.subset, args.mask, expert)) { continue; }
            auto& group = work.scratch.groups[indices[expert]];
            work.scratch.members[group.first_member + group.count++] = {t, args.coefficients[slot]};
        }
    }
    for (int g = 0; g < work.group_count; ++g) {
        const auto& group = work.scratch.groups[g];
        for (int first = 0; first < group.count;) {
            const int count = std::min(work.plan.max_columns, group.count - first);
            work.scratch.tasks[work.task_count++] = {
                .group_index = g, .first_member = group.first_member + first, .columns = count};
            first += count;
        }
    }
}

void launch(const Invocation& args, const Plan& plan, MemoryArena& arena,
            CpuExecutionView execution) {
    Work work{
        .args = args, .plan = plan, .scratch = allocate(arena, args.tokens, plan.token_block)};
    prepare_chunk(work, 0, std::min(args.tokens, plan.token_block));
    nvtx::ScopedRange compute(nvtx::Name::MoeCpuCompute, nvtx::Category::Moe, work.group_count);
    execution.run(plan.run, &work);
}

} // namespace ninfer::ops::detail::moe_cpu::nvfp4
