#include "ops/moe/kernels/cpu/groupwise_int/work.h"
#include "core/layout.h"

#include <algorithm>
#include <array>

namespace ninfer::ops::detail::moe_cpu::groupwise_int {
namespace {
template <class Arena>
Scratch allocate(Arena& arena, int tokens, int token_block) {
    const int columns = std::min(token_block, tokens);
    return {static_cast<Group*>(arena.alloc_bytes(sizeof(Group) * top_k * columns).data),
            static_cast<Task*>(arena.alloc_bytes(sizeof(Task) * top_k * columns).data),
            static_cast<float*>(arena.alloc(DType::FP32, {hidden, columns}).data),
            static_cast<float*>(arena.alloc(DType::FP32, {intermediate, top_k * columns}).data)};
}

MatrixBank matrix_bank(const WeightBankView& bank) {
    const auto& layout = bank.layout;
    return {{static_cast<const std::uint8_t*>(bank.plane(layout.codes, 0).data),
             static_cast<const std::uint8_t*>(bank.plane(layout.high, 0).data),
             static_cast<const std::uint16_t*>(bank.plane(layout.scales, 0).data)},
            layout.codes.expert_stride,
            layout.high.expert_stride,
            layout.scales.expert_stride};
}
} // namespace

std::size_t capacity(int tokens, int token_block) {
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, tokens, token_block);
    return layout.peak_bytes(1);
}

void launch(const Invocation& args, const Plan& plan, MemoryArena& arena,
            CpuExecutionView execution) {
    Work work{.args    = args,
              .plan    = plan,
              .gate_up = matrix_bank(args.experts.gate_up),
              .down    = matrix_bank(args.experts.down),
              .scratch = allocate(arena, args.tokens, plan.token_block)};
    if (plan.kernel->first_chunk == FirstChunkPreparation::Caller) {
        prepare_chunk(work, 0, std::min(args.tokens, plan.token_block));
    }
    execution.run(plan.kernel->run, &work);
}

void prepare_chunk(Work& work, int token_begin, int columns) noexcept {
    std::array<int, experts> indexes;
    indexes.fill(-1);
    work.group_count = work.task_count = 0;
    work.has_four_columns              = false;
    for (int t = 0; t < columns; ++t) {
        const auto offset = static_cast<std::size_t>(token_begin + t) * top_k;
        for (int j = 0; j < top_k; ++j) {
            const int expert = work.args.ids[offset + j];
            if (!participates(work.args.subset, work.args.mask, expert)) { continue; }
            auto& index = indexes[expert];
            if (index < 0) {
                index         = work.group_count++;
                auto& group   = work.scratch.groups[index];
                group.gate_up = work.gate_up.matrix(expert);
                group.down    = work.down.matrix(expert);
                group.count   = 0;
            }
            auto& group                = work.scratch.groups[index];
            const int member           = group.count++;
            group.tokens[member]       = t;
            group.slots[member]        = t * top_k + j;
            group.coefficients[member] = work.args.coefficients[offset + j];
        }
    }
    for (int g = 0; g < work.group_count; ++g) {
        for (int first = 0; first < work.scratch.groups[g].count;) {
            const int remaining = work.scratch.groups[g].count - first;
            const int tile      = remaining >= 4 ? 4 : remaining >= 2 ? 2 : 1;
            const int count     = std::min(work.plan.max_columns, tile);

            work.scratch.tasks[work.task_count++] = {
                .group_index = g, .first_column = first, .columns = count};
            work.has_four_columns |= count == 4;
            first += count;
        }
    }
}

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
