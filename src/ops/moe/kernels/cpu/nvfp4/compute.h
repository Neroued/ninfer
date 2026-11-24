#pragma once

#include "ops/moe/kernels/cpu/nvfp4/work.h"
#include "ops/common/numeric/math.h"
#include "ops/common/quant/nvfp4.h"

#include <array>
#include <algorithm>
#include <type_traits>

namespace ninfer::ops::detail::moe_cpu::nvfp4 {

inline constexpr int channel_block = 16;

// The grouping stage emits one through max_columns contiguous members per task.
template <int MaxColumns, class Function>
inline void with_columns(int columns, Function&& function) noexcept {
    switch (columns) {
    case 1:
        return function.template operator()<1>();
    case 2:
        return function.template operator()<2>();
    case 3:
        if constexpr (MaxColumns >= 4) { return function.template operator()<3>(); }
        break;
    case 4:
        if constexpr (MaxColumns >= 4) { return function.template operator()<4>(); }
        break;
    case 5:
        if constexpr (MaxColumns >= 8) { return function.template operator()<5>(); }
        break;
    case 6:
        if constexpr (MaxColumns >= 8) { return function.template operator()<6>(); }
        break;
    case 7:
        if constexpr (MaxColumns >= 8) { return function.template operator()<7>(); }
        break;
    case 8:
        if constexpr (MaxColumns >= 8) { return function.template operator()<8>(); }
        break;
    }
    __builtin_unreachable();
}

template <class Simd, int K, int Rows, int Columns, class Input>
inline void dot(const Matrix& matrix, const int (&rows)[Rows], const Input* const (&x)[Columns],
                float (&out)[Rows][Columns]) noexcept {
    using V = typename Simd::Float;
    V sums[Rows][Columns];
    const std::uint8_t* codes[Rows];
    const std::uint8_t* scales[Rows];
    for (int r = 0; r < Rows; ++r) {
        codes[r]  = matrix.codes + rows[r] * (K / 2);
        scales[r] = matrix.scales + common::nvfp4_scale_byte_offset(rows[r], 0, K);
        for (int c = 0; c < Columns; ++c) { sums[r][c] = Simd::zero(); }
    }
    for (int k = 0; k < K; k += Simd::lanes) {
        V input[Columns];
        for (int c = 0; c < Columns; ++c) {
            if constexpr (std::is_same_v<Input, float>) {
                input[c] = Simd::load(x[c] + k);
            } else {
                input[c] = Simd::bf16(x[c] + k);
            }
        }
        const int s = (k / 64) * 512 + ((k / 16) & 3);
        for (int r = 0; r < Rows; ++r) {
            const auto w = Simd::decode(codes[r] + k / 2, scales[r][s]);
            for (int c = 0; c < Columns; ++c) { sums[r][c] = Simd::fma(w, input[c], sums[r][c]); }
        }
    }
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { out[r][c] = Simd::sum(sums[r][c]) / matrix.divisor; }
    }
}

template <class Simd, int Columns>
inline void gate_up(Work& work, const Task& task, int channel, int begin,
                    const std::array<bool, max_token_block>& safe) noexcept {
    constexpr int rows = Columns > 4 ? 2 : Simd::rows;
    const auto& matrix = work.scratch.groups[task.group_index].gate_up;
    const std::uint16_t* input[Columns];
    bool use_bf16 = Simd::bf16_gate;
    for (int c = 0; c < Columns; ++c) {
        const int token = work.scratch.members[task.first_member + c].token;
        input[c]        = work.args.input + std::size_t(begin + token) * hidden;
        use_bf16 &= safe[token];
    }
    for (int i = channel; i < channel + channel_block; i += rows / 2) {
        int indices[rows];
        for (int r = 0; r < rows / 2; ++r) {
            indices[2 * r]     = i + r;
            indices[2 * r + 1] = intermediate + i + r;
        }
        float values[rows][Columns];
        if constexpr (Simd::bf16_gate) {
            if (use_bf16) {
                Simd::gate_dot(matrix, indices, input, values);
            } else {
                dot<Simd, hidden>(matrix, indices, input, values);
            }
        } else {
            dot<Simd, hidden>(matrix, indices, input, values);
        }
        for (int r = 0; r < rows / 2; ++r) {
            for (int c = 0; c < Columns; ++c) {
                const float gate = values[2 * r][c];
                const float up   = values[2 * r + 1][c];
                work.scratch.activation[(task.first_member + c) * intermediate + i + r] =
                    common::silu_stable(gate) * up;
            }
        }
    }
}

template <class Simd, int Columns>
inline void down(Work& work, const Task& task, int channel,
                 float (&out)[max_token_block][channel_block]) noexcept {
    constexpr int rows = Columns > 4 ? 2 : Simd::rows;
    const auto& matrix = work.scratch.groups[task.group_index].down;
    const float* input[Columns];
    for (int c = 0; c < Columns; ++c) {
        input[c] = work.scratch.activation + (task.first_member + c) * intermediate;
    }
    for (int i = 0; i < channel_block; i += rows) {
        int indices[rows];
        for (int r = 0; r < rows; ++r) { indices[r] = channel + i + r; }
        float values[rows][Columns];
        dot<Simd, intermediate>(matrix, indices, input, values);
        for (int c = 0; c < Columns; ++c) {
            const auto& member = work.scratch.members[task.first_member + c];
            for (int r = 0; r < rows; ++r) {
                out[member.token][i + r] += member.coefficient * values[r][c];
            }
        }
    }
}

template <class Simd>
void compute(void* argument, const CpuWorkerContext& worker) noexcept {
    auto& work = *static_cast<Work*>(argument);
    for (int begin = 0; begin < work.args.tokens;) {
        const int columns = std::min(work.plan.token_block, work.args.tokens - begin);
        if (begin != 0) {
            if (worker.rank() == 0) { prepare_chunk(work, begin, columns); }
            worker.barrier();
        }
        std::array<bool, max_token_block> safe{};
        if constexpr (Simd::bf16_gate) {
            for (int t = 0; t < columns; ++t) {
                safe[t] = Simd::safe(work.args.input + std::size_t(begin + t) * hidden);
            }
        }
        constexpr int blocks = intermediate / channel_block;
        const auto tasks     = worker.range(work.task_count * blocks);
        for (auto i = tasks.begin; i < tasks.end; ++i) {
            const auto& task  = work.scratch.tasks[i / blocks];
            const int channel = (i % blocks) * channel_block;
            with_columns<Simd::max_columns>(task.columns, [&]<int Columns>() {
                gate_up<Simd, Columns>(work, task, channel, begin, safe);
            });
        }
        worker.barrier();
        const auto outputs = worker.range(hidden / channel_block);
        for (auto block = outputs.begin; block < outputs.end; ++block) {
            const int channel = block * channel_block;
            float accumulated[max_token_block][channel_block];
            for (int t = 0; t < columns; ++t) { std::fill_n(accumulated[t], channel_block, 0.0F); }
            for (int i = 0; i < work.task_count; ++i) {
                const auto& task = work.scratch.tasks[i];
                with_columns<Simd::max_columns>(task.columns, [&]<int Columns>() {
                    down<Simd, Columns>(work, task, channel, accumulated);
                });
            }
            for (int t = 0; t < columns; ++t) {
                std::copy_n(accumulated[t], channel_block,
                            work.args.output + std::size_t(begin + t) * hidden + channel);
            }
        }
        begin += columns;
        if (begin < work.args.tokens) { worker.barrier(); }
    }
}

} // namespace ninfer::ops::detail::moe_cpu::nvfp4
