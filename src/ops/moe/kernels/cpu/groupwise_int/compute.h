#pragma once

#include "ops/moe/kernels/cpu/groupwise_int/work.h"
#include "ops/common/numeric/math.h"

#include <algorithm>
#include <type_traits>

namespace ninfer::ops::detail::moe_cpu::groupwise_int {

// Instantiated only in an ISA-specific C++ object. A decode feeds up to four columns;
// group scales are applied to vector partials, with one horizontal reduction per output.
template <class Simd, int Bits, int K, int Rows, int Columns, class Input>
inline void dot(const Matrix& matrix, const int (&rows)[Rows], const Input* const (&x)[Columns],
                float (&out)[Rows][Columns]) noexcept {
    using V = typename Simd::Float;
    V total[Rows][Columns];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { total[r][c] = Simd::zero(); }
    }
    for (int group = 0; group < K / 64; ++group) {
        V partial[Rows][Columns];
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) { partial[r][c] = Simd::zero(); }
        }
        for (int k = 0; k < 64; k += Simd::lanes) {
            V input[Columns];
            for (int c = 0; c < Columns; ++c) {
                if constexpr (std::is_same_v<Input, float>) {
                    input[c] = Simd::load(x[c] + group * 64 + k);
                } else {
                    input[c] = Simd::bf16(x[c] + group * 64 + k);
                }
            }
            for (int r = 0; r < Rows; ++r) {
                const auto* codes        = matrix.codes + rows[r] * (K / 2) + group * 32 + k / 2;
                const std::uint8_t* high = nullptr;
                if constexpr (Bits > 4) {
                    high = matrix.high + (rows[r] * (K / 8) + group * 8 + k / 8) * (Bits - 4);
                }
                const auto values = Simd::template decode<Bits>(codes, high);
                for (int c = 0; c < Columns; ++c) {
                    partial[r][c] = Simd::fma(values, input[c], partial[r][c]);
                }
            }
        }
        for (int r = 0; r < Rows; ++r) {
            const auto scale = Simd::scale(matrix.scales[rows[r] * (K / 64) + group]);
            for (int c = 0; c < Columns; ++c) {
                total[r][c] = Simd::fma(partial[r][c], scale, total[r][c]);
            }
        }
    }
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { out[r][c] = Simd::sum(total[r][c]); }
    }
}

inline constexpr int channel_block = 16; // One cache line of FP32 output per ownership unit.

template <class Simd, int Columns>
inline void gate_up(Work& work, const Group& group, int first, int channel, int begin,
                    unsigned bf16_columns) noexcept {
    constexpr int rows = Columns == 4 ? Simd::rows / 2 : Simd::rows;
    const std::conditional_t<Simd::bf16_gate, std::uint16_t, float>* input[Columns];
    for (int c = 0; c < Columns; ++c) {
        if constexpr (Simd::bf16_gate) {
            input[c] = work.args.input +
                       static_cast<std::size_t>(begin + group.tokens[first + c]) * hidden;
        } else {
            input[c] = work.scratch.input + group.tokens[first + c] * hidden;
        }
    }
    for (int i = channel; i < channel + channel_block; i += rows / 2) {
        int indices[rows];
        for (int r = 0; r < rows / 2; ++r) {
            indices[2 * r]     = i + r;
            indices[2 * r + 1] = intermediate + i + r;
        }
        float values[rows][Columns];
        if constexpr (Simd::bf16_gate) {
            unsigned mask = 0;
            for (int c = 0; c < Columns; ++c) { mask |= 1U << group.tokens[first + c]; }
            if ((bf16_columns & mask) == mask) {
                Simd::gate_dot(group.gate_up, indices, input, values);
            } else {
                dot<Simd, 4, hidden>(group.gate_up, indices, input, values);
            }
        } else {
            dot<Simd, 4, hidden>(group.gate_up, indices, input, values);
        }
        for (int r = 0; r < rows / 2; ++r) {
            for (int c = 0; c < Columns; ++c) {
                work.scratch.activation[group.slots[first + c] * intermediate + i + r] =
                    common::silu_stable(values[2 * r][c]) * values[2 * r + 1][c];
            }
        }
    }
}

template <class Simd, int DownBits, int Columns>
inline void down(Work& work, const Group& group, int first, int channel,
                 float (&accumulated)[max_token_block][channel_block]) noexcept {
    constexpr int rows = Columns == 4 ? Simd::rows / 2 : Simd::rows;
    const float* input[Columns];
    for (int c = 0; c < Columns; ++c) {
        input[c] = work.scratch.activation + group.slots[first + c] * intermediate;
    }
    for (int i = 0; i < channel_block; i += rows) {
        int indices[rows];
        for (int r = 0; r < rows; ++r) { indices[r] = channel + i + r; }
        float values[rows][Columns];
        dot<Simd, DownBits, intermediate>(group.down, indices, input, values);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < Columns; ++c) {
                accumulated[group.tokens[first + c]][i + r] +=
                    group.coefficients[first + c] * values[r][c];
            }
        }
    }
}

template <class Simd, int DownBits, bool FourColumns>
[[gnu::always_inline]] inline void compute_chunk(Work& work, const CpuWorkerContext& worker,
                                                 int begin, int columns,
                                                 unsigned bf16_columns) noexcept {
    constexpr int blocks = intermediate / channel_block;
    const auto tasks     = worker.range(work.task_count * blocks);
    for (auto index = tasks.begin; index < tasks.end; ++index) {
        const auto& task  = work.scratch.tasks[index / blocks];
        const auto& group = work.scratch.groups[task.group_index];
        const int channel = (index % blocks) * channel_block;
        if constexpr (FourColumns) {
            if (task.columns == 4) {
                gate_up<Simd, 4>(work, group, task.first_column, channel, begin, bf16_columns);
                continue;
            }
        }
        if (task.columns == 2) {
            gate_up<Simd, 2>(work, group, task.first_column, channel, begin, bf16_columns);
        } else {
            gate_up<Simd, 1>(work, group, task.first_column, channel, begin, bf16_columns);
        }
    }
    worker.barrier();
    const auto outputs = worker.range(hidden / channel_block);
    for (auto block = outputs.begin; block < outputs.end; ++block) {
        const int channel = block * channel_block;
        float accumulated[max_token_block][channel_block]{};
        if constexpr (FourColumns) {
            for (int index = 0; index < work.task_count; ++index) {
                const auto& task  = work.scratch.tasks[index];
                const auto& group = work.scratch.groups[task.group_index];
                if (task.columns == 4) {
                    down<Simd, DownBits, 4>(work, group, task.first_column, channel, accumulated);
                    continue;
                }
                if (task.columns == 2) {
                    down<Simd, DownBits, 2>(work, group, task.first_column, channel, accumulated);
                } else {
                    down<Simd, DownBits, 1>(work, group, task.first_column, channel, accumulated);
                }
            }
        } else {
            for (int g = 0; g < work.group_count; ++g) {
                const auto& group = work.scratch.groups[g];
                for (int first = 0; first < group.count; first += 2) {
                    if (first + 1 < group.count) {
                        down<Simd, DownBits, 2>(work, group, first, channel, accumulated);
                    } else {
                        down<Simd, DownBits, 1>(work, group, first, channel, accumulated);
                    }
                }
            }
        }
        for (int t = 0; t < columns; ++t) {
            auto* output =
                work.args.output + static_cast<std::size_t>(begin + t) * hidden + channel;
            for (int i = 0; i < channel_block; ++i) { output[i] = accumulated[t][i]; }
        }
    }
}

// Keep the wide-column specializations outside the worker's live register range.
template <class Simd, int DownBits, bool FourColumns>
[[gnu::noinline]] void compute_wide_chunk(Work& work, const CpuWorkerContext& worker, int begin,
                                          int columns, unsigned bf16_columns) noexcept {
    compute_chunk<Simd, DownBits, FourColumns>(work, worker, begin, columns, bf16_columns);
}

template <class Simd, int DownBits>
void compute(void* argument, const CpuWorkerContext& worker) noexcept {
    auto& work = *static_cast<Work*>(argument);
    for (int begin = 0; begin < work.args.tokens;) {
        const int columns = std::min(work.plan.token_block, work.args.tokens - begin);
        // Caller-prepared metadata is published with run(); worker-prepared metadata
        // is published with the input barrier. Later chunks reuse joined scratch.
        if (begin != 0 || Simd::first_chunk == FirstChunkPreparation::Worker) {
            if (worker.rank() == 0) { prepare_chunk(work, begin, columns); }
        }
        if constexpr (!Simd::bf16_gate) {
            const auto input_range = worker.range(hidden * columns / Simd::lanes);
            for (auto v = input_range.begin; v < input_range.end; ++v) {
                const auto index = v * Simd::lanes;
                Simd::store(
                    work.scratch.input + index,
                    Simd::bf16(work.args.input + static_cast<std::size_t>(begin) * hidden + index));
            }
        }
        if (begin != 0 || !Simd::bf16_gate) { worker.barrier(); }
        unsigned bf16_columns = 0;
        if constexpr (Simd::bf16_gate) {
            bf16_columns = Simd::safe_columns(
                work.args.input + static_cast<std::size_t>(begin) * hidden, columns);
        }
        if constexpr (Simd::max_columns == 4) {
            if (work.has_four_columns) {
                compute_wide_chunk<Simd, DownBits, true>(work, worker, begin, columns,
                                                         bf16_columns);
            } else {
                compute_wide_chunk<Simd, DownBits, false>(work, worker, begin, columns,
                                                          bf16_columns);
            }
        } else {
            compute_chunk<Simd, DownBits, false>(work, worker, begin, columns, bf16_columns);
        }
        begin += columns;
        // Protect shared scratch before the next chunk; the final join is provided by run().
        if (begin < work.args.tokens) { worker.barrier(); }
    }
}

template <class Simd>
constexpr Kernel make_kernel() {
    return {.run =
                [](void* argument, const CpuWorkerContext& worker) noexcept {
                    if (static_cast<Work*>(argument)->plan.down_bits == 6) {
                        compute<Simd, 6>(argument, worker);
                    } else {
                        compute<Simd, 5>(argument, worker);
                    }
                },
            .first_chunk = Simd::first_chunk};
}

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
