#pragma once

#include "ops/moe/kernels/cpu/invocation.h"

namespace ninfer::ops::detail::moe_cpu::groupwise_int {

inline constexpr int hidden = 2048, intermediate = 512, experts = 256, top_k = 8;
inline constexpr int max_token_block = 8;

enum class FirstChunkPreparation { Caller, Worker };

struct Kernel {
    CpuWork run;
    FirstChunkPreparation first_chunk;
};

struct Plan {
    const Kernel* kernel;
    int token_block;
    int down_bits;
    int max_columns;
};

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint16_t* scales;
};

struct MatrixBank {
    Matrix base;
    std::size_t code_stride;
    std::size_t high_stride;
    std::size_t scale_stride;

    [[nodiscard]] Matrix matrix(int expert) const noexcept {
        return {base.codes + expert * code_stride,
                base.high ? base.high + expert * high_stride : nullptr,
                reinterpret_cast<const std::uint16_t*>(
                    reinterpret_cast<const std::byte*>(base.scales) + expert * scale_stride)};
    }
};

struct Group {
    Matrix gate_up;
    Matrix down;
    int count;
    int tokens[max_token_block];
    int slots[max_token_block];
    float coefficients[max_token_block];
};

struct Task {
    int group_index;
    int first_column;
    int columns;
};

struct Scratch {
    Group* groups;
    Task* tasks;
    float* input;
    float* activation;
};

struct Work {
    Invocation args;
    Plan plan;
    MatrixBank gate_up;
    MatrixBank down;
    Scratch scratch;
    int group_count       = 0;
    int task_count        = 0;
    bool has_four_columns = false;
};

void prepare_chunk(Work&, int token_begin, int columns) noexcept;
extern const Kernel avx2;
extern const Kernel avx512;
extern const Kernel avx512_bf16;
std::size_t capacity(int tokens, int token_block);
void launch(const Invocation&, const Plan&, MemoryArena&, CpuExecutionView);

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
