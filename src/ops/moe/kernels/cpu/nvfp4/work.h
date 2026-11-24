#pragma once

#include "ops/moe/kernels/cpu/invocation.h"

namespace ninfer::ops::detail::moe_cpu::nvfp4 {

inline constexpr int hidden = 2560, intermediate = 640, experts = 512, top_k = 10;
inline constexpr int max_token_block = 256;

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    float divisor;
};

struct Group {
    Matrix gate_up;
    Matrix down;
    int first_member;
    int count;
};

struct Member {
    int token;
    float coefficient;
};

struct Task {
    int group_index;
    int first_member;
    int columns;
};

struct Scratch {
    Group* groups;
    Task* tasks;
    Member* members;
    float* activation;
};

struct Plan {
    CpuWork run;
    int token_block;
    int max_columns;
};

struct Work {
    Invocation args;
    Plan plan;
    Scratch scratch;
    int group_count = 0;
    int task_count  = 0;
};

std::size_t capacity(int tokens, int token_block);

void prepare_chunk(Work&, int begin, int columns) noexcept;
void launch(const Invocation&, const Plan&, MemoryArena&, CpuExecutionView);
void run_avx2(void*, const CpuWorkerContext&) noexcept;
void run_avx512(void*, const CpuWorkerContext&) noexcept;
void run_avx512_bf16(void*, const CpuWorkerContext&) noexcept;

} // namespace ninfer::ops::detail::moe_cpu::nvfp4
