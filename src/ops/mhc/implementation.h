#pragma once

#include "ninfer/ops/mhc.h"

namespace ninfer::ops::detail::mhc {
inline constexpr int kHidden = 4096, kStreams = 4, kRows = 24, kWidth = kHidden * kStreams;
// Each hidden tile owns every stream, including all writes in an in-place post.
enum class Form { Pre, Post, ExpandPre, PostPre, PostMeanNorm };

struct Operands {
    const void* residual   = nullptr;
    const void* block      = nullptr;
    const float* post      = nullptr;
    const float* mix       = nullptr;
    const void* projection = nullptr;
    const void* base       = nullptr;
    const void* scale      = nullptr;
    const void* gamma      = nullptr;
    void* updated          = nullptr;
    void* output           = nullptr;
    float* next_post       = nullptr;
    float* next_mix        = nullptr;
    int columns            = 0;
    MhcPreParameters parameters{};
};

enum class Method {
    Projection,
    VectorProjection,
    ClusterProjection,
    Post,
    Mean,
    SplitMean,
    ClusterMean
};

struct Plan {
    Method method;
    int hidden_tile;
    int token_tile;
    int partials;
    int output_threads;
    int output_vector;
    int interval_end; // Inclusive end of this route; also owns interval capacity traversal.
};

Plan select(Form form, int columns, int sm_count);
std::size_t scratch_bytes(Plan plan, int columns);
std::size_t capacity(Form form, const MhcGeometry& geometry, int sm_count, int first, int last);
void validate_geometry(const MhcGeometry& geometry);
void launch(Form form, Operands operands, Plan plan, void* scratch, CudaExecutionView execution);
} // namespace ninfer::ops::detail::mhc
