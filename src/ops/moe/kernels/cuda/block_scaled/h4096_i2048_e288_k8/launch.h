#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8 {

inline constexpr int kHidden = 4096, kIntermediate = 2048, kExperts = 288, kTopK = 8;

struct Plan {
    // Policies select these widths from slice M, but each tile covers assignments
    // belonging to one expert. Public entries split total M into slices <= 4096.
    QType format     = QType::NVFP4;
    int gate_columns = 0, down_columns = 0;
    bool vector                = false;
    bool atomic_output         = false;
    bool single_block_grouping = false;
    // GU consumes the original expert-major jobs; only down may reorder them.
    bool column_major_down = false;
    // Keep the original expert-major jobs when every expert has at most two jobs.
    bool preserve_balanced_down = false;

    int queued_ctas_per_sm = 6;
    float clamp_limit      = 0;

    int job_columns() const { return gate_columns > down_columns ? gate_columns : down_columns; }
};

struct Groups {
    int* counts;
    int* offsets;
    int* cursors;
    int* job_experts;
    int* job_columns;
    int* jobs;
    int* assignments;
    int* inverse;
};

struct Storage {
    std::uint8_t* input_codes   = nullptr;
    void* input_scales          = nullptr;
    std::uint8_t* product_codes = nullptr;
    void* product_scales        = nullptr;
    float* contributions        = nullptr;
    float* vector_activation    = nullptr;
    float* routed_sum           = nullptr;
    int* active_tokens          = nullptr;
    Groups groups{};
};

void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            ExpertSubset subset, MoeEpilogue mode, Tensor& destination, const Plan& plan,
            const Storage& storage, CudaExecutionView execution);

// May reorder job_experts/job_columns after GU has finished on the same stream.
// preserve_balanced keeps expert-major pairs when Device counts need at most two jobs each.
// Counts, offsets, packed assignments and all quantized payloads remain unchanged.
void launch_column_jobs(Groups groups, int tokens, int columns, bool preserve_balanced,
                        CudaExecutionView execution);

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8
