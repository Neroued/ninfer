#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled {

inline constexpr int kHidden = 2560, kIntermediate = 640, kExperts = 512, kTopK = 10;
inline constexpr int kScoreRows = (kExperts + 4) / 4 * 4;

enum class SharedAlgorithm { Vector, Mma };

struct Plan {
    // Policies select these widths from slice M, but each tile covers assignments
    // belonging to one expert. Public entries split total M into slices <= 4096.
    QType format     = QType::NVFP4;
    int gate_columns = 0, down_columns = 0;
    bool vector                = false;
    bool atomic_output         = false;
    bool single_block_grouping = false;
    // GU consumes the original expert-major jobs; only down may reorder them.
    bool column_major_down      = false;
    bool preserve_balanced_down = false;

    SharedAlgorithm shared_algorithm = SharedAlgorithm::Vector;
    bool shared_split_activation     = true;
    // Complete may widen shared/control tiles independently of routed-shared.
    bool network_topk      = false;
    int shared_columns     = 32;
    int queued_ctas_per_sm = 0;

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
    void* shared_activation     = nullptr;
    Groups groups{};
};

struct Preparation {
    bool shared_activation = false;
    bool route_counts      = false;
};

void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue mode,
            Tensor& destination, const Plan& plan, const Storage& storage,
            CudaExecutionView execution, Preparation prepared = {});
void launch_shared(const Tensor& x, const ScaledSwiGLUTerm& shared, const ExpertRoutes& routes,
                   MoeEpilogue mode, Tensor& destination, const Storage& storage, const Plan& plan,
                   CudaExecutionView execution, bool prepared = false);

void launch_control_shared(const Tensor& x, const MoeWeights& weights, const ExpertRoutes& routes,
                           ExpertSubset subset, const Tensor& shared, float* scores,
                           const Storage& storage, const Plan& plan, CudaExecutionView execution);

// Overwrites job_experts/job_columns after GU has finished on the same stream.
// Counts, offsets, packed assignments and all quantized payloads remain unchanged.
void launch_column_jobs(Groups groups, int tokens, int columns, bool preserve_balanced,
                        CudaExecutionView execution);

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
