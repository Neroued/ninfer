#pragma once

#include "ops/moe/kernels/cuda/launch.h"

namespace ninfer::ops::detail::moe_cuda::groupwise_int {

struct GroupedInfo {
    int jobs;
    int direct;
    int spread;
};

struct GroupedStorage {
    int* local_ranks  = nullptr;
    int* tile_counts  = nullptr;
    int* tile_bases   = nullptr;
    int* offsets      = nullptr;
    int* job_experts  = nullptr;
    int* job_columns  = nullptr;
    int* inverse      = nullptr;
    int* tokens       = nullptr;
    GroupedInfo* info = nullptr;
};

struct TokenPlan {
    int paths_per_block     = 1;
    int output_rows         = 1;
    bool single_token       = false;
    bool programmatic_input = false;
};

struct SharedPlan {
    TokenPlan token;
    int columns = 0; // Zero selects SIMT; otherwise the MMA column tile.
};

struct GroupedPlan {
    int columns            = 0;
    bool adaptive          = false;
    int spread_numerator   = 0;
    int spread_denominator = 0;
    int queued_ctas_per_sm = 0;
    int shared_columns     = 0;
    int direct_numerator   = 0;
    int direct_denominator = 0;
};

struct Plan {
    TokenPlan token;
    GroupedPlan group;
    SharedPlan shared;
    bool grouped = false;
};

struct ExpertStorage {
    float* token_activation  = nullptr;
    void* grouped_activation = nullptr;
    void* grouped_io         = nullptr;
    void* shared_activation  = nullptr;
    float* routed_sum        = nullptr;
    GroupedStorage groups;
};

void launch_group_count(const ExpertRoutes& routes, ExpertSubset subset,
                        const GroupedStorage& storage, CudaExecutionView execution);
void launch_group_scan(const ExpertRoutes& routes, const GroupedPlan& plan,
                       const GroupedStorage& storage, CudaExecutionView execution);
void launch_group_index(const ExpertRoutes& routes, const GroupedStorage& storage,
                        CudaExecutionView execution);

void launch_token_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                         const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                         ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue epilogue,
                         Tensor& destination, const TokenPlan& plan, const ExpertStorage& storage,
                         const MoeExecutionHints& hints, CudaExecutionView execution,
                         bool adaptive = false);
void launch_grouped_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                           const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                           const ScaledSwiGLUTerm* shared, MoeEpilogue epilogue,
                           Tensor& destination, const GroupedPlan& plan,
                           const ExpertStorage& storage, CudaExecutionView execution);
void launch_shared_swiglu(const Tensor& x, const ScaledSwiGLUTerm& shared, MoeEpilogue epilogue,
                          Tensor& destination, const ExpertStorage& storage, const SharedPlan& plan,
                          CudaExecutionView execution);
} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
