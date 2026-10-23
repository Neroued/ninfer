#pragma once

#include "ninfer/ops/moe.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/schedule.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8 {

inline constexpr int kHidden = 4096, kIntermediate = 2048, kExperts = 288, kTopK = 8;

enum class ExpertLoadRegime { All, ExcludeBroad, OnlyBroad, DenseWhenBroad };

struct Fp8PopulationTuning {
    static constexpr int kWideDownColumns = 256;
    int row_window                        = 0;
    int ordered_expert_limit              = 0;
    int wide_min_average                  = 0;
    bool wide_down                        = false;
    bool dense_down                       = false;
};

struct Plan {
    // Policies select these widths from slice M, but each tile covers assignments
    // belonging to one expert. The entry policy owns the maximum slice size.
    QType format     = QType::NVFP4;
    int gate_columns = 0, down_columns = 0;
    bool vector                = false;
    bool atomic_output         = false;
    bool single_block_grouping = false;
    // GU consumes the original expert-major jobs; only down may reorder them.
    bool column_major_down = false;
    // Keep the original expert-major jobs when every expert has at most two jobs.
    bool preserve_balanced_down = false;

    int queued_ctas_per_sm      = 6;
    int vector_gate_ctas_per_sm = 0;
    float clamp_limit           = 0;

    Fp8PopulationTuning fp8;
    Nvfp4Tuning nvfp4;
    int group_columns = 0, small_expert_limit = 0;
    Nvfp4LoadSchedule medium, dense, broad;
    int broad_short_limit = 0, broad_long_limit = 0;

    int job_columns() const {
        int columns = gate_columns > down_columns ? gate_columns : down_columns;
        if (fp8.wide_down && Fp8PopulationTuning::kWideDownColumns > columns)
            columns = Fp8PopulationTuning::kWideDownColumns;
        return columns > group_columns ? columns : group_columns;
    }
};

struct Groups {
    int* counts;
    int* offsets;
    int* cursors;
    int* job_experts;
    int* job_columns;
    // Total jobs, maximum jobs per expert; FP8 also publishes active experts and
    // the maximum assignments per expert, independent of the selected job width.
    int* jobs;
    int* assignments;
    int* inverse;
    int* load_range;
};

struct Storage {
    bool partial_output         = false;
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
