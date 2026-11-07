#pragma once

#include "ninfer/ops/moe.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/schedule.h"
#include <climits>

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped {

inline constexpr int kHidden = 4096, kIntermediate = 2048, kExperts = 288, kTopK = 8;

enum class ExpertLoadRegime { All, ExcludeBroad, OnlyBroad, DenseWhenBroad };
enum class Nvfp4Arithmetic { DivideA4, MultiplyA4, MultiplyA16 };

struct Fp8PopulationTuning {
    static constexpr int kWideDownColumns = 256;
    int row_window                        = 0;
    int ordered_expert_limit              = 0;
    int wide_min_average                  = 0;
    bool wide_down                        = false;
    bool dense_down                       = false;
    // Zero partitions expert loads at k256_max_load: K256 for short groups,
    // K128 with the large column tiles for long groups.
    int k_step             = 128;
    int k256_max_load      = 0;
    int large_gate_columns = 0, large_down_columns = 0;
};

struct Nvfp4A16Tile {
    int gate_rows = 64, down_rows = 64;
    int k_step = 128, stages = 2;
    int queued_ctas_per_sm = 0;
    bool full_rows_when_mixed = false;
    bool gate_register_pipeline = false;
};

struct Nvfp4A16Bucket {
    int maximum_load = 0;
    int gate_columns = 0, down_columns = 0;
    Nvfp4A16Tile tile;
};

struct Nvfp4A16Dispatch {
    int maximum_load[5]{}, columns[5]{};
    int vector_limit = 0;
    bool full_rows_when_mixed = false;
#if defined(__CUDACC__)
    __host__ __device__
#endif
        int
        column_tile(int load) const {
        if (load <= vector_limit) return vector_limit;
        for (int i = 0; i < 5; ++i)
            if (columns[i] && load <= maximum_load[i]) return columns[i];
        return 0;
    }
#if defined(__CUDACC__)
    __host__ __device__
#endif
        int
        cohort_load(int low, int high) const {
        if (!high || high - low > 1) return 0;
        if (low == high) return high;
        // Neighboring populations use the wider of their two tiles. This
        // also covers a narrower tile becoming useful after a partial job.
        const int low_columns = column_tile(low), high_columns = column_tile(high);
        if (low_columns == high_columns) return 0;
        return low_columns > high_columns ? low : high;
    }
};

struct Nvfp4A16Tuning {
    Nvfp4A16Bucket buckets[5];
    int vector_max_load = 0;
    // Normalize actual FP32 product columns before one FP16 down contraction.
    bool fp16_product = false;

    Nvfp4A16Dispatch dispatch(bool gate) const {
        Nvfp4A16Dispatch out;
        out.vector_limit = vector_max_load;
        for (int i = 0; i < 5; ++i) {
            out.maximum_load[i] = buckets[i].maximum_load;
            out.columns[i]      = gate ? buckets[i].gate_columns : buckets[i].down_columns;
        }
        return out;
    }
};

struct Plan {
    // Policies select these widths from slice M, but each tile covers assignments
    // belonging to one expert. The entry policy owns the maximum slice size.
    QType format               = QType::NVFP4;
    Nvfp4Arithmetic arithmetic = Nvfp4Arithmetic::DivideA4;
    int intermediate           = kIntermediate;
    int gate_columns = 0, down_columns = 0;
    int dense_down_columns     = 0;
    bool direct_assignments    = false;
    bool atomic_output         = false;
    bool single_block_grouping = false;
    // GU consumes the original expert-major jobs; only down may reorder them.
    bool column_major_down = false;
    // Keep the original expert-major jobs when every expert has at most two jobs.
    bool preserve_balanced_down = false;
    // Preserve token order when every selected expert covers all input columns.
    bool token_ordered_groups = false;

    int queued_ctas_per_sm      = 6;
    int vector_gate_ctas_per_sm = 0;
    float clamp_limit           = 0;

    int pool_min_experts = 0, pool_expert_limit = 0, pool_columns = 0;
    int pool_row_window = 0, pool_queued_ctas_per_sm = 0;

    Fp8PopulationTuning fp8;
    Nvfp4Tuning nvfp4;
    Nvfp4A16Tuning a16;
    int group_columns = 0, small_expert_limit = 0;
    Nvfp4LoadSchedule medium, dense, broad;
    int broad_short_limit = 0, broad_long_limit = 0;
    int native_a4_vector_max_load = 0;
    // Quantize directly into expert-major rows after grouping. Both code and
    // scale planes use the packed assignment index consumed by contractions.
    bool native_a4_grouped_input = false;

    int job_columns() const {
        int columns = gate_columns > down_columns ? gate_columns : down_columns;
        if (arithmetic == Nvfp4Arithmetic::MultiplyA16)
            for (const auto& bucket : a16.buckets) {
                if (bucket.gate_columns > columns) columns = bucket.gate_columns;
                if (bucket.down_columns > columns) columns = bucket.down_columns;
            }
        if (fp8.k_step == 0) {
            if (fp8.large_gate_columns > columns) columns = fp8.large_gate_columns;
            if (fp8.large_down_columns > columns) columns = fp8.large_down_columns;
        }
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

// Compensated BF16 stores two [assignment,I] planes. The FP16 route first
// stages FP32 columns in values. SIMT consumes them directly; MMA columns use
// a normalized FP16 prefix and an exponent byte in their unused second half.
struct A16Product {
    std::uint16_t* values   = nullptr;
    std::uint16_t* residual = nullptr;
};

struct Storage {
    bool partial_output         = false;
    std::uint8_t* input_codes   = nullptr;
    void* input_scales          = nullptr;
    std::uint8_t* product_codes = nullptr;
    void* product_scales        = nullptr;
    float* contributions        = nullptr;
    float* direct_product       = nullptr;
    A16Product a16_product;
    float* routed_sum  = nullptr;
    int* active_tokens = nullptr;
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

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped
