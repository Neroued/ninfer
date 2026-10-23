#pragma once

#include "ninfer/ops/gdn_input_proj.h"
#include <span>

namespace ninfer::ops::detail::gdn_input {

enum class Format { Q4Q5, Q8, Fp8, Nvfp4 };

struct Geometry {
    std::int32_t input_rows;
    std::int32_t value_rows;
    std::int32_t query_rows = 2048;
    std::int32_t key_rows   = 2048;

    [[nodiscard]] std::int32_t channels() const { return query_rows + key_rows + value_rows; }
};

struct WeightProfile {
    Format format;
    Geometry geometry;
    LinearPolicy policy;
};

struct WeightOperands : WeightProfile {
    const Weight* first;
    const Weight* second;
};

struct Batch {
    std::int32_t tokens;
    std::int32_t rows;

    [[nodiscard]] std::int32_t columns() const { return tokens * rows; }
};

enum class Publication { Snapshot, Update, Record };

struct HistoryDestination {
    void* data                         = nullptr;
    const std::int64_t* byte_offsets   = nullptr;
    const std::int32_t* snapshot_bases = nullptr;
    std::size_t field_offset           = 0;
    std::size_t snapshot_stride        = 0;
};

// Metadata-only admission is also used by capacity queries before execution.
WeightProfile describe_profile(const ProjectionProfile& weights);
WeightOperands describe_weights(const ProjectionWeights& weights);
void validate_weights(const WeightOperands& weights);
void validate_interval(std::int32_t first, std::int32_t last);
void validate_batch_interval(Publication publication, std::int32_t batch, std::int32_t first,
                             std::int32_t last);
Batch validate_convolution(const Tensor& x, const WeightOperands& weights,
                           const Tensor& convolution, const ConstStateFieldView& history,
                           const Tensor& valid_columns, const GdnConvOutputs& outputs,
                           Publication publication, const Tensor& destination);
void validate_storage(const WeightOperands& weights, std::span<const Tensor* const> tensors,
                      CudaExecutionView execution, MemorySpan scratch = {},
                      const Tensor* alias_first = nullptr, const Tensor* alias_second = nullptr);

} // namespace ninfer::ops::detail::gdn_input
