#pragma once
#include "core/weight.h"
#include "ops/linear/topk/workspace.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class LinearTopKProfile { Q8Full, Fp8Full, Q4Mapped };
using LinearTopKProducer = void (*)(const Tensor&, const Weight&, const Tensor*,
                                    const LinearTopKWorkspace&, cudaStream_t);

struct LinearTopKPlan {
    LinearTopKProducer producer;
    int rows_per_producer;
    int row_window = 0;
};

LinearTopKProfile select_linear_topk_profile(QType type, int rows, int k);
int linear_topk_large_threshold(LinearTopKProfile);
LinearTopKPlan select_linear_topk(LinearTopKProfile profile, int columns);
std::size_t linear_topk_capacity(LinearTopKProfile profile, int head_rows, int first, int last);
void validate_topk_io(const Tensor& hidden, const Tensor& ids, const Tensor& scores);
void validate_topk_head(const Weight& head, LinearTopKProfile profile, const Tensor* id_map);
void validate_topk_storage(const Weight& head, const Tensor& hidden, const Tensor& ids,
                           const Tensor& scores, const Tensor* id_map,
                           const LinearTopKWorkspace& scratch, const Tensor& accumulated);
void launch_linear_topk_merge(const LinearTopKWorkspace&, Tensor& ids, Tensor& scores,
                              cudaStream_t);
LinearTopKPlan select_topk_q8_n248320_k5120_large();
LinearTopKPlan select_topk_q4_n131072_k5120_large();
LinearTopKPlan select_topk_fp8_n248320_k5120_large();
void launch_linear_topk_merge_keys(const LinearTopKWorkspace&, Tensor& keys, cudaStream_t);
void launch_linear_topk_merge_keys_accumulate(const LinearTopKWorkspace&, Tensor& keys,
                                              cudaStream_t);
void launch_linear_topk_merge_final(const LinearTopKWorkspace&, Tensor& keys, Tensor& ids,
                                    Tensor& scores, cudaStream_t);

LinearTopKProducer select_topk_q8_n248320_k5120_sliced(int columns);
LinearTopKPlan select_topk_q8_n248320_k5120_mma(int columns);
LinearTopKProducer select_topk_fp8_n248320_k5120_sliced(int columns);
LinearTopKPlan select_topk_fp8_n248320_k5120_mma(int columns);
LinearTopKProducer select_topk_q4_n131072_k5120_sliced(int columns);
LinearTopKPlan select_topk_q4_n131072_k5120_mma(int columns);
} // namespace ninfer::ops::detail
