#include "ops/softmax_attention/kpool_mla/sparse_mla/plan.h"

#include <stdexcept>

namespace ninfer::ops::detail::kpool {
void sparse_mla_geometry(const KpoolSparseMlaProfile&, KpoolMlaStorage, KpoolExecutionEnvelope, int,
                         int, int, int) {
    throw std::invalid_argument("Kpool Sparse MLA: no implementation for this target");
}

std::size_t sparse_mla_capacity(KpoolMlaStorage, int, int, int, KpoolExecutionEnvelope, int) {
    throw std::invalid_argument("Kpool Sparse MLA: no implementation for this target");
}

SparseMlaPlan sparse_mla_plan(KpoolMlaStorage, int, int, KpoolExecutionEnvelope, int) {
    throw std::invalid_argument("Kpool Sparse MLA: no implementation for this target");
}

void launch_sparse_mla(SparseMlaArguments, MemorySpan, KpoolMlaStorage, KpoolExecutionEnvelope, int,
                       cudaStream_t) {
    throw std::invalid_argument("Kpool Sparse MLA: no implementation for this target");
}
} // namespace ninfer::ops::detail::kpool
