#include "ops/softmax_attention/kpool_mla/prepare/arguments.h"

#include <stdexcept>

namespace ninfer::ops::detail::kpool {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument("Kpool Prepare: no implementation for this target");
}
} // namespace

PrepareGeometry prepare_geometry(const KpoolMlaPrepareProfile&, KpoolMlaStorage,
                                 KvCacheVectorFormat, int, int, int, int) {
    unsupported();
}

PrepareProjectionPlan prepare_projection_plan(const PrepareGeometry&, int) { unsupported(); }

int prepare_window_columns() { unsupported(); }

void launch_prepare(const PrepareArguments&, const PrepareGeometry&, int, cudaStream_t) {
    unsupported();
}
} // namespace ninfer::ops::detail::kpool
