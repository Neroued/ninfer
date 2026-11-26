#include "kernel.cuh"

#include <algorithm>

namespace ninfer::ops::short_conv_detail {
namespace {

template <int V, int Threads, bool Ragged, bool Snapshot>
__global__ void short_conv_kernel(Conv a) {
    const int m       = blockIdx.x;
    const auto col    = locate<Ragged>(a, m);
    const auto first  = (blockIdx.y * Threads + threadIdx.x) * V;
    const auto stride = gridDim.y * Threads * V;
    if (col.pos >= col.length) {
        for (std::int64_t c = first; c < a.d; c += stride)
            store_bf16<V>(a.out + std::int64_t(m) * a.d + c, {});
        return;
    }
    const Work<Snapshot> work(a, col);
    for (std::int64_t c = first; c < a.d; c += stride) {
        for (int pos = col.pos; pos < work.end; ++pos) {
            const auto y = convolve<V>(a, col, pos, int(c));
            store_bf16<V>(a.out + (std::int64_t(col.first) + pos) * a.d + c, y);
            if constexpr (Snapshot) {
                if (work.writes_snapshot(pos)) update_state<V, true>(a, col, pos, int(c));
            }
        }
        const int last = work.final_write();
        if (last >= 0) update_state<V, Snapshot>(a, col, last, int(c));
    }
}

} // namespace

void launch(Conv a, CudaExecutionView execution) {
    constexpr int threads = 64;
    auto submit           = [&]<int V, bool Ragged, bool Snapshot>() {
        const auto groups = (std::int64_t(a.d) + threads * V - 1) / (threads * V);
        short_conv_kernel<V, threads, Ragged, Snapshot>
            <<<dim3(a.columns, std::min<std::int64_t>(groups, 65535)), threads, 0,
                         execution.stream()>>>(a);
    };
    auto layout = [&]<int V>() {
        if (a.prefix)
            submit.template operator()<V, true, false>();
        else if (a.snapshot_slots)
            submit.template operator()<V, false, true>();
        else
            submit.template operator()<V, false, false>();
    };
    if (a.d % 8 == 0)
        layout.template operator()<8>();
    else
        layout.template operator()<1>();
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::short_conv_detail
