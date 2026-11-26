#include "kernel.cuh"

#include <algorithm>

namespace ninfer::ops::short_conv_detail {
namespace {

template <int V, int Tile, bool Snapshot>
__device__ __forceinline__ void convolve_interior(const Conv& a, Column col, int c) {
    const auto d      = std::int64_t(a.d);
    const auto offset = (std::int64_t(col.first) + col.pos) * d + c;
    const auto* input = a.x + offset;
    auto* out         = a.out + offset;
    const auto w0     = load_bf16<V>(a.weight + c);
    const auto w1     = load_bf16<V>(a.weight + d + c);
    const auto w2     = load_bf16<V>(a.weight + 2 * d + c);
    const auto w3     = load_bf16<V>(a.weight + 3 * d + c);
    auto x0           = load_bf16<V>(input - 3 * d);
    auto x1           = load_bf16<V>(input - 2 * d);
    auto x2           = load_bf16<V>(input - d);
    const Work<Snapshot> work(a, col);
#pragma unroll 1
    for (int i = 0; i < Tile; ++i) {
        const auto x3 = load_bf16<V>(input);
        Values<V> result;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            float sum   = fmaf(w0.x[v], x0.x[v], 0.0f);
            sum         = fmaf(w1.x[v], x1.x[v], sum);
            sum         = fmaf(w2.x[v], x2.x[v], sum);
            sum         = fmaf(w3.x[v], x3.x[v], sum);
            result.x[v] = (sum + x3.x[v]) * kConvScale;
        }
        if constexpr (V == 8) {
            *reinterpret_cast<uint4*>(out) = round_conv_pack(a, input, result);
        } else {
            result.x[0] = __bfloat162float(__float2bfloat16_rn(result.x[0]));
            add_residual(a, input, result);
            store_bf16<V>(out, result);
        }
        if constexpr (Snapshot)
            if (work.writes_snapshot(col.pos + i)) update_state<V, true>(a, col, col.pos + i, c);
        input += d;
        out += d;
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }
    if constexpr (!Snapshot) {
        col.pos += Tile - 1;
        if (col.pos == col.length - 1 && !aliases_history(a, col))
            update_state<V, false>(a, col, col.pos, c);
    }
}

template <int V, int Threads, int Tile, bool Ragged, bool Snapshot>
__global__ __launch_bounds__(Threads) void short_conv_kernel(Conv a) {
    const int start   = blockIdx.x * Tile;
    const auto first  = (blockIdx.y * Threads + threadIdx.x) * V;
    const auto stride = gridDim.y * Threads * V;
    if constexpr (Tile > 1) {
        const auto col = locate<Ragged>(a, start);
        if (col.pos >= 3 && col.pos <= col.length - Tile) {
            for (unsigned c = first; c < unsigned(a.d); c += stride)
                convolve_interior<V, Tile, Snapshot>(a, col, int(c));
            return;
        }
    }
    const int count = min(Tile, a.columns - start);
    for (int i = 0; i < count; ++i) {
        const int m    = start + i;
        const auto col = locate<Ragged>(a, m);
        if (col.pos >= col.length) {
            for (unsigned c = first; c < unsigned(a.d); c += stride)
                store_bf16<V>(a.out + std::int64_t(m) * a.d + c, {});
            continue;
        }
        const Work<Snapshot> work(a, col);
        for (unsigned c = first; c < unsigned(a.d); c += stride) {
            for (int pos = col.pos; pos < work.end; ++pos) {
                auto* out = a.out + (std::int64_t(col.first) + pos) * a.d + c;
                if constexpr (V == 8)
                    *reinterpret_cast<uint4*>(out) = convolve_pack(a, col, pos, int(c));
                else
                    store_bf16<V>(out, convolve<V>(a, col, pos, int(c)));
                if constexpr (Snapshot)
                    if (work.writes_snapshot(pos)) update_state<V, true>(a, col, pos, int(c));
            }
            const int last = work.final_write();
            if (last >= 0) update_state<V, Snapshot>(a, col, last, int(c));
        }
    }
}

} // namespace

void launch(Conv a, CudaExecutionView execution) {
    auto submit = [&]<int V, int Threads, int Tile, bool Ragged, bool Snapshot>() {
        const auto groups = (std::int64_t(a.d) + Threads * V - 1) / (Threads * V);
        short_conv_kernel<V, Threads, Tile, Ragged, Snapshot>
            <<<dim3((a.columns - 1) / Tile + 1, std::min<std::int64_t>(groups, 65535)), Threads, 0,
               execution.stream()>>>(a);
    };
    auto geometry = [&]<int V, bool Ragged, bool Snapshot>() {
        const auto groups        = (std::int64_t(a.d) + 64 * V - 1) / (64 * V);
        const auto channel_tiles = a.columns * groups;
        const int sms            = execution.multiprocessor_count();
        if (channel_tiles < sms) submit.template operator()<V, 32, 1, Ragged, Snapshot>();
        // The four-column tile needs enough work to hide its longer per-CTA loop.
        // Short segments would mostly execute the boundary path.
        else if (channel_tiles >= 48LL * sms && a.columns >= 12LL * a.rows)
            submit.template operator()<V, 64, 4, Ragged, Snapshot>();
        else
            submit.template operator()<V, 64, 1, Ragged, Snapshot>();
    };
    auto layout = [&]<int V>() {
        if (a.prefix)
            geometry.template operator()<V, true, false>();
        else if (a.snapshot_slots)
            geometry.template operator()<V, false, true>();
        else
            geometry.template operator()<V, false, false>();
    };
    if (a.d % 8 == 0)
        layout.template operator()<8>();
    else
        layout.template operator()<1>();
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::short_conv_detail
