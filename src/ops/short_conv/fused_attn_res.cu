#include "kernel.cuh"

namespace ninfer::ops::short_conv_detail {
namespace {
using namespace attn_res_detail;

struct Partial {
    const Conv& conv;
    Column col;
    int pos;
    bool publish;

    template <int V>
    __device__ __forceinline__ Values<V> load(int c) const {
        const auto y = convolve<V>(conv, col, pos, c);
        if (publish) store_bf16<V>(conv.out + (std::int64_t(col.first) + pos) * conv.d + c, y);
        return y;
    }
};

struct ResidentPartial {
    uint4 packed;

    template <int V>
    __device__ __forceinline__ Values<V> load(int) const {
        static_assert(V == 8);
        return decode_bf16(packed);
    }
};

template <int V, int Q, int Threads, int Packs, bool Exact, bool Norm, bool Ragged, bool Snapshot>
__global__ __launch_bounds__(Threads, Exact&& Q == 1 ? 3 : 1) void fused_kernel(Conv a, Mix mix) {
    __shared__ MixScratch<Threads, Q, Packs> reduction;
    const int m    = blockIdx.x;
    const auto col = locate<Ragged>(a, m);
    if (col.pos >= col.length) {
        for (std::int64_t c = threadIdx.x * V; c < a.d; c += Threads * V)
            store_bf16<V>(a.out + std::int64_t(m) * a.d + c, {});
        zero_mix<V, Norm>(mix, m);
        return;
    }
    const Work<Snapshot> work(a, col);
    for (int pos = col.pos; pos < work.end; ++pos) {
        if constexpr (Exact) {
            const int c       = int(threadIdx.x) * V;
            const auto offset = (std::int64_t(col.first) + pos) * a.d + c;
            const ResidentPartial partial{convolve_pack(a, col, pos, c)};
            *reinterpret_cast<uint4*>(a.out + offset) = partial.packed;
            for (int q = 0; q < mix.q; q += Q) {
                mix_resident<V, Q, Threads, Packs, Exact, Norm>(mix, col.first + pos, partial, q,
                                                                reduction);
                if (q + Q < mix.q) __syncthreads();
            }
        } else {
            for (int q = 0; q < mix.q; q += Q) {
                const Partial partial{a, col, pos, Packs != 0 && q + Q == mix.q};
                if constexpr (Packs)
                    mix_resident<V, Q, Threads, Packs, Exact, Norm>(mix, col.first + pos, partial,
                                                                    q, reduction);
                else
                    mix_general<V, Threads, Norm>(mix, col.first + pos, partial, q, reduction);
                if (q + Q < mix.q) __syncthreads();
            }
            if constexpr (!Packs) {
                // The unbounded-channel route recomputes tiles. Publish after all
                // queries have read the original residual, which may alias out.
                const Partial partial{a, col, pos, true};
                for (std::int64_t c = threadIdx.x * V; c < a.d; c += Threads * V)
                    (void)partial.template load<V>(int(c));
            }
        }
        if constexpr (Snapshot) {
            if (work.writes_snapshot(pos)) {
                for (std::int64_t c = threadIdx.x * V; c < a.d; c += Threads * V)
                    update_state<V, true>(a, col, pos, int(c));
            }
        }
        if (pos + 1 < work.end) __syncthreads();
    }
    const int last = work.final_write();
    if (last >= 0) {
        for (std::int64_t c = threadIdx.x * V; c < a.d; c += Threads * V)
            update_state<V, Snapshot>(a, col, last, int(c));
    }
}

} // namespace

void launch_fused(Conv a, attn_res_detail::Mix mix, CudaExecutionView execution) {
    auto layout = [&]<bool Ragged, bool Snapshot>() {
        attn_res_detail::dispatch_mix(
            mix, [&]<int V, int Q, int Threads, int Packs, bool Exact, bool Norm>() {
                fused_kernel<V, Q, Threads, Packs, Exact, Norm, Ragged, Snapshot>
                    <<<a.columns, Threads, 0, execution.stream()>>>(a, mix);
            });
    };
    if (a.prefix)
        layout.template operator()<true, false>();
    else if (a.snapshot_slots)
        layout.template operator()<false, true>();
    else
        layout.template operator()<false, false>();
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::short_conv_detail
