#include "staged.cuh"

namespace ninfer::ops::attn_res_detail {
namespace {

struct Partial {
    const Word* data;

    template <int V>
    __device__ __forceinline__ Values<V> load(int c) const {
        return load_bf16<V>(data + c);
    }

    template <int V>
    __device__ __forceinline__ Packed<V> packed(int c) const {
        return load_pack<V>(data + c);
    }
};

template <int V, int Q, int Threads, int Packs, bool Norm>
__global__ __launch_bounds__(Threads, 1) void block_attn_res_staged(Mix a, const Word* partial) {
    extern __shared__ __align__(16) Word words[];
    __shared__ StageScratch<Threads, Q> scratch;
    const int m = blockIdx.x;
    const Partial source{partial + std::int64_t(m) * a.d};
    mix_staged<V, Q, Threads, Packs, Norm>(a, m, source, words, scratch);
}

template <int V, int Q, int Threads, int Packs, bool Exact, bool Norm>
__device__ __forceinline__ void mix_column(const Mix& a, const Word* partial) {
    __shared__ MixScratch<Threads, Q, Packs> reduction;
    const int m = blockIdx.x;
    const Partial source{partial + std::int64_t(m) * a.d};
    const int query_extent = Q == 1 ? a.q : Q;
    for (int q = 0; q < query_extent; q += Q) {
        if constexpr (Packs)
            mix_resident<V, Q, Threads, Packs, Exact, Norm>(a, m, source, q, reduction);
        else
            mix_general<V, Threads, Norm>(a, m, source, q, reduction);
        if (q + Q < query_extent) __syncthreads();
    }
}

template <int V, int Q, int Threads, int Packs, bool Norm>
__global__ void block_attn_res_kernel(Mix a, const Word* partial) {
    mix_column<V, Q, Threads, Packs, false, Norm>(a, partial);
}

template <int V, int Q, int Threads, int Packs, bool Norm>
__global__ __launch_bounds__(Threads, 2) void block_attn_res_resident(Mix a, const Word* partial) {
    mix_column<V, Q, Threads, Packs, true, Norm>(a, partial);
}

} // namespace

void launch(Mix mix, const Word* partial, CudaExecutionView execution) {
    if (dispatch_staged(mix, [&]<int V, int Q, int Threads, int Packs, bool Norm>(int bytes) {
            constexpr auto kernel = block_attn_res_staged<V, Q, Threads, Packs, Norm>;
            kernel<<<mix.columns, Threads, prepare_stage<kernel>(bytes), execution.stream()>>>(
                mix, partial);
        })) {
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    dispatch_mix(mix, [&]<int V, int Q, int Threads, int Packs, bool Exact, bool Norm>() {
        if constexpr (Exact)
            block_attn_res_resident<V, Q, Threads, Packs, Norm>
                <<<mix.columns, Threads, 0, execution.stream()>>>(mix, partial);
        else
            block_attn_res_kernel<V, Q, Threads, Packs, Norm>
                <<<mix.columns, Threads, 0, execution.stream()>>>(mix, partial);
    });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::attn_res_detail
