#include "mix.cuh"

namespace ninfer::ops::attn_res_detail {
namespace {

struct Partial {
    const Word* data;

    template <int V>
    __device__ __forceinline__ Values<V> load(int c) const {
        return load_bf16<V>(data + c);
    }
};

template <int V, int Q, int Threads, int Packs, bool Exact, bool Norm>
__global__ void block_attn_res_kernel(Mix a, const Word* partial) {
    __shared__ MixScratch<Threads, Q, Packs> reduction;
    const int m = blockIdx.x;
    const Partial source{partial + std::int64_t(m) * a.d};
    for (int q = 0; q < a.q; q += Q) {
        if constexpr (Packs)
            mix_resident<V, Q, Threads, Packs, Exact, Norm>(a, m, source, q, reduction);
        else
            mix_general<V, Threads, Norm>(a, m, source, q, reduction);
        if (q + Q < a.q) __syncthreads();
    }
}

} // namespace

void launch(Mix mix, const Word* partial, CudaExecutionView execution) {
    dispatch_mix(mix, [&]<int V, int Q, int Threads, int Packs, bool Exact, bool Norm>() {
        block_attn_res_kernel<V, Q, Threads, Packs, Exact, Norm>
            <<<mix.columns, Threads, 0, execution.stream()>>>(mix, partial);
    });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::attn_res_detail
