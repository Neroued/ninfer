#include "ops/softmax_attention/kpool_mla/sparse_mla/plan.h"
#include "ops/softmax_attention/kpool_mla/sparse_mla/factor.cuh"
#include "ops/softmax_attention/kpool_mla/sparse_mla/attention.cuh"
#include "ops/common/host/launch.h"

#include <cuda.h>
#include <string>

namespace ninfer::ops::detail::kpool {
namespace {
CUtensorMap latent_map(const SparseMlaArguments& a) {
    CUtensorMap map{};
    // The nonmonotonic strides transpose a contiguous pool into [D/64, 4, 64]
    // in shared memory. Z addresses 128-byte sectors of the native page arena;
    // page offsets need only the public 256-byte alignment, not a 4-KiB alignment.
    const cuuint64_t dimensions[]{64, 4, a.latent_bytes / 128}, strides[]{1024, 128};
    const unsigned box[]{64, 4, 8}, step[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(a.latent), dimensions,
        strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("Sparse MLA latent TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return map;
}

template <bool Key, bool Direct, int Queries, int Rows, int K, bool Merge = false>
void factor(SparseMlaArguments a, cudaStream_t stream) {
    constexpr auto kernel = mla_factor_kernel<Key, Queries, Rows, K, Direct, Merge>;
    constexpr int stages  = K == (Key ? 256 : 512) ? 1 : 2;
    constexpr int bytes   = stages * (Rows + Queries) * K * 2;
    const int shared      = common::prepare_shared<bytes, kernel, true>();
    const dim3 grid =
        Key && a.direct
            ? dim3(1)
            : dim3(((a.width + Queries - 1) / Queries) * ((Key ? 512 : 256) / Rows), 64);
    constexpr int threads = Rows > 64 ? 256 : 128;
    kernel<<<grid, threads, shared, stream>>>(a);
}

template <bool Key, bool Direct = false>
void factor(SparseMlaArguments a, MlaFactorSchedule schedule, cudaStream_t stream) {
    switch (schedule) {
    case MlaFactorSchedule::Small:
        factor<Key, Direct, 16, 32, 256>(a, stream);
        break;
    case MlaFactorSchedule::Medium:
        factor<Key, Direct, 32, 64, Key ? 128 : 512>(a, stream);
        break;
    case MlaFactorSchedule::Bulk:
        factor<Key, Direct, 64, 128, 64>(a, stream);
        break;
    }
}

template <int Tokens, int HeadGroups>
void attention(SparseMlaArguments a, const CUtensorMap& map, cudaStream_t stream) {
    constexpr bool independent = Tokens == 32;
    constexpr auto kernel      = mla_attention_kernel<Tokens, HeadGroups, independent>;
    const int shared =
        common::prepare_shared<sizeof(MlaAttentionStorage<Tokens, HeadGroups, independent>), kernel,
                               true>();
    const dim3 grid = a.direct ? dim3(1) : dim3(a.width * (4 / HeadGroups), a.splits);
    kernel<<<grid, 64 * HeadGroups, shared, stream>>>(a, map);
}
} // namespace

void launch_sparse_mla(SparseMlaArguments a, MemorySpan scratch, KpoolExecutionEnvelope envelope,
                       int sms, cudaStream_t stream) {
    // By-value kernel parameters own the descriptor during asynchronous execution
    // and Graph replay. Encoding uses no device allocation or persistent host state.
    const auto map   = latent_map(a);
    const auto total = std::int64_t(a.tokens) * a.batch;
    for (a.begin = 0; a.begin < total; a.begin += kMlaWindow) {
        a.width         = std::min<std::int64_t>(kMlaWindow, total - a.begin);
        const auto plan = sparse_mla_plan(a.width, a.batch, envelope, sms);
        a.splits        = plan.splits;
        a.tile          = plan.tile;
        a.direct        = plan.direct;
        if (plan.bytes) a.mapped = static_cast<__nv_bfloat16*>(plan.mapped.bind(scratch).data);
        if (plan.numerator.region.bytes) {
            a.numerator   = static_cast<float*>(plan.numerator.bind(scratch).data);
            a.maximum     = static_cast<float*>(plan.maximum.bind(scratch).data);
            a.denominator = static_cast<float*>(plan.denominator.bind(scratch).data);
        }
        // Four ordered nodes per fixed window also cover singleton/empty envelopes.
        // This keeps Graph topology stable when the envelope changes.
        factor<true>(a, plan.factor, stream);
        if (plan.attention == MlaAttentionSchedule::Small)
            attention<16, 2>(a, map, stream);
        else
            attention<32, 4>(a, map, stream);
        // Retain the node for Graph updates without scheduling an idle CTA per head.
        auto merge = a;
        if (plan.merge_value) merge.splits = 1;
        const int merge_blocks = merge.splits > 1 ? a.width * 64 : 1;
        mla_merge_kernel<<<merge_blocks, 256, 0, stream>>>(merge);
        if (plan.merge_value)
            factor<false, false, 8, 64, 128, true>(a, stream);
        else if (a.direct)
            factor<false, true>(a, plan.factor, stream);
        else
            factor<false>(a, plan.factor, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::kpool
