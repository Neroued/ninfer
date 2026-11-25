#pragma once

#include "ops/softmax_attention/kpool_mla/sparse_mla/plan.h"
#include "ops/softmax_attention/kpool_mla/sparse_mla/factor.cuh"
#include "ops/softmax_attention/kpool_mla/sparse_mla/attention.cuh"
#include "ops/softmax_attention/kpool_mla/sparse_mla/attention_fp8.cuh"
#include "ops/common/host/launch.h"

#include <cuda.h>
#include <string>

namespace ninfer::ops::detail::kpool {
namespace {
#ifdef NINFER_KPOOL_PACKED
using LaunchArguments = PackedSparseMlaArguments;
#define NINFER_MLA_KERNEL(name) packed_##name
#else
using LaunchArguments = SparseMlaArguments;
#define NINFER_MLA_KERNEL(name) name
#endif
CUtensorMap latent_map(const SparseMlaArguments& a, bool fp8) {
    CUtensorMap map{};
    // Nonmonotonic strides transpose Kpool4 into 128-byte chunks in shared
    // memory. Z addresses 128-byte sectors of the native page arena;
    // page offsets need only the public 256-byte alignment, not a 4-KiB alignment.
    const cuuint64_t dimensions[]{fp8 ? 128u : 64u, 4, a.latent_bytes / 128},
        strides[]{fp8 ? 512u : 1024u, 128};
    const unsigned box[]{fp8 ? 128u : 64u, 4, fp8 ? 4u : 8u}, step[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &map, fp8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
        fp8 ? static_cast<void*>(const_cast<std::uint8_t*>(a.latent_fp8))
            : static_cast<void*>(const_cast<__nv_bfloat16*>(a.latent)),
        dimensions, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("Sparse MLA latent TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return map;
}

template <bool Key, int Queries, int Rows, int K, bool Merge, bool Fp8, bool Mxfp8>
void factor_format(LaunchArguments a, cudaStream_t stream) {
    constexpr auto kernel = NINFER_MLA_KERNEL(mla_factor_kernel)<Key, Queries, Rows, K, Merge, Fp8, Mxfp8>;
    constexpr int stages  = K == (Key ? 256 : 512) ? 1 : 2;
    constexpr int bytes   = stages * (Rows + Queries) * K * 2;
    const int shared      = common::prepare_shared<bytes, kernel, true>();
    const dim3 grid(((a.width + Queries - 1) / Queries) * ((Key ? 512 : 256) / Rows), 64);
    constexpr int threads = Rows > 128 ? 512 : Rows > 64 ? 256 : 128;
    kernel<<<grid, threads, shared, stream>>>(a);
}

template <bool Key, int Queries, int Rows, int K, bool Merge = false, bool Fp8 = false>
void factor(LaunchArguments a, cudaStream_t stream) {
    if (Key ? a.key_mxfp8 : a.value_mxfp8)
        factor_format<Key, Queries, Rows, K, Merge, Fp8, true>(a, stream);
    else
        factor_format<Key, Queries, Rows, K, Merge, Fp8, false>(a, stream);
}

template <bool Key>
void factor(LaunchArguments a, MlaFactorSchedule schedule, cudaStream_t stream) {
    if constexpr (Key) {
        if (a.key_mxfp8 && a.width >= 9 && a.width <= 32) {
            factor<Key, 32, 128, 64>(a, stream);
            return;
        }
        if (a.key_mxfp8 && a.width >= 33 && a.width <= 64) {
            factor<Key, 64, 128, 64>(a, stream);
            return;
        }
    }
    if (!Key && a.value_mxfp8 && a.width >= 25 && a.width <= 32) {
        factor<Key, 32, 64, 64>(a, stream);
        return;
    }
    if (!Key && a.value_mxfp8 && a.width >= 33 && a.width <= 48) {
        factor<Key, 16, 64, 64>(a, stream);
        return;
    }
    if ((Key ? a.key_mxfp8 : a.value_mxfp8) && a.width <= 8) {
        factor<Key, 8, 64, 64>(a, stream);
        return;
    }
    if (!Key && a.value_mxfp8 && a.width >= 33 && a.width <= 64) {
        factor<Key, 64, 128, 64>(a, stream);
        return;
    }
    switch (schedule) {
    case MlaFactorSchedule::Small:
        factor<Key, 16, 32, 256>(a, stream);
        break;
    case MlaFactorSchedule::Medium:
        factor<Key, 32, 64, Key ? 128 : 512>(a, stream);
        break;
    case MlaFactorSchedule::Bulk:
        factor<Key, 64, 128, 64>(a, stream);
        break;
    }
}

template <int Tokens, int HeadGroups>
void attention(LaunchArguments a, const CUtensorMap& map, cudaStream_t stream) {
    constexpr bool independent = Tokens == 32;
    constexpr auto kernel = NINFER_MLA_KERNEL(mla_attention_kernel)<Tokens, HeadGroups, independent>;
    const int shared =
        common::prepare_shared<sizeof(MlaAttentionStorage<Tokens, HeadGroups, independent>), kernel,
                               true>();
    const dim3 grid = dim3(a.width * (4 / HeadGroups), a.splits);
    kernel<<<grid, 64 * HeadGroups, shared, stream>>>(a, map);
}
} // namespace

#ifdef NINFER_KPOOL_PACKED
void launch_packed_sparse_mla(LaunchArguments a, MemorySpan scratch,
#else
void launch_sparse_mla(LaunchArguments a, MemorySpan scratch,
#endif
                       const KpoolSparseMlaProfile& profile, KpoolMlaStorage storage,
                       ContextBucket bucket, int sms, cudaStream_t stream) {
    // By-value kernel parameters own the descriptor during asynchronous execution
    // and Graph replay. Encoding uses no device allocation or persistent host state.
    const bool fp8   = storage == KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0;
    const auto map   = latent_map(a, fp8);
    const auto total = std::int64_t(a.tokens) * a.batch;
    for (a.begin = 0; a.begin < total; a.begin += kMlaWindow) {
        a.width            = std::min<std::int64_t>(kMlaWindow, total - a.begin);
        a.first_row        = a.begin / a.tokens;
        a.last_row         = (a.begin + a.width - 1) / a.tokens;
        const auto plan    = sparse_mla_plan(profile, storage, a.width, a.batch, bucket, sms);
        a.splits           = plan.splits;
        a.preferred_splits = plan.preferred_splits;
        a.split_wave_size  = sms;
        a.tile             = plan.tile;
        a.mapped           = static_cast<__nv_bfloat16*>(plan.mapped.bind(scratch).data);
        if (plan.numerator.region.bytes) {
            a.numerator   = static_cast<float*>(plan.numerator.bind(scratch).data);
            a.maximum     = static_cast<float*>(plan.maximum.bind(scratch).data);
            a.denominator = static_cast<float*>(plan.denominator.bind(scratch).data);
        }
        // Four ordered nodes per window, including singleton and empty selections.
        if (fp8) {
            a.query_fp8    = static_cast<std::uint8_t*>(plan.query_fp8.bind(scratch).data);
            a.query_scales = static_cast<std::uint8_t*>(plan.query_scales.bind(scratch).data);
            switch (plan.fp8_query) {
            case MlaFactorSchedule::Small:
                if (a.key_mxfp8 && a.width <= 8)
                    factor<true, 8, 128, 64, false, true>(a, stream);
                else
                    factor<true, 16, 128, 128, false, true>(a, stream);
                break;
            case MlaFactorSchedule::Medium:
                if (a.key_mxfp8)
                    factor<true, 32, 128, 64, false, true>(a, stream);
                else
                    factor<true, 32, 128, 128, false, true>(a, stream);
                break;
            case MlaFactorSchedule::Bulk:
                factor<true, 32, 128, 64, false, true>(a, stream);
                break;
            }
            const auto submit = [&]<int Tokens, int HeadGroups>() {
                constexpr auto kernel = NINFER_MLA_KERNEL(mla_attention_fp8_kernel)<Tokens, HeadGroups>;
                const int shared = common::prepare_shared<sizeof(MlaFp8Storage<Tokens, HeadGroups>),
                                                          kernel, true>();
                const dim3 grid  = dim3(a.width * (4 / HeadGroups), a.splits);
                kernel<<<grid, 64 * HeadGroups, shared, stream>>>(a, map);
            };
            if (plan.tile == 64)
                submit.template operator()<64, 4>();
            else if (plan.attention == MlaAttentionSchedule::Small)
                submit.template operator()<16, 2>();
            else
                submit.template operator()<32, 4>();
        } else {
            factor<true>(a, plan.factor, stream);
            if (plan.attention == MlaAttentionSchedule::Small)
                attention<16, 2>(a, map, stream);
            else
                attention<32, 4>(a, map, stream);
        }
        // Retain the node for Graph updates without scheduling an idle CTA per head.
        auto merge = a;
        if (plan.merge_value) {
            merge.splits           = 1;
            merge.preferred_splits = 0;
        }
        const int merge_blocks = merge.splits > 1 ? a.width * 64 : 1;
        NINFER_MLA_KERNEL(mla_merge_kernel)<<<merge_blocks, 256, 0, stream>>>(merge);
        if (plan.merge_value) {
            if (a.width <= 8)
                factor<false, 8, 256, 64, true>(a, stream);
            else if (fp8 && a.value_mxfp8)
                factor<false, 32, 128, 64, true>(a, stream);
            else
                factor<false, 8, 64, 128, true>(a, stream);
        } else
            factor<false>(a, plan.factor, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
#undef NINFER_MLA_KERNEL
} // namespace ninfer::ops::detail::kpool
