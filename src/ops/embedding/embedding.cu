#include "ninfer/ops/embedding.h"
#include "core/device.h"
#include "core/platform.h"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/math.h"
#include "ops/embedding/kernel.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
namespace kernels = detail::embedding;

std::int32_t align_up_i32(std::int32_t x, std::int32_t m) {
    const std::int64_t y =
        common::round_up(static_cast<std::int64_t>(x), static_cast<std::int64_t>(m));
    if (y > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("embedding: padded shape overflows int32");
    }
    return static_cast<std::int32_t>(y);
}

void require_ids_shape(const Tensor& ids) {
    if (ids.ne[1] != 1 || ids.ne[2] != 1 || ids.ne[3] != 1) {
        throw std::invalid_argument("embedding: ids must have shape [T]");
    }
}

void require_out_shape(const Tensor& ids, const Tensor& out) {
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("embedding: out must have shape [d,T]");
    }
    if (out.ne[1] != ids.ne[0]) {
        throw std::invalid_argument("embedding: out T dimension must match ids");
    }
}

void require_weight_2d(const Weight& table) {
    if (table.ndim != 2) { throw std::invalid_argument("embedding: table must be 2-D [vocab,d]"); }
    if (table.shape[0] <= 0 || table.shape[1] <= 0) {
        throw std::invalid_argument("embedding: table shape must be positive");
    }
}

void require_dense_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("embedding: BF16 table must be Contiguous");
    }
    require_weight_2d(table);
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: dense table d must match out.ne[0]");
    }
    if (table.qhigh != nullptr || table.high_plane_bytes != 0) {
        throw std::invalid_argument("embedding: dense table high plane must be null");
    }
    const std::uint64_t expected = common::operand::bytes(
        common::operand::bytes(static_cast<std::uint64_t>(table.shape[0]),
                               static_cast<std::uint64_t>(table.shape[1]), "embedding"),
        2, "embedding");
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: dense payload is too small");
    }
}

void require_q6_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("embedding: Q6_G64_FP16 table must be RowSplit");
    }
    require_weight_2d(table);
    if (table.group_size != 64 || table.group != 64) {
        throw std::invalid_argument("embedding: Q6_G64_FP16 table group must be 64");
    }
    if (table.scale_dtype != DType::FP16) {
        throw std::invalid_argument("embedding: Q6_G64_FP16 table scale dtype must be FP16");
    }
    if (table.padded_shape[0] != table.shape[0] ||
        table.padded_shape[1] != align_up_i32(table.shape[1], 128)) {
        throw std::invalid_argument("embedding: Q6_G64_FP16 padded shape is invalid");
    }
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: Q6_G64_FP16 table d must match out.ne[0]");
    }
    const std::uint64_t kg                 = static_cast<std::uint64_t>(table.padded_shape[1] / 64);
    const std::uint64_t nibble_plane_bytes = common::operand::bytes(
        common::operand::bytes(static_cast<std::uint64_t>(table.shape[0]), kg, "embedding"), 32,
        "embedding");
    const std::uint64_t high_plane_bytes = common::operand::bytes(
        common::operand::bytes(static_cast<std::uint64_t>(table.shape[0]), kg, "embedding"), 16,
        "embedding");
    const std::uint64_t scale_plane_bytes = common::operand::bytes(
        common::operand::bytes(static_cast<std::uint64_t>(table.shape[0]), kg, "embedding"), 2,
        "embedding");
    const std::uint64_t high_plane_off = ((nibble_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t scale_plane_off =
        high_plane_off + ((high_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t expected = scale_plane_off + scale_plane_bytes;
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: Q6_G64_FP16 payload is too small");
    }
    if (table.qdata == nullptr || table.qhigh == nullptr || table.scales == nullptr) {
        throw std::invalid_argument("embedding: Q6_G64_FP16 planes must be non-null");
    }
    if (table.high_plane_bytes < high_plane_bytes) {
        throw std::invalid_argument("embedding: Q6_G64_FP16 high plane is too small");
    }
}

void require_q8_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("embedding: Q8_G32_FP16 table must be RowSplit");
    }
    require_weight_2d(table);
    if (table.group_size != 32 || table.group != 32) {
        throw std::invalid_argument("embedding: Q8_G32_FP16 table group must be 32");
    }
    if (table.scale_dtype != DType::FP16) {
        throw std::invalid_argument("embedding: Q8_G32_FP16 table scale dtype must be FP16");
    }
    if (table.padded_shape[0] != table.shape[0] ||
        table.padded_shape[1] != align_up_i32(table.shape[1], 128)) {
        throw std::invalid_argument("embedding: Q8_G32_FP16 padded shape is invalid");
    }
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: Q8_G32_FP16 table d must match out.ne[0]");
    }
    const std::uint64_t kg               = static_cast<std::uint64_t>(table.padded_shape[1] / 32);
    const std::uint64_t code_plane_bytes = common::operand::bytes(
        common::operand::bytes(static_cast<std::uint64_t>(table.shape[0]), kg, "embedding"), 32,
        "embedding");
    const std::uint64_t scale_plane_bytes = common::operand::bytes(
        common::operand::bytes(static_cast<std::uint64_t>(table.shape[0]), kg, "embedding"), 2,
        "embedding");
    const std::uint64_t scale_plane_off = ((code_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t expected        = scale_plane_off + scale_plane_bytes;
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: Q8_G32_FP16 payload is too small");
    }
    if (table.qdata == nullptr || table.scales == nullptr) {
        throw std::invalid_argument("embedding: Q8_G32_FP16 planes must be non-null");
    }
    if (table.qhigh != nullptr || table.high_plane_bytes != 0) {
        throw std::invalid_argument("embedding: Q8_G32_FP16 high plane must be empty");
    }
}

void require_fp8_metadata(const Weight& table, const Tensor& out) {
    constexpr std::int32_t kVocabulary = 248320;
    constexpr std::int32_t kHidden     = 5120;
    if (table.n != kVocabulary || table.k != kHidden || out.ne[0] != kHidden) {
        throw std::invalid_argument("embedding: unsupported FP8 table shape");
    }
    if ((reinterpret_cast<std::uintptr_t>(out.data) &
         (static_cast<std::uintptr_t>(alignof(std::uint32_t)) - 1)) != 0) {
        throw std::invalid_argument("embedding: FP8 output must be 4-byte aligned");
    }
    (void)common::validate_fp8_weight(table, "embedding");
}

bool is_empty_T(const Tensor& ids, const Tensor& out) { return ids.ne[0] == 0 || out.ne[1] == 0; }

void require_non_empty_tensors(const Tensor& ids, const Tensor& out) {
    if (!ids.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("embedding: ids/out must be contiguous");
    }
    if (ids.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("embedding: ids/out data must be non-null");
    }
}

using EmbeddingLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

constexpr int kBlock = 128;

int linear_grid(std::int64_t count) {
    return static_cast<int>(std::max<std::int64_t>(1, common::div_up(count, std::int64_t{kBlock})));
}

void launch_bf16_scalar(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    kernels::embed_gather_dense_kernel<<<linear_grid(out.numel()), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const __nv_bfloat16*>(table.qdata),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0], ids.ne[0]);
}

#if NINFER_TARGET_SPARK
void launch_bf16_d2560(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    kernels::embed_gather_bf16_d2560_kernel<5, 64><<<ids.ne[0] * 5, 64, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const uint4*>(table.qdata),
        static_cast<uint4*>(out.data));
}
#endif

EmbeddingLaunch select_bf16(const Weight& table, const Tensor& out) {
#if NINFER_TARGET_SPARK
    if (out.ne[0] == 2560 && common::operand::aligned(table.qdata, 16) &&
        common::operand::aligned(out.data, 16))
        return launch_bf16_d2560;
#endif
    return launch_bf16_scalar;
}

template <bool Grouped>
void launch_q6(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    const auto* codes  = static_cast<const std::uint8_t*>(table.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(table.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(table.scales);
    if constexpr (Grouped) {
        constexpr int block = kernels::kEmbedGatherQ6Group * kernels::kEmbedGatherQ6GroupsPerBlock;
        const auto groups   = common::div_up(out.ne[0] / kernels::kEmbedGatherQ6Group,
                                             kernels::kEmbedGatherQ6GroupsPerBlock);
        const auto grid =
            static_cast<int>(std::max<std::int64_t>(1, std::int64_t{ids.ne[0]} * groups));
        kernels::embed_gather_q6_grouped_kernel<<<grid, block, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, high, scales,
            static_cast<__nv_bfloat16*>(out.data), out.ne[0], ids.ne[0]);
    } else {
        kernels::embed_gather_q6_kernel<<<linear_grid(out.numel()), kBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, high, scales,
            static_cast<__nv_bfloat16*>(out.data), out.ne[0], ids.ne[0], table.padded_shape[1]);
    }
}

EmbeddingLaunch select_q6(const Weight& table, const Tensor& out) {
    return out.ne[0] == table.padded_shape[1] && out.ne[0] % kernels::kEmbedGatherQ6Group == 0
               ? launch_q6<true>
               : launch_q6<false>;
}

template <bool Grouped>
void launch_q8_2048(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    const auto* codes  = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(table.scales);
    if constexpr (Grouped)
        kernels::embed_gather_q8_grouped_2048_kernel<<<ids.ne[0] * kernels::kEmbedGatherQ8Groups,
                                                       32, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
    else
        kernels::embed_gather_q8_row_2048_kernel<<<ids.ne[0], 256, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
}

template <int Blocks, bool PairStore>
void launch_q8_packed(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    constexpr int threads = 128;
    kernels::embed_gather_q8_packed_5120_kernel<Blocks, threads, PairStore>
        <<<ids.ne[0] * Blocks, threads, 0, stream>>>(static_cast<const std::int32_t*>(ids.data),
                                                     static_cast<const std::uint8_t*>(table.qdata),
                                                     static_cast<const std::uint8_t*>(table.scales),
                                                     static_cast<__nv_bfloat16*>(out.data));
}

void launch_q8_scalar(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    kernels::embed_gather_q8_kernel<<<linear_grid(out.numel()), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const std::uint8_t*>(table.scales), static_cast<__nv_bfloat16*>(out.data),
        out.ne[0], ids.ne[0], table.padded_shape[1]);
}

EmbeddingLaunch select_q8(const Weight& table, const Tensor& out) {
    if (out.ne[0] == kernels::kEmbedGatherQ8D &&
        table.padded_shape[1] == kernels::kEmbedGatherQ8D) {
        if (out.ne[1] <= 6) return launch_q8_2048<true>;
        if (common::operand::aligned(table.qdata, 8) && common::operand::aligned(out.data, 4))
            return launch_q8_2048<false>;
        return launch_q8_scalar;
    }
    if (out.ne[0] == 5120 && table.padded_shape[1] == 5120 &&
        common::operand::aligned(table.qdata, 4)) {
        const bool paired = common::operand::aligned(out.data, 4);
        if (out.ne[1] <= 128)
            return paired ? launch_q8_packed<10, true> : launch_q8_packed<10, false>;
        return paired ? launch_q8_packed<5, true> : launch_q8_packed<5, false>;
    }
    return launch_q8_scalar;
}

template <int Blocks>
void launch_fp8(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    constexpr int threads = 128;
    kernels::embed_gather_fp8_kernel<Blocks, threads><<<ids.ne[0] * Blocks, threads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const __nv_bfloat16*>(table.scales), static_cast<__nv_bfloat16*>(out.data));
}

EmbeddingLaunch select_fp8(const Tensor& out) {
    return out.ne[1] <= 176 ? launch_fp8<10> : launch_fp8<5>;
}

} // namespace

void embedding(const Tensor& ids, const Weight& table, Tensor& out, CudaExecutionView execution) {
    if (ids.device != execution.device() || table.device != execution.device() ||
        out.device != execution.device()) {
        throw std::invalid_argument("embedding: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    if (ids.dtype != DType::I32) { throw std::invalid_argument("embedding: ids must be I32"); }
    if (out.dtype != DType::BF16) { throw std::invalid_argument("embedding: out must be BF16"); }

    (void)common::operand::numel(ids, "embedding");
    (void)common::operand::numel(out, "embedding");
    require_ids_shape(ids);
    require_out_shape(ids, out);

    switch (table.qtype) {
    case QType::BF16: {
        require_dense_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        if (table.qdata == nullptr) {
            throw std::invalid_argument("embedding: dense table data must be non-null");
        }
        select_bf16(table, out)(ids, table, out, stream);
    } break;
    case QType::Q6_G64_FP16:
        require_q6_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        select_q6(table, out)(ids, table, out, stream);
        break;
    case QType::Q8_G32_FP16:
        require_q8_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        select_q8(table, out)(ids, table, out, stream);
        break;
    case QType::FP8_E4M3FN_ROW_BF16:
        require_fp8_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        select_fp8(out)(ids, table, out, stream);
        break;
    default:
        throw std::invalid_argument("embedding: unsupported table qtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
