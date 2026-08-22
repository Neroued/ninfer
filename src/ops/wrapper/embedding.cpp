// ninfer::ops - embedding wrapper: public api validation and qtype dispatch.
#include "ninfer/ops/embedding.h"

#include "ops/common/math.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/launcher/embed_gather.h" // detail::embed_gather_*_launch
#include "core/verbose.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

// Verbose: true if the stream is in CUDA graph capture mode (blocking readbacks
// are illegal then).
bool verbose_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
           capture != cudaStreamCaptureStatusNone;
}

// Verbose probe: validate that each device pointer is a real device allocation,
// print the weight metadata, and dump the actual token ids (device->host) so an
// out-of-range row (the usual cause of an illegal address in a gather kernel)
// is visible. Runs before the launch, so the CUDA context is still clean.
void verbose_probe_pointers(const char* tag, const Tensor& ids, const Weight& table,
                            const Tensor& out, cudaStream_t stream) {
    if (!verbose_enabled()) { return; }
    auto describe = [](const char* name, const void* p) {
        if (p == nullptr) {
            std::fprintf(stderr, "[verbose]   %-8s = (null)\n", name);
            return;
        }
        cudaPointerAttributes attrs {};
        const cudaError_t err = cudaPointerGetAttributes(&attrs, p);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[verbose]   %-8s = %p  (cudaPointerGetAttributes FAILED: %s)\n",
                         name, p, cudaGetErrorString(err));
            return;
        }
        std::fprintf(stderr, "[verbose]   %-8s = %p  type=%d device=%d devptr=%p\n", name, p,
                     static_cast<int>(attrs.type), attrs.device, attrs.devicePointer);
    };
    const std::int32_t T = ids.ne[0];
    std::fprintf(stderr,
                 "[verbose] embedding(%s): T=%d vocab(n)=%d hidden(k)=%d out_d=%d "
                 "payload_bytes=%llu layout=%d scale_dtype=%d padded=[%d,%d,%d,%d]\n",
                 tag, T, table.n, table.k, out.ne[0],
                 static_cast<unsigned long long>(table.payload_bytes),
                 static_cast<int>(table.layout), static_cast<int>(table.scale_dtype),
                 table.padded_shape[0], table.padded_shape[1], table.padded_shape[2],
                 table.padded_shape[3]);
    describe("ids", ids.data);
    describe("qdata", table.qdata);
    describe("scales", table.scales);
    describe("out", out.data);
    if (ids.data != nullptr && T > 0 && !verbose_stream_capturing(stream)) {
        std::vector<std::int32_t> host_ids(static_cast<std::size_t>(T));
        const cudaError_t err = cudaMemcpy(host_ids.data(), ids.data,
                                           static_cast<std::size_t>(T) * sizeof(std::int32_t),
                                           cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[verbose]   ids dump FAILED: %s\n", cudaGetErrorString(err));
        } else {
            std::fprintf(stderr, "[verbose]   ids = [");
            for (std::int32_t i = 0; i < T; ++i) {
                std::fprintf(stderr, "%s%d%s", i ? ", " : "", host_ids[i],
                             host_ids[i] >= table.n ? " OOB!" : "");
            }
            std::fprintf(stderr, "]\n");
        }
    }
}

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("embedding: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("embedding: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::overflow_error("embedding: weight payload size overflows uint64");
    }
    return a * b;
}

std::int32_t align_up_i32(std::int32_t x, std::int32_t m) {
    const std::int64_t y = round_up(static_cast<std::int64_t>(x), static_cast<std::int64_t>(m));
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
        throw std::invalid_argument("embedding: BF16_CTRL table must be Contiguous");
    }
    require_weight_2d(table);
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: dense table d must match out.ne[0]");
    }
    if (table.qhigh != nullptr || table.high_plane_bytes != 0) {
        throw std::invalid_argument("embedding: dense table high plane must be null");
    }
    const std::uint64_t expected =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]),
                                        static_cast<std::uint64_t>(table.shape[1])),
                        2);
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: dense payload is too small");
    }
}

void require_q6_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("embedding: Q6G64_F16S table must be RowSplit");
    }
    require_weight_2d(table);
    if (table.group_size != 64 || table.group != 64) {
        throw std::invalid_argument("embedding: Q6G64_F16S table group must be 64");
    }
    if (table.scale_dtype != DType::FP16) {
        throw std::invalid_argument("embedding: Q6G64_F16S table scale dtype must be FP16");
    }
    if (table.padded_shape[0] != table.shape[0] ||
        table.padded_shape[1] != align_up_i32(table.shape[1], 128)) {
        throw std::invalid_argument("embedding: Q6G64_F16S padded shape is invalid");
    }
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: Q6G64_F16S table d must match out.ne[0]");
    }
    const std::uint64_t kg = static_cast<std::uint64_t>(table.padded_shape[1] / 64);
    const std::uint64_t nibble_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 32);
    const std::uint64_t high_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 16);
    const std::uint64_t scale_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 2);
    const std::uint64_t high_plane_off = ((nibble_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t scale_plane_off =
        high_plane_off + ((high_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t expected = scale_plane_off + scale_plane_bytes;
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: Q6G64_F16S payload is too small");
    }
    if (table.qdata == nullptr || table.qhigh == nullptr || table.scales == nullptr) {
        throw std::invalid_argument("embedding: Q6G64_F16S planes must be non-null");
    }
    if (table.high_plane_bytes < high_plane_bytes) {
        throw std::invalid_argument("embedding: Q6G64_F16S high plane is too small");
    }
}

void require_w8_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("embedding: W8G32_F16S table must be RowSplit");
    }
    require_weight_2d(table);
    if (table.group_size != 32 || table.group != 32) {
        throw std::invalid_argument("embedding: W8G32_F16S table group must be 32");
    }
    if (table.scale_dtype != DType::FP16) {
        throw std::invalid_argument("embedding: W8G32_F16S table scale dtype must be FP16");
    }
    if (table.padded_shape[0] != table.shape[0] ||
        table.padded_shape[1] != align_up_i32(table.shape[1], 128)) {
        throw std::invalid_argument("embedding: W8G32_F16S padded shape is invalid");
    }
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: W8G32_F16S table d must match out.ne[0]");
    }
    const std::uint64_t kg = static_cast<std::uint64_t>(table.padded_shape[1] / 32);
    const std::uint64_t code_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 32);
    const std::uint64_t scale_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 2);
    const std::uint64_t scale_plane_off = ((code_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t expected        = scale_plane_off + scale_plane_bytes;
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: W8G32_F16S payload is too small");
    }
    if (table.qdata == nullptr || table.scales == nullptr) {
        throw std::invalid_argument("embedding: W8G32_F16S planes must be non-null");
    }
    if (table.qhigh != nullptr || table.high_plane_bytes != 0) {
        throw std::invalid_argument("embedding: W8G32_F16S high plane must be empty");
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
    (void)detail::validate_fp8_weight(table, "embedding");
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

} // namespace

void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    if (ids.dtype != DType::I32) { throw std::invalid_argument("embedding: ids must be I32"); }
    if (out.dtype != DType::BF16) { throw std::invalid_argument("embedding: out must be BF16"); }

    (void)numel_allow_zero(ids, "ids");
    (void)numel_allow_zero(out, "out");
    require_ids_shape(ids);
    require_out_shape(ids, out);

    switch (table.qtype) {
    case QType::BF16_CTRL: {
        require_dense_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        if (table.qdata == nullptr) {
            throw std::invalid_argument("embedding: dense table data must be non-null");
        }
        const Tensor dense = as_dense(table);
        detail::embed_gather_dense_launch(ids, dense, out, stream);
    } break;
    case QType::Q6G64_F16S:
        require_q6_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        detail::embed_gather_q6_launch(ids, table, out, stream);
        break;
    case QType::W8G32_F16S:
        require_w8_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        detail::embed_gather_w8_launch(ids, table, out, stream);
        break;
    case QType::FP8_E4M3FN_ROW_BF16S:
        require_fp8_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        verbose_probe_pointers("fp8", ids, table, out, stream);
        detail::embed_gather_fp8_launch(ids, table, out, stream);
        break;
    default:
        throw std::invalid_argument("embedding: unsupported table qtype");
    }
}

} // namespace ninfer::ops
