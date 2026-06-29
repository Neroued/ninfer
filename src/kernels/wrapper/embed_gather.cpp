// qus::kernels - embed_gather wrapper: public api validation and qtype dispatch.
#include "qus/kernels/embed_gather.h"

#include "kernels/launcher/embed_gather.h" // detail::embed_gather_*_launch
#include "qus/core/weight.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("embed_gather: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("embed_gather: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::overflow_error("embed_gather: weight payload size overflows uint64");
    }
    return a * b;
}

void require_ids_shape(const Tensor& ids) {
    if (ids.ne[1] != 1 || ids.ne[2] != 1 || ids.ne[3] != 1) {
        throw std::invalid_argument("embed_gather: ids must have shape [T]");
    }
}

void require_out_shape(const Tensor& ids, const Tensor& out) {
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("embed_gather: out must have shape [d,T]");
    }
    if (out.ne[1] != ids.ne[0]) {
        throw std::invalid_argument("embed_gather: out T dimension must match ids");
    }
}

void require_weight_2d(const Weight& table) {
    if (table.ndim != 2) {
        throw std::invalid_argument("embed_gather: table must be 2-D [vocab,d]");
    }
    if (table.shape[0] <= 0 || table.shape[1] <= 0) {
        throw std::invalid_argument("embed_gather: table shape must be positive");
    }
}

void require_dense_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("embed_gather: BF16_CTRL table must be Contiguous");
    }
    require_weight_2d(table);
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embed_gather: dense table d must match out.ne[0]");
    }
    const std::uint64_t expected = checked_mul_u64(
        checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]),
                        static_cast<std::uint64_t>(table.shape[1])),
        2);
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embed_gather: dense payload is too small");
    }
}

void require_q6_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("embed_gather: Q6G64_F16S table must be RowSplit");
    }
    require_weight_2d(table);
    if (table.group_size != 64 || table.group != 64) {
        throw std::invalid_argument("embed_gather: Q6G64_F16S table group must be 64");
    }
    if (table.q5090_scale_dtype != ScaleDType::FP16) {
        throw std::invalid_argument("embed_gather: Q6G64_F16S table scale dtype must be FP16");
    }
    if (table.padded_shape[0] != table.shape[0] || table.padded_shape[1] < table.shape[1] ||
        table.padded_shape[1] % 64 != 0) {
        throw std::invalid_argument("embed_gather: Q6G64_F16S padded shape is invalid");
    }
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embed_gather: Q6G64_F16S table d must match out.ne[0]");
    }
    const std::uint64_t kg = static_cast<std::uint64_t>(table.padded_shape[1] / 64);
    const std::uint64_t code_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 48);
    const std::uint64_t scale_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 2);
    const std::uint64_t scale_plane_off = ((code_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t expected = scale_plane_off + scale_plane_bytes;
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embed_gather: Q6G64_F16S payload is too small");
    }
    if (table.qdata == nullptr || table.scales == nullptr) {
        throw std::invalid_argument("embed_gather: Q6G64_F16S code and scale planes must be non-null");
    }
}

bool is_empty_T(const Tensor& ids, const Tensor& out) {
    return ids.ne[0] == 0 || out.ne[1] == 0;
}

void require_non_empty_tensors(const Tensor& ids, const Tensor& out) {
    if (!ids.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("embed_gather: ids/out must be contiguous");
    }
    if (ids.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("embed_gather: ids/out data must be non-null");
    }
}

} // namespace

void embed_gather(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    if (ids.dtype != DType::I32) {
        throw std::invalid_argument("embed_gather: ids must be I32");
    }
    if (out.dtype != DType::BF16) {
        throw std::invalid_argument("embed_gather: out must be BF16");
    }

    (void) numel_allow_zero(ids, "ids");
    (void) numel_allow_zero(out, "out");
    require_ids_shape(ids);
    require_out_shape(ids, out);

    switch (table.qtype) {
    case QType::BF16_CTRL: {
        require_dense_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        if (table.qdata == nullptr) {
            throw std::invalid_argument("embed_gather: dense table data must be non-null");
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
    default:
        throw std::invalid_argument("embed_gather: unsupported table qtype");
    }
}

} // namespace qus::kernels
