#include "ninfer/ops/linear_pair.h"

#include "ops/linear_pair/w8/w8_pair_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("linear_pair: invalid ") + label);
    }
}

std::uint64_t required_payload_bytes(std::int32_t rows, std::int32_t k) {
    const std::uint64_t r = static_cast<std::uint64_t>(rows);
    const std::uint64_t h = static_cast<std::uint64_t>(k);
    if (h > std::numeric_limits<std::uint64_t>::max() / r) {
        throw std::overflow_error("linear_pair: weight payload size overflows");
    }
    const std::uint64_t code_bytes = r * h;
    const std::uint64_t groups     = h / 32;
    if (groups > std::numeric_limits<std::uint64_t>::max() / (r * 2u)) {
        throw std::overflow_error("linear_pair: scale payload size overflows");
    }
    return code_bytes + r * groups * 2u;
}

void require_weight(const Weight& weight, std::int32_t k, const char* label) {
    const std::uint64_t payload_bytes = required_payload_bytes(1024, k);
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != 1024 || weight.k != k || weight.shape[0] != 1024 ||
        weight.shape[1] != k || weight.padded_shape[0] != 1024 || weight.padded_shape[1] != k ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < payload_bytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string("linear_pair: invalid ") + label);
    }
}

void require_nonoverlap(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                        const Tensor& first_out, const Tensor& second_out) {
    struct Range {
        const void* pointer;
        std::uint64_t bytes;
        const char* label;
    };

    const std::uint64_t cols        = static_cast<std::uint64_t>(x.ne[1]);
    const std::uint64_t k           = static_cast<std::uint64_t>(x.ne[0]);
    const std::uint64_t code_bytes  = 1024u * k;
    const std::uint64_t scale_bytes = 1024u * (k / 32u) * 2u;
    const std::array<Range, 7> ranges{{
        {x.data, k * cols * 2u, "x"},
        {first_out.data, 1024u * cols * 2u, "first output"},
        {second_out.data, 1024u * cols * 2u, "second output"},
        {first_weight.qdata, code_bytes, "first codes"},
        {first_weight.scales, scale_bytes, "first scales"},
        {second_weight.qdata, code_bytes, "second codes"},
        {second_weight.scales, scale_bytes, "second scales"},
    }};
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto first_begin = reinterpret_cast<std::uintptr_t>(ranges[i].pointer);
        const auto first_end   = first_begin + ranges[i].bytes;
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            const auto second_begin = reinterpret_cast<std::uintptr_t>(ranges[j].pointer);
            const auto second_end   = second_begin + ranges[j].bytes;
            if (first_begin < second_end && second_begin < first_end) {
                throw std::invalid_argument(std::string("linear_pair: ") + ranges[i].label +
                                            " overlaps " + ranges[j].label);
            }
        }
    }
}

} // namespace

void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (x.ne[0] != 5120 && x.ne[0] != 2048) {
        throw std::invalid_argument("linear_pair: x K must be 5120 or 2048");
    }
    require_matrix(x, x.ne[0], cols, "x");
    require_matrix(first_out, 1024, cols, "first output");
    require_matrix(second_out, 1024, cols, "second output");
    require_weight(first_weight, x.ne[0], "first weight");
    require_weight(second_weight, x.ne[0], "second weight");
    require_nonoverlap(x, first_weight, second_weight, first_out, second_out);

    (void)ws;
    detail::w8_pair_dispatch(x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops
