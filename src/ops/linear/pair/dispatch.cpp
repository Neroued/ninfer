#include "ops/linear/pair/dispatch.h"
#include "ops/common/host/operands.h"
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
void require_adjacent_parent_views(const Weight& first_weight, const Weight& second_weight) {
    constexpr std::int32_t kParentRows     = 6144;
    constexpr std::int32_t kHidden         = 2048;
    constexpr std::int32_t kFirstRow       = 4096;
    constexpr std::int32_t kSecondRow      = 5120;
    constexpr std::uint64_t kCodeBytes     = static_cast<std::uint64_t>(kParentRows) * kHidden;
    constexpr std::uint64_t kScaleRowBytes = (kHidden / 32) * 2;
    constexpr std::uint64_t kPayloadBytes =
        kCodeBytes + static_cast<std::uint64_t>(kParentRows) * kScaleRowBytes;

    const auto* payload = static_cast<const std::byte*>(first_weight.payload);
    if (payload == nullptr || second_weight.payload != first_weight.payload ||
        first_weight.payload_bytes < kPayloadBytes || second_weight.payload_bytes < kPayloadBytes ||
        first_weight.qdata != payload + static_cast<std::uint64_t>(kFirstRow) * kHidden ||
        second_weight.qdata != payload + static_cast<std::uint64_t>(kSecondRow) * kHidden ||
        first_weight.scales !=
            payload + kCodeBytes + static_cast<std::uint64_t>(kFirstRow) * kScaleRowBytes ||
        second_weight.scales !=
            payload + kCodeBytes + static_cast<std::uint64_t>(kSecondRow) * kScaleRowBytes) {
        throw std::invalid_argument(
            "q8 pair: [1024,2048] weights must be exact adjacent parent row views");
    }
}

void require_weight(const Weight& w, int k) {
    const std::uint64_t payload_bytes =
        std::uint64_t(1024) * k + std::uint64_t(1024) * (k / 32) * 2;
    if (w.qtype != QType::Q8_G32_FP16 || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != 32 || w.group_size != 32 || w.ndim != 2 ||
        w.n != 1024 || w.k != k || w.shape[0] != 1024 || w.shape[1] != k ||
        w.padded_shape[0] != 1024 || w.padded_shape[1] != k || w.payload_bytes < payload_bytes ||
        w.qhigh != nullptr || w.high_plane_bytes != 0 || !common::operand::aligned(w.qdata, 16) ||
        !common::operand::aligned(w.scales, 16))
        throw std::invalid_argument(
            "linear_pair: invalid Q8 row-split weight representation or alignment");
}
} // namespace

LinearPairLaunch select_linear_pair(int k, int tokens) {
    if (tokens > 0) {
        if (k == 2048) return select_q8_pair_n1024_k2048(tokens);
        if (k == 5120) return select_q8_pair_n1024_k5120(tokens);
    }
    throw std::invalid_argument("linear_pair: unsupported shape or token extent");
}

void validate_pair_weights(const Weight& first, const Weight& second, int max_tokens) {
    if (first.qtype == QType::BF16 && second.qtype == QType::BF16) {
        if (max_tokens <= 0 || (first.n != 4096 && first.n != 8192) || first.k != 128 ||
            second.n != first.n || second.k != first.k)
            throw std::invalid_argument("linear_pair: unsupported BF16 geometry");
        for (const auto* w : {&first, &second})
            if (w->ndim != 2 || w->shape[0] != w->n || w->shape[1] != w->k ||
                w->padded_shape[0] != w->n || w->padded_shape[1] != w->k ||
                w->layout != QuantLayout::Contiguous || w->group || w->group_size || w->scales ||
                w->qhigh || w->high_plane_bytes ||
                w->payload_bytes < std::uint64_t(w->n) * w->k * 2 ||
                !common::operand::aligned(w->qdata, 16))
                throw std::invalid_argument("linear_pair: invalid BF16 weight");
        return;
    }
    (void)select_linear_pair(first.k, max_tokens);
    require_weight(first, first.k);
    require_weight(second, first.k);
    if (first.k == 2048) require_adjacent_parent_views(first, second);
}
} // namespace ninfer::ops::detail
