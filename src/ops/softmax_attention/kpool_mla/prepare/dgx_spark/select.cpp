#include "ops/softmax_attention/kpool_mla/prepare/plan.h"

#include <stdexcept>

namespace ninfer::ops::detail::kpool {

PrepareGeometry prepare_geometry(const KpoolMlaPrepareProfile& p, KpoolMlaStorage storage,
                                 KvCacheVectorFormat index_format, int batch, int lo, int hi,
                                 int sms) {
    int q = 0, heads = 0;
    switch (p.projection) {
    case KpoolMlaPrepareProjection::Full:
        q     = 1536;
        heads = 32;
        break;
    case KpoolMlaPrepareProjection::QueryAndHistory:
        q = 1536;
        break;
    case KpoolMlaPrepareProjection::History:
        break;
    default:
        throw std::invalid_argument("Kpool Prepare: invalid projection");
    }
    if ((storage != KpoolMlaStorage::BFloat16 &&
         storage != KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0) ||
        (index_format != KvCacheVectorFormat::BFloat16 &&
         index_format != KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0) ||
        p.rows.latent.rows != 512 || p.rows.index_key.rows != 128 || p.rows.pool_gate.rows != 128 ||
        p.rows.query.rows != q || p.rows.coefficients.rows != heads ||
        p.pool_size != kKpoolPoolSize || batch < 1 || batch > 8 || lo <= 0 || hi < lo || sms <= 0)
        throw std::invalid_argument("Kpool Prepare: unsupported geometry, format or extent");
    const auto valid_weight = [](const LinearProfile& w) {
        return w.k == 4096 && w.n > 0 && valid_linear_policy(w.policy) &&
               ((w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous) ||
                (w.qtype == QType::MXFP8 && w.layout == QuantLayout::RowGroupScale) ||
                (w.qtype == QType::FP8_E4M3FN_BLOCK128X128_FP32 &&
                 w.layout == QuantLayout::BlockScaleM128K128 && w.n % 128 == 0));
    };
    if (!valid_weight(p.weights.first) || (p.weights.second && !valid_weight(*p.weights.second)))
        throw std::invalid_argument("Kpool Prepare: unsupported projection format or policy");
    const std::array roles{p.rows.latent, p.rows.index_key, p.rows.pool_gate, p.rows.query,
                           p.rows.coefficients};
    int covered[2]{};
    for (std::size_t i = 0; i < roles.size(); ++i) {
        const auto& r = roles[i];
        if (!r.rows) continue;
        if (r.parent < 0 || r.parent > int(p.weights.second.has_value()) || r.row_begin < 0 ||
            r.row_begin % 4 || r.rows % 4)
            throw std::invalid_argument("Kpool Prepare: invalid projection row mapping");
        const auto& w = r.parent ? *p.weights.second : p.weights.first;
        if (std::int64_t(r.row_begin) + r.rows > w.n)
            throw std::invalid_argument("Kpool Prepare: projection role exceeds parent");
        for (std::size_t j = 0; j < i; ++j) {
            const auto& other = roles[j];
            if (other.rows && r.parent == other.parent &&
                r.row_begin < other.row_begin + other.rows &&
                other.row_begin < r.row_begin + r.rows)
                throw std::invalid_argument("Kpool Prepare: overlapping projection roles");
        }
        covered[r.parent] += r.rows;
    }
    if (covered[0] != p.weights.first.n ||
        covered[1] != (p.weights.second ? p.weights.second->n : 0))
        throw std::invalid_argument("Kpool Prepare: projection roles do not cover parents");
    return {768 + q + heads, q, heads, p.weights};
}

int prepare_window_columns() { return 1024; }

struct SelectedProjection {
    ProjectionRoute route;
    int last_column;
};

static SelectedProjection select_bf16_projection(int columns, int rows) {
    if (columns == 1) return {ProjectionRoute::Gemv, 1};
    if (columns == 2) return {ProjectionRoute::Simt2, 2};
    if (rows == 768) {
        if (columns <= 8) return {ProjectionRoute::Sliced8, 8};
        if (columns <= 16) return {ProjectionRoute::SlicedHistory, 16};
        if (columns <= 64) return {ProjectionRoute::Split32, 64};
        if (columns <= 128) return {ProjectionRoute::Split64, 128};
        if (columns <= 512) return {ProjectionRoute::TmaSmall, 512};
        return {ProjectionRoute::TmaHistory, std::numeric_limits<int>::max()};
    }
    if (columns <= 4) return {ProjectionRoute::Simt4, 4};
    if (columns <= 8) return {ProjectionRoute::Sliced8, 8};
    if (columns <= 16) return {ProjectionRoute::Sliced16, 16};
    if (columns <= 64) return {ProjectionRoute::Mma32, 64};
    if (columns <= 128) return {ProjectionRoute::TmaSmall, 128};
    return {ProjectionRoute::Tma128, std::numeric_limits<int>::max()};
}

PrepareProjectionPlan prepare_projection_plan(const PrepareGeometry& geometry, int columns) {
    const bool single_bf16 =
        !geometry.weights.second && geometry.weights.first.qtype == QType::BF16;
    const auto select = [&](const LinearProfile& weight) -> SelectedProjection {
        if (weight.qtype == QType::BF16) {
            if (!single_bf16 && weight.n <= 288 && columns > 32 && columns <= 256)
                return {ProjectionRoute::Mma32, 256};
            auto selected = select_bf16_projection(columns, weight.n);
            if (!single_bf16 && projection_splits(selected.route) > 1)
                selected.route = ProjectionRoute::Mma32;
            return selected;
        }
        if (weight.qtype == QType::MXFP8 && allows_a8(weight.policy) && columns > 32) {
            if (columns <= 128) return {ProjectionRoute::Mxfp8A8Short, 128};
            if (columns <= 256) return {ProjectionRoute::Mxfp8A8Medium, 256};
            return {ProjectionRoute::Mxfp8A8Large, std::numeric_limits<int>::max()};
        }
        if (columns == 1) return {ProjectionRoute::Fp8Simt1, 1};
        if (columns <= 4 && weight.n <= 768) return {ProjectionRoute::Fp8Simt4, 4};
        if (columns <= 8) return {ProjectionRoute::Fp8Sliced8, 8};
        if (columns <= 32)
            return {weight.n <= 768 ? ProjectionRoute::Fp8Sliced8 : ProjectionRoute::Fp8Sliced16,
                    32};
        return columns <= 256
                   ? SelectedProjection{ProjectionRoute::Fp8Mma32, 256}
                   : SelectedProjection{ProjectionRoute::Fp8Mma64, std::numeric_limits<int>::max()};
    };
    const auto first  = select(geometry.weights.first);
    const auto second = geometry.weights.second ? select(*geometry.weights.second) : first;
    auto paired       = PairedProjectionRoute::None;
    const auto encoded_format =
        geometry.weights.first.qtype == QType::BF16 && geometry.weights.second
            ? geometry.weights.second->qtype
            : geometry.weights.first.qtype;
    const int pair_limit = geometry.query_rank == 0 ? 64 : encoded_format == QType::MXFP8 ? 48 : 32;
    if (geometry.weights.second && columns <= pair_limit &&
        ((geometry.weights.first.qtype == QType::BF16) !=
         (geometry.weights.second->qtype == QType::BF16))) {
        paired = columns == 1   ? PairedProjectionRoute::Simt1
                 : columns <= 8 ? PairedProjectionRoute::Sliced8
                                : PairedProjectionRoute::Sliced16;
    }
    const bool activation_quantization =
        paired == PairedProjectionRoute::None &&
        (projection_uses_a8(first.route) ||
         (geometry.weights.second && projection_uses_a8(second.route)));
    const int last = paired == PairedProjectionRoute::Simt1      ? 1
                     : paired == PairedProjectionRoute::Sliced8  ? 8
                     : paired == PairedProjectionRoute::Sliced16 ? pair_limit
                                                                 : std::numeric_limits<int>::max();
    return {first.route,
            second.route,
            activation_quantization,
            single_bf16 ? projection_splits(first.route) : 1,
            paired,
            paired == PairedProjectionRoute::None || columns == 1 ||
                (columns > 1 &&
                 !(encoded_format == QType::MXFP8 && (geometry.query_rank == 0 || columns > 32 ||
                                                      (geometry.heads == 0 && columns > 16)))),
            std::min({first.last_column, second.last_column, last})};
}

} // namespace ninfer::ops::detail::kpool
