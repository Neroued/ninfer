#include "ops/softmax_attention/kpool_mla/prepare/plan.h"

#include <stdexcept>

namespace ninfer::ops::detail::kpool {

PrepareGeometry prepare_geometry(const KpoolMlaPrepareProfile& p, KpoolMlaStorage storage,
                                 int batch, int lo, int hi, int sms) {
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
    if (storage != KpoolMlaStorage::BFloat16 || p.hidden != 4096 || p.latent_dim != 512 ||
        p.index_dim != 128 || p.query_rank != q || p.index_heads != heads || p.pool_size != 4 ||
        batch < 1 || batch > 8 || lo <= 0 || hi < lo || sms <= 0)
        throw std::invalid_argument("Kpool Prepare: unsupported BF16 geometry or extent");
    return {768 + q + heads, q, heads};
}

int prepare_window_columns() { return 1024; }

ProjectionRoute select_projection(int columns, int rows) {
    if (columns == 1) return ProjectionRoute::Gemv;
    if (columns == 2) return ProjectionRoute::Simt2;
    if (rows == 768) {
        if (columns <= 8) return ProjectionRoute::Sliced8;
        if (columns <= 16) return ProjectionRoute::SlicedHistory;
        if (columns <= 64) return ProjectionRoute::Split32;
        if (columns <= 128) return ProjectionRoute::Split64;
        if (columns <= 512) return ProjectionRoute::TmaSmall;
        return ProjectionRoute::TmaHistory;
    }
    if (columns <= 4) return ProjectionRoute::Simt4;
    if (columns <= 8) return ProjectionRoute::Sliced8;
    if (columns <= 16) return ProjectionRoute::Sliced16;
    if (columns <= 64) return ProjectionRoute::Mma32;
    if (columns <= 128) return ProjectionRoute::TmaSmall;
    return ProjectionRoute::Tma128;
}
} // namespace ninfer::ops::detail::kpool
