#include "ops/softmax_attention/kpool_mla/sparse_mla/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

#include <numeric>

namespace ninfer::ops::detail::kpool {
void sparse_mla_geometry(const KpoolSparseMlaProfile& p, KpoolMlaStorage storage,
                         KpoolExecutionEnvelope envelope, int batch, int lo, int hi, int sms) {
    require(p.query_heads == 64 && p.query_dim == 256 && p.value_dim == 256 &&
                p.latent_dim == 512 && p.pool_size == 4 && p.selected_pools == 512,
            "unsupported Sparse MLA profile");
    require(storage == KpoolMlaStorage::BFloat16, "Sparse MLA requires BF16 latent");
    require(lo > 0 && hi >= lo && batch >= 1 && batch <= 8 && sms > 0, "Sparse MLA T/B/device");
    require(envelope.max_visible_tokens <= kKpoolMaxVisibleTokens, "Sparse MLA context limit");
}

SparseMlaPlan sparse_mla_plan(int columns, int batch, KpoolExecutionEnvelope envelope, int sms) {
    const int bound = std::min(envelope.max_visible_tokens, 2051u);
    SparseMlaPlan p{};
    p.factor = columns <= 32   ? MlaFactorSchedule::Small
               : columns <= 64 ? MlaFactorSchedule::Medium
                               : MlaFactorSchedule::Bulk;
    // Narrow head tiles fill a sparse final SM wave. Once the request working
    // set streams from DRAM, the smaller token tile also limits outstanding data.
    const bool short_context = bound <= 128;
    const bool short_wave = batch == 1 && envelope.max_visible_tokens <= 65536 && columns > sms &&
                            columns <= sms * 4 / 3;
    const bool streaming = envelope.max_visible_tokens >= 262144 && columns >= 128;
    p.attention          = columns <= 3 || short_context || short_wave || streaming
                               ? MlaAttentionSchedule::Small
                               : MlaAttentionSchedule::Bulk;
    p.tile               = p.attention == MlaAttentionSchedule::Small ? 16 : 32;
    p.direct             = bound <= 1;
    p.splits             = 1;
    if (p.direct) return p;
    if (columns <= 8) {
        const auto target = (columns <= 4 ? std::int64_t(sms) : std::int64_t(sms) * 4 / 3);
        p.splits          = std::clamp<std::int64_t>(target / (columns * 2), 1, 32);
    } else if (columns <= sms / 2)
        p.splits = std::min(32, sms / columns);
    if (short_context && columns >= 4) p.splits = 1;
    p.splits = std::min(p.splits, (bound + p.tile - 1) / p.tile);
    // Two partials fit the fused value map's on-chip reduction without the
    // repeated partial traffic that outweighs fusion at larger split counts.
    p.merge_value = columns > 16 && p.splits == 2;
    LayoutBuilder builder;
    // Unsplit attention replaces q_latent with u after the last read. Split
    // partials feed either a merge into this region or the fused value map.
    p.mapped = builder.add_tensor(DType::BF16, {512, 64, columns}, 256, "MLA mapped rows");
    if (p.splits > 1) {
        p.numerator = builder.add_tensor(DType::FP32, {512, 64, p.splits, columns}, 256,
                                         "MLA split numerator");
        p.maximum =
            builder.add_tensor(DType::FP32, {64, p.splits, columns}, 256, "MLA split maximum");
        p.denominator =
            builder.add_tensor(DType::FP32, {64, p.splits, columns}, 256, "MLA split denominator");
    }
    p.bytes = builder.finish(1);
    return p;
}

std::size_t sparse_mla_capacity(int lo, int hi, int batch, KpoolExecutionEnvelope e, int sms) {
    std::size_t peak = 0;
    if (std::int64_t(hi) * batch >= kMlaWindow)
        peak = sparse_mla_plan(kMlaWindow, batch, e, sms).bytes;
    // Full windows are fixed. Tail widths repeat every window/gcd(window,B) values of T.
    const int period = kMlaWindow / std::gcd(kMlaWindow, batch);
    const auto stop  = std::min<std::int64_t>(hi, std::int64_t(lo) + period - 1);
    for (std::int64_t t = lo; t <= stop; ++t) {
        const int tail = t * batch % kMlaWindow;
        if (tail) peak = std::max(peak, sparse_mla_plan(tail, batch, e, sms).bytes);
    }
    return peak;
}
} // namespace ninfer::ops::detail::kpool
