#include "ops/softmax_attention/kpool_mla/sparse_mla/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

#include <numeric>

namespace ninfer::ops::detail::kpool {
void sparse_mla_geometry(const KpoolSparseMlaProfile& p, KpoolMlaStorage storage,
                         KpoolExecutionEnvelope envelope, int batch, int lo, int hi, int sms) {
    require(p.query_heads == 64 && p.query_dim == 256 && p.value_dim == 256 &&
                p.latent_dim == 512 && p.pool_size == 4 && p.selected_pools == 512,
            "unsupported Sparse MLA profile");
    require((p.key_format == QType::BF16 || p.key_format == QType::MXFP8) &&
                (p.value_format == QType::BF16 || p.value_format == QType::MXFP8),
            "unsupported Sparse MLA factor format");
    require(storage == KpoolMlaStorage::BFloat16 ||
                storage == KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0,
            "unsupported Sparse MLA latent format");
    require(lo > 0 && hi >= lo && batch >= 1 && batch <= 8 && sms > 0, "Sparse MLA T/B/device");
    require(envelope.max_visible_tokens <= kKpoolMaxVisibleTokens, "Sparse MLA context limit");
}

SparseMlaPlan sparse_mla_plan(const KpoolSparseMlaProfile& profile, KpoolMlaStorage storage,
                              int columns, int batch, KpoolExecutionEnvelope envelope, int sms) {
    const bool fp8  = storage == KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0;
    const int bound = std::min(envelope.max_visible_tokens, 2051u);
    SparseMlaPlan p{};
    p.factor                 = columns <= 32   ? MlaFactorSchedule::Small
                               : columns <= 64 ? MlaFactorSchedule::Medium
                                               : MlaFactorSchedule::Bulk;
    const bool short_context = bound <= 128;
    if (fp8) {
        // Q8 epilogues own a complete G128. Wider query tiles amortize the
        // weight traffic; wide calls use a smaller K stage to keep two CTAs live.
        p.fp8_query = columns <= 8    ? MlaFactorSchedule::Small
                      : columns <= 64 ? MlaFactorSchedule::Medium
                                      : MlaFactorSchedule::Bulk;
        p.attention = columns <= 3 || short_context ? MlaAttentionSchedule::Small
                                                    : MlaAttentionSchedule::Bulk;
    } else {
        // BF16's larger cache tiles have their own wave and streaming policy.
        const bool short_wave = batch == 1 && envelope.max_visible_tokens <= 65536 &&
                                columns > sms && columns <= sms * 4 / 3;
        const bool streaming = envelope.max_visible_tokens >= 262144 && columns >= 128;
        p.attention          = columns <= 3 || short_context || short_wave || streaming
                                   ? MlaAttentionSchedule::Small
                                   : MlaAttentionSchedule::Bulk;
    }
    p.tile   = p.attention == MlaAttentionSchedule::Small ? 16 : fp8 && columns >= 49 ? 64 : 32;
    p.direct = bound <= 1;
    p.splits = 1;
    if (p.direct) return p;
    if (columns <= 8) {
        const auto target = (columns <= 4 ? std::int64_t(sms) : std::int64_t(sms) * 4 / 3);
        p.splits          = std::clamp<std::int64_t>(target / (columns * 2), 1, 32);
    } else if (columns <= sms / 2)
        p.splits = std::min(32, sms / columns);
    const bool native_value = profile.value_format == QType::MXFP8;
    const bool widen_small =
        fp8 ? !(envelope.max_visible_tokens > 65536 && !native_value && columns >= 5 &&
                columns <= 6)
            : native_value && envelope.max_visible_tokens <= 65536 && columns >= 6;
    if (widen_small && batch == 1 && columns >= 4 && columns <= 8)
        p.splits = std::max(1, sms / columns);
    if (short_context && columns >= 4) p.splits = 1;
    if (fp8 && p.tile == 64) p.splits = 1;
    p.splits = std::min(p.splits, std::max(1, (bound + p.tile - 1) / p.tile));
    // Tiny native-factor calls give one CTA the complete output head, so merge
    // reads each partial once. Wider calls fuse only the two-partial route.
    p.merge_value =
        (profile.value_format == QType::MXFP8 && columns >= 5 && columns <= 8 && p.splits > 1) ||
        (columns > 16 && p.splits == 2);
    // In this narrow wave, padding can leave half the SMs idle. Reserve both
    // decompositions; device validity selects the wider split only when its
    // active CTAs still fit one wave. Full batches retain their shorter split.
    if (fp8 && batch > 1 && columns >= 13 && columns <= 24 && !short_context && p.splits < 4) {
        p.preferred_splits = p.splits;
        p.splits           = 4;
    }
    LayoutBuilder builder;
    if (fp8) {
        p.query_fp8 =
            builder.add_tensor(DType::FP8_E4M3FN, {512, 64, columns}, 256, "MLA FP8 query");
        p.query_scales = builder.add_tensor(DType::U8, {4, 64, columns}, 256, "MLA query scales");
    }
    // BF16 unsplit attention replaces q_latent with u after the last read.
    // Compact Q8 remains disjoint: its row stride differs from u's. Split
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

std::size_t sparse_mla_capacity(const KpoolSparseMlaProfile& profile, KpoolMlaStorage storage,
                                int lo, int hi, int batch, KpoolExecutionEnvelope e, int sms) {
    std::size_t peak = 0;
    if (std::int64_t(hi) * batch >= kMlaWindow)
        peak = sparse_mla_plan(profile, storage, kMlaWindow, batch, e, sms).bytes;
    // Full windows are fixed. Tail widths repeat every window/gcd(window,B) values of T.
    const int period = kMlaWindow / std::gcd(kMlaWindow, batch);
    const auto stop  = std::min<std::int64_t>(hi, std::int64_t(lo) + period - 1);
    for (std::int64_t t = lo; t <= stop; ++t) {
        const int tail = t * batch % kMlaWindow;
        if (tail)
            peak = std::max(peak, sparse_mla_plan(profile, storage, tail, batch, e, sms).bytes);
    }
    return peak;
}
} // namespace ninfer::ops::detail::kpool
