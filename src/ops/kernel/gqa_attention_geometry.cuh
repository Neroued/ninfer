#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    // Ceiling on split-K over the KV length for the small-T decode attention kernel.
    //
    // Was 85, which put both registered geometries at 4 x 85 = 340 CTAs -- about one wave. That
    // is the right target only while the split *policy* asks for less: gqa_small_t_split_upper_bound
    // asks for window/480 above 16390 keys, so the ceiling starts binding at ~41k context and
    // then holds the split count fixed while each CTA's key range grows without bound. At 82k
    // that is 965 keys per CTA instead of the 479 the policy wanted.
    //
    // Measured on the 27B geometry (d256-h24-kv4, int8 cache, width 4), one attention call:
    //
    //   ceiling:      85      170     340
    //   ctx 81920:  6562us   5052us  5037us      (policy wants 171; 170 already captures it)
    //
    // and end to end on an 81626-token prompt, same server command otherwise:
    //
    //   ceiling 85 : decode 21.0 tok/s, 3.60 tok/round -> 171.4ms per round
    //   ceiling 560: decode 24.5 tok/s, 3.55 tok/round -> 145.1ms per round   (+16.7%)
    //
    // The 26.5ms per round recovered matches 16 layers x the 1.52ms per call the op bench shows.
    // Nothing below ~41k context changes -- 16k and 32k measure identically either side -- and the
    // extra fp32 partial buffers cost nothing that shows: the server reports the same 14.09 GiB
    // runtime at max-context 262144 both ways, because the workspace is sized from the actual
    // envelope rather than this ceiling.
    //
    // 560 is chosen so the ceiling never binds within the engine's 262144 max context
    // (262144 / 480 = 546). It is a bound, not a target; the policy above still picks the count.
    static constexpr int DecodeSplits     = 560 * DecodeSplitScale;
};

using Gqa27Geometry = GqaGeometry<24, 4, 1>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;

} // namespace ninfer::ops
