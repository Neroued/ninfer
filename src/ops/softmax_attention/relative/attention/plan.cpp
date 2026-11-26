#include "ops/softmax_attention/relative/attention/plan.h"

#include "core/platform.h"
#include "ops/common/host/graph_profile.h"
#include "ops/common/host/operands.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::ops::relative {

void validate_profile(AttentionHeadGeometry g, KvCacheStorage storage, int sms) {
#if !NINFER_TARGET_RTX5090
    (void)g;
    (void)storage;
    (void)sms;
    throw std::runtime_error("relative_softmax_attention: requires the RTX 5090 build target");
#else
    if (g.head_dim != kDim || g.query_heads != kHeads || g.kv_heads != kKvHeads || sms <= 0 ||
        (storage != KvCacheStorage::BFloat16 && storage != KvCacheStorage::Fp8E4m3fnGroup128Ue8m0))
        throw std::invalid_argument("relative_softmax_attention: unsupported profile");
#endif
}

void validate_window(std::uint32_t window) {
    if (window != kWindow)
        throw std::invalid_argument("relative_softmax_attention: expected window 1024");
}

DecodePlan select_decode(bool swa, KvCacheStorage storage, ContextBucket bucket, int batch,
                         int width, int sms) {
    const int visible = swa ? kWindow : static_cast<int>(context_bucket_tokens(bucket));
    if (batch < 1 || width < 1 || width > kMaxWidth)
        throw std::invalid_argument(
            "relative_softmax_attention: expected positive B and W in 1..16");
    const bool simt  = width == 1;
    const auto heads = std::int64_t(batch) * kKvHeads;
    // Target several resident CTAs for streaming memory, up to one CTA per small key
    // interval. The split budget falls with the number of query fragments per CTA.
    const int query_fragments = (width * kGroup + 15) / 16;
    const bool swa_fp8        = swa && storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    // SWA BF16 decode saturates bandwidth at 2 waves while smaller FP8 tiles need 5 waves
    // to hide memory latency; SWA verify uses >=4 waves so narrow widths (qf=1) split
    // across SMs while wider verify (qf>=2) keeps combine traffic bounded.
    const int waves = swa ? (simt ? (swa_fp8 ? 5 : 2) : std::max(4, 2 * query_fragments))
                          : (simt ? 5 : 8);
    // Compressed Decode can keep more CTAs in flight without increasing cache traffic.
    const bool compressed_decode = simt && storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    const auto budget =
        std::int64_t(waves) * sms + (compressed_decode ? 2 * std::int64_t(sms) / 3 : 0);
    const auto target  = std::max<std::int64_t>(1, budget / (heads * query_fragments));
    const int min_keys = !swa && storage == KvCacheStorage::BFloat16 ? 2 * kKeyTile : kKeyTile;
    const int splits   = static_cast<int>(
        std::min({target, std::int64_t(visible / min_keys), std::int64_t(kMaxSplits)}));
    return {batch, width, splits, simt};
}

std::size_t DecodePlan::rows() const { return std::uint64_t(batch) * width * kHeads * splits; }

std::size_t DecodePlan::bytes() const {
    if (splits == 1) return 0;
    return common::operand::bytes(rows(), (kDim + 1) * sizeof(float), kOp);
}

} // namespace ninfer::ops::relative

namespace ninfer::ops {
namespace {

using namespace relative;

std::size_t decode_capacity(AttentionHeadGeometry geometry, KvCacheStorage storage, bool swa,
                            ContextBucketRange buckets, int batch, int lo, int hi, int sms) {
    validate_profile(geometry, storage, sms);
    if (lo > hi || lo < 1 || hi > kMaxWidth)
        throw std::invalid_argument("relative_softmax_attention: invalid width envelope");
    std::size_t peak = 0;
    buckets.for_each([&](auto bucket) {
        for (int width = lo; width <= hi; ++width)
            peak = std::max(peak, select_decode(swa, storage, bucket, batch, width, sms).bytes());
    });
    return peak;
}

OpGraphProfile graph_profile(AttentionHeadGeometry geometry, KvCacheStorage storage, bool swa,
                             ContextBucket bucket, int batch, int width, int sms, bool masked) {
    validate_profile(geometry, storage, sms);
    (void)context_bucket_tokens(bucket);
    const auto plan = select_decode(swa, storage, bucket, batch, width, sms);
    // Both W=1 SWA forms bind the optional chunk in the same kernel argument.
    // Presence requires neither a different launch sequence nor different scratch.
    const std::array words{std::uint64_t(swa),         std::uint64_t(storage),
                           std::uint64_t(batch),       std::uint64_t(width),
                           std::uint64_t(plan.splits), std::uint64_t(plan.simt),
                           std::uint64_t(masked)};
    return detail::make_op_graph_profile(words);
}

} // namespace

std::size_t relative_softmax_attention_swa_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, std::uint32_t window,
    ContextBucketRange buckets, int batch, int lo, int hi, int sms) {
    validate_window(window);
    return decode_capacity(geometry, storage, true, buckets, batch, lo, hi, sms);
}

std::size_t relative_softmax_attention_full_workspace_capacity_bytes(AttentionHeadGeometry geometry,
                                                                     KvCacheStorage storage,
                                                                     ContextBucketRange buckets,
                                                                     int batch, int lo, int hi,
                                                                     int sms) {
    return decode_capacity(geometry, storage, false, buckets, batch, lo, hi, sms);
}

std::size_t relative_softmax_attention_ragged_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, bool, ContextBucketRange buckets,
    int min_tokens, int max_tokens, int min_segments, int max_segments, int sms) {
    validate_profile(geometry, storage, sms);
    buckets.validate();
    common::operand::ragged_envelope(min_tokens, max_tokens, min_segments, max_segments, kOp);
    const int segments = std::min(max_segments, max_tokens);
    if (segments == 1) return 0;
    return common::operand::bytes(ragged_tiles(max_tokens, segments), sizeof(RaggedTile), kOp);
}

OpGraphProfile relative_softmax_attention_swa_graph_profile(AttentionHeadGeometry geometry,
                                                            KvCacheStorage storage,
                                                            std::uint32_t window,
                                                            ContextBucket bucket, int batch,
                                                            int width, int sms, bool masked) {
    validate_window(window);
    return graph_profile(geometry, storage, true, bucket, batch, width, sms, masked);
}

OpGraphProfile relative_softmax_attention_full_graph_profile(AttentionHeadGeometry geometry,
                                                             KvCacheStorage storage,
                                                             ContextBucket bucket, int batch,
                                                             int width, int sms, bool masked) {
    return graph_profile(geometry, storage, false, bucket, batch, width, sms, masked);
}

} // namespace ninfer::ops
