#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;
} // namespace

Bf16KvCausalPlan select_bf16_kv_causal_plan(int heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        batch > 8 || (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("BF16 attention: invalid plan inputs");
    const int kv_heads = heads == 24 ? 4 : 2, group = heads / kv_heads;
    const auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    const int visible   = envelope.max_visible_keys;
    if (width > kGroupedPrefillMaxWidth) {
        const auto instance =
            heads == 16 && width >= 1024 && width % 128 == 0 && visible - width >= 8192
                ? Bf16KvInstance::Tiled128
                : Bf16KvInstance::Tiled64;
        return {instance, {}, heads, width, batch, envelope};
    }
    const int rows                  = width * group;
    const auto instance             = width == 1   ? Bf16KvInstance::GroupedDecode
                                      : rows <= 32 ? Bf16KvInstance::Grouped32
                                                   : Bf16KvInstance::Grouped64;
    const auto description          = bf16_kv_instance_description(instance);
    const int tiles                 = ceil_div(rows, description.query_rows);
    const int independent_tiles     = batch * kv_heads * tiles;
    const bool many_memory_tiles    = description.query_rows == 32 && independent_tiles >= 16;
    const bool multiple_query_tiles = tiles > 1;
    const std::int64_t sms          = multiprocessor_count;
    // Small queries permit up to six CTAs per SM, with a half-SM-count split budget per
    // independent query tile. Wider prefill targets two CTAs per SM to limit partial traffic.
    const auto long_ctas = width <= 128 / group && (many_memory_tiles || multiple_query_tiles)
                               ? std::clamp((sms / 2) * independent_tiles, 2 * sms, 6 * sms)
                               : 2 * sms;
    Bf16KvPartition partition{1, causal_partition_target(2 * sms, independent_tiles),
                              causal_partition_target(long_ctas, independent_tiles),
                              description.key_rows};
    // The envelope bounds the largest live row. Other batch rows may be shorter.
    const int low      = batch == 1 ? static_cast<int>(envelope.min_visible_keys) : 1;
    partition.capacity = std::max(
        partition.interval_capacity(low, std::min(visible, 32767), partition.normal_target),
        partition.interval_capacity(std::max(low, 32768), visible, partition.long_target));
    return {instance, partition, heads, width, batch, envelope};
}

std::size_t bf16_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            select_bf16_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, plan.partition.capacity, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

kv::PagedWriteInstance select_bf16_kv_write(int heads) {
    return heads == 4 ? kv::PagedWriteInstance::Copy128 : kv::PagedWriteInstance::Copy96;
}

} // namespace ninfer::ops::detail
