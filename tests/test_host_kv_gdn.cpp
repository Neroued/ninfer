#include "core/device.h"
#include "core/host_kv_budget.h"
#include "core/linear_attention_state.h"

#include "targets/qwen3_6/impl/runtime/host_kv_sequence.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using ninfer::targets::qwen3_6::detail::park_gdn_slot;
using ninfer::targets::qwen3_6::detail::restore_gdn_slot;

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%-64s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) { ++failures; }
}

// A deterministic byte pattern over the whole pool backing, so a park/restore
// that drops or scrambles any GDN byte is caught.
std::vector<std::uint8_t> pattern(std::size_t bytes, std::uint8_t seed) {
    std::vector<std::uint8_t> out(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        out[i] = static_cast<std::uint8_t>((i * 31 + seed) & 0xFF);
    }
    return out;
}

}  // namespace

int main() {
    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }

    ninfer::DeviceContext ctx(0);
    const cudaStream_t stream = ctx.stream;

    // A small GDN pool: 3 layers, 2 slots (the current and the
    // rewrite-checkpoint slot, as LinearStateSlots lays them out).
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_linear_attention_state_pool(
        builder, ninfer::LinearAttentionStatePoolSpec{.layers         = 3,
                                                      .conv_channels  = 10,
                                                      .conv_width     = 3,
                                                      .value_heads    = 4,
                                                      .value_head_dim = 5,
                                                      .key_head_dim   = 6,
                                                      .slot_count     = 2,
                                                      .conv_dtype     = ninfer::DType::BF16});
    const std::size_t bytes = builder.finish(256);
    ninfer::DeviceArena arena(bytes);
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, layout);

    // Fill the whole backing with a known pattern, then capture it.
    const auto before = pattern(bytes, 7);
    (void)cudaMemcpyAsync(arena.base(), before.data(), bytes, cudaMemcpyHostToDevice, stream);
    (void)cudaStreamSynchronize(stream);

    // Park both slots into a slab, exactly as park_lane_to_host does. The budget
    // must hold the aligned request: acquire() rounds the region up to 64 bytes,
    // so a budget of exactly 2 * slot_bytes would be too small when it is not
    // 64-aligned.
    const std::size_t slot_bytes = pool.slot_bytes();
    const std::size_t gdn_budget = (2 * slot_bytes + 63) & ~std::size_t(63);
    ninfer::HostKvBudget slab_arena(gdn_budget);
    ninfer::HostKvSlab* slab = slab_arena.acquire(2 * slot_bytes);
    check(slab != nullptr, "a slab is available for the GDN state");
    std::size_t gdn_bytes = park_gdn_slot(pool, 0, *slab, 0, stream);
    gdn_bytes += park_gdn_slot(pool, 1, *slab, gdn_bytes, stream);
    (void)cudaStreamSynchronize(stream);
    check(gdn_bytes == 2 * slot_bytes, "parking both slots writes exactly 2 * slot_bytes");
    // park_lane_to_host records the filled extent on the slab after the park.
    slab->set_filled(gdn_bytes);
    check(slab->filled() == 2 * slot_bytes, "the slab records the parked GDN bytes");

    // Simulate the side request that runs in the same lane: ordered_reset()
    // zeroes the current slot, and a new checkpoint capture overwrites the
    // checkpoint slot. Without parking the GDN state, a restore would put the
    // sequence back with a zeroed recurrent state and produce wrong tokens.
    pool.zero_slot(0, stream);
    pool.zero_slot(1, stream);
    (void)cudaStreamSynchronize(stream);

    // Restore both slots from the slab, exactly as restore_lane_from_host does.
    restore_gdn_slot(pool, 0, *slab, 0, stream);
    restore_gdn_slot(pool, 1, *slab, slot_bytes, stream);
    (void)cudaStreamSynchronize(stream);

    std::vector<std::uint8_t> after(bytes);
    (void)cudaMemcpyAsync(after.data(), arena.base(), bytes, cudaMemcpyDeviceToHost, stream);
    (void)cudaStreamSynchronize(stream);
    check(before == after, "GDN state round-trips bitwise across a lane zero-out");

    // A slot that was not parked must stay zero: the restore only touches the
    // regions the park wrote, so a missing park cannot be masked by a restore.
    std::vector<std::uint8_t> zeroed(bytes, 0);
    (void)cudaMemcpyAsync(arena.base(), zeroed.data(), bytes, cudaMemcpyHostToDevice, stream);
    (void)cudaStreamSynchronize(stream);
    restore_gdn_slot(pool, 0, *slab, 0, stream);  // restore only slot 0
    (void)cudaStreamSynchronize(stream);

    // Expected: the slot-0 regions (every layer's conv + recurrent, slot 0)
    // hold the pattern again; the slot-1 regions and all padding are zero.
    std::vector<std::uint8_t> expected(bytes, 0);
    const auto* arena_base = static_cast<const std::uint8_t*>(arena.base());
    for (std::uint32_t layer = 0; layer < pool.layer_count(); ++layer) {
        const auto c0 = pool.conv_slot(layer, 0);
        const auto r0 = pool.recurrent_slot(layer, 0);
        auto* c0p = static_cast<std::uint8_t*>(c0.data);
        auto* r0p = static_cast<std::uint8_t*>(r0.data);
        for (std::size_t i = 0; i < c0.bytes(); ++i) {
            expected[c0p - arena_base + i] = before[c0p - arena_base + i];
        }
        for (std::size_t i = 0; i < r0.bytes(); ++i) {
            expected[r0p - arena_base + i] = before[r0p - arena_base + i];
        }
    }

    std::vector<std::uint8_t> partial(bytes);
    (void)cudaMemcpyAsync(partial.data(), arena.base(), bytes, cudaMemcpyDeviceToHost, stream);
    (void)cudaStreamSynchronize(stream);
    check(partial == expected, "a parked slot is restored; an un-parked slot stays zero");

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}