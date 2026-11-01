#include "core/device.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

struct PlannedPool {
    q36::StateImageDeviceLayout layout;
    std::size_t bytes = 0;
};

PlannedPool plan_pool(bool dflash, std::int32_t slots = 4, bool dflash2 = false, bool ple = false) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = slots,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 7,
    };
    if (dflash) {
        spec.dflash_local = dflash2
                                ? q36::DFlashLocalStateSpec{.layers   = 5,
                                                            .capacity = 2048,
                                                            .kv_heads = 8,
                                                            .head_dim = 128}
                                : q36::DFlashLocalStateSpec{
                                      .layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    // Qwen4Exp PLE convolution history: (kernel-1)*dilation = 9 columns of 5 channels.
    if (ple) { spec.ple = q36::PleStateSpec{.layers = 1, .rows = 45}; }
    ninfer::LayoutBuilder builder;
    q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    return {.layout = std::move(layout), .bytes = builder.finish(256)};
}

void set_bytes(const ninfer::Tensor& tensor, unsigned char value) {
    CUDA_CHECK(cudaMemset(tensor.data, value, tensor.bytes()));
}

void expect_bytes(const ninfer::Tensor& tensor, unsigned char expected, std::string_view label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (const unsigned char value : host) {
        if (value != expected) {
            ++failures;
            std::cerr << "FAIL: " << label << " differs\n";
            return;
        }
    }
}

void fill_slot(q36::StateImageDevicePool& pool, std::int32_t slot, unsigned char base) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        set_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer));
        set_bytes(pool.linear().recurrent_slot(layer, slot),
                  static_cast<unsigned char>(base + 0x10 + layer));
    }
    set_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20));
    for (std::uint32_t layer = 0; layer < pool.ple_layers(); ++layer) {
        set_bytes(pool.ple_states(layer).slice(1, slot, 1),
                  static_cast<unsigned char>(base + 0x50 + layer));
    }
    if (ninfer::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            set_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer));
            set_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer));
        }
    }
    // The fills run on the default stream, which the pool's copies on the context's non-blocking
    // streams do not wait for.
    CUDA_CHECK(cudaDeviceSynchronize());
}

void expect_slot(q36::StateImageDevicePool& pool, std::int32_t slot, unsigned char base,
                 std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer),
                     label);
        expect_bytes(pool.linear().recurrent_slot(layer, slot),
                     static_cast<unsigned char>(base + 0x10 + layer), label);
    }
    expect_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20),
                 label);
    for (std::uint32_t layer = 0; layer < pool.ple_layers(); ++layer) {
        expect_bytes(pool.ple_states(layer).slice(1, slot, 1),
                     static_cast<unsigned char>(base + 0x50 + layer), label);
    }
    if (ninfer::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer),
                         label);
            expect_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer),
                         label);
        }
    }
}

void expect_zero_slot(q36::StateImageDevicePool& pool, std::int32_t slot, std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), 0, label);
        expect_bytes(pool.linear().recurrent_slot(layer, slot), 0, label);
    }
    expect_bytes(pool.continuation_hidden_slot(slot), 0, label);
    for (std::uint32_t layer = 0; layer < pool.ple_layers(); ++layer) {
        expect_bytes(pool.ple_states(layer).slice(1, slot, 1), 0, label);
    }
    if (ninfer::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), 0, label);
            expect_bytes(view.v.slice(3, slot, 1), 0, label);
        }
    }
}

void test_host_roundtrip(bool dflash, ninfer::DeviceContext& device, bool dflash2 = false,
                         bool ple = false) {
    PlannedPool planned = plan_pool(dflash, 2, dflash2, ple);
    if (dflash2) {
        expect(q36::dflash_local_transfer_work(planned.layout.host).payload_bytes ==
                   40ULL * 1024 * 1024,
               "DFlash2 local snapshot must transfer exactly 40 MiB");
    }
    ninfer::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    fill_slot(pool, 0, dflash ? 0x19 : 0x25);
    pool.zero_slot(1, device.stream);

    q36::HostStatePool host(planned.layout.host, 1);
    const auto handle = host.allocate();
    expect(handle.has_value(), "HostStatePool allocates its fixed slot");
    expect(!host.allocate().has_value(), "HostStatePool reports capacity exhaustion");
    expect(host.occupied() == 1, "HostStatePool occupied count after allocation");

    pool.copy_to_host(0, host.writable_view(*handle), device.stream);
    device.synchronize();
    pool.copy_from_host(host.view(*handle), 1, device.stream);
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x19 : 0x25,
                dflash ? "DFlash Host roundtrip" : "common Host roundtrip");

    const q36::HostStateSlotHandle stale = *handle;
    expect(host.release(stale), "HostStatePool releases a live handle");
    expect(host.occupied() == 0, "HostStatePool occupied count after release");
    bool stale_view_rejected = false;
    try {
        (void)host.view(stale);
    } catch (const std::invalid_argument&) { stale_view_rejected = true; }
    expect(stale_view_rejected, "HostStatePool rejects a stale view");
    const auto reused = host.allocate();
    expect(reused && reused->index == stale.index && reused->generation != stale.generation,
           "HostStatePool reuse advances generation");
    expect(!host.release(stale), "HostStatePool rejects stale release");
    expect(host.release(*reused), "HostStatePool releases the reused slot");
}

// tp 2: rank 1's pool mirrors rank 0's slots; a Host image holds rank 0's image, then rank 1's
// (here without the drafter rings, as when the DFlash2 drafter is whole on rank 0).
void test_mirrored_host_roundtrip(ninfer::DeviceContext& device, bool dflash, bool ple) {
    PlannedPool origin_plan = plan_pool(dflash, 2, false, ple);
    PlannedPool mirror_plan = plan_pool(false, 2, false, ple);
    ninfer::DeviceArena origin_arena(origin_plan.bytes);
    ninfer::DeviceArena mirror_arena(mirror_plan.bytes);
    q36::StateImageDevicePool origin({origin_arena.base(), origin_arena.capacity()},
                                     origin_plan.layout);
    q36::StateImageDevicePool mirror({mirror_arena.base(), mirror_arena.capacity()},
                                     mirror_plan.layout);
    ninfer::CudaCompletionEvent order(device);
    origin.attach_mirror(mirror, device.device, device.stream, device.transfer_stream, &order);
    const q36::StateImageHostLayout& paired = origin.host_layout();
    expect(paired.peer_offset == origin_plan.layout.host.image_bytes &&
               paired.peer_bytes == mirror_plan.layout.host.image_bytes &&
               paired.image_bytes == paired.peer_offset + paired.peer_bytes,
           "mirrored StateImage Host image holds rank 0's image, then rank 1's");

    fill_slot(origin, 0, 0x21);
    fill_slot(mirror, 0, 0x71);
    origin.zero_slot(1, device.stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_zero_slot(mirror, 1, "mirrored zero_slot replayed on rank 1");

    q36::HostStatePool single(origin_plan.layout.host, 1);
    bool single_rejected = false;
    try {
        origin.copy_to_host(0, single.writable_view(*single.allocate()), device.transfer_stream);
    } catch (const std::invalid_argument&) { single_rejected = true; }
    expect(single_rejected, "mirrored StateImage rejects a one-rank Host image");

    q36::HostStatePool host(paired, 1);
    const auto handle = host.allocate();
    origin.copy_to_host(0, host.writable_view(*handle), device.transfer_stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    origin.copy_from_host(host.view(*handle), 1, device.transfer_stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_slot(origin, 1, 0x21, "mirrored Host roundtrip, rank 0");
    expect_slot(mirror, 1, 0x71, "mirrored Host roundtrip, rank 1");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    ninfer::DeviceContext device(0);
    PlannedPool planned = plan_pool(true);
    ninfer::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);

    expect(pool.slot_count() == 4, "StateImage slot count");
    expect(pool.linear().layer_count() == 2, "StateImage Linear Attention layer count");
    expect(pool.continuation_hidden_slot(0).dtype == ninfer::DType::BF16 &&
               pool.continuation_hidden_slot(0).ne[0] == 7,
           "StateImage continuation hidden geometry");
    expect(pool.dflash_local() != nullptr && pool.dflash_local()->layer_count() == 2 &&
               pool.dflash_local()->lane_capacity() == 4,
           "StateImage DFlash local geometry");
    const q36::StateImageDeviceSlotView complete = pool.slot_view(0);
    expect(complete.linear.layers == 2 && complete.continuation_hidden.ne[0] == 7 &&
               complete.dflash_local && complete.dflash_local->layers == 2,
           "StateImage slot view exposes every fixed component");

    fill_slot(pool, 0, 0x11);
    fill_slot(pool, 1, 0x52);
    pool.copy_slot(0, 2, device.stream);
    device.synchronize();
    expect_slot(pool, 0, 0x11, "StateImage D2D source isolation");
    expect_slot(pool, 2, 0x11, "StateImage D2D complete destination");

    pool.zero_slot(1, device.stream);
    device.synchronize();
    expect_zero_slot(pool, 1, "StateImage zero complete destination");
    expect_slot(pool, 0, 0x11, "StateImage zero source isolation");

    test_host_roundtrip(false, device);
    test_host_roundtrip(true, device);
    test_host_roundtrip(true, device, true);

    // Qwen4Exp: the PLE convolution history is StateImage state (copy, zero, Host round trip).
    {
        PlannedPool ple_planned = plan_pool(false, 4, false, true);
        ninfer::DeviceArena ple_arena(ple_planned.bytes);
        q36::StateImageDevicePool ple_pool({ple_arena.base(), ple_arena.capacity()},
                                           ple_planned.layout);
        expect(ple_pool.ple_layers() == 1 && ple_pool.ple_states(0).ne[0] == 45 &&
                   ple_pool.ple_states(0).ne[1] == 4 && ple_pool.ple_states(0).is_contiguous(),
               "StateImage PLE geometry");
        const auto common = plan_pool(false, 4);
        expect(q36::state_image_transfer_work(ple_planned.layout.host).payload_bytes ==
                   q36::state_image_transfer_work(common.layout.host).payload_bytes + 45U * 2U,
               "StateImage transfer work counts the PLE history of one image");
        fill_slot(ple_pool, 0, 0x13);
        fill_slot(ple_pool, 1, 0x61);
        ple_pool.copy_slot(0, 3, device.stream);
        ple_pool.zero_slot(1, device.stream);
        device.synchronize();
        expect_slot(ple_pool, 3, 0x13, "StateImage PLE D2D destination");
        expect_zero_slot(ple_pool, 1, "StateImage PLE zero");
        expect_slot(ple_pool, 0, 0x13, "StateImage PLE source isolation");
        test_host_roundtrip(false, device, false, true);
    }
    test_mirrored_host_roundtrip(device, false, false);
    test_mirrored_host_roundtrip(device, true, false);
    test_mirrored_host_roundtrip(device, false, true);

    return failures == 0 ? 0 : 1;
}
