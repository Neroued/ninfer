#include "core/device.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
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

PlannedPool plan_pool(bool dflash, std::int32_t slots = 4, bool dflash2 = false) {
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
    if (ninfer::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            set_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer));
            set_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer));
        }
    }
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
    if (ninfer::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), 0, label);
            expect_bytes(view.v.slice(3, slot, 1), 0, label);
        }
    }
}

void test_host_roundtrip(bool dflash, ninfer::DeviceContext& device, bool dflash2 = false) {
    PlannedPool planned = plan_pool(dflash, 2, dflash2);
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

// Segmented Host images (hybrid prefix cache slabs): the packed image split into fixed-size
// segments at scattered addresses must hold exactly the contiguous packed bytes, and restore them.
// An odd segment size splits components mid-way.
void test_segmented_roundtrip(bool dflash, ninfer::DeviceContext& device, bool dflash2 = false) {
    PlannedPool planned = plan_pool(dflash, 2, dflash2);
    ninfer::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    fill_slot(pool, 0, dflash ? 0x31 : 0x47);
    pool.zero_slot(1, device.stream);

    q36::HostStatePool contiguous(planned.layout.host, 1);
    const auto handle = contiguous.allocate();
    pool.copy_to_host(0, contiguous.writable_view(*handle), device.stream);

    const std::size_t image_bytes   = planned.layout.host.image_bytes;
    const std::size_t segment_bytes = 777;
    const std::size_t segments      = (image_bytes + segment_bytes - 1) / segment_bytes;
    ninfer::PinnedHostBuffer backing(segments * segment_bytes);
    auto* base = static_cast<std::byte*>(backing.data());
    std::vector<std::byte*> scattered;
    for (std::size_t segment = 0; segment < segments; ++segment) {
        scattered.push_back(base + (segments - 1U - segment) * segment_bytes);
    }
    pool.copy_to_host_segments(0, scattered, segment_bytes, device.stream);
    device.synchronize();

    const std::byte* packed = contiguous.view(*handle).data;
    bool equal              = true;
    for (std::size_t offset = 0; offset < image_bytes; ++offset) {
        equal =
            equal && scattered[offset / segment_bytes][offset % segment_bytes] == packed[offset];
    }
    expect(equal, "segmented StateImage bytes differ from the packed image");

    const std::vector<const std::byte*> sources(scattered.begin(), scattered.end());
    pool.copy_from_host_segments(sources, segment_bytes, 1, device.stream);
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x31 : 0x47, "segmented Host roundtrip");

    // Part by part (the order a hybrid restore lands them: the rest, then each linear layer)
    // rebuilds the same slot.
    pool.zero_slot(1, device.stream);
    pool.copy_from_host_segments(sources, segment_bytes, 1,
                                 q36::StateImagePart{.kind = q36::StateImagePart::Kind::Rest},
                                 device.stream);
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        pool.copy_from_host_segments(
            sources, segment_bytes, 1,
            q36::StateImagePart{.kind = q36::StateImagePart::Kind::LinearLayer, .layer = layer},
            device.stream);
    }
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x31 : 0x47, "per-part segmented Host roundtrip");
    bool layer_rejected = false;
    try {
        pool.copy_from_host_segments(
            sources, segment_bytes, 1,
            q36::StateImagePart{.kind  = q36::StateImagePart::Kind::LinearLayer,
                                .layer = pool.linear().layer_count()},
            device.stream);
    } catch (const std::out_of_range&) { layer_rejected = true; }
    expect(layer_rejected, "an out-of-range StateImage linear layer was accepted");

    bool short_rejected = false;
    try {
        pool.copy_to_host_segments(0, std::span<std::byte* const>(scattered.data(), 1),
                                   segment_bytes, device.stream);
    } catch (const std::invalid_argument&) { short_rejected = true; }
    expect(short_rejected || segments == 1, "segments that do not cover the image were accepted");
    (void)contiguous.release(*handle);
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
    test_segmented_roundtrip(false, device);
    test_segmented_roundtrip(true, device);
    test_segmented_roundtrip(true, device, true);

    return failures == 0 ? 0 : 1;
}
