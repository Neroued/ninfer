#include "core/device.h"
#include "core/buffer.h"
#include "core/paged_kv_geometry.h"
#include "models/common/context/directory/allocation.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36     = ninfer::models::qwen3_5;
namespace context = ninfer::models::context;

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
        .slot_count = slots,
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden       = 7,
        .dflash_local = std::nullopt,
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

void set_bytes(const ninfer::Tensor& tensor, unsigned char value,
               ninfer::CudaExecutionView execution) {
    CUDA_CHECK(cudaMemsetAsync(tensor.data, value, tensor.bytes(), execution.stream()));
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

void fill_slot(q36::StateImageViews& pool, std::int32_t slot, unsigned char base,
               ninfer::CudaExecutionView execution) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        set_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer),
                  execution);
        set_bytes(pool.linear().recurrent_slot(layer, slot),
                  static_cast<unsigned char>(base + 0x10 + layer), execution);
    }
    set_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20),
              execution);
    if (ninfer::CyclicKVCacheView* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            set_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer),
                      execution);
            set_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer),
                      execution);
        }
    }
}

void expect_slot(q36::StateImageViews& pool, std::int32_t slot, unsigned char base,
                 std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer),
                     label);
        expect_bytes(pool.linear().recurrent_slot(layer, slot),
                     static_cast<unsigned char>(base + 0x10 + layer), label);
    }
    expect_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20),
                 label);
    if (ninfer::CyclicKVCacheView* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer),
                         label);
            expect_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer),
                         label);
        }
    }
}

void expect_zero_slot(q36::StateImageViews& pool, std::int32_t slot, std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), 0, label);
        expect_bytes(pool.linear().recurrent_slot(layer, slot), 0, label);
    }
    expect_bytes(pool.continuation_hidden_slot(slot), 0, label);
    if (ninfer::CyclicKVCacheView* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), 0, label);
            expect_bytes(view.v.slice(3, slot, 1), 0, label);
        }
    }
}

void test_host_roundtrip(bool dflash, ninfer::DeviceContext& device, bool dflash2 = false) {
    PlannedPool planned = plan_pool(dflash, 2, dflash2);
    ninfer::DeviceBuffer storage(device.device(), planned.bytes);
    q36::StateImageViews pool(storage.span(), planned.layout);
    if (dflash2) {
        expect(pool.dflash_local_transfer_work().payload_bytes == 40ULL * 1024 * 1024,
               "DFlash2 local snapshot transfers 40 MiB");
    }
    fill_slot(pool, 0, dflash ? 0x19 : 0x25, device.execution_view());
    pool.storage().clear(1, device.execution_view());
    context::HostAllocator host(0, planned.layout.images.image_stride,
                                planned.layout.images.image_stride);
    ninfer::PinnedHostBuffer host_bytes(host.capacity_bytes());
    auto allocation = host.allocate(pool.storage().image_bytes());
    expect(allocation && !host.can_allocate(pool.storage().image_bytes()), "Host reservation");
    pool.storage().copy_to_host(
        0, host_bytes.span().subspan(allocation->extent().offset, allocation->bytes()),
        device.execution_view());
    device.synchronize();
    allocation->publish();
    expect(host.reserved_bytes() == 0 && host.live_bytes() == pool.storage().image_bytes(),
           "Host publication preserves one allocation");
    pool.storage().copy_from_host(
        host_bytes.span().subspan(allocation->extent().offset, allocation->bytes()), 1,
        device.execution_view());
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x19 : 0x25, "complete Host roundtrip");
    if (dflash) {
        fill_slot(pool, 1, 0x61, device.execution_view());
        pool.storage().copy_group(pool.dflash_local_group(), 0, 1, device.execution_view());
        device.synchronize();
        expect_bytes(pool.linear().conv_slot(0, 1), 0x61, "local copy preserves GDN");
        expect_bytes(pool.continuation_hidden_slot(1), 0x81, "local copy preserves hidden");
        for (std::uint32_t layer = 0; layer < pool.dflash_local()->layer_count(); ++layer) {
            auto view = pool.dflash_local()->layer_view(layer);
            expect_bytes(view.k.slice(3, 1, 1), 0x19 + 0x30 + layer, "local K copy");
            expect_bytes(view.v.slice(3, 1, 1), 0x19 + 0x40 + layer, "local V copy");
        }
    }
    allocation.reset();
    expect(host.occupied_bytes() == 0, "Host image owner releases its allocation");
}

void test_shared_host_capacity() {
    const PlannedPool planned           = plan_pool(false, 2);
    const ninfer::HostKVPageLayout page = ninfer::plan_host_kv_page_layout(
        {.page_tokens = 1,
         .planes      = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 1}}});
    const std::size_t image_bytes = planned.layout.images.image_stride;
    expect(image_bytes % page.page_stride == 0,
           "shared Host fixture image consists of complete KV page extents");
    const auto pages_per_image = static_cast<std::uint32_t>(image_bytes / page.page_stride);
    context::HostAllocator backing(0, image_bytes * 2, std::min(image_bytes, page.page_stride));
    auto source                = backing.allocate(image_bytes);
    const auto original_offset = source->extent().offset;
    source->publish();
    auto destination = backing.allocate(page.page_stride * pages_per_image);
    expect(destination && backing.live_bytes() == image_bytes &&
               backing.reserved_bytes() == image_bytes && !backing.can_allocate(image_bytes) &&
               !backing.allocate(image_bytes) && !backing.allocate(page.page_stride),
           "State and in-flight KV destination consume the same actual backing");
    destination.reset();
    expect(backing.reserved_bytes() == 0 && backing.live_bytes() == image_bytes &&
               source->extent().offset == original_offset,
           "KV transfer cancellation retains the pinned State source");
    expect(source->release(), "shared Host state release");
    auto all_kv = backing.allocate(page.page_stride * pages_per_image * 2);
    expect(all_kv && all_kv->extent().offset == original_offset &&
               backing.occupied_bytes() == backing.capacity_bytes(),
           "KV reuses the exact physical extent previously occupied by State");
    all_kv->publish();
    auto parts = backing.split(std::move(*all_kv), page.page_stride * pages_per_image);
    expect(!all_kv->valid() && backing.live_bytes() == image_bytes * 2,
           "KV split transfers ownership while keeping unique byte accounting");
    (void)parts.first.release();
    auto recovered = backing.allocate(image_bytes);
    expect(recovered && recovered->extent().offset == original_offset &&
               backing.occupied_bytes() == backing.capacity_bytes(),
           "State reuses the exact physical extent released by split KV");
    expect(recovered->release(), "shared Host recovered state release");
    (void)parts.second.release();
    expect(backing.occupied_bytes() == 0 && backing.allocation_count() == 0 &&
               backing.capacity_bytes() == image_bytes * 2,
           "shared Host typed owners return all live and reserved extents");
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
    ninfer::DeviceBuffer arena_storage(device.device(), planned.bytes);
    ninfer::MemoryArena arena(arena_storage.span());
    q36::StateImageViews pool(arena.storage(), planned.layout);

    expect(pool.storage().slot_count() == 4, "StateImage slot count");
    expect(pool.linear().layer_count() == 2, "StateImage Linear Attention layer count");
    expect(pool.continuation_hidden_slot(0).dtype == ninfer::DType::BF16 &&
               pool.continuation_hidden_slot(0).ne[0] == 7,
           "StateImage continuation hidden geometry");
    expect(pool.dflash_local() != nullptr && pool.dflash_local()->layer_count() == 2 &&
               pool.dflash_local()->slot_count() == 4,
           "StateImage DFlash local geometry");

    fill_slot(pool, 0, 0x11, device.execution_view());
    fill_slot(pool, 1, 0x52, device.execution_view());
    pool.storage().copy(0, 2, device.execution_view());
    device.synchronize();
    expect_slot(pool, 0, 0x11, "StateImage D2D source isolation");
    expect_slot(pool, 2, 0x11, "StateImage D2D complete destination");

    pool.storage().clear(1, device.execution_view());
    device.synchronize();
    expect_zero_slot(pool, 1, "StateImage zero complete destination");
    expect_slot(pool, 0, 0x11, "StateImage zero source isolation");

    test_host_roundtrip(false, device);
    test_host_roundtrip(true, device);
    test_host_roundtrip(true, device, true);
    test_shared_host_capacity();

    return failures == 0 ? 0 : 1;
}
