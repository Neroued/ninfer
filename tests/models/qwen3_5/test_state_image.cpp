#include "core/device.h"
#include "core/buffer.h"
#include "core/context_pool.h"
#include "core/paged_kv_geometry.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
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

q36::StateImageDeviceLayout plan_image(bool dflash, bool dflash2 = false) {
    q36::StateImageSpec spec{
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
    return q36::plan_state_image_layout(spec);
}

ninfer::PoolReservation reserve(ninfer::ContextPool& pool,
                                std::initializer_list<ninfer::RecordRequest> demand) {
    auto result = pool.reserve(demand);
    if (!result.reservation || !result.reservation->ready()) {
        throw std::runtime_error("fixture requires an immediately available reservation");
    }
    return std::move(*result.reservation);
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

void fill_image(q36::StateImageViews& pool, std::int64_t offset, unsigned char base,
                ninfer::CudaExecutionView execution) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        set_bytes(pool.linear().conv_tensor(layer, offset),
                  static_cast<unsigned char>(base + layer), execution);
        set_bytes(pool.linear().recurrent_tensor(layer, offset),
                  static_cast<unsigned char>(base + 0x10 + layer), execution);
    }
    set_bytes(pool.continuation_hidden(offset), static_cast<unsigned char>(base + 0x20), execution);
    if (ninfer::CyclicKVCacheView* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer, {});
            set_bytes(view.k.tensor(offset), static_cast<unsigned char>(base + 0x30 + layer),
                      execution);
            set_bytes(view.v.tensor(offset), static_cast<unsigned char>(base + 0x40 + layer),
                      execution);
        }
    }
}

void expect_image(q36::StateImageViews& pool, std::int64_t offset, unsigned char base,
                  std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_tensor(layer, offset),
                     static_cast<unsigned char>(base + layer), label);
        expect_bytes(pool.linear().recurrent_tensor(layer, offset),
                     static_cast<unsigned char>(base + 0x10 + layer), label);
    }
    expect_bytes(pool.continuation_hidden(offset), static_cast<unsigned char>(base + 0x20), label);
    if (ninfer::CyclicKVCacheView* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer, {});
            expect_bytes(view.k.tensor(offset), static_cast<unsigned char>(base + 0x30 + layer),
                         label);
            expect_bytes(view.v.tensor(offset), static_cast<unsigned char>(base + 0x40 + layer),
                         label);
        }
    }
}

void expect_zero_image(q36::StateImageViews& pool, std::int64_t offset, std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_tensor(layer, offset), 0, label);
        expect_bytes(pool.linear().recurrent_tensor(layer, offset), 0, label);
    }
    expect_bytes(pool.continuation_hidden(offset), 0, label);
    if (ninfer::CyclicKVCacheView* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer, {});
            expect_bytes(view.k.tensor(offset), 0, label);
            expect_bytes(view.v.tensor(offset), 0, label);
        }
    }
}

void test_host_roundtrip(bool dflash, ninfer::DeviceContext& device, bool dflash2 = false) {
    const auto layout              = plan_image(dflash, dflash2);
    const auto image_bytes         = layout.images.image_bytes;
    const std::int64_t source      = layout.images.alignment;
    const std::int64_t destination = source + image_bytes + 2 * layout.images.alignment;
    ninfer::DeviceBuffer storage(device.device(), destination + image_bytes);
    q36::StateImageViews pool(storage.span(), layout);
    if (dflash2) {
        expect(pool.dflash_local_transfer_work().payload_bytes == 40ULL * 1024 * 1024,
               "DFlash2 local snapshot transfers 40 MiB");
    }
    fill_image(pool, source, dflash ? 0x19 : 0x25, device.execution_view());
    pool.storage().clear(destination, device.execution_view());
    ninfer::PinnedHostBuffer host_bytes(image_bytes);
    pool.storage().copy_to_host(source, host_bytes.span(), device.execution_view());
    device.synchronize();
    pool.storage().copy_from_host(host_bytes.span(), destination, device.execution_view());
    device.synchronize();
    expect_image(pool, source, dflash ? 0x19 : 0x25, "Host roundtrip source isolation");
    expect_image(pool, destination, dflash ? 0x19 : 0x25, "complete Host roundtrip");
    if (dflash) {
        fill_image(pool, destination, 0x61, device.execution_view());
        pool.storage().copy_group(pool.dflash_local_group(), source, destination,
                                  device.execution_view());
        device.synchronize();
        expect_bytes(pool.linear().conv_tensor(0, destination), 0x61, "local copy preserves GDN");
        expect_bytes(pool.continuation_hidden(destination), 0x81, "local copy preserves hidden");
        for (std::uint32_t layer = 0; layer < pool.dflash_local()->layer_count(); ++layer) {
            auto view = pool.dflash_local()->layer_view(layer, {});
            expect_bytes(view.k.tensor(destination), 0x19 + 0x30 + layer, "local K copy");
            expect_bytes(view.v.tensor(destination), 0x19 + 0x40 + layer, "local V copy");
        }
    }
}

void test_shared_host_capacity() {
    const auto image = plan_image(false);
    const auto page  = ninfer::plan_kv_page_layout(
        {.page_tokens = 1,
          .planes      = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 1}}});
    const std::size_t image_bytes = image.images.image_bytes;
    const auto pages_per_image    = static_cast<std::uint32_t>(image_bytes / page.page_bytes);
    const std::array specs{
        ninfer::RecordSpec{image_bytes, image.images.alignment},
        ninfer::RecordSpec{page.page_bytes, page.alignment},
    };
    ninfer::ContextPool backing(0, image_bytes * 2, specs);
    const auto state_class     = backing.record_class(0);
    const auto page_class      = backing.record_class(1);
    auto source_reservation    = reserve(backing, {{state_class, 1}});
    const auto source          = source_reservation.materialize(state_class);
    const auto original_offset = backing.resolve(source).offset;
    auto destination           = reserve(backing, {{page_class, pages_per_image}});
    const std::array more_state{ninfer::RecordRequest{state_class, 1}};
    const std::array more_page{ninfer::RecordRequest{page_class, 1}};
    expect(backing.usage().live_bytes == image_bytes &&
               backing.usage().reserved_bytes == image_bytes &&
               !backing.assess(more_state).feasible() && !backing.reserve(more_state).reservation &&
               !backing.reserve(more_page).reservation,
           "State and in-flight KV destination consume the same actual backing");
    destination.cancel();
    expect(backing.usage().reserved_bytes == 0 && backing.usage().live_bytes == image_bytes &&
               backing.resolve(source).offset == original_offset,
           "KV transfer cancellation retains the pinned State source");
    backing.release(source);
    auto all_kv = reserve(backing, {{page_class, pages_per_image * 2}});
    std::vector<ninfer::AllocationId> pages;
    for (std::uint32_t i = 0; i < pages_per_image * 2; ++i) {
        pages.push_back(all_kv.materialize(page_class));
    }
    expect(backing.usage().live_bytes == backing.capacity_bytes(),
           "KV reuses the capacity previously occupied by State");
    std::vector<ninfer::AllocationId> retained;
    for (const auto page_id : pages) {
        const auto offset = backing.resolve(page_id).offset;
        if (offset >= original_offset && offset < original_offset + image_bytes)
            backing.release(page_id);
        else
            retained.push_back(page_id);
    }
    auto recovered             = reserve(backing, {{state_class, 1}});
    const auto recovered_state = recovered.materialize(state_class);
    expect(backing.usage().live_bytes == backing.capacity_bytes(),
           "State reuses capacity released by independently owned KV records");
    backing.release(recovered_state);
    for (const auto page_id : retained) backing.release(page_id);
    expect(backing.usage().live_bytes == 0 && backing.usage().reserved_bytes == 0 &&
               backing.usage().free_block_bytes == backing.capacity_bytes(),
           "shared Host typed owners return all live and reserved capacity");
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
    const auto layout      = plan_image(true);
    const auto image_bytes = layout.images.image_bytes;
    const auto alignment   = layout.images.alignment;
    const std::array<std::int64_t, 3> offsets{
        static_cast<std::int64_t>(alignment),
        static_cast<std::int64_t>(image_bytes + 3 * alignment),
        static_cast<std::int64_t>(2 * image_bytes + 6 * alignment),
    };
    ninfer::DeviceBuffer arena_storage(device.device(), offsets.back() + image_bytes);
    q36::StateImageViews pool(arena_storage.span(), layout);

    expect(pool.linear().layer_count() == 2, "StateImage Linear Attention layer count");
    expect(pool.continuation_hidden(offsets[0]).dtype == ninfer::DType::BF16 &&
               pool.continuation_hidden(offsets[0]).ne[0] == 7,
           "StateImage continuation hidden geometry");
    expect(pool.dflash_local() != nullptr && pool.dflash_local()->layer_count() == 2,
           "StateImage DFlash local geometry");

    fill_image(pool, offsets[0], 0x11, device.execution_view());
    fill_image(pool, offsets[1], 0x52, device.execution_view());
    pool.storage().copy(offsets[0], offsets[2], device.execution_view());
    device.synchronize();
    expect_image(pool, offsets[0], 0x11, "StateImage D2D source isolation");
    expect_image(pool, offsets[1], 0x52, "StateImage D2D unrelated image isolation");
    expect_image(pool, offsets[2], 0x11, "StateImage D2D complete destination");

    pool.storage().clear(offsets[1], device.execution_view());
    device.synchronize();
    expect_zero_image(pool, offsets[1], "StateImage zero complete destination");
    expect_image(pool, offsets[0], 0x11, "StateImage zero source isolation");

    test_host_roundtrip(false, device);
    test_host_roundtrip(true, device);
    test_host_roundtrip(true, device, true);
    test_shared_host_capacity();

    return failures == 0 ? 0 : 1;
}
