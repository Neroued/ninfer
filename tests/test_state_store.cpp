#include "ninfer/core/device.h"
#include "ninfer/core/state_store.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw expected exception\n";
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& t, const std::int32_t (&expected)[4], const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got " << t.ne[i]
                      << '\n';
        }
    }
    return failures;
}

int expect_device_byte(const ninfer::Tensor& tensor, unsigned char expected, const char* label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (unsigned char value : host) {
        if (value != expected) {
            std::cerr << label << " expected byte 0x" << std::hex << static_cast<int>(expected)
                      << ", got 0x" << static_cast<int>(value) << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);
    ninfer::DeviceArena cache_arena(16384);
    ninfer::GdnState state(cache_arena, 3, 10, 3, 4, 5, 6, ninfer::DType::BF16);

    failures += expect_size(state.layer_count(), 3, "state.layer_count");
    failures += expect_size(state.conv.size(), 3, "state.conv.size");
    failures += expect_size(state.ssm.size(), 3, "state.ssm.size");
    failures += expect_size(state.conv_width, 3, "state.conv_width");
    failures += expect_size(state.snapshot_slots, 1, "state.snapshot_slots");
    for (std::size_t layer = 0; layer < state.layer_count(); ++layer) {
        failures += check_shape(state.conv[layer], {10, 3, 1, 1}, "state.conv");
        failures += check_shape(state.ssm[layer], {6, 5, 4, 1}, "state.ssm");
        failures += check_shape(state.conv_slot(static_cast<std::uint32_t>(layer), 0),
                                {10, 3, 1, 1}, "state.conv_slot");
        failures += check_shape(state.ssm_slot(static_cast<std::uint32_t>(layer), 0),
                                {6, 5, 4, 1}, "state.ssm_slot");
        if (state.conv[layer].dtype != ninfer::DType::BF16) {
            ++failures;
            std::cerr << "conv dtype is not BF16\n";
        }
        if (state.ssm[layer].dtype != ninfer::DType::FP32) {
            ++failures;
            std::cerr << "ssm dtype is not FP32\n";
        }
        if (state.conv[layer].data == state.ssm[layer].data) {
            ++failures;
            std::cerr << "conv/ssm alias for layer " << layer << '\n';
        }
        failures += expect_device_byte(state.conv[layer], 0, "initial conv");
        failures += expect_device_byte(state.ssm[layer], 0, "initial ssm");
    }

    CUDA_CHECK(cudaMemset(state.conv[0].data, 0x7a, state.conv[0].bytes()));
    CUDA_CHECK(cudaMemset(state.ssm[1].data, 0x5c, state.ssm[1].bytes()));
    failures += expect_device_byte(state.conv[0], 0x7a, "sentinel conv0");
    failures += expect_device_byte(state.conv[1], 0, "untouched conv1");
    failures += expect_device_byte(state.ssm[0], 0, "untouched ssm0");
    failures += expect_device_byte(state.ssm[1], 0x5c, "sentinel ssm1");

    state.reset(ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(state.conv[0], 0, "reset conv0");
    failures += expect_device_byte(state.ssm[1], 0, "reset ssm1");

    ninfer::DeviceArena slotted_arena(16384);
    ninfer::GdnState slotted(slotted_arena, 1, 10, 3, 4, 5, 6, 3, ninfer::DType::BF16);
    failures += expect_size(slotted.snapshot_slots, 3, "slotted.snapshot_slots");
    failures += check_shape(slotted.conv[0], {10, 3, 3, 1}, "slotted.conv");
    failures += check_shape(slotted.ssm[0], {6, 5, 4, 3}, "slotted.ssm");
    failures += check_shape(slotted.conv_slot(0, 2), {10, 3, 1, 1}, "slotted.conv_slot");
    failures += check_shape(slotted.ssm_slot(0, 2), {6, 5, 4, 1}, "slotted.ssm_slot");
    failures += expect_throws<std::out_of_range>(
        [&] { (void)slotted.conv_slot(1, 0); }, "conv_slot invalid layer");
    failures += expect_throws<std::out_of_range>(
        [&] { (void)slotted.ssm_slot(0, 3); }, "ssm_slot invalid slot");

    ninfer::Tensor conv0 = slotted.conv_slot(0, 0);
    ninfer::Tensor conv1 = slotted.conv_slot(0, 1);
    ninfer::Tensor ssm0  = slotted.ssm_slot(0, 0);
    ninfer::Tensor ssm1  = slotted.ssm_slot(0, 1);
    CUDA_CHECK(cudaMemset(conv0.data, 0x7a, conv0.bytes()));
    CUDA_CHECK(cudaMemset(conv1.data, 0x6b, conv1.bytes()));
    CUDA_CHECK(cudaMemset(ssm0.data, 0x5c, ssm0.bytes()));
    CUDA_CHECK(cudaMemset(ssm1.data, 0x4d, ssm1.bytes()));
    failures += expect_device_byte(conv0, 0x7a, "slotted sentinel conv slot0");
    failures += expect_device_byte(conv1, 0x6b, "slotted sentinel conv slot1");
    failures += expect_device_byte(ssm0, 0x5c, "slotted sentinel ssm slot0");
    failures += expect_device_byte(ssm1, 0x4d, "slotted sentinel ssm slot1");
    slotted.reset(ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(slotted.conv_slot(0, 0), 0, "slotted reset conv slot0");
    failures += expect_device_byte(slotted.conv_slot(0, 1), 0x6b, "slotted reset keeps conv slot1");
    failures += expect_device_byte(slotted.ssm_slot(0, 0), 0, "slotted reset ssm slot0");
    failures += expect_device_byte(slotted.ssm_slot(0, 1), 0x4d, "slotted reset keeps ssm slot1");

    failures += expect_throws<std::invalid_argument>(
        [&] { ninfer::GdnState invalid(cache_arena, 0, 10, 3, 4, 5, 6); }, "zero layers");
    failures += expect_throws<std::invalid_argument>(
        [&] { ninfer::GdnState invalid(cache_arena, 1, 0, 3, 4, 5, 6); }, "zero conv dim");
    failures += expect_throws<std::invalid_argument>(
        [&] { ninfer::GdnState invalid(cache_arena, 1, 10, 0, 4, 5, 6); }, "zero conv width");
    failures += expect_throws<std::invalid_argument>(
        [&] { ninfer::GdnState invalid(cache_arena, 1, 10, 3, 0, 5, 6); }, "zero value heads");
    failures += expect_throws<std::invalid_argument>(
        [&] { ninfer::GdnState invalid(cache_arena, 1, 10, 3, 4, 0, 6); }, "zero value head dim");
    failures += expect_throws<std::invalid_argument>(
        [&] { ninfer::GdnState invalid(cache_arena, 1, 10, 3, 4, 5, 0); }, "zero key head dim");
    failures += expect_throws<std::invalid_argument>(
        [&] { ninfer::GdnState invalid(cache_arena, 1, 10, 3, 4, 5, 6, ninfer::DType::U8); },
        "invalid conv dtype");
    failures += expect_throws<std::invalid_argument>(
        [&] { ninfer::GdnState invalid(cache_arena, 1, 10, 3, 4, 5, 6, 0); },
        "zero snapshot slots");

    ninfer::DeviceArena small_arena(512);
    const std::size_t small_before = small_arena.used();
    failures += expect_throws<std::bad_alloc>(
        [&] { ninfer::GdnState too_big(small_arena, 3, 10, 3, 4, 5, 6); }, "undersized state arena");
    failures += expect_size(small_arena.used(), small_before, "small arena used after failure");

    ninfer::DeviceArena overflow_arena(1024);
    const std::size_t overflow_before = overflow_arena.used();
    failures += expect_throws<std::overflow_error>(
        [&] {
            ninfer::GdnState overflow(overflow_arena, std::numeric_limits<std::uint32_t>::max(),
                                   std::numeric_limits<std::int32_t>::max(), 1, 1, 1, 1);
        },
        "aggregate state overflow");
    failures +=
        expect_size(overflow_arena.used(), overflow_before, "overflow arena used after failure");

    return failures == 0 ? 0 : fail("state store test failed");
}
