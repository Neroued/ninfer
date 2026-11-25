#include "core/layout.h"
#include "models/qwen3_5/state/state_image.h"

#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace q36 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::StateImageDeviceLayout plan(bool dflash,
                                 ninfer::DType recurrent_dtype = ninfer::DType::FP32) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers          = 2,
                .conv_channels   = 5,
                .conv_width      = 3,
                .value_heads     = 2,
                .value_head_dim  = 4,
                .key_head_dim    = 3,
                .conv_dtype      = ninfer::DType::BF16,
                .recurrent_dtype = recurrent_dtype,
            },
        .hidden = 7,
    };
    if (dflash) {
        spec.dflash_local =
            q36::DFlashLocalStateSpec{.layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    return q36::plan_state_image_layout(spec);
}

} // namespace

int main() {
    const auto common      = plan(false);
    const auto common_work = common.images.transfer_work();
    expect(common_work.payload_bytes == common.images.image_bytes &&
               common_work.copy_operations == 1,
           "complete image work includes actual alignment bytes in one copy");
    const auto dflash      = plan(true);
    const auto full_work   = dflash.images.transfer_work();
    const auto local_bytes = dflash.dflash_local->group.bytes;
    expect(local_bytes == 2ULL * 2 * 4 * 128 * 2 * 2,
           "local group includes both K/V for every layer");
    expect(full_work.payload_bytes == common_work.payload_bytes + local_bytes &&
               full_work.copy_operations == 1,
           "full image copy includes common and local fields in one contiguous range");
    expect(dflash.images.image_bytes % dflash.images.alignment == 0,
           "complete image is independently aligned for either memory domain");

    // Verify BF16 recurrent state halves the recurrent allocation on realistic head dimensions.
    q36::StateImageSpec large_fp32{
        .linear = {.layers          = 4,
                   .conv_channels   = 4096,
                   .conv_width      = 3,
                   .value_heads     = 16,
                   .value_head_dim  = 128,
                   .key_head_dim    = 128,
                   .conv_dtype      = ninfer::DType::BF16,
                   .recurrent_dtype = ninfer::DType::FP32},
        .hidden = 4096,
    };
    auto large_bf16                   = large_fp32;
    large_bf16.linear.recurrent_dtype = ninfer::DType::BF16;
    const auto fp32_layout            = q36::plan_state_image_layout(large_fp32);
    const auto bf16_layout            = q36::plan_state_image_layout(large_bf16);
    const std::size_t expected_delta  = 4ULL * 16 * 128 * 128 * 2;
    expect(fp32_layout.linear.spec.recurrent_dtype == ninfer::DType::FP32 &&
               bf16_layout.linear.spec.recurrent_dtype == ninfer::DType::BF16,
           "linear state layout lost recurrent_dtype");
    expect(fp32_layout.images.image_bytes - bf16_layout.images.image_bytes == expected_delta &&
               bf16_layout.images.image_bytes % bf16_layout.images.alignment == 0 &&
               bf16_layout.images.transfer_work().payload_bytes == bf16_layout.images.image_bytes,
           "BF16 recurrent state did not halve recurrent image bytes cleanly");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
