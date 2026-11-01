#include "ops/linear/weights.h"
#include "ops/direct_bf16_weight.h"
#include "ops/linear_common.h"

namespace ninfer::test::linear_weights {

namespace {
WeightGeometry bank_geometry(int n, int k) {
    const std::array<std::uint64_t, 3> shape{2, std::uint64_t(n + 256), std::uint64_t(k)};
    return weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4, shape);
}
} // namespace

BankFixture::BankFixture(int n, int k)
    : packed(quantized_weight::make_patterned_weight(
          QType::NVFP4, n, k, 701U, {.weight_scale_divisor = 128.0F, .input_scale_divisor = 1.0F})),
      bytes(bank_geometry(n, k).bytes), storage(bytes.size()) {
    const auto geometry     = bank_geometry(n, k);
    const auto first_row    = geometry.shape[1] + 128;
    const auto code_offset  = first_row * k / 2;
    const auto scale_offset = geometry.scale_offset + first_row * k / 16;
    std::memcpy(bytes.data() + code_offset, packed.payload.data(), packed.code_plane_bytes);
    std::memcpy(bytes.data() + scale_offset, packed.payload.data() + packed.scale_plane_offset,
                packed.scale_plane_bytes);
    const std::array divisors{64.0F, 128.0F};
    std::memcpy(bytes.data() + geometry.divisor_offset, divisors.data(), sizeof(divisors));
    storage.copy_from_host(bytes.data(), bytes.size());
    const WeightParent parent{geometry, storage.device(),
                              static_cast<const std::byte*>(storage.data()), divisors};
    weight = native_weight(
        {{std::uint64_t(n), std::uint64_t(k)}, {{&parent, first_row * k, (first_row + n) * k}}},
        1.0F);
}

int BankFixture::verify_preserved() const {
    return storage.verify_guards("NVFP4 bank") +
           linear_common::verify_preserved(storage.data(), std::span(bytes), "NVFP4 bank");
}

quantized_weight::PackedWeight make_q4_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q5_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q6_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q6_G64_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q8_g32_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    return quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
}

quantized_weight::PackedWeight make_fp8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, n, k, seed);
}

quantized_weight::PackedWeight make_bf16_weight(std::int32_t n, std::int32_t k,
                                                std::uint32_t seed) {
    const auto host = direct_bf16_weight::make_patterned(n, k, seed);
    quantized_weight::PackedWeight result;
    result.payload.resize(host.bits.size() * sizeof(std::uint16_t));
    std::memcpy(result.payload.data(), host.bits.data(), result.payload.size());
    result.weight =
        host.device_weight({result.payload.data(), result.payload.size(), Device::cpu()});
    result.code_plane_bytes = result.payload.size();
    return result;
}


} // namespace ninfer::test::linear_weights
