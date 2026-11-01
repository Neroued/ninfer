#pragma once

#include "ops/quantized_weight.h"
#include "core/weight_view.h"
#include "ops/op_tester.h"

namespace ninfer::test::linear_weights {

class BankFixture {
public:
    BankFixture(int n, int k);
    int verify_preserved() const;
    quantized_weight::PackedWeight packed;
    std::vector<std::byte> bytes;
    GuardedDeviceBuffer storage;
    Weight weight;
};

using WeightGenerator = quantized_weight::PackedWeight (*)(std::int32_t, std::int32_t,
                                                           std::uint32_t);

quantized_weight::PackedWeight make_bf16_weight(std::int32_t n, std::int32_t k, std::uint32_t seed);

quantized_weight::PackedWeight make_q4_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q5_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q6_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q8_g32_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed);
quantized_weight::PackedWeight make_fp8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed);


} // namespace ninfer::test::linear_weights
