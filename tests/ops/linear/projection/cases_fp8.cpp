#include "core/weight.h"
#include "core/platform.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/linear/projection/runner.h"
#include <array>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

namespace {
using namespace ninfer::test::linear_weights;

using namespace ninfer;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

std::vector<Invocation> a16_capacity_calls() {
    std::vector<Invocation> calls;
    for (int t = 1; t <= 33; ++t) calls.push_back({t});
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024}) calls.push_back({t});
    for (int t : {3, 7, 11, 15, 19, 23, 25, 33})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    calls.push_back({1, CallForm::A16Convenience});
    for (int t : {64, 97, 129})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    return calls;
}

void run_fp8_a16(Suite& suite) {
    const auto attn_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight, {14336, 5120, 811U, Comparison::Sampled, true, attn_invocations});
    const auto gdn_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight, {16384, 5120, 817U, Comparison::Sampled, true, gdn_invocations});
    const auto mlp_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight, {34816, 5120, 821U, Comparison::Sampled, true, mlp_invocations});
    std::vector<Invocation> vocabulary_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{9, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{41, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{42, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{49, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{96, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{97, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{160, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{161, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    for (int t = 1; t <= 41; ++t)
        vocabulary_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
    for (int t : {7, 25, 41, 65, 128})
        vocabulary_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {248320, 5120, 823U, Comparison::Sampled, true, vocabulary_invocations});
    suite.add(linear_suite::make_case("fp8", 248320, 5120, 0, ops::LinearPolicy::A16Only,
                                      linear_suite::Execution::Eager, "vocabulary-workspace"),
              [] {
                  int failures = 0;
                  const std::array vocabulary_policies{
                      ops::LinearPolicy::A16Only,
                      ops::LinearPolicy::AllowA8,
                      ops::LinearPolicy::AllowA4,
                  };
                  for (const ops::LinearPolicy policy : vocabulary_policies) {
                      try {
                          const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                              QType::FP8_E4M3FN_ROW_BF16, 248320, 5120, policy, 1, 2048);
                          if (capacity != 0) {
                              std::cerr << "FP8 vocabulary A16 route reported nonzero workspace\n";
                              ++failures;
                          }
                      } catch (const std::exception& error) {
                          std::cerr << "FP8 vocabulary policy was rejected: " << error.what()
                                    << '\n';
                          ++failures;
                      }
                  }
                  return failures;
              });
    const auto residual6144_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {5120, 6144, 827U, Comparison::Sampled, true, residual6144_invocations});
    const auto residual17408_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {5120, 17408, 829U, Comparison::Sampled, true, residual17408_invocations});

    suite.add(linear_suite::make_case("fp8", 14336, 5120, 1, ops::LinearPolicy::A16Only,
                                      linear_suite::Execution::Eager, "metadata-s831"),
              [] {
                  int failures = 0;
                  auto packed  = make_fp8_weight(14336, 5120, 831U);
                  DeviceBuffer device_weight(ninfer::test::cuda_environment().context.device(),
                                             packed.payload.size());
                  device_weight.copy_from_host(packed.payload.data(), device_weight.size());
                  const Weight weight = packed.device_weight(device_weight.span());
                  const std::vector<std::uint16_t> input_bits(5120, 0);
                  auto device_input = test::to_device(input_bits);
                  test::GuardedDeviceBuffer device_output(14336 * sizeof(std::uint16_t));
                  device_output.fill(0xff);
                  Tensor input(device_input.span(), DType::BF16, {5120, 1});
                  Tensor output(device_output.span(), DType::BF16, {14336, 1});
                  ninfer::DeviceBuffer workspace_storage(
                      ninfer::test::cuda_environment().context.device(), 256);
                  MemoryArena workspace(workspace_storage.span());
                  const auto invoke = [&](const Weight& candidate) {
                      ops::linear(input, candidate, output, ops::LinearPolicy::A16Only, workspace,
                                  ninfer::test::cuda_environment().default_execution);
                  };
                  try {
                      invoke(weight);
                      test::cuda_synchronize();
                  } catch (const std::exception& error) {
                      std::cerr << "valid FP8 metadata was rejected: " << error.what() << '\n';
                      ++failures;
                  }
                  const auto expect_invalid = [&](const char* label, Weight invalid) {
                      try {
                          invoke(invalid);
                          test::cuda_synchronize();
                          std::cerr << "invalid FP8 " << label << " was accepted\n";
                          ++failures;
                      } catch (const std::invalid_argument&) {}
                  };
                  Weight invalid = weight;
                  invalid.layout = QuantLayout::Contiguous;
                  expect_invalid("layout", invalid);
                  invalid             = weight;
                  invalid.scale_nb[1] = invalid.scale_nb[1] - 2;
                  expect_invalid("scale stride", invalid);
                  invalid               = weight;
                  invalid.payload_bytes = invalid.payload_bytes - 1;
                  expect_invalid("payload bound", invalid);
                  invalid = weight;
                  invalid.scales = weight.qdata;
                  expect_invalid("overlapping planes", invalid);
                  invalid = weight;
                  invalid.qdata = static_cast<const std::byte*>(weight.qdata) + 256;
                  expect_invalid("code range overlapping scales", invalid);
                  failures += device_output.verify_guards("FP8 metadata output");
                  failures += test::linear_common::verify_preserved(
                      device_input.data(), std::as_bytes(std::span(input_bits)),
                      "FP8 metadata input");
                  failures += test::linear_common::verify_preserved(
                      device_weight.data(), std::as_bytes(std::span(packed.payload)),
                      "FP8 metadata weight");
                  return failures;
              });
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{248320, 5120}, std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        add_workspace_cases(suite, QType::FP8_E4M3FN_ROW_BF16, n, k);
    }
}

} // namespace

namespace {
using namespace ninfer::test::linear_weights;

using namespace ninfer;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

void run_fp8_a8(Suite& suite) {
    const std::array attn_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {14336, 5120, 827U, Comparison::Sampled, true, attn_decode_invocations});
    const std::array attn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1023, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {14336, 5120, 829U, Comparison::Sampled, true, attn_invocations});
    const std::array attn_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {14336, 5120, 833U, Comparison::Sampled, true, attn_bulk_invocations});
    const std::array gdn_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {16384, 5120, 837U, Comparison::Sampled, true, gdn_decode_invocations});
    const std::array gdn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {16384, 5120, 839U, Comparison::Sampled, true, gdn_invocations});
    const std::array gdn_bulk_invocations{
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {16384, 5120, 841U, Comparison::Sampled, true, gdn_bulk_invocations});
    const std::array mlp_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {34816, 5120, 851U, Comparison::Sampled, true, mlp_decode_invocations});
    const std::array mlp_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {34816, 5120, 853U, Comparison::Sampled, true, mlp_invocations});

    const std::array mlp_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{511, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {34816, 5120, 863U, Comparison::Sampled, true, mlp_bulk_invocations});

    const std::array residual6144_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {5120, 6144, 855U, Comparison::Sampled, true, residual6144_decode_invocations});
    const std::array residual6144_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {5120, 6144, 857U, Comparison::Sampled, true, residual6144_invocations});
    const std::array residual17408_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {5120, 17408, 867U, Comparison::Sampled, true, residual17408_decode_invocations});
    const std::array residual17408_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {5120, 17408, 859U, Comparison::Sampled, true, residual17408_invocations});
}

} // namespace

void ninfer::test::linear::add_fp8_cases(Suite& suite) {
    run_fp8_a16(suite);
    run_fp8_a8(suite);
#if NINFER_TARGET_RTX5090
    const std::vector<Invocation> a16{
        {1, CallForm::A16Convenience}, {2}, {16}, {24}, {32}, {64}, {128}, {2048}};
    add_shape(suite, QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight, {14976, 5120, 861U, Comparison::Sampled, true, a16});
    const std::vector<Invocation> a8{
        {16, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {17, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        {32, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {64, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        {1536, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        {512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {2048, CallForm::Policy, ops::LinearPolicy::AllowA8}};
    add_shape(suite, QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed,
              make_fp8_weight, {14976, 5120, 863U, Comparison::Sampled, true, a8});
    const std::vector<Invocation> tail{{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true}};
    add_shape(suite, QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed,
              make_fp8_weight,
              {14976, 5120, 865U, Comparison::Sampled, true, tail, ActivationPattern::KTail});
    add_workspace_cases(suite, QType::FP8_E4M3FN_ROW_BF16, 14976, 5120);
    const std::vector<Invocation> stem_a16{
        {1, CallForm::A16Convenience}, {2}, {16}, {24}, {32}, {64}, {128}, {2048}};
    add_shape(suite, QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight, {5120, 5120, 875U, Comparison::Sampled, true, stem_a16});
    const std::vector<Invocation> stem_a8{
        {16, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {17, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        {32, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {64, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        {512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        {2048, CallForm::Policy, ops::LinearPolicy::AllowA8},
        {4096, CallForm::Policy, ops::LinearPolicy::AllowA8, true}};
    add_shape(suite, QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed,
              make_fp8_weight, {5120, 5120, 877U, Comparison::Sampled, true, stem_a8});
    const std::vector<Invocation> stem_tail{
        {2052, CallForm::Policy, ops::LinearPolicy::AllowA8, true}};
    add_shape(suite, QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed,
              make_fp8_weight,
              {5120, 5120, 879U, Comparison::Sampled, true, stem_tail, ActivationPattern::KTail});
    add_workspace_cases(suite, QType::FP8_E4M3FN_ROW_BF16, 5120, 5120);
#endif
}
