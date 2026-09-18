#include "ops/linear_add/linear_add_test_common.h"

#include "ninfer/ops/linear_add.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int cancellation_conformance() {
    using namespace ninfer;
    namespace qw    = test::quantized_weight;
    int failures    = 0;
    constexpr int t = 129;
    for (const auto [n, k] : std::array<std::array<int, 2>, 4>{
             {{2048, 4096}, {2048, 6144}, {5120, 6144}, {5120, 17408}}}) {
        auto packed = qw::make_patterned_weight(QType::Q8_G32_FP16, n, k, 427U);
        std::fill_n(packed.payload.begin(), packed.code_plane_bytes, std::uint8_t{1});
        constexpr std::uint16_t scale = 0x3000; // FP16 0.125
        for (std::size_t offset = packed.scale_plane_offset;
             offset < packed.scale_plane_offset + packed.scale_plane_bytes; offset += 2) {
            std::memcpy(packed.payload.data() + offset, &scale, sizeof(scale));
        }
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * t,
                                              test::f32_to_bf16(0.125F));
        for (int col = 0; col < t; ++col)
            activation[static_cast<std::size_t>(col) * k] = test::f32_to_bf16(0.25F);
        const auto residual_bits = test::f32_to_bf16(-static_cast<float>(k) / 64.0F);
        std::vector<std::uint16_t> residual(static_cast<std::size_t>(n) * t, residual_bits);
        // Identical rows and columns share one independently decoded FP64 dot product.
        double expected = static_cast<double>(test::bf16_to_f32(residual_bits));
        for (int column = 0; column < k; ++column) {
            expected += qw::logical_weight_fp64(packed, 0, column) *
                        static_cast<double>(test::bf16_to_f32(activation[column]));
        }
        test::GuardedDeviceBuffer dw(packed.payload.size()), dx(activation.size() * 2),
            dr(residual.size() * 2);
        dw.copy_from_host(packed.payload.data(), dw.bytes());
        dx.copy_from_host(activation.data(), dx.bytes());
        dr.copy_from_host(residual.data(), dr.bytes());
        const auto weight = packed.device_weight(dw.data());
        Tensor input(dx.data(), DType::BF16, {k, t});
        Tensor output(dr.data(), DType::BF16, {n, t});
        WorkspaceArena workspace(256);
        ops::linear_add(input, weight, output, workspace, nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "synchronize cancellation LinearAdd");
        dr.copy_to_host(residual.data(), dr.bytes());
        std::vector<double> actual(residual.size()), reference(residual.size(), expected);
        std::transform(residual.begin(), residual.end(), actual.begin(),
                       [](std::uint16_t bits) { return double(test::bf16_to_f32(bits)); });
        failures += test::verify_reduction("Q8 LinearAdd cancellation", actual, reference,
                                           {1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0});
        failures += dr.verify_guards("Q8 LinearAdd cancellation residual");
    }
    return failures;
}

int q8_a16_conformance() {
    int failures = 0;

    constexpr std::array<std::int32_t, 4> kK4096RouteStarts{2, 49, 129, 641};
    constexpr std::array<std::int32_t, 5> kK4096RouteInteriors{1, 24, 96, 256, 1024};
    failures += ninfer::test::linear_add::run_shape(
        "Q8_A16 LinearAdd", WeightFormat::Q8G32F16S,
        ShapeCase{2048, 4096, 419U, kK4096RouteStarts, kK4096RouteInteriors});

    constexpr std::array<std::int32_t, 32> kK6144RouteStarts{
        2,    49,   129,  192,  193,  257,  385,  400,  401,  448,  449,
        481,  641,  673,  705,  785,  897,  961,  1024, 1025, 1121, 1281,
        1345, 1409, 1681, 1792, 1793, 1920, 1921, 2017, 2048, 2049,
    };
    constexpr std::array<std::int32_t, 33> kK6144RouteInteriors{
        1,    24,   96,   160,  192,  224,  320,  392,  400,  424,  448,
        464,  560,  656,  688,  744,  840,  928,  992,  1024, 1072, 1200,
        1312, 1376, 1544, 1736, 1792, 1856, 1920, 1968, 2032, 2048, 4096,
    };
    failures += ninfer::test::linear_add::run_shape(
        "Q8_A16 LinearAdd", WeightFormat::Q8G32F16S,
        ShapeCase{2048, 6144, 421U, kK6144RouteStarts, kK6144RouteInteriors});
    constexpr std::array<std::int32_t, 10> large_route_starts{5,  9,  17, 25, 33,
                                                              41, 49, 57, 65, 129};
    constexpr std::array<std::int32_t, 8> large_interiors{1, 4, 8, 96, 128, 512, 1024, 1025};
    constexpr std::array<std::int32_t, 6> graph_tokens{1, 8, 64, 65, 129, 512};
    constexpr std::array<std::int32_t, 3> full_tokens{1, 4, 8};
    for (const auto k : {6144, 17408}) {
        failures += ninfer::test::linear_add::run_shape(
            "Q8_A16 LinearAdd", WeightFormat::Q8G32F16S,
            ShapeCase{5120, k, 423U, large_route_starts, large_interiors, graph_tokens});
        failures += ninfer::test::linear_add::run_shape(
            "Q8_A16 LinearAdd full", WeightFormat::Q8G32F16S,
            ShapeCase{5120, k, 425U, {}, full_tokens, {}, true});
    }
    return failures + cancellation_conformance();
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q8_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q8_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
