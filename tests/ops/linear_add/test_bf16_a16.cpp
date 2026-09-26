#include "core/weight.h"
#include "ops/linear_add/linear_add_test_common.h"

#include "ninfer/ops/linear_add.h"
#include "ops/op_tester.h"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

template <class Callable>
int expect_invalid(Callable&& callable, const char* label) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int bf16_a16_rejections() {
    int failures = 0;
    failures += expect_invalid(
        [] {
            (void)ninfer::ops::linear_add_workspace_capacity_bytes(ninfer::QType::BF16, 5120, 6143,
                                                                   1, 32);
        },
        "BF16_A16 LinearAdd workspace shape");

    ninfer::DeviceBuffer input(static_cast<std::size_t>(6144) * sizeof(std::uint16_t));
    ninfer::DeviceBuffer residual(static_cast<std::size_t>(5120) * sizeof(std::uint16_t));
    ninfer::DeviceBuffer weight_storage(256);
    ninfer::Weight weight{};
    weight.qtype  = ninfer::QType::BF16;
    weight.layout = ninfer::QuantLayout::RowSplit;
    weight.qdata  = weight_storage.p;
    weight.n      = 5120;
    weight.k      = 6144;
    ninfer::Tensor x(input.p, ninfer::DType::BF16, {6144, 1});
    ninfer::Tensor out(residual.p, ninfer::DType::BF16, {5120, 1});
    ninfer::WorkspaceArena workspace(1);
    failures += expect_invalid([&] { ninfer::ops::linear_add(x, weight, out, workspace, nullptr); },
                               "BF16_A16 LinearAdd weight layout");
    return failures;
}

int bf16_a16_conformance() {
    constexpr std::array<std::int32_t, 7> kRouteStarts{2, 5, 33, 65, 97, 129, 257};
    constexpr std::array<std::int32_t, 10> kRouteInteriors{
        4, 8, 16, 32, 48, 96, 192, 256, 1024, 1536,
    };
    return ninfer::test::linear_add::run_shape(
               "BF16_A16 LinearAdd", WeightFormat::BF16,
               ShapeCase{5120, 6144, 431U, kRouteStarts, kRouteInteriors}) +
           bf16_a16_rejections();
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = bf16_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
