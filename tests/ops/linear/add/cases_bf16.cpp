#include "core/weight.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/linear/add/runner.h"
#include "ops/linear_common.h"

#include "ninfer/ops/linear_add.h"
#include "ops/op_tester.h"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::QType;

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

    ninfer::DeviceBuffer input(ninfer::test::cuda_environment().context.device(),
                               static_cast<std::size_t>(6144) * sizeof(std::uint16_t));
    ninfer::DeviceBuffer residual(ninfer::test::cuda_environment().context.device(),
                                  static_cast<std::size_t>(5120) * sizeof(std::uint16_t));
    ninfer::DeviceBuffer weight_storage(ninfer::test::cuda_environment().context.device(), 256);
    ninfer::Weight weight{};
    weight.device = weight_storage.device();
    weight.qtype  = ninfer::QType::BF16;
    weight.layout = ninfer::QuantLayout::RowSplit;
    weight.qdata  = weight_storage.data();
    weight.n      = 5120;
    weight.k      = 6144;
    ninfer::Tensor x(input.span(), ninfer::DType::BF16, {6144, 1});
    ninfer::Tensor out(residual.span(), ninfer::DType::BF16, {5120, 1});
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(), 1);
    ninfer::MemoryArena workspace(workspace_storage.span());
    failures += expect_invalid(
        [&] {
            ninfer::ops::linear_add(x, weight, out, workspace,
                                    ninfer::test::cuda_environment().default_execution);
        },
        "BF16_A16 LinearAdd weight layout");
    return failures;
}

void bf16_a16_conformance(ninfer::test::linear_suite::Suite& suite) {
    constexpr std::array<std::int32_t, 7> kRouteStarts{2, 5, 33, 65, 97, 129, 257};
    constexpr std::array<std::int32_t, 10> kRouteInteriors{
        4, 8, 16, 32, 48, 96, 192, 256, 1024, 1536,
    };
    ninfer::test::linear_add::add_shape(suite, QType::BF16,
                                        ShapeCase{.n               = 5120,
                                                  .k               = 6144,
                                                  .seed            = 431U,
                                                  .route_starts    = kRouteStarts,
                                                  .route_interiors = kRouteInteriors});
    suite.add(ninfer::test::linear_suite::make_case(
                  "bf16", 5120, 6144, 0, ninfer::ops::LinearPolicy::A16Only,
                  ninfer::test::linear_suite::Execution::Eager, "metadata"),
              bf16_a16_rejections);
}

} // namespace

void ninfer::test::linear_add::add_bf16_cases(Suite& suite) { bf16_a16_conformance(suite); }
