#include "core/weight.h"
#include "core/platform.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/linear/add/runner.h"
#include "ops/linear_common.h"

#include "ninfer/ops/linear_add.h"
#include "ops/op_tester.h"

#include <array>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

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

#if NINFER_TARGET_RTX5090
int inkling_bf16_rejections(int k) {
    using namespace ninfer;
    constexpr int n = 2304;
    int failures    = 0;
    if (ops::linear_add_workspace_capacity_bytes(QType::BF16, n, k, 1,
                                                 std::numeric_limits<int>::max()) != 0) {
        std::cerr << "BF16 LinearAdd expected zero workspace for K=" << k << '\n';
        ++failures;
    }
    failures += expect_invalid(
        [&] { (void)ops::linear_add_workspace_capacity_bytes(QType::BF16, n, k, 0, 1); },
        "BF16 LinearAdd invalid token interval");
    failures += expect_invalid(
        [&] { (void)ops::linear_add_workspace_capacity_bytes(QType::BF16, n, k - 8, 1, 1); },
        "BF16 LinearAdd unsupported geometry");

    const auto& env   = test::cuda_environment();
    const auto device = env.context.device();
    DeviceBuffer input(device, sizeof(std::uint16_t) * k);
    DeviceBuffer residual(device, sizeof(std::uint16_t) * n);
    DeviceBuffer weight_storage(device, sizeof(std::uint16_t) * n * k);
    DeviceBuffer workspace_storage(device, 1);
    MemoryArena workspace(workspace_storage.span());
    const Tensor x(input.span(), DType::BF16, {k, 1});
    Tensor out(residual.span(), DType::BF16, {n, 1});
    Weight weight{};
    weight.device = device;
    weight.qtype  = QType::BF16;
    weight.layout = QuantLayout::Contiguous;
    weight.qdata  = weight_storage.data();
    weight.n      = n;
    weight.k      = k;

    failures += expect_invalid(
        [&] {
            Tensor invalid = x;
            invalid.dtype  = DType::FP16;
            ops::linear_add(invalid, weight, out, workspace, env.default_execution);
        },
        "BF16 LinearAdd activation dtype");
    failures += expect_invalid(
        [&] {
            Tensor invalid = out;
            invalid.ne[1]  = 2;
            ops::linear_add(x, weight, invalid, workspace, env.default_execution);
        },
        "BF16 LinearAdd residual token extent");
    failures += expect_invalid(
        [&] {
            Tensor alias(input.span(), DType::BF16, {n, 1});
            ops::linear_add(x, weight, alias, workspace, env.default_execution);
        },
        "BF16 LinearAdd overlapping activation and residual");
    failures += expect_invalid(
        [&] {
            Weight invalid = weight;
            invalid.layout = QuantLayout::RowSplit;
            ops::linear_add(x, invalid, out, workspace, env.default_execution);
        },
        "BF16 LinearAdd weight layout");
    return failures;
}

void run_inkling_bf16_add(ninfer::test::linear_suite::Suite& suite, int k,
                          std::initializer_list<int> boundaries) {
    using namespace ninfer;
    using namespace ninfer::test::linear_add;
    constexpr int n = 2304;
    std::set<int> points{1, 4, 8, 1023, 1024, 1025, 4096, 4097};
    for (int boundary : boundaries)
        for (int tokens : {boundary - 1, boundary, boundary + 1})
            if (tokens > 0) points.insert(tokens);

    std::vector<Invocation> calls;
    for (int tokens : points) {
        calls.push_back({.tokens = tokens, .graph = GraphReplay::OriginalZeroOriginal});
    }
    add_cases(suite, QType::BF16,
              {.n                = n,
               .k                = k,
               .seed             = static_cast<std::uint32_t>(n + k + 719),
               .invocations      = calls,
               .fixture          = FixturePattern::Hashed,
               .columns          = Samples::Uniform32,
               .interval_witness = true});
    namespace linear_suite = ninfer::test::linear_suite;
    suite.add(linear_suite::make_case("bf16", n, k, 0, ops::LinearPolicy::A16Only,
                                      linear_suite::Execution::Eager, "metadata"),
              [k] { return inkling_bf16_rejections(k); });
}
#endif

} // namespace

void ninfer::test::linear_add::add_bf16_cases(Suite& suite) {
    bf16_a16_conformance(suite);
#if NINFER_TARGET_RTX5090
    run_inkling_bf16_add(suite, 3072,
                         {1, 8, 16, 64, 128, 224, 256, 288, 448, 512, 576, 672, 896, 1152, 1344});
    run_inkling_bf16_add(suite, 9216, {1,   8,   16,  32,  64,   72,   80,   96,   224,  256, 288,
                                       448, 576, 672, 896, 1152, 1344, 1792, 2304, 3136, 3584});
#endif
}
