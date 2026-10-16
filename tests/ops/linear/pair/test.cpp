#include "core/weight.h"
#include "cuda_fixture.h"
#include "ops/linear/support.h"
#include "core/decode_graph.h"

#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
enum class WeightParents : std::uint8_t {
    Separate,
    Shared,
};

struct ShapeCase {
    std::int32_t k;
    std::uint32_t seed;
    WeightParents parents;
    std::int32_t parent_rows;
    std::int32_t first_row;
    std::int32_t second_row;
    std::span<const std::int32_t> route_starts;
    std::span<const std::int32_t> route_interiors;
};

struct Invocation {
    int tokens;
    bool graph = false;
    std::string id;
};

constexpr std::int32_t kOutputRows     = 1024;
constexpr std::size_t kOutputScanWords = 1U << 20;

using linear_common::checked_elements;
using linear_common::sampled_indices;
using linear_common::verify_preserved;

// Both observable projections use the same A16 arithmetic profile. T and the selected pair
// launcher never select another correctness criterion.
constexpr ReductionCriterion kLinearPairA16Tolerance{
    2.9e-3,
    4.0e-3,
    3.8e-3,
};

struct PairFixture {
    bool shared_payload;
    quantized_weight::PackedWeight first_storage;
    quantized_weight::PackedWeight second_storage;
};

std::vector<std::int32_t> conformance_tokens(const ShapeCase& shape) {
    std::vector<std::int32_t> result{1};
    for (const std::int32_t boundary : shape.route_starts) {
        if (boundary <= 1) {
            throw std::invalid_argument("linear_pair test: route starts must be greater than one");
        }
        result.push_back(boundary - 1);
        result.push_back(boundary);
        if (boundary != std::numeric_limits<std::int32_t>::max()) {
            result.push_back(boundary + 1);
        }
    }
    for (const std::int32_t interior : shape.route_interiors) {
        if (interior <= 0) {
            throw std::invalid_argument("linear_pair test: route interior must be positive");
        }
        result.push_back(interior);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

PairFixture make_pair_fixture(const ShapeCase& shape) {
    if (shape.parent_rows < kOutputRows || shape.first_row < 0 || shape.second_row < 0 ||
        shape.first_row > shape.parent_rows - kOutputRows ||
        shape.second_row > shape.parent_rows - kOutputRows) {
        throw std::invalid_argument("linear_pair test: row view is outside its parent");
    }
    const quantized_weight::PatternedWeightOptions options{
        quantized_weight::RowSplitScalePattern::Tiny};
    PairFixture fixture{
        shape.parents == WeightParents::Shared,
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, shape.parent_rows, shape.k,
                                                shape.seed, options),
        {},
    };
    if (!fixture.shared_payload) {
        fixture.second_storage = quantized_weight::make_patterned_weight(
            QType::Q8_G32_FP16, shape.parent_rows, shape.k, shape.seed + 1U, options);
    }
    return fixture;
}

std::vector<std::int32_t> physical_rows(std::span<const std::int32_t> logical_rows,
                                        std::int32_t row_begin) {
    std::vector<std::int32_t> result(logical_rows.begin(), logical_rows.end());
    for (std::int32_t& row : result) { row += row_begin; }
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> result(checked_elements(k, t, "activation"));
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            const std::uint64_t token_block = static_cast<std::uint64_t>(token / 256);
            const std::uint64_t coordinate =
                static_cast<std::uint64_t>(column) * 17U + static_cast<std::uint64_t>(token) * 31U +
                static_cast<std::uint64_t>(seed) * 13U +
                token_block * (static_cast<std::uint64_t>(column / 256) * 13U + 47U);
            const int raw = static_cast<int>(coordinate & 0xffU);
            result[static_cast<std::size_t>(token) * k + column] =
                test::f32_to_bf16(static_cast<float>(raw - 128) * (1.0F / 256.0F));
        }
    }
    return result;
}

std::vector<double> linear_pair_oracle(std::span<const float> weight, std::int32_t k,
                                       std::int32_t oracle_n,
                                       const std::vector<std::uint16_t>& activation,
                                       std::span<const std::int32_t> columns) {
    std::vector<double> result(
        checked_elements(oracle_n, static_cast<std::int32_t>(columns.size()), "oracle output"));
    for (std::size_t selected_column = 0; selected_column < columns.size(); ++selected_column) {
        const std::int32_t token = columns[selected_column];
        const std::uint16_t* activation_column =
            activation.data() + static_cast<std::size_t>(token) * k;
        for (std::int32_t oracle_row = 0; oracle_row < oracle_n; ++oracle_row) {
            double sum              = 0.0;
            const float* weight_row = weight.data() + static_cast<std::size_t>(oracle_row) * k;
            for (std::int32_t column = 0; column < k; ++column) {
                sum += static_cast<double>(weight_row[column]) *
                       static_cast<double>(test::bf16_to_f32(activation_column[column]));
            }
            result[selected_column * static_cast<std::size_t>(oracle_n) + oracle_row] = sum;
        }
    }
    return result;
}

struct OutputRead {
    int failures = 0;
    std::vector<double> selected;
};

OutputRead read_output(const void* device, std::int32_t t, std::span<const std::int32_t> rows,
                       std::span<const std::int32_t> columns, std::string_view label) {
    const std::size_t total_words = checked_elements(kOutputRows, t, "output");
    std::vector<std::size_t> wanted;
    wanted.reserve(rows.size() * columns.size());
    for (const std::int32_t column : columns) {
        for (const std::int32_t row : rows) {
            wanted.push_back(static_cast<std::size_t>(column) * kOutputRows + row);
        }
    }

    OutputRead result;
    result.selected.resize(wanted.size());
    std::vector<std::uint16_t> chunk(std::min(kOutputScanWords, total_words));
    std::size_t wanted_index    = 0;
    std::size_t poison_count    = 0;
    std::size_t nonfinite_count = 0;
    for (std::size_t begin = 0; begin < total_words; begin += chunk.size()) {
        const std::size_t count = std::min(chunk.size(), total_words - begin);
        test::cuda_check(
            cudaMemcpy(chunk.data(),
                       static_cast<const std::uint8_t*>(device) + begin * sizeof(std::uint16_t),
                       count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "copy linear_pair output");
        for (const std::uint16_t bits : std::span<const std::uint16_t>(chunk.data(), count)) {
            if (bits == 0xffffU) { ++poison_count; }
            if ((bits & 0x7f80U) == 0x7f80U) { ++nonfinite_count; }
        }
        while (wanted_index < wanted.size() && wanted[wanted_index] < begin + count) {
            result.selected[wanted_index] =
                static_cast<double>(test::bf16_to_f32(chunk[wanted[wanted_index] - begin]));
            ++wanted_index;
        }
    }
    if (poison_count != 0) {
        std::cerr << label << ": output retains " << poison_count << " poison values\n";
        ++result.failures;
    }
    if (nonfinite_count != 0) {
        std::cerr << label << ": output contains " << nonfinite_count << " non-finite values\n";
        ++result.failures;
    }
    return result;
}

int compare_output(std::string_view label, std::span<const double> actual,
                   std::span<const double> reference) {
    return verify_reduction(label, actual, reference, kLinearPairA16Tolerance);
}

int run_q8_a16_shape(std::string_view label, const ShapeCase& shape,
                     std::span<const Invocation> calls) {
    int maximum_t = 0;
    for (const auto& call : calls) maximum_t = std::max(maximum_t, call.tokens);

    const std::vector<std::int32_t> oracle_rows = sampled_indices(kOutputRows);
    PairFixture fixture                         = make_pair_fixture(shape);
    const quantized_weight::PackedWeight& second_storage =
        fixture.shared_payload ? fixture.first_storage : fixture.second_storage;
    const std::vector<std::int32_t> first_physical_rows =
        physical_rows(oracle_rows, shape.first_row);
    const std::vector<std::int32_t> second_physical_rows =
        physical_rows(oracle_rows, shape.second_row);
    const std::vector<float> first_oracle =
        quantized_weight::materialize_rows_fp32(fixture.first_storage, first_physical_rows);
    const std::vector<float> second_oracle =
        quantized_weight::materialize_rows_fp32(second_storage, second_physical_rows);
    const std::vector<std::uint16_t> activation =
        make_activation(shape.k, maximum_t, shape.seed + 2U);

    test::GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    test::GuardedDeviceBuffer first_device_weight(fixture.first_storage.payload.size());
    first_device_weight.copy_from_host(fixture.first_storage.payload.data(),
                                       fixture.first_storage.payload.size());
    test::GuardedDeviceBuffer second_device_weight(
        fixture.shared_payload ? 1U : fixture.second_storage.payload.size());
    if (!fixture.shared_payload) {
        second_device_weight.copy_from_host(fixture.second_storage.payload.data(),
                                            fixture.second_storage.payload.size());
    }

    const auto second_payload =
        fixture.shared_payload ? first_device_weight.span() : second_device_weight.span();
    const auto parent = [](const quantized_weight::PackedWeight& source, ConstMemorySpan payload) {
        const std::array shape{static_cast<std::uint64_t>(source.weight.n),
                               static_cast<std::uint64_t>(source.weight.k)};
        return WeightParent{weight_geometry(source.weight.qtype, source.weight.layout, shape),
                            payload.device, static_cast<const std::byte*>(payload.data)};
    };
    const auto first_parent  = parent(fixture.first_storage, first_device_weight.span());
    const auto second_parent = parent(second_storage, second_payload);
    const auto region        = [](const WeightParent& storage, std::uint64_t row) {
        const auto k = storage.geometry.shape[1];
        return WeightView{{kOutputRows, k}, {{&storage, row * k, (row + kOutputRows) * k}}};
    };
    const auto first_view = region(first_parent, shape.first_row);
    const auto second_view =
        region(fixture.shared_payload ? first_parent : second_parent, shape.second_row);
    const Weight first_weight  = ops::prepare_linear_weight({first_view}).weight;
    const Weight second_weight = ops::prepare_linear_weight({second_view}).weight;

    int failures = 0;
    DeviceContext context;
    for (const auto& call : calls) {
        const int t                    = call.tokens;
        const std::size_t output_words = checked_elements(kOutputRows, t, "output");
        test::GuardedDeviceBuffer first_output(output_words * sizeof(std::uint16_t));
        test::GuardedDeviceBuffer second_output(output_words * sizeof(std::uint16_t));

        Tensor input(device_activation.span(), DType::BF16, {shape.k, t});
        Tensor first(first_output.span(), DType::BF16, {kOutputRows, t});
        Tensor second(second_output.span(), DType::BF16, {kOutputRows, t});

        const auto& case_label = call.id;
        const auto profile     = ops::LinearProfile{QType::Q8_G32_FP16, kOutputRows, shape.k};
        if (ops::linear_pair_workspace_capacity_bytes(first_weight, second_weight, t, t) != 0 ||
            ops::linear_pair_workspace_capacity_bytes(profile, profile, t, t) != 0)
            throw std::runtime_error("LinearPair must not require scratch");
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        const auto launch = [&] {
            ops::linear_pair(input, first_weight, second_weight, first, second,
                             context.execution_view());
        };
        cuda_synchronize();
        if (call.graph) {
            definition.capture(context.stream(), launch);
            graph.instantiate(definition);
        }
        std::vector<std::uint16_t> negative;
        if (call.graph) {
            negative = activation;
            for (auto& bits : negative) bits ^= 0x8000U;
        }
        for (int phase = 0; phase < (call.graph ? 2 : 1); ++phase) {
            const auto& represented_input = phase ? negative : activation;
            if (phase)
                device_activation.copy_from_host(represented_input.data(),
                                                 device_activation.bytes());
            first_output.fill(0xff);
            second_output.fill(0xff);
            cuda_synchronize();
            if (call.graph)
                graph.launch(context.stream());
            else
                launch();
            context.synchronize();
            failures += first_output.verify_guards((case_label + " first").c_str());
            failures += second_output.verify_guards((case_label + " second").c_str());
            const std::vector<std::int32_t> columns = sampled_indices(t);
            const OutputRead first_actual =
                read_output(first_output.data(), t, oracle_rows, columns, case_label + " first");
            const OutputRead second_actual =
                read_output(second_output.data(), t, oracle_rows, columns, case_label + " second");
            failures += first_actual.failures + second_actual.failures;

            const std::vector<double> first_reference = linear_pair_oracle(
                first_oracle, shape.k, static_cast<std::int32_t>(oracle_rows.size()),
                represented_input, columns);
            const std::vector<double> second_reference = linear_pair_oracle(
                second_oracle, shape.k, static_cast<std::int32_t>(oracle_rows.size()),
                represented_input, columns);
            failures +=
                compare_output(case_label + " first", first_actual.selected, first_reference);
            failures +=
                compare_output(case_label + " second", second_actual.selected, second_reference);
            if (call.graph)
                failures +=
                    verify_preserved(device_activation.data(),
                                     std::as_bytes(std::span(represented_input)), case_label);
        }
        if (call.graph)
            device_activation.copy_from_host(activation.data(), device_activation.bytes());
    }

    failures += device_activation.verify_guards("linear_pair activation");
    failures += first_device_weight.verify_guards("linear_pair first weight");
    if (!fixture.shared_payload) {
        failures += second_device_weight.verify_guards("linear_pair second weight");
    }
    failures += verify_preserved(device_activation.data(), std::as_bytes(std::span(activation)),
                                 "linear_pair activation");
    failures += verify_preserved(first_device_weight.data(),
                                 std::as_bytes(std::span(fixture.first_storage.payload)),
                                 "linear_pair first weight");
    if (!fixture.shared_payload) {
        failures += verify_preserved(second_device_weight.data(),
                                     std::as_bytes(std::span(fixture.second_storage.payload)),
                                     "linear_pair second weight");
    }
    return failures;
}

void add_shape(linear_suite::Suite& suite, const ShapeCase& shape) {
    std::vector<Invocation> calls;
    for (int t : conformance_tokens(shape)) calls.push_back({t});
    calls.push_back({shape.k == 2048 ? 16 : 24, true});
    std::vector<linear_suite::Case> cases;
    for (auto& call : calls) {
        auto test = linear_suite::make_case(
            "q8", kOutputRows, shape.k, call.tokens, ops::LinearPolicy::A16Only,
            call.graph ? linear_suite::Execution::Graph : linear_suite::Execution::Eager,
            std::string(shape.parents == WeightParents::Shared ? "shared-parent"
                                                               : "separate-parents") +
                "/s" + std::to_string(shape.seed));
        cases.push_back(std::move(test));
    }
    auto fixture            = shape;
    fixture.route_starts    = {};
    fixture.route_interiors = {};
    suite.add(std::move(cases),
              [shape = fixture, calls = std::move(calls)](auto selected, auto declared) {
                  std::vector<Invocation> chosen;
                  for (auto index : selected) {
                      chosen.push_back(calls[index]);
                      chosen.back().id = declared[index].id;
                  }
                  return run_q8_a16_shape("LinearPair", shape, chosen);
              });
    suite.add(linear_suite::make_case("q8", kOutputRows, shape.k, 0, ops::LinearPolicy::A16Only,
                                      linear_suite::Execution::Eager, "workspace-domain"),
              [shape = fixture] {
                  const auto profile = ops::LinearProfile{QType::Q8_G32_FP16, kOutputRows, shape.k};
                  int failures       = 0;
                  for (auto [first, last] : {std::pair{0, 1}, std::pair{-1, 8}, std::pair{8, 7}}) {
                      try {
                          (void)ops::linear_pair_workspace_capacity_bytes(profile, profile, first,
                                                                          last);
                          ++failures;
                      } catch (const std::invalid_argument&) {}
                  }
                  return failures;
              });
}
} // namespace

int main(int argc, char** argv) {
    linear_suite::Suite suite;
    constexpr std::array<std::int32_t, 2> kK5120RouteStarts{86, 961};
    constexpr std::array<std::int32_t, 3> kK5120RouteInteriors{24, 512, 1024};
    add_shape(suite, ShapeCase{.k               = 5120,
                               .seed            = 431U,
                               .parents         = WeightParents::Separate,
                               .parent_rows     = 1024,
                               .first_row       = 0,
                               .second_row      = 0,
                               .route_starts    = kK5120RouteStarts,
                               .route_interiors = kK5120RouteInteriors});

    constexpr std::array<std::int32_t, 36> kK2048RouteStarts{
        2,    33,   49,   65,   81,   89,   97,   105,  113,  129,  161,  193,
        385,  481,  641,  642,  673,  681,  785,  897,  961,  977,  1281, 1317,
        1345, 1346, 1441, 1467, 1681, 1709, 1921, 1923, 2017, 2019, 2209, 2271,
    };
    constexpr std::array<std::int32_t, 37> kK2048RouteInteriors{
        1,    16,   40,   56,   72,   84,   92,   100,  108,  120,  144,  176,  288,
        432,  560,  641,  656,  676,  736,  840,  928,  968,  1120, 1296, 1332, 1345,
        1392, 1456, 1576, 1696, 1816, 1921, 1968, 2017, 2112, 2240, 4096,
    };
    add_shape(suite, ShapeCase{.k               = 2048,
                               .seed            = 433U,
                               .parents         = WeightParents::Shared,
                               .parent_rows     = 6144,
                               .first_row       = 4096,
                               .second_row      = 5120,
                               .route_starts    = kK2048RouteStarts,
                               .route_interiors = kK2048RouteInteriors});
    return suite.main(argc, argv);
}
