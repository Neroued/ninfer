#include "core/weight.h"
#include "ops/linear_pair/linear_pair_test_common.h"

#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/weight_input.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::linear_pair {
namespace {

constexpr std::int32_t kOutputRows       = 1024;
constexpr std::size_t kOutputScanWords   = 1U << 20;
constexpr std::int32_t kDFlashParentRows = 6144;
constexpr std::int32_t kDFlashFirstRow   = 4096;
constexpr std::int32_t kDFlashSecondRow  = 5120;

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
    std::int32_t first_row;
    std::int32_t second_row;
};

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear_pair test: invalid ") + label + " extent");
    }
    const std::size_t a = static_cast<std::size_t>(first);
    const std::size_t b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear_pair test: ") + label + " size overflow");
    }
    return a * b;
}

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index :
         {0, 1, extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1}) {
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

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

PairFixture make_pair_fixture(std::int32_t k, std::uint32_t seed) {
    const quantized_weight::PatternedWeightOptions options{
        quantized_weight::RowSplitScalePattern::Tiny};
    if (k == 5120) {
        return {
            false,
            quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, kOutputRows, k, seed,
                                                    options),
            quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, kOutputRows, k, seed + 1U,
                                                    options),
            0,
            0,
        };
    }
    if (k == 2048) {
        return {
            true,
            quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, kDFlashParentRows, k, seed,
                                                    options),
            {},
            kDFlashFirstRow,
            kDFlashSecondRow,
        };
    }
    throw std::invalid_argument("linear_pair test: unsupported K");
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

int verify_preserved(const test::GuardedDeviceBuffer& device,
                     std::span<const std::uint8_t> expected, const char* label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": input payload was modified\n";
    return 1;
}

} // namespace

bool cuda_available() { return !test::cuda_unavailable(); }

int run_q8_a16_shape(std::string_view label, const ShapeCase& shape) {
    const std::vector<std::int32_t> tokens = conformance_tokens(shape);
    if (tokens.empty()) { throw std::invalid_argument("linear_pair test: no token cases"); }
    const std::int32_t maximum_t = tokens.back();

    const std::vector<std::int32_t> oracle_rows = sampled_indices(kOutputRows);
    PairFixture fixture                         = make_pair_fixture(shape.k, shape.seed);
    const quantized_weight::PackedWeight& second_storage =
        fixture.shared_payload ? fixture.first_storage : fixture.second_storage;
    const std::vector<std::int32_t> first_physical_rows =
        physical_rows(oracle_rows, fixture.first_row);
    const std::vector<std::int32_t> second_physical_rows =
        physical_rows(oracle_rows, fixture.second_row);
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

    void* const second_payload =
        fixture.shared_payload ? first_device_weight.data() : second_device_weight.data();
    const auto parent = [](const quantized_weight::PackedWeight& source, const void* payload) {
        const std::array shape{static_cast<std::uint64_t>(source.weight.n),
                               static_cast<std::uint64_t>(source.weight.k)};
        return WeightParent{weight_geometry(source.weight.qtype, source.weight.layout, shape),
                            static_cast<const std::byte*>(payload)};
    };
    const auto first_parent  = parent(fixture.first_storage, first_device_weight.data());
    const auto second_parent = parent(second_storage, second_payload);
    const auto region        = [](const WeightParent& storage, std::uint64_t row) {
        const auto k = storage.geometry.shape[1];
        return WeightView{{kOutputRows, k}, {{&storage, row * k, (row + kOutputRows) * k}}};
    };
    const auto first_view = region(first_parent, fixture.first_row);
    const auto second_view =
        region(fixture.shared_payload ? first_parent : second_parent, fixture.second_row);
    const Weight first_weight  = ops::prepare_linear_weight({first_view}).weight;
    const Weight second_weight = ops::prepare_linear_weight({second_view}).weight;

    int failures = 0;
    for (const std::int32_t t : tokens) {
        const std::size_t output_words = checked_elements(kOutputRows, t, "output");
        test::GuardedDeviceBuffer first_output(output_words * sizeof(std::uint16_t));
        test::GuardedDeviceBuffer second_output(output_words * sizeof(std::uint16_t));
        first_output.fill(0xff);
        second_output.fill(0xff);

        Tensor input(device_activation.data(), DType::BF16, {shape.k, t});
        Tensor first(first_output.data(), DType::BF16, {kOutputRows, t});
        Tensor second(second_output.data(), DType::BF16, {kOutputRows, t});

        const std::string case_label =
            std::string(label) + " [1024," + std::to_string(shape.k) + "] T=" + std::to_string(t);
        try {
            ops::linear_pair(input, first_weight, second_weight, first, second, nullptr);
            test::cuda_check(cudaDeviceSynchronize(), "synchronize linear_pair");
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }

        failures += first_output.verify_guards((case_label + " first").c_str());
        failures += second_output.verify_guards((case_label + " second").c_str());
        const std::vector<std::int32_t> columns = sampled_indices(t);
        const OutputRead first_actual =
            read_output(first_output.data(), t, oracle_rows, columns, case_label + " first");
        const OutputRead second_actual =
            read_output(second_output.data(), t, oracle_rows, columns, case_label + " second");
        failures += first_actual.failures + second_actual.failures;

        const std::vector<double> first_reference =
            linear_pair_oracle(first_oracle, shape.k, static_cast<std::int32_t>(oracle_rows.size()),
                               activation, columns);
        const std::vector<double> second_reference =
            linear_pair_oracle(second_oracle, shape.k,
                               static_cast<std::int32_t>(oracle_rows.size()), activation, columns);
        failures += compare_output(case_label + " first", first_actual.selected, first_reference);
        failures +=
            compare_output(case_label + " second", second_actual.selected, second_reference);
    }

    failures += device_activation.verify_guards("linear_pair activation");
    failures += first_device_weight.verify_guards("linear_pair first weight");
    if (!fixture.shared_payload) {
        failures += second_device_weight.verify_guards("linear_pair second weight");
    }
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "linear_pair activation");
    failures += verify_preserved(first_device_weight, fixture.first_storage.payload,
                                 "linear_pair first weight");
    if (!fixture.shared_payload) {
        failures += verify_preserved(second_device_weight, fixture.second_storage.payload,
                                     "linear_pair second weight");
    }
    return failures;
}

} // namespace ninfer::test::linear_pair
