#pragma once

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::test::input_projection {

inline std::vector<std::int32_t> sampled_rows(std::int32_t rows, std::int32_t sample_count = 7) {
    if (rows <= 0 || sample_count <= 0) {
        throw std::invalid_argument("sampled_rows requires positive extents");
    }
    std::vector<std::int32_t> result;
    const std::int32_t selected = std::min(rows, sample_count);
    for (std::int32_t sample = 0; sample < selected; ++sample) {
        const std::int32_t row =
            selected == 1 ? 0
                          : static_cast<std::int32_t>(
                                (static_cast<std::int64_t>(rows - 1) * sample) / (selected - 1));
        if (std::find(result.begin(), result.end(), row) == result.end()) { result.push_back(row); }
    }
    return result;
}

inline std::vector<float> make_bf16_activation(std::int32_t rows, std::int32_t tokens,
                                               std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(rows) * tokens);
    fill_uniform(values, seed, -0.01F, 0.01F);
    round_to_bf16(values);
    return values;
}

class DevicePackedWeight {
public:
    explicit DevicePackedWeight(quantized_weight::PackedWeight packed)
        : host(std::move(packed)), device(host.payload.size()) {
        device.copy_from_host(host.payload.data(), host.payload.size());
    }

    Weight view() const { return host.device_weight(device.p); }

    int verify_preserved(std::string_view label) const {
        std::vector<std::uint8_t> after(host.payload.size());
        device.copy_to_host(after.data(), after.size());
        if (after == host.payload) { return 0; }
        std::cerr << label << ": packed weight was modified\n";
        return 1;
    }

    quantized_weight::PackedWeight host;
    DeviceBuffer device;
};

class GuardedBf16Tensor {
public:
    GuardedBf16Tensor(std::int32_t rows, std::int32_t tokens)
        : rows_(rows), tokens_(tokens),
          payload_bytes_(checked_elements(rows, tokens) * sizeof(std::uint16_t)),
          storage_(payload_bytes_) {
        storage_.fill(kPoisonByte);
    }

    void* data() { return storage_.data(); }

    const void* data() const { return storage_.data(); }

    Tensor tensor() { return Tensor(data(), DType::BF16, {rows_, tokens_}); }

    void copy_from_bits(std::span<const std::uint16_t> values) {
        if (values.size_bytes() != payload_bytes_) {
            throw std::invalid_argument("guarded BF16 tensor host payload size mismatch");
        }
        storage_.copy_from_host(values.data(), values.size_bytes());
    }

    std::vector<std::uint16_t> bits() const {
        return from_device<std::uint16_t>(data(), payload_bytes_ / sizeof(std::uint16_t));
    }

    std::vector<double> values() const {
        return from_device_bf16(data(), payload_bytes_ / sizeof(std::uint16_t));
    }

    int verify_guards(std::string_view label) const { return storage_.verify_guards(label); }

    int verify_fully_written(std::string_view label) const {
        const std::vector<std::uint16_t> output = bits();
        for (std::size_t index = 0; index < output.size(); ++index) {
            if (!std::isfinite(bf16_to_f32(output[index]))) {
                std::cerr << label << ": output element " << index << " was not finite\n";
                return 1;
            }
        }
        return 0;
    }

private:
    static std::size_t checked_elements(std::int32_t rows, std::int32_t tokens) {
        if (rows <= 0 || tokens <= 0) {
            throw std::invalid_argument("guarded BF16 tensor requires positive extents");
        }
        const std::size_t row_count   = static_cast<std::size_t>(rows);
        const std::size_t token_count = static_cast<std::size_t>(tokens);
        if (row_count > std::numeric_limits<std::size_t>::max() / token_count) {
            throw std::overflow_error("guarded BF16 tensor size overflow");
        }
        return row_count * token_count;
    }

    static constexpr std::uint8_t kPoisonByte = 0xff;

    std::int32_t rows_;
    std::int32_t tokens_;
    std::size_t payload_bytes_;
    GuardedDeviceBuffer storage_;
};

inline std::vector<double> gather_rows(const std::vector<double>& full, std::int32_t full_rows,
                                       std::int32_t row_offset, std::int32_t rows,
                                       std::int32_t tokens, std::int32_t sample_count = 7) {
    std::vector<double> gathered;
    const std::vector<std::int32_t> selected = sampled_rows(rows, sample_count);
    gathered.reserve(selected.size() * static_cast<std::size_t>(tokens));
    for (const std::int32_t local_row : selected) {
        const std::int32_t global_row = row_offset + local_row;
        for (std::int32_t token = 0; token < tokens; ++token) {
            gathered.push_back(full[static_cast<std::size_t>(token) * full_rows + global_row]);
        }
    }
    return gathered;
}

inline std::vector<double>
projection_oracle(const quantized_weight::PackedWeight& weight, std::int32_t weight_row_offset,
                  std::int32_t output_rows, const std::vector<float>& activation,
                  std::int32_t hidden, std::int32_t tokens, std::int32_t sample_count = 7) {
    std::vector<double> expected;
    const std::vector<std::int32_t> selected = sampled_rows(output_rows, sample_count);
    expected.reserve(selected.size() * static_cast<std::size_t>(tokens));
    for (const std::int32_t local_row : selected) {
        // Decode each represented coefficient once; every token still evaluates the full
        // naive FP64 dot product, without production staging or intermediate rounding.
        std::vector<double> decoded(hidden);
        for (std::int32_t column = 0; column < hidden; ++column)
            decoded[column] = quantized_weight::logical_weight_fp64(
                weight, weight_row_offset + local_row, column);
        for (std::int32_t token = 0; token < tokens; ++token) {
            const float* input = activation.data() + static_cast<std::size_t>(token) * hidden;
            double sum         = 0.0;
            for (std::int32_t column = 0; column < hidden; ++column)
                sum += decoded[column] * static_cast<double>(input[column]);
            expected.push_back(sum);
        }
    }
    return expected;
}

inline int compare(std::string_view label, const std::vector<double>& actual,
                   const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(std::string(label).c_str(), actual, expected, criterion);
}

inline int verify_preserved(std::string_view label, const DeviceBuffer& device,
                            std::span<const std::uint16_t> before) {
    const std::vector<std::uint16_t> after = from_device<std::uint16_t>(device, before.size());
    if (std::equal(after.begin(), after.end(), before.begin(), before.end())) { return 0; }
    std::cerr << label << ": BF16 input was modified\n";
    return 1;
}

inline int verify_preserved(std::string_view label, const DeviceBuffer& device,
                            std::span<const std::int32_t> before) {
    const std::vector<std::int32_t> after = from_device<std::int32_t>(device, before.size());
    if (std::equal(after.begin(), after.end(), before.begin(), before.end())) { return 0; }
    std::cerr << label << ": I32 input was modified\n";
    return 1;
}

inline std::vector<std::uint16_t> bf16_bits(std::span<const float> values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

} // namespace ninfer::test::input_projection
