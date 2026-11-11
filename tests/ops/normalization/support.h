#pragma once

#include "core/buffer.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::test::norm {

inline constexpr float kEps = 1.0e-6F;

struct Shape {
    std::int32_t d;
    std::int32_t rows;
    std::int32_t tokens = 1;

    std::size_t elements() const {
        return static_cast<std::size_t>(d) * static_cast<std::size_t>(rows) *
               static_cast<std::size_t>(tokens);
    }
};

inline Tensor tensor_for(MemorySpan memory, const Shape& shape) {
    if (shape.tokens == 1) return Tensor(memory, DType::BF16, {shape.d, shape.rows});
    return Tensor(memory, DType::BF16, {shape.d, shape.rows, shape.tokens});
}

struct DeviceInput {
    DeviceBuffer storage;
    MemorySpan memory;
    std::vector<std::uint16_t> expected;
};

inline DeviceInput make_input(const std::vector<float>& values, std::size_t leading) {
    DeviceInput input;
    input.expected.resize(leading + values.size(), 0x5a5aU);
    for (std::size_t index = 0; index < values.size(); ++index) {
        input.expected[leading + index] = f32_to_bf16(values[index]);
    }
    input.storage = to_device(input.expected);
    input.memory  = input.storage.span().subspan(leading * sizeof(std::uint16_t));
    return input;
}

inline int verify_preserved(const std::string& label, const DeviceInput& input) {
    return verify_exact(label.c_str(),
                        from_device<std::uint16_t>(input.storage, input.expected.size()),
                        input.expected);
}

inline int verify_output_storage(const std::string& label, const GuardedDeviceBuffer& output,
                                 std::size_t leading) {
    int failures = output.verify_guards(label);
    if (leading) {
        failures += verify_exact((label + " prefix").c_str(),
                                 from_device<std::uint16_t>(output.data(), leading),
                                 std::vector<std::uint16_t>(leading, 0xffffU));
    }
    return failures;
}

} // namespace ninfer::test::norm
