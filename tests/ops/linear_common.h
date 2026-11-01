#pragma once

#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::test::linear_common {

inline std::size_t checked_elements(std::int32_t first, std::int32_t second,
                                    std::string_view label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument("linear test: invalid " + std::string(label) + " extent");
    }
    const auto a = static_cast<std::size_t>(first);
    const auto b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("linear test: " + std::string(label) + " size overflow");
    }
    return a * b;
}

inline std::vector<std::int32_t> all_indices(std::int32_t extent) {
    std::vector<std::int32_t> result(static_cast<std::size_t>(extent));
    std::iota(result.begin(), result.end(), 0);
    return result;
}

// The existing seven-coordinate sample used by LinearAdd and LinearPair. Other sampling
// recipes remain explicit in their owning test family.
inline std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
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

inline std::vector<std::int32_t> evenly_spaced_indices(std::int32_t extent,
                                                       std::int32_t count = 32) {
    if (count < 2) {
        throw std::invalid_argument("linear test: evenly spaced sample count must be at least two");
    }
    std::vector<std::int32_t> result;
    for (std::int32_t sample = 0; sample < count; ++sample) {
        const auto index = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(extent - 1) * sample) / (count - 1));
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// The caller completes Device work and separately checks any allocation guards.
inline int verify_preserved(const void* device, std::span<const std::byte> expected,
                            std::string_view label) {
    std::vector<std::byte> actual(expected.size());
    cuda_check(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost),
               "copy preserved linear input");
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": input payload was modified\n";
    return 1;
}

} // namespace ninfer::test::linear_common
