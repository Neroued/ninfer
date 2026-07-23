#include "ninfer/ops/position.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int position_case(int count, int start, int delta_value, bool in_place) {
    DBuf filled(static_cast<std::size_t>(count) * sizeof(int));
    Tensor filled_tensor(filled.p, DType::I32, {count});
    ops::fill_i32_positions(filled_tensor, start, nullptr);
    cudaDeviceSynchronize();

    std::vector<int> expected_filled(count);
    for (int i = 0; i < count; ++i) { expected_filled[static_cast<std::size_t>(i)] = start + i; }
    if (from_device_i32(filled, count) != expected_filled) {
        std::cerr << "position fill mismatch count=" << count << '\n';
        return 1;
    }

    DBuf delta = to_device_i32({delta_value});
    DBuf offset(static_cast<std::size_t>(count) * sizeof(int));
    Tensor delta_tensor(delta.p, DType::I32, {1});
    Tensor offset_tensor(offset.p, DType::I32, {count});
    Tensor& destination = in_place ? filled_tensor : offset_tensor;
    ops::offset_i32_positions(filled_tensor, delta_tensor, destination, nullptr);
    cudaDeviceSynchronize();

    std::vector<int> expected_offset(count);
    for (int i = 0; i < count; ++i) {
        expected_offset[static_cast<std::size_t>(i)] = start + delta_value + i;
    }
    const auto got = from_device_i32(in_place ? filled : offset, count);
    if (got != expected_offset) {
        std::cerr << "position offset mismatch count=" << count << " in_place=" << in_place << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    for (int count = 1; count <= 16; ++count) {
        failures += position_case(count, 262144 - count, -17, false);
        failures += position_case(count, 262144 - 2 * count, count, true);
    }
    failures += position_case(1024, 131072, -17, false);
    std::cout << (failures ? "FAIL" : "OK") << " position Ops\n";
    return failures ? 1 : 0;
}
