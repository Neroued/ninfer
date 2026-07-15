#include "ninfer/ops/cast.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    const std::vector<float> source{-10.5f,          -1.00390625f, -0.0f,      0.0f,
                                    0.333251953125f, 1.00390625f,  3.1415927f, 65504.0f};
    DBuf device_source = to_device_f32(source);
    DBuf device_destination(source.size() * sizeof(std::uint16_t));
    Tensor source_tensor(device_source.p, DType::FP32,
                         {4, static_cast<std::int32_t>(source.size() / 4)});
    Tensor destination_tensor(device_destination.p, DType::BF16,
                              {4, static_cast<std::int32_t>(source.size() / 4)});
    ops::cast_fp32_to_bf16(source_tensor, destination_tensor, nullptr);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> actual(source.size());
    cudaMemcpy(actual.data(), device_destination.p, device_destination.bytes,
               cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (actual[i] != f32_to_bf16(source[i])) {
            std::cerr << "cast_fp32_to_bf16 mismatch at " << i << '\n';
            return 1;
        }
    }
    std::cout << "OK cast_fp32_to_bf16\n";
    return 0;
}
