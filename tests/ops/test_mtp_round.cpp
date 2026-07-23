#include "ninfer/ops/mtp_round.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

template <class T>
int expect_eq(const std::string& label, const std::vector<T>& got, const std::vector<T>& expected) {
    if (got == expected) { return 0; }
    std::cerr << label << ": mismatch\n";
    return 1;
}

int alignment_case(std::int32_t accepted_value, const std::vector<std::int32_t>& expected) {
    constexpr int k  = 5;
    auto d_verify    = to_device_i32({40, 41, 42, 43, 44, 45});
    auto d_token     = to_device_i32({99});
    auto d_accepted  = to_device_i32({accepted_value});
    auto d_alignment = to_device_i32({-1, -1, -1, -1, -1, -1});

    Tensor verify(d_verify.p, DType::I32, {k + 1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor alignment(d_alignment.p, DType::I32, {k + 1});
    ops::mtp_prepare_alignment_ids(verify, token, accepted, alignment, nullptr);
    cudaDeviceSynchronize();

    return expect_eq("MTP alignment ids", from_device_i32(d_alignment, k + 1), expected);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    failures += alignment_case(2, {41, 42, 99, 44, 45, -1});
    failures += alignment_case(5, {41, 42, 43, 44, 45, 99});
    if (failures == 0) { std::cout << "mtp round tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
