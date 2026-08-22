// Column-wise token-count invariance probe for the linear families.
//
// For a registered (n, k) problem, column 0 of the activation is held fixed while the batch
// token count t varies. Because every output column depends only on its own activation column,
// a batch-composition-invariant implementation must produce bitwise-identical out[:, 0] for
// every t. Differences mark the t boundaries at which the selected kernel (or its internal
// schedule) changes its reduction order — the source of greedy-decode drift under concurrent
// load. This is an instrument: it always exits 0 and prints the per-shape map.

#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include "core/arena.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

constexpr std::int32_t kTokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 24, 25, 32, 33, 40, 48, 49, 64};

struct Shape {
    const char* family;
    WeightGenerator generator;
    std::int32_t n;
    std::int32_t k;
};

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::uint16_t> out(count);
    for (auto& v : out) { v = f32_to_bf16(dist(rng)); }
    return out;
}

void probe(const Shape& shape) {
    const std::int32_t t_max = *std::max_element(std::begin(kTokens), std::end(kTokens));
    quantized_weight::PackedWeight host_weight = shape.generator(shape.n, shape.k, 7u);
    DeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.bytes);
    const Weight weight = host_weight.device_weight(device_weight.p);

    // Column 0 is the probe column; the remaining columns are arbitrary batch neighbours.
    const std::vector<std::uint16_t> activation =
        random_bf16(static_cast<std::size_t>(shape.k) * static_cast<std::size_t>(t_max), 11u);
    DeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes);
    DeviceBuffer device_out(static_cast<std::size_t>(shape.n) * static_cast<std::size_t>(t_max) *
                            sizeof(std::uint16_t));

    std::vector<std::uint16_t> baseline;
    std::vector<std::uint16_t> previous;
    std::string map;
    bool invariant = true;
    for (const std::int32_t t : kTokens) {
        Tensor input(device_activation.p, DType::BF16, {shape.k, t});
        Tensor destination(device_out.p, DType::BF16, {shape.n, t});
        std::vector<std::uint16_t> column(static_cast<std::size_t>(shape.n));
        try {
            const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                weight.qtype, shape.n, shape.k, ops::LinearPolicy::A16Only, t, t);
            DeviceArena workspace(std::max<std::size_t>(capacity, 256));
            ops::linear(input, weight, destination, ops::LinearPolicy::A16Only, workspace,
                        nullptr);
            cuda_check(cudaDeviceSynchronize(), "synchronize linear");
        } catch (const std::exception& error) {
            map += " t" + std::to_string(t) + ":unsupported";
            continue;
        }
        device_out.copy_to_host(column.data(), column.size() * sizeof(std::uint16_t));
        if (baseline.empty()) {
            baseline = column;
            previous = column;
            map += " t" + std::to_string(t) + ":base";
            continue;
        }
        std::size_t differing = 0;
        float max_delta       = 0.0f;
        for (std::size_t i = 0; i < column.size(); ++i) {
            if (column[i] != baseline[i]) {
                ++differing;
                max_delta = std::max(max_delta, std::fabs(bf16_to_f32(column[i]) -
                                                          bf16_to_f32(baseline[i])));
            }
        }
        const bool same_as_previous = column == previous;
        previous                    = column;
        if (differing == 0) {
            map += " t" + std::to_string(t) + ":=";
        } else {
            invariant = false;
            map += " t" + std::to_string(t) + ":" + (same_as_previous ? "~" : "!") +
                   std::to_string(differing) + "/" + std::to_string(max_delta);
        }
    }
    std::cout << shape.family << " [" << shape.n << "," << shape.k << "] "
              << (invariant ? "INVARIANT" : "VARIANT") << map << '\n';
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    const Shape shapes[] = {
        {"W8", make_w8g32_f16s_weight, 1024, 5120},   {"W8", make_w8g32_f16s_weight, 6144, 5120},
        {"W8", make_w8g32_f16s_weight, 14336, 5120},  {"W8", make_w8g32_f16s_weight, 34816, 5120},
        {"W8", make_w8g32_f16s_weight, 5120, 6144},   {"W8", make_w8g32_f16s_weight, 5120, 17408},
        {"W8", make_w8g32_f16s_weight, 5120, 10240},  {"W8", make_w8g32_f16s_weight, 248320, 5120},
        {"Q6", make_q6g64_f16s_weight, 248320, 5120}, {"Q4", make_q4g64_f16s_weight, 1024, 5120},
        {"Q4", make_q4g64_f16s_weight, 6144, 5120},   {"Q4", make_q4g64_f16s_weight, 34816, 5120},
        {"BF16", nullptr, 14336, 5120},
    };
    std::cout << "legend: '=' bitwise equal to t=1 column; '!' differs from t=1 and from the"
                 " previous t (new reduction order); '~' differs from t=1 but equals the previous"
                 " t (same class as previous); value = differing_elements/max_abs_delta\n";
    for (const Shape& shape : shapes) {
        if (shape.generator == nullptr) { continue; }
        try {
            probe(shape);
        } catch (const std::exception& error) {
            std::cout << shape.family << " [" << shape.n << "," << shape.k
                      << "] error: " << error.what() << '\n';
        }
    }
    std::cout << "t-invariance probe: DONE\n";
    return 0;
}
