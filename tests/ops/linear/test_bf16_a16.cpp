#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/linear/linear_test_common.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <set>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

std::vector<std::uint16_t> make_activation_bits(std::int32_t hidden, std::int32_t tokens) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < hidden; ++column) {
            const int centered = ((column * 29 + token * 71 + 17) & 0xff) - 128;
            result[static_cast<std::size_t>(token) * hidden + column] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<float> materialize(std::span<const std::uint16_t> bits) {
    std::vector<float> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = bf16_to_f32(bits[index]);
    }
    return result;
}

std::vector<double> oracle_all_rows(const HostWeight& weight, std::span<const float> activation) {
    std::vector<double> result(static_cast<std::size_t>(weight.n));
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(weight.n, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(weight.n) * thread) / threads);
        const std::int32_t end = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(weight.n) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) {
                result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return result;
}

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    std::vector<std::int32_t> result{0, 1, rows / 4, rows / 2, (3 * rows) / 4, rows - 2, rows - 1};
    if (rows == 14336) {
        result.insert(result.end(), {1023, 6143, 6144, 7167, 7168, 13311, 13312});
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int run_bf16_linear_case(DeviceWeight& weight, std::int32_t tokens, bool replay = false) {
    const std::int32_t rows                    = weight.host.n;
    const std::int32_t hidden                  = weight.host.k;
    std::vector<std::uint16_t> activation_bits = make_activation_bits(hidden, tokens);
    std::vector<float> activation              = materialize(activation_bits);
    DeviceBuffer device_activation             = to_device(activation_bits);
    GuardedDeviceBuffer guarded_output(static_cast<std::size_t>(rows) * tokens *
                                       sizeof(std::uint16_t));
    guarded_output.fill(0xff);

    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    DeviceArena workspace(256);
    ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
    cuda_synchronize();

    if (replay) {
        DeviceContext context;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, [&] {
            ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace,
                        context.stream);
        });
        graph.instantiate(definition);
        graph.launch(context.stream);
        cuda_synchronize();
        for (auto& bits : activation_bits) bits ^= 0x8000;
        activation = materialize(activation_bits);
        device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
        guarded_output.fill(0xff);
        cuda_synchronize();
        graph.launch(context.stream);
        cuda_synchronize();
    }
    const std::string suffix = " T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
    int failures             = guarded_output.verify_guards("BF16_A16 Linear output" + suffix);
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), static_cast<std::size_t>(rows) * tokens);
    for (std::size_t index = 0; index < output_bits.size(); ++index) {
        const std::uint16_t bits = output_bits[index];
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "BF16_A16 Linear output" << suffix << " element " << index
                      << " is not finite\n";
            ++failures;
            break;
        }
    }

    std::vector<double> actual;
    std::vector<double> expected;
    if (tokens == 1) {
        const std::vector<double> complete =
            oracle_all_rows(weight.host, std::span<const float>(activation));
        actual.reserve(rows);
        for (const std::uint16_t bits : output_bits) { actual.push_back(bf16_to_f32(bits)); }
        expected = complete;
    } else {
        const std::vector<std::int32_t> sampled       = sampled_rows(rows);
        const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
        actual.reserve(sampled.size() * token_samples.size());
        expected.reserve(actual.capacity());
        for (const std::int32_t row : sampled) {
            for (const std::int32_t token : token_samples) {
                actual.push_back(
                    bf16_to_f32(output_bits[static_cast<std::size_t>(token) * rows + row]));
                expected.push_back(dot_fp64(
                    weight.host, row,
                    std::span<const float>(
                        activation.data() + static_cast<std::size_t>(token) * hidden, hidden)));
            }
        }
    }
    failures += verify_reduction("BF16_A16 Linear [" + std::to_string(rows) + "," +
                                     std::to_string(hidden) + "]" + suffix,
                                 actual, expected, kA16Tolerance);
    const std::vector<std::uint16_t> activation_after =
        from_device<std::uint16_t>(device_activation, activation_bits.size());
    if (activation_after != activation_bits) {
        std::cerr << "BF16_A16 Linear" << suffix << " modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved("BF16_A16 Linear weight" + suffix);
    return failures;
}

int run_selector_linear() {
    constexpr int n = 256, k = 5120, max_t = 2048;
    DeviceWeight weight(make_patterned(n, k, 419U));
    const auto bits       = make_activation_bits(k, max_t);
    const auto activation = materialize(bits);
    std::vector<double> oracle(n * max_t);
    std::vector<std::thread> workers;
    const int threads = std::min(32U, std::max(1U, std::thread::hardware_concurrency()));
    for (int worker = 0; worker < threads; ++worker)
        workers.emplace_back([&, worker] {
            for (int index = worker; index < n * max_t; index += threads) {
                const int token = index / n, row = index % n;
                oracle[index] = dot_fp64(weight.host, row,
                                         std::span<const float>(activation.data() + token * k, k));
            }
        });
    for (auto& worker : workers) worker.join();
    DeviceBuffer input = to_device(bits);
    auto negative      = bits;
    for (auto& value : negative) value ^= 0x8000;
    DeviceContext context;
    int failures   = 0;
    const auto run = [&](int tokens, bool replay) {
        const auto capacity = ops::linear_workspace_capacity_bytes(
            QType::BF16, n, k, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena scratch(std::max<std::size_t>(capacity, 256));
        GuardedDeviceBuffer output_buffer(static_cast<std::size_t>(n) * tokens * 2);
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor output(output_buffer.data(), DType::BF16, {n, tokens});
        const auto launch = [&] {
            ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, scratch,
                        context.stream);
        };
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        cuda_synchronize();
        if (replay) {
            definition.capture(context.stream, launch);
            graph.instantiate(definition);
        }
        const std::string label =
            "BF16 selector T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
        for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
            const auto& represented = phase == 0 ? bits : negative;
            if (phase) input.copy_from_host(represented.data(), input.bytes);
            output_buffer.fill(0xff);
            cuda_synchronize();
            if (replay)
                graph.launch(context.stream);
            else
                launch();
            cuda_synchronize(context.stream);
            failures += output_buffer.verify_guards(label);
            if (scratch.peak_used() > capacity || scratch.used() != 0) {
                std::cerr << label << ": workspace query/scope mismatch\n";
                ++failures;
            }
            const auto actual_bits = from_device<std::uint16_t>(output_buffer.data(), n * tokens);
            std::vector<double> actual(actual_bits.size()),
                expected(oracle.begin(), oracle.begin() + n * tokens);
            for (std::size_t i = 0; i < actual.size(); ++i) actual[i] = bf16_to_f32(actual_bits[i]);
            if (phase)
                for (auto& value : expected) value = -value;
            failures += verify_reduction(label, actual, expected, kA16Tolerance);
            if (from_device<std::uint16_t>(input, bits.size()) != represented) {
                std::cerr << label << ": modified input\n";
                ++failures;
            }
        }
        if (replay) input.copy_from_host(bits.data(), input.bytes);
    };
    for (int tokens = 1; tokens <= 120; ++tokens) run(tokens, false);
    for (int tokens : {121, 127, 128,  129,  159,  160,  161,  162,  256,  639,  640,
                       641, 642, 1023, 1024, 1025, 1026, 1279, 1280, 1281, 1282, 2048})
        run(tokens, false);
    for (int tokens : {1,   7,   8,   15,  16,  63,  64,  65,   76,   77,   80,   81,
                       119, 120, 129, 160, 161, 640, 641, 1024, 1025, 1280, 1281, 2048})
        run(tokens, true);
    failures += weight.verify_preserved("BF16 selector weight");
    return failures;
}

int run_bf16_linear() {
    int failures = 0;
    DeviceWeight attention_weight(make_patterned(14336, 5120, 401U));
    DeviceWeight output_weight(make_patterned(5120, 6144, 409U));
    for (DeviceWeight* weight : {&attention_weight, &output_weight}) {
        for (int tokens = 1; tokens <= 33; ++tokens) {
            failures += run_bf16_linear_case(*weight, tokens);
        }
        for (int tokens :
             {63, 64, 65, 66, 95, 96, 97, 98, 127, 128, 129, 130, 191, 192, 193, 194, 1024, 1536}) {
            failures += run_bf16_linear_case(*weight, tokens);
        }
        for (int tokens : {3, 7, 13, 19, 23, 25, 29, 33, 64, 65, 97, 129, 193}) {
            failures += run_bf16_linear_case(*weight, tokens, true);
        }
    }
    failures += run_selector_linear();
    return failures;
}

enum class ColumnDomain { Text, RawPatch, Merger };

struct Geometry {
    int n, k;
    std::initializer_list<int> boundaries;
    ColumnDomain domain = ColumnDomain::Text;
};

constexpr std::array kNewGeometries{
    Geometry{48, 2560, {1, 8, 12, 16, 128}},
    Geometry{96, 2560, {1, 4, 16, 128, 512}},
    Geometry{1664, 2560, {1, 16, 48, 64, 96, 128, 512}},
    Geometry{324, 10240, {1, 8, 48, 96, 128, 512}},
    Geometry{320, 10240, {1, 8, 48, 96, 128, 512}},
    Geometry{10240, 320, {1, 16, 48, 64, 128}},
    Geometry{12800, 2560, {1, 8, 16, 32, 48, 64, 96, 128, 512}},
    Geometry{248320, 2560, {1, 8, 16, 32, 64, 96}},
    Geometry{13952, 2560, {1, 8, 16, 32, 48, 64, 96, 128, 512}},
    Geometry{2560, 6144, {1, 8, 16, 32, 64, 96, 128, 512}},
    Geometry{2560, 2560, {1, 8, 32, 48, 64, 96, 128, 512, 2048}},
    Geometry{1152, 1536, {8, 16, 32, 64, 128, 256, 512, 1024, 2048}, ColumnDomain::RawPatch},
    Geometry{3456, 1152, {4, 16, 96, 128, 256, 512, 1024}, ColumnDomain::RawPatch},
    Geometry{1152, 1152, {4, 16, 32, 40, 64, 128, 256, 512, 1024, 2048}, ColumnDomain::RawPatch},
    Geometry{4304, 1152, {32, 96, 128, 256, 512, 1024, 2048}, ColumnDomain::RawPatch},
    Geometry{1152, 4304, {16, 32, 40, 64, 116, 128, 256, 512, 1024, 2048}, ColumnDomain::RawPatch},
    Geometry{4608, 4608, {1, 2, 3, 4, 16, 32, 64, 96, 128, 256, 512}, ColumnDomain::Merger},
    Geometry{2560, 4608, {1, 2, 8, 24, 64, 96, 128, 256, 512}, ColumnDomain::Merger},
};

ninfer::test::quantized_weight::PackedWeight cancellation_weight(int n, int k, std::uint32_t seed) {
    auto result = ninfer::test::linear::make_bf16_weight(n, k, seed);
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < k; column += 2) {
            const auto offset = (static_cast<std::size_t>(row) * k + column) * 2;
            auto value =
                ninfer::test::quantized_weight::detail::load_u16_le(result.payload, offset);
            if (column == k - 2 && (value & 0x7fff) == 0) value = 0x3d80;
            ninfer::test::quantized_weight::detail::store_u16_le(result.payload, offset, value);
            ninfer::test::quantized_weight::detail::store_u16_le(result.payload, offset + 2, value);
        }
    }
    return result;
}

int run_new_bf16_geometry(const Geometry& shape) {
    using namespace ninfer::test::linear;
    const bool raw    = shape.domain == ColumnDomain::RawPatch;
    const bool text   = shape.domain == ColumnDomain::Text;
    const int step    = raw ? 4 : 1;
    const int maximum = raw ? 131072 : text ? std::numeric_limits<int>::max() : 32768;
    const bool wide   = shape.n == 2560 && shape.k == 2560;
    int failures      = 0;
    for (const auto policy :
         {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
        if (ops::linear_workspace_capacity_bytes(QType::BF16, shape.n, shape.k, policy, step,
                                                 maximum) != 0) {
            std::cerr << "BF16: expected zero workspace across the admitted column domain\n";
            ++failures;
        }
        std::vector<std::pair<int, int>> invalid{{0, step}, {-1, step}, {8, 4}};
        if (!text) invalid.push_back({step, maximum + step});
        if (raw) invalid.insert(invalid.end(), {{1, 4}, {4, 5}, {3, 8}});
        for (auto [lo, hi] : invalid) {
            try {
                (void)ops::linear_workspace_capacity_bytes(QType::BF16, shape.n, shape.k, policy,
                                                           lo, hi);
                std::cerr << "BF16: accepted an invalid column domain\n";
                ++failures;
            } catch (const std::invalid_argument&) {}
        }
        if (!text) {
            for (auto [lo, hi] :
                 {std::pair{step, 128}, std::pair{512, 1024}, std::pair{maximum - step, maximum}}) {
                const auto capacity = ops::linear_workspace_capacity_bytes(QType::BF16, shape.n,
                                                                           shape.k, policy, lo, hi);
                for (int t = lo; t <= hi; t += step) {
                    if (ops::linear_workspace_capacity_bytes(QType::BF16, shape.n, shape.k, policy,
                                                             t, t) > capacity) {
                        std::cerr << "BF16: insufficient vision workspace envelope\n";
                        ++failures;
                    }
                }
            }
        }
    }
    if (text) failures += verify_workspace_envelopes(QType::BF16, shape.n, shape.k);
    if (raw) {
        DeviceWeight weight(make_patterned(shape.n, shape.k, 503U));
        DeviceBuffer input(static_cast<std::size_t>(shape.k) * 8 * 2);
        DeviceBuffer output(static_cast<std::size_t>(shape.n) * 8 * 2);
        DeviceArena scratch(256);
        for (int t : {1, 3, 5, 7}) {
            Tensor x(input.p, DType::BF16, {shape.k, t});
            Tensor y(output.p, DType::BF16, {shape.n, t});
            for (bool convenience : {false, true}) {
                try {
                    if (convenience)
                        ops::linear(x, weight.view(), y, nullptr);
                    else
                        ops::linear(x, weight.view(), y, ops::LinearPolicy::A16Only, scratch,
                                    nullptr);
                    std::cerr << "BF16: accepted an invalid raw-patch extent\n";
                    ++failures;
                } catch (const std::invalid_argument&) {}
            }
        }
    }
    std::set<int> points;
    if (shape.n <= 640 || raw) {
        for (int t = step; t <= 128; t += step) points.insert(t);
    } else {
        for (int t : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128})
            points.insert(t);
    }
    points.insert({raw ? 132 : 129, 256, 512, 1024});
    if (!raw) points.insert(1025);
    if (raw || wide) points.insert({2048, 4096});
    if (!text) points.insert({maximum - step, maximum});
    for (int end : shape.boundaries)
        for (int t : {end - step, end, end + step})
            if (t >= step && t <= maximum && t % step == 0) points.insert(t);
    std::vector<Invocation> calls;
    for (int t : points) calls.push_back({t});
    std::set<int> anchors{step, 4, 8, 128, 512, 1024};
    if (!text) anchors.insert(256);
    if (raw || wide) anchors.insert({2048, 4096});
    for (int t : anchors) {
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
        calls.push_back({t, CallForm::A16Convenience});
    }
    for (int end : shape.boundaries) {
        for (int t : {end, end + step})
            if (t >= step && t <= maximum && t % step == 0)
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
        if (end >= step && end % step == 0) calls.push_back({end, CallForm::A16Convenience});
    }
    calls.push_back({raw ? 20 : 17, CallForm::Policy, ops::LinearPolicy::AllowA8});
    calls.push_back({64, CallForm::Policy, ops::LinearPolicy::AllowA4});
    const std::uint32_t seed = static_cast<std::uint32_t>(shape.n + shape.k + 431);
    failures += run_shape("BF16_A16", ActivationCompute::A16, make_bf16_weight,
                          {shape.n, shape.k, seed,
                           shape.n <= 640 ? Comparison::Full : Comparison::Sampled, true, calls});
    if (shape.n > 640) {
        std::set<int> full_points{step, 4, 8};
        std::vector<Invocation> full;
        for (int t : full_points) full.push_back({t});
        failures += run_shape("BF16_A16 full", ActivationCompute::A16, make_bf16_weight,
                              {shape.n, shape.k, seed + 1, Comparison::Full, true, full});
    }
    if (shape.n <= 640 || shape.k == 320 || shape.k % 64 != 0) {
        std::set<int> tail_points{step, 4, 8, 128};
        std::vector<Invocation> tail;
        for (int t : tail_points) tail.push_back({t});
        tail.push_back({128, CallForm::Policy, ops::LinearPolicy::A16Only, true});
        failures += run_shape("BF16_A16 K tail", ActivationCompute::A16, make_bf16_weight,
                              {shape.n, shape.k, seed + 2,
                               shape.n <= 640 ? Comparison::Full : Comparison::Sampled, true, tail,
                               ActivationPattern::KTail});
    }
    if (shape.n <= 96) {
        const std::array cancel{Invocation{1}, Invocation{4}, Invocation{8}, Invocation{128},
                                Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only, true}};
        failures += run_shape("BF16_A16 cancellation", ActivationCompute::A16, cancellation_weight,
                              {shape.n, shape.k, seed + 3, Comparison::Full, true, cancel,
                               ActivationPattern::Cancellation});
    }
    return failures;
}

int run_new_bf16(int selected_n = 0, int selected_k = 0) {
    int failures = 0;
    bool found   = selected_n == 0;
    for (const auto& shape : kNewGeometries) {
        if (selected_n && (shape.n != selected_n || shape.k != selected_k)) continue;
        found = true;
        failures += run_new_bf16_geometry(shape);
    }
    if (!found) throw std::invalid_argument("BF16 test: unknown geometry");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        int n = 0, k = 0;
        if (argc == 4 && std::string_view(argv[1]) == "--shape") {
            n = std::stoi(argv[2]);
            k = std::stoi(argv[3]);
            if (n <= 0 || k <= 0) throw std::invalid_argument("BF16 test: positive N/K required");
        } else if (argc != 1) {
            throw std::invalid_argument("usage: ninfer_linear_bf16_a16_test [--shape N K]");
        }
        const int failures = (n == 0 ? run_bf16_linear() : 0) + run_new_bf16(n, k);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
