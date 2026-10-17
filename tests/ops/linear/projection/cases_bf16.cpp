#include "core/weight.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ninfer/ops/linear.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/linear/projection/runner.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <set>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace ninfer::test::linear_weights;

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;
using ninfer::test::linear_suite::Suite;

std::vector<std::uint16_t> make_bf16_activation(std::int32_t hidden, std::int32_t tokens,
                                                std::uint32_t) {
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

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    auto result = linear_common::sampled_indices(rows);
    if (rows == 14336) {
        result.insert(result.end(), {1023, 6143, 6144, 7167, 7168, 13311, 13312});
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) return linear_common::all_indices(tokens);
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

ninfer::test::linear::ShapeCase
bf16_fixture(std::int32_t n, std::int32_t k, std::uint32_t seed,
             ninfer::test::linear::Comparison comparison,
             std::span<const ninfer::test::linear::Invocation> calls) {
    ninfer::test::linear::ShapeCase shape{n, k, seed, comparison, true, calls};
    shape.activation_generator   = make_bf16_activation;
    shape.row_sampler            = sampled_rows;
    shape.column_sampler         = sampled_tokens;
    shape.verify_each_invocation = true;
    return shape;
}

void run_bf16_projection(Suite& suite, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    using namespace ninfer::test::linear;
    namespace linear_suite = ninfer::test::linear_suite;
    // These geometries keep their original activation recipe, weight seeds, and coordinates:
    // T=1 checks every row; larger T checks the established row/token samples.
    const std::array full{Invocation{1}};
    add_shape(suite, ninfer::QType::BF16, ActivationRecipe::TokenBlocks, make_bf16_weight,
              bf16_fixture(n, k, seed, Comparison::Full, full));
    std::vector<Invocation> calls;
    for (int tokens = 2; tokens <= 33; ++tokens) calls.push_back({tokens});
    for (int tokens :
         {63, 64, 65, 66, 95, 96, 97, 98, 127, 128, 129, 130, 191, 192, 193, 194, 1024, 1536}) {
        calls.push_back({tokens});
    }
    for (int tokens : {3, 7, 13, 19, 23, 25, 29, 33, 64, 65, 97, 129, 193}) {
        // The original graph cases also made an eager call before capture.
        calls.push_back({tokens});
        calls.push_back({tokens, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    }
    add_shape(suite, ninfer::QType::BF16, ActivationRecipe::TokenBlocks, make_bf16_weight,
              bf16_fixture(n, k, seed, Comparison::Sampled, calls));
}

void run_selector_linear(Suite& suite) {
    using namespace ninfer::test::linear;
    namespace linear_suite = ninfer::test::linear_suite;
    std::vector<Invocation> calls;
    for (int tokens = 1; tokens <= 120; ++tokens) calls.push_back({tokens});
    for (int tokens : {121, 127, 128,  129,  159,  160,  161,  162,  256,  639,  640,
                       641, 642, 1023, 1024, 1025, 1026, 1279, 1280, 1281, 1282, 2048}) {
        calls.push_back({tokens});
    }
    for (int tokens : {1,   7,   8,   15,  16,  63,  64,  65,   76,   77,   80,   81,
                       119, 120, 129, 160, 161, 640, 641, 1024, 1025, 1280, 1281, 2048}) {
        calls.push_back({tokens, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    }
    add_shape(suite, ninfer::QType::BF16, ActivationRecipe::TokenBlocks, make_bf16_weight,
              bf16_fixture(256, 5120, 419U, Comparison::Full, calls));
}

enum class ColumnDomain { Text, RawPatch, Merger };

struct Geometry {
    int n, k;
    std::initializer_list<int> boundaries;
    ColumnDomain domain = ColumnDomain::Text;
};

const std::array kNewGeometries{
    Geometry{48, 2560, {1, 8, 12, 16, 128}},
    Geometry{96, 2560, {1, 4, 16, 128, 512}},
    Geometry{1664, 2560, {1,   16,  32,  48,   64,   96,   128,  160,  256,  336,  448,  512,
                          560, 640, 896, 1024, 1152, 1280, 1408, 1440, 1664, 1856, 2048, 4096}},
    Geometry{324, 10240, {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 512, 1024, 1536, 2048, 4096}},
    Geometry{320, 10240, {1, 4, 8, 16, 32, 48, 64, 96, 128, 512, 576, 768, 1216, 1536, 2048, 4096}},
    Geometry{10240, 320, {1, 16, 48, 64, 128, 448, 512, 576, 640, 896, 1024, 4096}},
    Geometry{12800, 2560, {1, 8, 16, 32, 48, 64, 96, 128, 384, 512, 3840, 4032, 4096}},
    Geometry{248320, 2560, {1, 8, 16, 32, 64, 96}},
    Geometry{13952, 2560, {1, 8, 16, 32, 48, 64, 96, 128, 256, 384, 512, 3840, 4096}},
    Geometry{2560, 6144, {1,    8,    16,   32,   64,   96,   128,  256,  512,
                          576,  896,  1152, 1280, 1344, 1920, 2048, 2112, 2304,
                          2688, 2880, 3328, 3456, 3968, 4032, 4096}},
    Geometry{2560, 2560, {1, 8, 32, 48, 64, 96, 128, 256, 512, 576, 896, 1152, 2048, 4096}},
    Geometry{1152,
             1536,
             {4, 8, 16, 32, 64, 128, 192, 256, 320, 512, 640, 768, 1024, 2048, 4096},
             ColumnDomain::RawPatch},
    Geometry{
        3456, 1152, {4, 16, 64, 96, 128, 256, 384, 512, 640, 1024, 4096}, ColumnDomain::RawPatch},
    Geometry{1152,
             1152,
             {4, 16, 32, 40, 64, 128, 192, 256, 320, 512, 640, 1024, 2048, 4096},
             ColumnDomain::RawPatch},
    Geometry{4304,
             1152,
             {32, 64, 96, 128, 256, 320, 384, 512, 1024, 2048, 4096},
             ColumnDomain::RawPatch},
    Geometry{1152, 4304, {16, 32, 40, 64, 116, 128, 256, 512, 1024, 2048}, ColumnDomain::RawPatch},
    Geometry{4608, 4608, {1, 2, 3, 4, 16, 32, 64, 96, 128, 256, 512}, ColumnDomain::Merger},
    Geometry{2560, 4608, {1, 2, 8, 24, 64, 96, 128, 256, 512}, ColumnDomain::Merger},
};

ninfer::test::quantized_weight::PackedWeight cancellation_weight(int n, int k, std::uint32_t seed) {
    auto result = ninfer::test::linear_weights::make_bf16_weight(n, k, seed);
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

void run_new_bf16_geometry(Suite& suite, const Geometry& shape) {
    using namespace ninfer::test::linear;
    namespace linear_suite = ninfer::test::linear_suite;
    const bool raw         = shape.domain == ColumnDomain::RawPatch;
    const bool text        = shape.domain == ColumnDomain::Text;
    const int step         = raw ? 4 : 1;
    const int maximum      = raw ? 131072 : text ? std::numeric_limits<int>::max() : 32768;
    const bool wide        = shape.n == 2560 && shape.k == 2560;
    auto test     = linear_suite::make_case("bf16", shape.n, shape.k, 0, ops::LinearPolicy::A16Only,
                                            linear_suite::Execution::Eager, "column-domain");
    test.policies = 7;
    suite.add(std::move(test), [shape, step, maximum, raw, text] {
        int failures = 0;
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
                    (void)ops::linear_workspace_capacity_bytes(QType::BF16, shape.n, shape.k,
                                                               policy, lo, hi);
                    std::cerr << "BF16: accepted an invalid column domain\n";
                    ++failures;
                } catch (const std::invalid_argument&) {}
            }
            if (!text) {
                for (auto [lo, hi] : {std::pair{step, 128}, std::pair{512, 1024},
                                      std::pair{maximum - step, maximum}}) {
                    const auto capacity = ops::linear_workspace_capacity_bytes(
                        QType::BF16, shape.n, shape.k, policy, lo, hi);
                    for (int t = lo; t <= hi; t += step) {
                        if (ops::linear_workspace_capacity_bytes(QType::BF16, shape.n, shape.k,
                                                                 policy, t, t) > capacity) {
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
            DeviceBuffer input(ninfer::test::cuda_environment().context.device(),
                               static_cast<std::size_t>(shape.k) * 8 * 2);
            DeviceBuffer output(ninfer::test::cuda_environment().context.device(),
                                static_cast<std::size_t>(shape.n) * 8 * 2);
            ninfer::DeviceBuffer scratch_storage(ninfer::test::cuda_environment().context.device(),
                                                 256);
            MemoryArena scratch(scratch_storage.span());
            for (int t : {1, 3, 5, 7}) {
                Tensor x(input.span(), DType::BF16, {shape.k, t});
                Tensor y(output.span(), DType::BF16, {shape.n, t});
                for (bool convenience : {false, true}) {
                    try {
                        if (convenience)
                            ops::linear(x, weight.view(), y,
                                        ninfer::test::cuda_environment().default_execution);
                        else
                            ops::linear(x, weight.view(), y, ops::LinearPolicy::A16Only, scratch,
                                        ninfer::test::cuda_environment().default_execution);
                        std::cerr << "BF16: accepted an invalid raw-patch extent\n";
                        ++failures;
                    } catch (const std::invalid_argument&) {}
                }
            }
        }
        return failures;
    });
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
    add_shape(suite, ninfer::QType::BF16, ActivationRecipe::TokenBlocks, make_bf16_weight,
              {shape.n, shape.k, seed, shape.n <= 640 ? Comparison::Full : Comparison::Sampled,
               true, calls});
    if (shape.n > 640) {
        std::set<int> full_points{step, 4, 8};
        std::vector<Invocation> full;
        for (int t : full_points) full.push_back({t});
        add_shape(suite, ninfer::QType::BF16, ActivationRecipe::TokenBlocks, make_bf16_weight,
                  {shape.n, shape.k, seed + 1, Comparison::Full, true, full});
    }
    if (shape.n <= 640 || shape.k == 320 || shape.k % 64 != 0) {
        std::set<int> tail_points{step, 4, 8, 128};
        std::vector<Invocation> tail;
        for (int t : tail_points) tail.push_back({t});
        tail.push_back({128, CallForm::Policy, ops::LinearPolicy::A16Only, true});
        add_shape(suite, ninfer::QType::BF16, ActivationRecipe::TokenBlocks, make_bf16_weight,
                  {shape.n, shape.k, seed + 2,
                   shape.n <= 640 ? Comparison::Full : Comparison::Sampled, true, tail,
                   ActivationPattern::KTail});
    }
    if (shape.n <= 96) {
        const std::array cancel{Invocation{1}, Invocation{4}, Invocation{8}, Invocation{128},
                                Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only, true}};
        add_shape(suite, ninfer::QType::BF16, ActivationRecipe::TokenBlocks, cancellation_weight,
                  {shape.n, shape.k, seed + 3, Comparison::Full, true, cancel,
                   ActivationPattern::Cancellation});
    }
}

void run_bf16(Suite& suite) {

    struct ProjectionFixture {
        std::int32_t n, k;
        std::uint32_t seed;
    };

    for (const auto shape :
         {ProjectionFixture{14336, 5120, 401U}, ProjectionFixture{5120, 6144, 409U}}) {
        run_bf16_projection(suite, shape.n, shape.k, shape.seed);
    }
    { run_selector_linear(suite); }
    for (const auto& shape : kNewGeometries) { run_new_bf16_geometry(suite, shape); }
}

} // namespace

void ninfer::test::linear::add_bf16_cases(Suite& suite) { run_bf16(suite); }
