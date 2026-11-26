#include "ops/suite.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim         = 128;
constexpr double kRelativeL2   = 1.85e-3;
constexpr double kPairRelative = 6.9e-3;

struct Profile {
    int query_heads, key_heads;
    float epsilon, theta;
};

constexpr Profile kQwen{32, 8, 1.e-6F, 1.e7F};
constexpr Profile kGlm{16, 4, 1.e-5F, 10000.F};
constexpr Profile kInkling{24, 8, 1.e-6F, 1.e6F};

struct PairStorage {
    std::size_t activation_offset = 0, weight_offset = 0;
    bool shared_weights = false;
};

struct OracleResult {
    std::vector<double> output;
    std::vector<double> pair_scale;
};

std::size_t dense_index(int heads, int token, int head, int dim) {
    return (static_cast<std::size_t>(token) * heads + head) * kHeadDim + dim;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return values;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](float value) { return f32_to_bf16(value); });
    return bits;
}

std::vector<std::int32_t> make_positions(int tokens, int first_position) {
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = first_position + token;
    }
    return positions;
}

// Independent FP64 oracle for the complete public formula. It does not call either standalone Op
// or reproduce the production warp reduction, coefficient table, or range reduction.
OracleResult fused_oracle(const std::vector<float>& input, const std::vector<float>& weight,
                          const std::vector<std::int32_t>& positions, int heads, Profile profile) {
    const int tokens = static_cast<int>(positions.size());
    OracleResult result{
        .output     = std::vector<double>(input.size()),
        .pair_scale = std::vector<double>(input.size()),
    };
    std::vector<double> normalized(kHeadDim);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            double sum_squares = 0.0;
            for (int dim = 0; dim < kHeadDim; ++dim) {
                const double value = input[dense_index(heads, token, head, dim)];
                sum_squares += value * value;
            }
            const double inverse =
                1.0 / std::sqrt(sum_squares / static_cast<double>(kHeadDim) + profile.epsilon);
            for (int dim = 0; dim < kHeadDim; ++dim) {
                normalized[static_cast<std::size_t>(dim)] =
                    static_cast<double>(input[dense_index(heads, token, head, dim)]) * inverse *
                    static_cast<double>(weight[static_cast<std::size_t>(dim)]);
            }
            for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                const double exponent = -2.0 * static_cast<double>(pair) / kHeadDim;
                const double phase =
                    static_cast<double>(positions[static_cast<std::size_t>(token)]) *
                    std::pow(double(profile.theta), exponent);
                const double cosine = std::cos(phase);
                const double sine   = std::sin(phase);
                const double first  = normalized[static_cast<std::size_t>(pair)];
                const double second = normalized[static_cast<std::size_t>(pair + kHeadDim / 2)];
                const double scale  = std::hypot(first, second);
                const std::size_t first_index = dense_index(heads, token, head, pair);
                const std::size_t second_index =
                    dense_index(heads, token, head, pair + kHeadDim / 2);
                result.output[first_index]      = first * cosine - second * sine;
                result.output[second_index]     = second * cosine + first * sine;
                result.pair_scale[first_index]  = scale;
                result.pair_scale[second_index] = scale;
            }
        }
    }
    return result;
}

int verify_profile(const std::string& label, const std::vector<double>& got,
                   const OracleResult& expected) {
    if (got.size() != expected.output.size() || got.size() != expected.pair_scale.size()) {
        std::cerr << label << ": result size mismatch\n";
        return 1;
    }
    double error_square_sum     = 0.0;
    double reference_square_sum = 0.0;
    double maximum_ratio        = 0.0;
    int violations              = 0;
    for (std::size_t index = 0; index < got.size(); ++index) {
        if (!std::isfinite(got[index]) || !std::isfinite(expected.output[index])) {
            std::cerr << label << ": non-finite value at index " << index << '\n';
            return 1;
        }
        const double error = std::abs(got[index] - expected.output[index]);
        const double scale = expected.pair_scale[index];
        const double limit = kPairRelative * scale;
        const double ratio = limit == 0.0
                                 ? (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                                 : error / limit;
        maximum_ratio      = std::max(maximum_ratio, ratio);
        if (error > limit) {
            ++violations;
            if (violations == 1) {
                std::cerr << label << ": pair-scaled mismatch at index " << index
                          << " error=" << error << " scale=" << scale << '\n';
            }
        }
        error_square_sum += error * error;
        reference_square_sum += expected.output[index] * expected.output[index];
    }
    const double relative_l2 = std::sqrt(error_square_sum / reference_square_sum);
    if (relative_l2 > kRelativeL2) {
        std::cerr << label << ": relative L2=" << relative_l2 << " exceeds " << kRelativeL2 << '\n';
        ++violations;
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=rmsnorm_rope count=" << got.size()
                  << " relative_l2=" << relative_l2 << " max_pair_limit_ratio=" << maximum_ratio
                  << " case=" << label << '\n';
    }
    if (violations != 0) {
        std::cerr << label << ": " << violations << " numerical violations\n";
        return 1;
    }
    return 0;
}

template <class Launch, class Reset, class Check>
int execute(Launch launch, Reset reset, Check check, bool graph) {
    if (!graph) {
        reset(cuda_environment().default_execution, 0);
        launch(cuda_environment().default_execution);
        cuda_synchronize();
        return check();
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    int failures                = 0;
    {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(stream, [&] { launch(stream_execution); });
        executable.instantiate(definition);
        for (int i = 0; i < 2; ++i) {
            reset(stream_execution, i);
            executable.launch(stream);
            cuda_check(cudaStreamSynchronize(stream), "graph replay synchronize");
            failures += check();
        }
    }
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    return failures;
}

int run_pair_case(int width, int batch, int first_position, std::uint32_t seed, bool graph = false,
                  Profile profile = kQwen, float magnitude = 4.F, PairStorage storage = {}) {
    const int tokens          = width * batch;
    const std::size_t q_count = static_cast<std::size_t>(kHeadDim) * profile.query_heads * tokens;
    const std::size_t k_count = static_cast<std::size_t>(kHeadDim) * profile.key_heads * tokens;
    auto q                    = make_bf16_values(q_count, seed, -magnitude, magnitude);
    auto k                    = make_bf16_values(k_count, seed + 1U, -magnitude, magnitude);
    const auto q_weight       = make_bf16_values(kHeadDim, seed + 2U, 0.25F, 1.75F);
    const auto k_weight       = make_bf16_values(kHeadDim, seed + 3U, 0.25F, 1.75F);
    auto positions            = make_positions(tokens, first_position);
    auto q_bits               = bf16_bits(q);
    auto k_bits               = bf16_bits(k);
    const auto q_weight_bits  = bf16_bits(q_weight);
    const auto k_weight_bits  = bf16_bits(k_weight);

    GuardedDeviceBuffer q_device(q_count * sizeof(std::uint16_t), 256 + storage.activation_offset);
    GuardedDeviceBuffer k_device(k_count * sizeof(std::uint16_t), 256 + storage.activation_offset);
    GuardedDeviceBuffer q_weight_device(kHeadDim * sizeof(std::uint16_t),
                                        256 + storage.weight_offset);
    GuardedDeviceBuffer k_weight_device(kHeadDim * sizeof(std::uint16_t),
                                        256 + storage.weight_offset);
    q_weight_device.copy_from_host(q_weight_bits.data(), q_weight_device.bytes());
    k_weight_device.copy_from_host(k_weight_bits.data(), k_weight_device.bytes());
    const auto& actual_k_weight  = storage.shared_weights ? q_weight : k_weight;
    auto& actual_k_weight_device = storage.shared_weights ? q_weight_device : k_weight_device;
    DeviceBuffer position_device = to_device(positions);
    Tensor q_tensor(q_device.span(), DType::BF16, {kHeadDim, profile.query_heads, width, batch});
    Tensor k_tensor(k_device.span(), DType::BF16, {kHeadDim, profile.key_heads, width, batch});
    Tensor q_weight_tensor(q_weight_device.span(), DType::BF16, {kHeadDim});
    Tensor k_weight_tensor(actual_k_weight_device.span(), DType::BF16, {kHeadDim});
    Tensor position_tensor(position_device.span(), DType::I32, {width, batch});
    const std::string label = "rmsnorm_rope pair Hq=" + std::to_string(profile.query_heads) +
                              " W=" + std::to_string(width) + " B=" + std::to_string(batch) +
                              " graph=" + std::to_string(graph) +
                              " theta=" + std::to_string(profile.theta) +
                              " activation_offset=" + std::to_string(storage.activation_offset) +
                              " weight_offset=" + std::to_string(storage.weight_offset);
    int failures = execute(
        [&](CudaExecutionView cuda) {
            ops::rmsnorm_rope(position_tensor, q_weight_tensor, k_weight_tensor, profile.epsilon,
                              profile.theta, q_tensor, k_tensor, cuda);
        },
        [&](CudaExecutionView cuda, int replay) {
            const auto stream = cuda.stream();
            for (int b = 0; b < batch; ++b)
                for (int i = 0; i < width; ++i)
                    positions[b * width + i] = (b % 2 ? 17 : first_position) + i + replay * 123;
            if (profile.query_heads == kInkling.query_heads) {
                if (tokens > 2) positions[1] = 0;
                positions.back() = -first_position - replay * 123;
            }
            if (replay != 0) {
                std::rotate(q.begin(), q.begin() + 17, q.end());
                std::rotate(k.begin(), k.begin() + 31, k.end());
                q_bits = bf16_bits(q);
                k_bits = bf16_bits(k);
            }
            cuda_check(cudaMemcpyAsync(position_device.data(), positions.data(),
                                       positions.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream),
                       "update positions");
            cuda_check(cudaMemcpyAsync(q_device.data(), q_bits.data(), q_device.bytes(),
                                       cudaMemcpyHostToDevice, stream),
                       "reset q");
            cuda_check(cudaMemcpyAsync(k_device.data(), k_bits.data(), k_device.bytes(),
                                       cudaMemcpyHostToDevice, stream),
                       "reset k");
        },
        [&] {
            return verify_profile(
                       label + " q", from_device_bf16(q_device.data(), q_count),
                       fused_oracle(q, q_weight, positions, profile.query_heads, profile)) +
                   verify_profile(
                       label + " k", from_device_bf16(k_device.data(), k_count),
                       fused_oracle(k, actual_k_weight, positions, profile.key_heads, profile));
        },
        graph);

    failures += q_device.verify_guards(label + " q guards");
    failures += k_device.verify_guards(label + " k guards");
    failures += q_weight_device.verify_guards(label + " q weight guards");
    failures += k_weight_device.verify_guards(label + " k weight guards");
    failures +=
        verify_exact((label + " positions").c_str(),
                     from_device<std::int32_t>(position_device, positions.size()), positions);
    failures += verify_exact(
        (label + " q weight").c_str(),
        from_device<std::uint16_t>(q_weight_device.data(), q_weight_bits.size()), q_weight_bits);
    failures += verify_exact(
        (label + " k weight").c_str(),
        from_device<std::uint16_t>(k_weight_device.data(), k_weight_bits.size()), k_weight_bits);
    return failures;
}

int run_single_case(int tokens, int first_position, std::uint32_t seed, bool graph = false,
                    Profile profile = kQwen, float magnitude = 4.F) {
    const std::size_t count = static_cast<std::size_t>(kHeadDim) * profile.key_heads * tokens;
    const auto input        = make_bf16_values(count, seed, -magnitude, magnitude);
    const auto weight       = make_bf16_values(kHeadDim, seed + 1U, 0.25F, 1.75F);
    auto positions          = make_positions(tokens, first_position);
    const auto input_bits   = bf16_bits(input);
    const auto weight_bits  = bf16_bits(weight);

    GuardedDeviceBuffer input_device(count * sizeof(std::uint16_t));
    input_device.copy_from_host(input_bits.data(), input_device.bytes());
    DeviceBuffer weight_device   = to_device(weight_bits);
    DeviceBuffer position_device = to_device(positions);
    Tensor input_tensor(input_device.span(), DType::BF16, {kHeadDim, profile.key_heads, tokens});
    Tensor weight_tensor(weight_device.span(), DType::BF16, {kHeadDim});
    Tensor position_tensor(position_device.span(), DType::I32, {tokens});
    const std::string label = "rmsnorm_rope single Hkv=" + std::to_string(profile.key_heads) +
                              " T=" + std::to_string(tokens) + " graph=" + std::to_string(graph);
    int failures = execute(
        [&](CudaExecutionView cuda) {
            ops::rmsnorm_rope(position_tensor, weight_tensor, profile.epsilon, profile.theta,
                              input_tensor, cuda);
        },
        [&](CudaExecutionView cuda, int replay) {
            const auto stream = cuda.stream();
            positions         = make_positions(tokens, first_position + replay * 123);
            cuda_check(cudaMemcpyAsync(position_device.data(), positions.data(),
                                       positions.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream),
                       "update positions");
            cuda_check(cudaMemcpyAsync(input_device.data(), input_bits.data(), input_device.bytes(),
                                       cudaMemcpyHostToDevice, stream),
                       "reset K");
        },
        [&] {
            return verify_profile(
                label, from_device_bf16(input_device.data(), count),
                fused_oracle(input, weight, positions, profile.key_heads, profile));
        },
        graph);

    failures += input_device.verify_guards(label + " guards");
    failures +=
        verify_exact((label + " positions").c_str(),
                     from_device<std::int32_t>(position_device, positions.size()), positions);
    failures +=
        verify_exact((label + " weight").c_str(),
                     from_device<std::uint16_t>(weight_device, weight_bits.size()), weight_bits);
    return failures;
}

int run_large_gain_case() {
    constexpr int width = 2;
    std::vector<float> q(kHeadDim * kInkling.query_heads * width, 1.F);
    const std::vector<float> k(kHeadDim * kInkling.key_heads * width, 1.F);
    std::vector<float> weight(kHeadDim, 1.F);
    const std::vector<std::int32_t> positions(width, 1);
    for (std::size_t i = 0; i < q.size(); i += kHeadDim) q[i] = 1.25F;
    // The first normalized value exceeds FP32 range. Both rotated values fit in BF16.
    weight.front() = bf16_to_f32(f32_to_bf16(3.e38F));
    auto q_device  = to_device(bf16_bits(q));
    auto k_device  = to_device(bf16_bits(k));
    auto w_device  = to_device(bf16_bits(weight));
    auto p_device  = to_device(positions);
    Tensor query(q_device.span(), DType::BF16, {kHeadDim, kInkling.query_heads, width});
    Tensor key(k_device.span(), DType::BF16, {kHeadDim, kInkling.key_heads, width});
    Tensor w(w_device.span(), DType::BF16, {kHeadDim});
    Tensor p(p_device.span(), DType::I32, {width});
    ops::rmsnorm_rope(p, w, w, kInkling.epsilon, kInkling.theta, query, key,
                      cuda_environment().default_execution);
    cuda_synchronize();
    return verify_profile("rmsnorm_rope large gain q", from_device_bf16(q_device.data(), q.size()),
                          fused_oracle(q, weight, positions, kInkling.query_heads, kInkling)) +
           verify_profile("rmsnorm_rope large gain k", from_device_bf16(k_device.data(), k.size()),
                          fused_oracle(k, weight, positions, kInkling.key_heads, kInkling));
}

int contract() {
    const auto e = cuda_environment().default_execution;
    DeviceBuffer storage(e.device(), 16384);
    Tensor q(storage.span().subspan(0, 8192), DType::BF16, {128, 16, 2}),
        k(storage.span().subspan(8192, 2048), DType::BF16, {128, 4, 2}),
        weight(storage.span().subspan(10240, 256), DType::BF16, {128}),
        positions(storage.span().subspan(10496, 8), DType::I32, {2});
    int failures = 0;
    auto reject  = [&](auto launch, const char* label = "legacy contract") {
        try {
            launch();
            std::cerr << "rmsnorm_rope accepted invalid input: " << label << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    for (float invalid : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        reject([&] { ops::rmsnorm_rope(positions, weight, weight, invalid, 10000.F, q, k, e); });
        reject([&] { ops::rmsnorm_rope(positions, weight, weight, 1.e-5F, invalid, q, k, e); });
        reject([&] { ops::rmsnorm_rope(positions, weight, invalid, 10000.F, k, e); });
        reject([&] { ops::rmsnorm_rope(positions, weight, 1.e-5F, invalid, k, e); });
    }
    k.ne[1] = 8;
    reject([&] { ops::rmsnorm_rope(positions, weight, weight, 1.e-5F, 10000.F, q, k, e); });
    k.ne[1] = 3;
    reject([&] { ops::rmsnorm_rope(positions, weight, 1.e-5F, 10000.F, k, e); });
    k.ne[1] = 4;
    k.data  = q.data;
    reject([&] { ops::rmsnorm_rope(positions, weight, weight, 1.e-5F, 10000.F, q, k, e); });

    DeviceBuffer q_storage(e.device(), 128 * 32 * 17 * sizeof(std::uint16_t));
    DeviceBuffer k_storage(e.device(), 128 * 16 * 17 * sizeof(std::uint16_t));
    DeviceBuffer p_storage(e.device(), 17 * sizeof(std::int32_t));
    const auto reject_pair = [&](int query_heads, int key_heads, int width) {
        Tensor query(q_storage.span(), DType::BF16, {128, query_heads, width});
        Tensor key(k_storage.span(), DType::BF16, {128, key_heads, width});
        Tensor p(p_storage.span(), DType::I32, {width});
        reject(
            [&] {
                ops::rmsnorm_rope(p, weight, weight, kInkling.epsilon, kInkling.theta, query, key,
                                  e);
            },
            "pair geometry");
    };
    reject_pair(24, 4, 2);
    reject_pair(32, 4, 2);
    reject_pair(24, 16, 2);
    reject_pair(24, 8, 1);
    reject_pair(24, 8, 17);

    Tensor query(q_storage.span(), DType::BF16, {128, 24, 2});
    Tensor key(k_storage.span(), DType::BF16, {128, 8, 2});
    Tensor aliased_key(q_storage.span().subspan(4, key.bytes()), DType::BF16, {128, 8, 2});
    Tensor aliased_weight(q_storage.span().subspan(query.bytes() - 4, weight.bytes()), DType::BF16,
                          {128});
    Tensor aliased_positions(k_storage.span().subspan(key.bytes() - 4, positions.bytes()),
                             DType::I32, {2});
    const auto inkling = [&](const Tensor& p, const Tensor& kw, Tensor& key) {
        ops::rmsnorm_rope(p, weight, kw, kInkling.epsilon, kInkling.theta, query, key, e);
    };
    reject([&] { inkling(positions, weight, aliased_key); }, "q/k overlap");
    reject([&] { inkling(positions, aliased_weight, key); }, "q/key weight overlap");
    reject([&] { inkling(aliased_positions, weight, key); }, "k/positions overlap");
    query.ne[3] = std::numeric_limits<int>::max() / 2 + 1;
    reject([&] { inkling(positions, weight, key); }, "pair token extent overflow");
    return failures;
}

} // namespace

void add_rmsnorm_rope_cases(ninfer::test::OpSuite& suite) {
    const auto pair = [&](int width, int batch, int position, unsigned seed, bool graph = false) {
        std::vector<std::pair<std::string, std::string>> tags{
            {"execution", graph ? "graph" : "eager"}};
        if (batch > 8) tags.emplace_back("suite", "concurrency");
        suite.add(
            "rmsnorm_rope/pair/w" + std::to_string(width) + "/b" + std::to_string(batch) +
                (graph ? "/graph" : "/eager"),
            [=] { return run_pair_case(width, batch, position, seed, graph); }, std::move(tags));
    };
    const auto single = [&](int tokens, int position, unsigned seed, bool graph = false) {
        std::vector<std::pair<std::string, std::string>> tags{
            {"execution", graph ? "graph" : "eager"}};
        if (tokens > 2048) tags.emplace_back("suite", "concurrency");
        suite.add(
            "rmsnorm_rope/single/t" + std::to_string(tokens) + (graph ? "/graph" : "/eager"),
            [=] { return run_single_case(tokens, position, seed, graph); }, std::move(tags));
    };
    for (int width = 2; width <= 16; ++width)
        for (int batch = 1; batch <= 8; ++batch)
            pair(width, batch, width == 2 && batch == 1 ? 0 : 262000, 0x1000U + width * 8 + batch);
    pair(3, 3, 0, 0x1001U, true);
    pair(16, 8, 262000, 0x1002U, true);
    pair(16, 17, 262000, 0x1003U, true);
    pair(2, 64, 262000, 0x1004U, true);
    single(1, 0, 0x2001U, true);
    single(2048, 260000, 0x2002U, true);
    single(2049, 260000, 0x2003U, true);
    single(1, 0, 0x2001U);
    single(8, 131'069, 0x2002U);
    single(64, 262'080, 0x2003U);
    single(1024, 130'048, 0x2004U);
    single(2048, 260'032, 0x2005U);
    for (const auto [width, batch] : {std::pair{2, 1}, {4, 2}, {8, 3}, {16, 8}}) {
        suite.add("rmsnorm_rope/glm/pair/w" + std::to_string(width) + "/b" + std::to_string(batch),
                  [=] { return run_pair_case(width, batch, 1'048'560, 0x3000U, false, kGlm); });
    }
    suite.add("rmsnorm_rope/glm/pair/graph",
              [] { return run_pair_case(8, 8, 1'048'560, 0x3001U, true, kGlm); },
              {{"execution", "graph"}});
    suite.add("rmsnorm_rope/glm/pair/epsilon",
              [] { return run_pair_case(2, 1, 0, 0x3002U, false, kGlm, 1.e-4F); });
    for (int width = 2; width <= 16; ++width) {
        constexpr int batches[]{1, 3, 8};
        const int batch = batches[(width - 2) % 3];
        suite.add(
            "rmsnorm_rope/inkling/pair/w" + std::to_string(width) + "/b" + std::to_string(batch),
            [=] {
                return run_pair_case(width, batch, 1'048'560, 0x6000U + width, false, kInkling);
            });
    }
    for (const auto [width, batch] : {std::pair{8, 32}, {16, 64}}) {
        suite.add("rmsnorm_rope/inkling/pair/w" + std::to_string(width) + "/b" +
                      std::to_string(batch),
                  [=] { return run_pair_case(width, batch, 1'048'560, 0x6100U, false, kInkling); },
                  {{"suite", "concurrency"}});
    }
    suite.add("rmsnorm_rope/inkling/pair/unaligned",
              [] { return run_pair_case(7, 3, 1'048'560, 0x6200U, false, kInkling, 4.F, {4, 4}); });
    suite.add("rmsnorm_rope/inkling/pair/shared_weight_graph",
              [] {
                  return run_pair_case(8, 8, 1'048'560, 0x6201U, true, kInkling, 4.F,
                                       {.shared_weights = true});
              },
              {{"execution", "graph"}});
    suite.add("rmsnorm_rope/inkling/pair/wide_angles",
              [] {
                  auto wide  = kInkling;
                  wide.theta = 1.e-4F;
                  return run_pair_case(5, 3, 1'048'560, 0x6202U, true, wide, .01F, {0, 4, true}) +
                         run_pair_case(8, 4, 1'048'560, 0x6203U, false, wide);
              },
              {{"execution", "graph"}});
    suite.add("rmsnorm_rope/inkling/pair/exponent_range", [] {
        auto tiny_epsilon    = kInkling;
        tiny_epsilon.epsilon = std::numeric_limits<float>::denorm_min();
        return run_pair_case(2, 1, 1'048'560, 0x6204U, false, kInkling, 1.e30F) +
               run_pair_case(3, 1, 1'048'560, 0x6205U, false, tiny_epsilon, 1.e-22F);
    });
    suite.add("rmsnorm_rope/inkling/pair/large_gain", run_large_gain_case);
    for (int tokens : {1, 16, 256, 512, 2048}) {
        suite.add("rmsnorm_rope/glm/single/t" + std::to_string(tokens),
                  [=] { return run_single_case(tokens, 1'048'560, 0x4000U, tokens == 2048, kGlm); },
                  {{"execution", tokens == 2048 ? "graph" : "eager"}});
    }
    suite.add("rmsnorm_rope/parameters",
              [] {
                  const Profile other{16, 4, 3.e-4F, 12345.F};
                  return run_pair_case(3, 2, 1'048'560, 0x5000U, true, other, .01F) +
                         run_single_case(7, 1'048'560, 0x5001U, false, other, .01F) +
                         run_single_case(2, 1'048'560, 0x5002U, false,
                                         Profile{16, 4, 3.e-4F, 1.e-4F}, .01F);
              },
              {{"execution", "graph"}});
    suite.add("rmsnorm_rope/contract", contract);
}
