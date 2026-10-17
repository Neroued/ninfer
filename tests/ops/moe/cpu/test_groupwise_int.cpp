#include "ops/suite.h"
#include <map>
#include <memory>
#include "core/buffer.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
constexpr int H = 2048, I = 512, E = 256, K = 8;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid CPU Op input was accepted");
}

double half(std::uint16_t word) {
    const int exponent = (word >> 10) & 31, fraction = word & 1023;
    return exponent ? std::ldexp(1024.0 + fraction, exponent - 25)
                    : std::ldexp(static_cast<double>(fraction), -24);
}

double bf16(std::uint16_t word) {
    const int exponent = (word >> 7) & 255, fraction = word & 127;
    const double magnitude = exponent ? std::ldexp(128.0 + fraction, exponent - 134)
                                      : std::ldexp(static_cast<double>(fraction), -133);
    return word & 0x8000 ? -magnitude : magnitude;
}

std::uint32_t next(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

std::span<std::byte> bytes(HostBuffer& buffer) {
    return {static_cast<std::byte*>(buffer.data()), buffer.size()};
}

std::span<const std::byte> bytes(const HostBuffer& buffer) {
    return {static_cast<const std::byte*>(buffer.data()), buffer.size()};
}

struct Matrix {
    int bits;
    HostBuffer storage;
    Weight weight;
    std::vector<float> represented;

    Matrix(int bits, int rows, int columns, unsigned seed, bool corners)
        : bits(bits), storage(static_cast<std::size_t>(rows) * columns * (4 * bits + 1) / 32),
          represented(static_cast<std::size_t>(rows) * columns) {
        const auto format = bits == 4   ? QType::Q4_G64_FP16
                            : bits == 5 ? QType::Q5_G64_FP16
                                        : QType::Q6_G64_FP16;
        const std::array<std::uint64_t, 2> shape{static_cast<unsigned>(rows),
                                                 static_cast<unsigned>(columns)};
        WeightParent parent{weight_geometry(format, QuantLayout::RowSplit, shape), Device::cpu(),
                            bytes(storage).data()};
        weight =
            native_weight({{shape.begin(), shape.end()}, {{&parent, 0, parent.geometry.elements}}});
        auto* base                   = reinterpret_cast<std::uint8_t*>(bytes(storage).data());
        const std::size_t code_bytes = static_cast<std::size_t>(rows) * columns / 2;
        const std::size_t high_bytes = static_cast<std::size_t>(rows) * columns * (bits - 4) / 8;
        std::fill(bytes(storage).begin(), bytes(storage).end(), std::byte{});
        for (std::size_t group = 0; group < static_cast<std::size_t>(rows) * columns / 64;
             ++group) {
            std::uint16_t scale = 0x1400 + ((next(seed) % 3) << 10) + (next(seed) & 1023);
            if (group % 31 == 0) scale = 0;
            if (corners && group % 7 == 1) scale = 1;
            if (corners && group % 257 == 2) scale = 0x7bff;
            base[code_bytes + high_bytes + 2 * group]     = scale & 255;
            base[code_bytes + high_bytes + 2 * group + 1] = scale >> 8;
            for (unsigned k = 0; k < 64; ++k) {
                const unsigned code = scale ? next(seed) & ((1U << bits) - 1) : 0;
                const auto index    = group * 64 + k;
                base[index / 2] |= (code & 15) << (4 * (index % 2));
                if (bits > 4) {
                    const auto high_bit = index * (bits - 4);
                    base[code_bytes + high_bit / 8] |= (code >> 4) << (high_bit % 8);
                }
            }
        }
        decode_reference();
    }

    void decode_reference() {
        // Independent decoding of actual stored planes, not the producer's code values.
        const auto* base      = reinterpret_cast<const std::uint8_t*>(bytes(storage).data());
        const auto code_bytes = represented.size() / 2;
        const auto high_bytes = represented.size() * (bits - 4) / 8;
        for (std::size_t index = 0; index < represented.size(); ++index) {
            unsigned code = (base[index / 2] >> (4 * (index % 2))) & 15;
            if (bits > 4) {
                const auto high_bit = index * (bits - 4);
                code |=
                    ((base[code_bytes + high_bit / 8] >> (high_bit % 8)) & ((1U << (bits - 4)) - 1))
                    << 4;
            }
            const int signed_code =
                code >= (1U << (bits - 1)) ? static_cast<int>(code) - (1 << bits) : code;
            const auto offset         = code_bytes + high_bytes + 2 * (index / 64);
            const std::uint16_t scale = base[offset] + 256 * base[offset + 1];
            represented[index]        = static_cast<float>(signed_code * half(scale));
        }
    }
};

struct Expert {
    Matrix gu, down;

    Expert(int down_bits, unsigned seed, bool corners = false)
        : gu(4, 2 * I, H, seed, corners), down(down_bits, H, I, seed + 99827, corners) {}
};

ops::RoutedSwiGLUProfile make_profile(int down_bits) {
    return {{H, I, QType::Q4_G64_FP16, down_bits == 5 ? QType::Q5_G64_FP16 : QType::Q6_G64_FP16,
             QuantLayout::RowSplit, QuantLayout::RowSplit},
            E,
            K};
}

double gamma(int operations) {
    constexpr double u = 0x1p-24;
    return operations * u / (1 - operations * u);
}

// The oracle evaluates the complete mathematical formula. Its separate forward-error
// budget uses conservative serial G64 and group-sum lengths, a SiLU Lipschitz bound of 2,
// stable exp/div/multiply rounding, and absolute gradual-underflow allowances.
struct Reference {
    std::vector<double> result, budget;
};

std::size_t source_index(int expert, std::size_t count) {
    return (5 * expert + expert / 8) % count;
}

Reference reference(const std::vector<Expert>& experts, const std::vector<std::uint16_t>& input,
                    const std::vector<int>& ids, const std::vector<float>& coefficients,
                    const std::array<int, E>& mask, ops::ExpertSubsetKind kind, int tokens) {
    constexpr double u = 0x1p-24, tiny = 0x1p-149;
    Reference result{std::vector<double>(H * tokens), std::vector<double>(H * tokens)};
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> x(H), magnitude(H);
        for (int k = 0; k < H; ++k) x[k] = bf16(input[t * H + k]);
        int assignments = 0;
        for (int j = 0; j < K; ++j) {
            const int e = ids[t * K + j];
            if (kind == ops::ExpertSubsetKind::None ||
                (kind == ops::ExpertSubsetKind::MaskOnes && !mask[e]) ||
                (kind == ops::ExpertSubsetKind::MaskZeros && mask[e]))
                continue;
            ++assignments;
            const auto& expert = experts[source_index(e, experts.size())];
            std::vector<double> gu(2 * I), error(2 * I), z(I), z_error(I);
            for (int row = 0; row < 2 * I; ++row) {
                double sum = 0, absolute = 0;
                for (int k = 0; k < H; ++k) {
                    const double value =
                        static_cast<double>(expert.gu.represented[row * H + k]) * x[k];
                    sum += value;
                    absolute += std::abs(value);
                }
                gu[row]    = sum;
                error[row] = gamma(64 + H / 64 + 8) * absolute + (H + 64) * tiny * 65504;
            }
            for (int i = 0; i < I; ++i) {
                const double activation = gu[i] / (1 + std::exp(-gu[i]));
                z[i]                    = activation * gu[I + i];
                const double activation_error =
                    2 * error[i] + 16 * u * std::abs(activation) + (2 + std::abs(gu[i])) * tiny;
                z_error[i] = activation_error * (std::abs(gu[I + i]) + error[I + i]) +
                             std::abs(activation) * error[I + i] +
                             u * (std::abs(activation) + activation_error) *
                                 (std::abs(gu[I + i]) + error[I + i]) +
                             tiny;
            }
            const double coefficient = coefficients[t * K + j];
            for (int row = 0; row < H; ++row) {
                double sum = 0, absolute = 0, propagated = 0;
                for (int k = 0; k < I; ++k) {
                    const double w     = expert.down.represented[row * I + k];
                    const double value = w * z[k];
                    sum += value;
                    absolute += std::abs(value);
                    propagated += std::abs(w) * z_error[k];
                }
                const double linear_error = gamma(64 + I / 64 + 8) * absolute +
                                            (1 + gamma(64 + I / 64 + 8)) * propagated +
                                            (I + 64) * tiny * 65504;
                result.result[t * H + row] += coefficient * sum;
                result.budget[t * H + row] += std::abs(coefficient) * linear_error;
                magnitude[row] += std::abs(coefficient * sum);
            }
        }
        for (int row = 0; row < H; ++row) {
            result.budget[t * H + row] = (1 + gamma(assignments + 1)) * result.budget[t * H + row] +
                                         gamma(assignments + 1) * magnitude[row] +
                                         (assignments + 1) * tiny;
        }
    }
    return result;
}

struct NativeBank {
    HostBuffer storage;
    WeightParent parent;
    WeightBankView view;

    NativeBank(QType format, int rows, int columns, bool strided = false) {
        const std::uint64_t row_begin  = strided ? rows : 0;
        const std::uint64_t row_stride = rows + (strided ? 16 : 0);
        const std::array<std::uint64_t, 2> shape{row_begin + E * row_stride,
                                                 static_cast<unsigned>(columns)};
        parent.geometry = weight_geometry(format, QuantLayout::RowSplit, shape);
        storage         = HostBuffer(parent.geometry.bytes);
        parent.device   = Device::cpu();
        parent.data     = static_cast<const std::byte*>(storage.data());
        const WeightBankRegion region{row_begin, E, static_cast<unsigned>(rows), row_stride};
        view = bind_weight_bank(parent, describe_weight_bank(parent.geometry, region));
        std::memset(storage.data(), 0, storage.size());
    }

    void publish(int expert, const Matrix& source) {
        for (auto [plane, pointer] : {std::pair{view.layout.codes, source.weight.qdata},
                                      std::pair{view.layout.high, source.weight.qhigh},
                                      std::pair{view.layout.scales, source.weight.scales}}) {
            if (plane.expert_bytes) {
                const auto destination = view.plane(plane, expert);
                std::memcpy(const_cast<void*>(destination.data), pointer, destination.bytes);
            }
        }
    }
};

struct Fixture {
    ops::RoutedSwiGLUProfile profile;
    std::vector<Expert> experts;
    NativeBank gate_up, down;

    explicit Fixture(int down_bits, bool corners = false, bool strided = false)
        : profile(make_profile(down_bits)), gate_up(profile.ffn.gate_up_type, 2 * I, H, strided),
          down(profile.ffn.down_type, H, I, strided) {
        for (unsigned e = 0; e < 8; ++e) experts.emplace_back(down_bits, 719 + 1709 * e, corners);
        publish();
    }

    void publish() {
        for (int e = 0; e < E; ++e) {
            const auto& source = experts[source_index(e, experts.size())];
            gate_up.publish(e, source.gu);
            down.publish(e, source.down);
        }
    }

    ops::CpuSwiGLUExperts weights() const { return {gate_up.view, down.view}; }
};

enum class InputCase {
    Random,
    ExpertCancellation,
    TinyCancellation,
    MixedTiny,
    Coherent,
    CodeSweep,
    LocalRoutes,
    RaggedRoutes
};

void qualify(Fixture& f, CpuExecutor& executor, int tokens, int active, ops::ExpertSubsetKind kind,
             bool overlap, InputCase input_case = InputCase::Random,
             std::uint16_t tiny_base = 0x0380) {
    std::vector<std::uint16_t> input(H * tokens);
    std::vector<int> ids(K * tokens);
    std::vector<float> coefficients(K * tokens), output(H * tokens + 2, -123.5F);
    std::array<int, E> mask{};
    for (int e = 0; e < E; ++e) mask[e] = e % K < active;
    if (kind == ops::ExpertSubsetKind::MaskZeros)
        for (auto& bit : mask) bit = 1 - bit;
    unsigned random = 1783;
    for (auto& value : input) value = 0x3d00 + (next(random) & 127) + ((next(random) & 1) << 15);
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < K; ++j) {
            ids[t * K + j]          = ((overlap ? 13 : (13 + t)) * K + j) % E;
            coefficients[t * K + j] = (j % 3 == 0 ? -1.0F : 1.0F) * (j + 1) / 7.0F;
        }
        if (input_case == InputCase::ExpertCancellation) {
            ids[t * K]              = 0;
            ids[t * K + 1]          = 64;
            coefficients[t * K]     = 1.0F;
            coefficients[t * K + 1] = -1.0F;
            for (int j = 2; j < K; ++j) coefficients[t * K + j] = 0;
        }
        if (input_case == InputCase::TinyCancellation || input_case == InputCase::MixedTiny) {
            std::fill(input.begin() + t * H, input.begin() + (t + 1) * H, 0);
            const auto base =
                input_case == InputCase::MixedTiny ? (t == 0 ? 0x0400 : 0x03ff) : tiny_base;
            input[t * H]     = base;
            input[t * H + 1] = base + 1;
            input[t * H + 2] = 0x60ad;
        }
        if (input_case == InputCase::Coherent)
            std::fill(input.begin() + t * H, input.begin() + (t + 1) * H, 0x3d00);
        if (input_case == InputCase::TinyCancellation || input_case == InputCase::MixedTiny ||
            input_case == InputCase::Coherent || input_case == InputCase::CodeSweep) {
            for (int j = 0; j < K; ++j) ids[t * K + j] = j;
            coefficients[t * K] = 1;
        }
        if (input_case == InputCase::LocalRoutes)
            for (int j = 0; j < K; ++j) ids[t * K + j] = (64 * t + j) % E;
        if (input_case == InputCase::RaggedRoutes) {
            const int count = t >= 8 && t < 16 ? 0 : t % 9;
            for (int j = 0; j < K; ++j) ids[t * K + j] = j % 4 + 8 * (j / 4) + (j < count ? 0 : 4);
        }
    }
    const auto expected     = reference(f.experts, input, ids, coefficients, mask, kind, tokens);
    const auto before_input = input;
    const auto before_ids   = ids;
    const auto before_coefficients = coefficients;
    const auto capacity = ops::routed_swiglu_workspace_capacity_bytes(f.profile, kind, 1, tokens);
    HostBuffer scratch(capacity + 256);
    std::fill(bytes(scratch).begin(), bytes(scratch).end(), std::byte{0x6d});
    Tensor x(input.data(), Device::cpu(), DType::BF16, {H, tokens});
    Tensor out(output.data() + 1, Device::cpu(), DType::FP32, {H, tokens});
    ops::ExpertRoutes routes{Tensor(ids.data(), Device::cpu(), DType::I32, {K, tokens}),
                             Tensor(coefficients.data(), Device::cpu(), DType::FP32, {K, tokens})};
    const auto weights =
        kind == ops::ExpertSubsetKind::None ? ops::CpuSwiGLUExperts{} : f.weights();
    if (kind == ops::ExpertSubsetKind::None) { routes = {}; }
    const ops::ExpertSubset subset{kind, Tensor(mask.data(), Device::cpu(), DType::I32, {E})};
    MemoryArena arena(scratch.span().subspan(0, capacity));
    ops::routed_swiglu(x, f.profile, weights, routes, subset, out, arena,
                       executor.execution_view());
    check(arena.used() == 0 && arena.peak_used() <= capacity, "workspace cursor escaped call");
    double worst_rms = 0, worst_point = 0;
    for (int t = 0; t < tokens; ++t) {
        double errors = 0, budgets = 0, max_error = 0, max_budget = 0;
        for (int h = 0; h < H; ++h) {
            const auto index   = t * H + h;
            const double error = std::abs(output[index + 1] - expected.result[index]);
            check(std::isfinite(error), "nonfinite CPU SwiGLU result/error");
            errors += error * error;
            budgets += expected.budget[index] * expected.budget[index];
            max_error  = std::max(max_error, error);
            max_budget = std::max(max_budget, expected.budget[index]);
        }
        check(errors <= budgets && max_error <= max_budget, "RoutedSwiGLUF32Budget exceeded");
        worst_rms   = std::max(worst_rms, std::sqrt(errors / budgets));
        worst_point = std::max(worst_point, max_error / max_budget);
    }
    check(output.front() == -123.5F && output.back() == -123.5F, "output guard overwritten");
    check(std::all_of(bytes(scratch).begin() + capacity, bytes(scratch).end(),
                      [](auto b) { return b == std::byte{0x6d}; }),
          "workspace guard overwritten");
    check(input == before_input && ids == before_ids && coefficients == before_coefficients,
          "CPU Op modified an input");
    if (capacity) {
        MemoryArena short_arena(scratch.span().subspan(0, capacity - 1));
        rejects([&] {
            ops::routed_swiglu(x, f.profile, weights, routes, subset, out, short_arena,
                               executor.execution_view());
        });
    }
    std::cout << "down_bits=" << f.experts[0].down.bits << " T=" << tokens << " active=" << active
              << " workers=" << executor.plan().worker_cpus.size()
              << " subset=" << static_cast<int>(kind) << " overlap=" << overlap
              << " rms/budget=" << worst_rms << " max/budget=" << worst_point << '\n';
}

void code_sweep(Fixture& fixture, CpuExecutor& executor) {
    auto& matrix = fixture.experts[0].down;
    std::fill(bytes(matrix.storage).begin(), bytes(matrix.storage).end(), std::byte{});
    auto* base            = reinterpret_cast<std::uint8_t*>(bytes(matrix.storage).data());
    const auto code_bytes = matrix.represented.size() / 2;
    const auto high_bytes = matrix.represented.size() * (matrix.bits - 4) / 8;
    for (int row = 0; row < H; ++row) {
        // Every signed code at varying lanes/groups, with otherwise zero down rows.
        // GU produces nonuniform activations: moving a high field to another lane
        // cannot silently preserve the row sum as it would with constant activations.
        const int q         = row % (1 << matrix.bits) - (1 << (matrix.bits - 1));
        const unsigned code = q & ((1 << matrix.bits) - 1);
        const int column    = ((row / 64) * 17 + (row % 64) * 7) % I;
        const auto index    = static_cast<std::size_t>(row) * I + column;
        base[index / 2] |= (code & 15) << (4 * (index % 2));
        const auto bit = index * (matrix.bits - 4);
        base[code_bytes + bit / 8] |= (code >> 4) << (bit % 8);
        const auto scale = code_bytes + high_bytes + 2 * (index / 64);
        base[scale]      = 0;
        base[scale + 1]  = 0x3c; // FP16 1.
    }
    matrix.decode_reference();
    fixture.publish();
    for (int tokens : {1, 4})
        qualify(fixture, executor, tokens, 1, ops::ExpertSubsetKind::MaskOnes, true,
                InputCase::CodeSweep);
}

void tiny_cancellation(Fixture& fixture, CpuExecutor& executor) {
    // Normal BF16 inputs can cancel to a subnormal FP32 dot. A large independent up
    // channel makes that difference observable after SwiGLU, so unconditional BF16-dot
    // flushing would fail the same oracle and budget as ordinary inputs.
    auto& expert = fixture.experts[0];
    for (auto* matrix : {&expert.gu, &expert.down}) {
        std::fill(bytes(matrix->storage).begin(), bytes(matrix->storage).end(), std::byte{});
        std::fill(matrix->represented.begin(), matrix->represented.end(), 0);
    }
    auto* gu        = reinterpret_cast<std::uint8_t*>(bytes(expert.gu.storage).data());
    auto* gu_scales = reinterpret_cast<std::uint16_t*>(gu + 2 * I * H / 2);
    for (int row = 0; row < I; ++row) {
        gu[row * H / 2]           = 0xf1;
        gu[(I + row) * H / 2 + 1] = 1;
        gu_scales[row * (H / 64)] = gu_scales[(I + row) * (H / 64)] = 0x7bff;
        expert.gu.represented[row * H]                              = 65504;
        expert.gu.represented[row * H + 1]                          = -65504;
        expert.gu.represented[(I + row) * H + 2]                    = 65504;
    }
    auto* down = reinterpret_cast<std::uint8_t*>(bytes(expert.down.storage).data());
    auto* down_scales =
        reinterpret_cast<std::uint16_t*>(down + H * I / 2 + H * I * (expert.down.bits - 4) / 8);
    for (int row = 0; row < H; ++row) {
        down[row * I / 2]                = 1;
        down_scales[row * (I / 64)]      = 0x3c00;
        expert.down.represented[row * I] = 1;
    }
    fixture.publish();
    for (std::uint16_t base : {0x0001, 0x0380, 0x03ff, 0x0400, 0x0401})
        qualify(fixture, executor, 1, 1, ops::ExpertSubsetKind::MaskOnes, false,
                InputCase::TinyCancellation, base);
    qualify(fixture, executor, 2, 1, ops::ExpertSubsetKind::MaskOnes, true, InputCase::MixedTiny);
    qualify(fixture, executor, 4, 1, ops::ExpertSubsetKind::MaskOnes, true, InputCase::MixedTiny);
    qualify(fixture, executor, 7, 1, ops::ExpertSubsetKind::MaskOnes, true, InputCase::MixedTiny);
}

std::uint64_t weight_fingerprint(const Fixture& fixture) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const auto* bank : {&fixture.gate_up, &fixture.down}) {
        for (auto byte : bytes(bank->storage)) {
            hash ^= static_cast<std::uint8_t>(byte);
            hash *= 0x100000001b3ULL;
        }
    }
    return hash;
}

void coherent_precision(Fixture& fixture, CpuExecutor& executor) {
    auto& expert = fixture.experts[0];
    for (auto* matrix : {&expert.gu, &expert.down}) {
        auto* code =
            const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(matrix->weight.qdata));
        const auto elements = matrix->represented.size();
        std::fill(code, code + elements / 2, 0x11);
        if (matrix->weight.qhigh) {
            auto* high =
                const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(matrix->weight.qhigh));
            std::fill(high, high + elements * (matrix->bits - 4) / 8, 0);
        }
        auto* scales =
            const_cast<std::uint16_t*>(static_cast<const std::uint16_t*>(matrix->weight.scales));
        std::fill(scales, scales + elements / 64, 0x1800);
        std::fill(matrix->represented.begin(), matrix->represented.end(), 1.0F / 512);
    }
    // Positive rank-one matrices remove cancellation amplification from the error budget.
    // An unintended BF16 intermediate rounding is larger than this same FP32 allowance.
    fixture.publish();
    qualify(fixture, executor, 1, 1, ops::ExpertSubsetKind::MaskOnes, false, InputCase::Coherent);
    qualify(fixture, executor, 4, 1, ops::ExpertSubsetKind::MaskOnes, true, InputCase::Coherent);
}

void validation(Fixture& f, CpuExecutor& executor) {
    using ops::ExpertSubsetKind;
    const auto view = executor.execution_view();
    for (const auto [first, last] : {std::pair{0, 1}, std::pair{2, 1}}) {
        rejects([&] {
            (void)ops::routed_swiglu_workspace_capacity_bytes(f.profile, ExpertSubsetKind::All,
                                                              first, last);
        });
    }
    auto unsupported          = f.profile;
    unsupported.ffn.down_type = QType::Q8_G32_FP16;
    rejects([&] {
        (void)ops::routed_swiglu_workspace_capacity_bytes(unsupported, ExpertSubsetKind::All, 1, 1);
    });
    rejects([&] {
        (void)ops::routed_swiglu_workspace_capacity_bytes(f.profile,
                                                          static_cast<ExpertSubsetKind>(255), 1, 1);
    });
    const auto capacity =
        ops::routed_swiglu_workspace_capacity_bytes(f.profile, ExpertSubsetKind::All, 1, 1);
    HostBuffer storage(capacity + 512), x_storage(H * 2), out_storage(H * 4);
    std::memset(x_storage.data(), 0, x_storage.size());
    std::array<int, K> ids{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<float, K> coefficients{1, 1, 1, 1, 1, 1, 1, 1};
    std::array<int, E> mask{};
    Tensor x(x_storage.span(), DType::BF16, {H, 1}), out(out_storage.span(), DType::FP32, {H, 1});
    ops::ExpertRoutes routes{Tensor(ids.data(), Device::cpu(), DType::I32, {K, 1}),
                             Tensor(coefficients.data(), Device::cpu(), DType::FP32, {K, 1})};
    auto weights = f.weights();
    MemoryArena arena(storage.span());
    (void)arena.alloc_bytes(256);
    auto call = [&](ops::ExpertSubset subset = {}) {
        ops::routed_swiglu(x, f.profile, weights, routes, subset, out, arena, view);
    };
    call();
    check(arena.used() == 256 && arena.peak_used() == 256 + capacity,
          "nonzero workspace cursor was not scoped");
    check(std::all_of(static_cast<float*>(out.data), static_cast<float*>(out.data) + H,
                      [](float value) { return value == 0; }),
          "zero input did not yield zero output");
    weights.down.layout.format = QType::Q4_G64_FP16;
    rejects([&] { call(); });
    weights = f.weights();
    --weights.down.layout.high.expert_bytes;
    rejects([&] { call(); });
    weights             = f.weights();
    auto wrong_parent   = f.down.parent;
    wrong_parent.device = Device::cuda(0);
    weights.down.parent = &wrong_parent;
    rejects([&] { call(); });
    weights = f.weights();
    ids[1]  = ids[0];
    rejects([&] { call(); });
    ids[1] = 1;
    ids[0] = E;
    rejects([&] { call(); });
    ids[0]          = 0;
    coefficients[0] = std::numeric_limits<float>::infinity();
    rejects([&] { call(); });
    coefficients[0] = 1;
    mask[0]         = 2;
    const Tensor mask_view(mask.data(), Device::cpu(), DType::I32, {E});
    rejects([&] { call({ExpertSubsetKind::MaskOnes, mask_view}); });
    mask[0]  = 0;
    out.data = x.data;
    rejects([&] { call(); });
    out.data = out_storage.data();
    x.device = Device::cuda(0);
    rejects([&] { call(); });
    x.device          = Device::cpu();
    routes.ids.device = Device::cuda(0);
    rejects([&] { call(); });
    routes.ids.device = Device::cpu();
    out.data          = storage.data();
    rejects([&] { call(); });
    out.data              = out_storage.data();
    const auto original_x = x.data;
    x.data                = static_cast<std::byte*>(storage.data()) + 256;
    rejects([&] { call(); });
    x.data            = original_x;
    const auto before = arena.used();
    (void)arena.alloc_bytes(1, 1);
    rejects([&] { call(); });
    check(arena.used() == before + 1, "rejected call changed the workspace cursor");

    // Empty contributions need no banks; None also needs no routes and no aligned scratch cursor.
    weights = {};
    MemoryArena no_scratch(storage.span().subspan(0, 1));
    (void)no_scratch.alloc_bytes(1, 1);
    ops::routed_swiglu(x, f.profile, {}, {}, {ExpertSubsetKind::None, {}}, out, no_scratch, view);
    check(no_scratch.used() == 1, "None rewound its caller's cursor");
    arena.reset();
    call({ExpertSubsetKind::MaskOnes, mask_view});
    std::fill(mask.begin(), mask.end(), 1);
    call({ExpertSubsetKind::MaskZeros, mask_view});
}
} // namespace

int main(int argc, char** argv) {
    using namespace ninfer::test;
    OpSuite suite(false);

    struct Case {
        int tokens, active, workers;
        ops::ExpertSubsetKind subset;
        bool overlap;
        InputCase input;
    };

    for (int bits : {5, 6}) {
        const std::string format = "q4_q" + std::to_string(bits);
        std::vector<Case> cases;
        std::vector<OpCase> catalogue;
        const auto add = [&](int tokens, int active, ops::ExpertSubsetKind subset, bool overlap,
                             InputCase input = InputCase::Random, int workers = 8) {
            cases.push_back({tokens, active, workers, subset, overlap, input});
            catalogue.push_back(
                {"cpu_moe/" + format + "/t" + std::to_string(tokens) + "/active" +
                     std::to_string(active) + "/subset" + std::to_string(int(subset)) + "/workers" +
                     std::to_string(workers) + (overlap ? "/overlap" : "/disjoint") + "/input" +
                     std::to_string(int(input)),
                 {{"format", format}}});
        };
        using ops::ExpertSubsetKind;
        add(1, 8, ExpertSubsetKind::All, false, InputCase::Random, 1);
        for (int tokens : {1, 2, 3, 7, 8, 9, 17})
            add(tokens, tokens % 3 == 0 ? 2 : 4,
                tokens % 2 ? ExpertSubsetKind::MaskOnes : ExpertSubsetKind::MaskZeros,
                tokens % 2 == 0);
        add(8, 8, ExpertSubsetKind::All, true);
        for (int tokens : {5, 6, 7, 9}) add(tokens, 2, ExpertSubsetKind::MaskOnes, true);
        add(25, 4, ExpertSubsetKind::MaskOnes, false, InputCase::RaggedRoutes);
        for (int tokens : {32, 64, 72}) add(tokens, 1, ExpertSubsetKind::MaskOnes, false);
        add(1, 0, ExpertSubsetKind::MaskOnes, false);
        add(9, 0, ExpertSubsetKind::None, false);
        add(1, 8, ExpertSubsetKind::All, false, InputCase::ExpertCancellation);
        add(3, 1, ExpertSubsetKind::MaskZeros, false, InputCase::Random, 3);
        add(1, 8, ExpertSubsetKind::All, false, InputCase::Random, 16);
        suite.add(std::move(catalogue), [=](auto selected) {
            CpuTopology topology;
            const auto available = cpu_worker_order(topology.info()).size();
            std::map<int, std::unique_ptr<CpuExecutor>> executors;
            const auto executor = [&](int workers) -> CpuExecutor& {
                auto& value = executors[workers];
                if (!value)
                    value = std::make_unique<CpuExecutor>(
                        topology,
                        plan_cpu_team(topology.info(), std::min<std::size_t>(workers, available)));
                return *value;
            };
            const auto features = executor(1).execution_view().features();
            if (!features.has(CpuFeature::Avx2) || !features.has(CpuFeature::Fma) ||
                !features.has(CpuFeature::F16c))
                throw OpTestSkip("CPU requires AVX2, FMA and F16C");
            Fixture fixture(bits);
            const auto initial_weights = weight_fingerprint(fixture);
            for (auto i : selected) {
                const auto c = cases[i];
                qualify(fixture, executor(c.workers), c.tokens, c.active, c.subset, c.overlap,
                        c.input);
            }
            check(weight_fingerprint(fixture) == initial_weights,
                  "CPU Op modified immutable weights");
            return 0;
        });
        enum class Recipe { Contract, CodeSweep, Corners, Tiny, Coherent };
        for (auto [recipe, name] : {std::pair{Recipe::Contract, "contract"},
                                    {Recipe::CodeSweep, "code_sweep"},
                                    {Recipe::Corners, "corners"},
                                    {Recipe::Tiny, "tiny"},
                                    {Recipe::Coherent, "coherent"}})
            suite.add(
                "cpu_moe/" + format + "/" + name,
                [=] {
                    CpuTopology topology;
                    CpuExecutor executor(
                        topology,
                        plan_cpu_team(topology.info(),
                                      recipe == Recipe::Contract
                                          ? 1
                                          : std::min<std::size_t>(
                                                8, cpu_worker_order(topology.info()).size())));
                    const auto features = executor.execution_view().features();
                    if (!features.has(CpuFeature::Avx2) || !features.has(CpuFeature::Fma) ||
                        !features.has(CpuFeature::F16c))
                        throw OpTestSkip("CPU requires AVX2, FMA and F16C");
                    const bool corners = recipe == Recipe::Corners || recipe == Recipe::Tiny ||
                                         recipe == Recipe::Coherent;
                    Fixture fixture(bits, corners, corners);
                    if (recipe == Recipe::Contract)
                        validation(fixture, executor);
                    else if (recipe == Recipe::CodeSweep)
                        code_sweep(fixture, executor);
                    else if (recipe == Recipe::Tiny)
                        tiny_cancellation(fixture, executor);
                    else if (recipe == Recipe::Coherent)
                        coherent_precision(fixture, executor);
                    else {
                        qualify(fixture, executor, 3, 1, ExpertSubsetKind::MaskOnes, false,
                                InputCase::LocalRoutes);
                        qualify(fixture, executor, 2, 4, ExpertSubsetKind::MaskOnes, false);
                    }
                    return 0;
                },
                {{"format", format}});
    }
    return suite.main(argc, argv);
}
