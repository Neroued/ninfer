#include "ops/suite.h"
#include "core/buffer.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
constexpr int H = 2560, I = 640, E = 512, K = 10;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& f) {
    bool caught = false;
    try {
        f();
    } catch (const std::invalid_argument&) { caught = true; }
    check(caught, "invalid CPU NVFP4 operand accepted");
}

unsigned random(unsigned& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

double e4m3(unsigned code) {
    const int e = code >> 3, m = code & 7;
    return e ? std::ldexp(8.0 + m, e - 10) : std::ldexp(double(m), -9);
}

double bf16(std::uint16_t code) { return double(std::bit_cast<float>(std::uint32_t(code) << 16)); }

struct Matrix {
    int rows, columns;
    std::vector<std::uint8_t> codes, scales;
    std::vector<double> represented;

    Matrix(int n, int k, unsigned seed, bool coherent)
        : rows(n), columns(k), codes(std::size_t(n) * k / 2), scales(std::size_t(n) * k / 16),
          represented(std::size_t(n) * k) {
        for (auto& code : codes) code = coherent ? 0x11 : random(seed) & 255;
        // Write the tiled scale order directly; decode below uses logical coordinates.
        for (auto& scale : scales) scale = coherent ? 32 : random(seed) % 127;
        decode_reference();
    }

    void decode_reference() {
        const int n = rows, k = columns;
        constexpr double table[]{0, 0.5, 1, 1.5, 2, 3, 4, 6};
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < k; ++col) {
                const auto index = std::size_t(row) * k + col;
                const auto code  = (codes[index / 2] >> (4 * (col & 1))) & 15;
                const int group  = col / 16;
                const auto scale_index =
                    (((row / 128) * (k / 64) + group / 4) * 32 + row % 32) * 16 +
                    ((row % 128) / 32) * 4 + group % 4;
                represented[index] =
                    (code & 8 ? -table[code & 7] : table[code]) * e4m3(scales[scale_index]);
            }
        }
    }
};

struct Bank {
    HostBuffer storage;
    std::array<float, E> divisors;
    WeightParent parent;
    WeightBankView view;
    std::array<Matrix, 2> matrices;

    Bank(int rows, int columns, unsigned seed)
        : matrices{Matrix(rows, columns, seed, false), Matrix(rows, columns, seed + 1, true)} {
        // A 128-row prefix per physical matrix exercises nonzero origins and expert strides.
        const std::array<std::uint64_t, 3> dims{E, unsigned(rows + 128), unsigned(columns)};
        parent.geometry =
            weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4, dims);
        storage = HostBuffer(parent.geometry.bytes);
        for (int e = 0; e < E; ++e) divisors[e] = 1024.375F + ((e * 137 + seed) % 2048);
        parent.device          = Device::cpu();
        parent.data            = static_cast<const std::byte*>(storage.data());
        parent.weight_divisors = divisors;
        std::memcpy(static_cast<std::byte*>(storage.data()) + parent.geometry.divisor_offset,
                    divisors.data(), divisors.size() * sizeof(float));
        view = bind_weight_bank(
            parent,
            describe_weight_bank(parent.geometry, {128, E, unsigned(rows), unsigned(rows + 128)}));
        // Only selected experts are backed by initialized pages; no model or disk fixture.
        for (int e : used_experts()) publish(e, matrices[e == 511]);
    }

    static std::vector<int> used_experts() {
        std::vector<int> ids(20);
        std::iota(ids.begin(), ids.end(), 0);
        ids.insert(ids.end(), {254, 255, 510, 511});
        return ids;
    }

    void publish(int e, const Matrix& m) {
        std::memcpy(const_cast<void*>(view.plane(view.layout.codes, e).data), m.codes.data(),
                    m.codes.size());
        std::memcpy(const_cast<void*>(view.plane(view.layout.scales, e).data), m.scales.data(),
                    m.scales.size());
    }
};

struct Fixture {
    Bank gu{2 * I, H, 87121}, down{H, I, 419341};
    ops::RoutedSwiGLUProfile profile{
        {H, I, QType::NVFP4, QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
         QuantLayout::BlockScaleK16M128x4, ops::LinearPolicy::AllowA4, ops::LinearPolicy::AllowA4},
        E,
        K};
    std::array<std::vector<std::uint16_t>, 2> inputs;
    std::map<std::pair<int, int>, std::vector<double>> oracle;

    Fixture() {
        unsigned seed = 783411;
        for (int pattern = 0; pattern < 2; ++pattern) {
            inputs[pattern].resize(H);
            for (auto& x : inputs[pattern])
                x = 0x3c80 + (random(seed) & 511) + ((random(seed) & 1) << 15);
            inputs[pattern][H - 1] = 0x3e89;
        }
    }

    const std::vector<double>& contribution(int expert, int pattern) {
        auto [it, inserted] = oracle.try_emplace({expert, pattern});
        if (!inserted) return it->second;
        const auto& gu_matrix   = gu.matrices[expert == 511];
        const auto& down_matrix = down.matrices[expert == 511];
        std::vector<double> projected(2 * I), activation(I);
        for (int r = 0; r < 2 * I; ++r) {
            double sum = 0;
            for (int k = 0; k < H; ++k)
                sum += (gu_matrix.represented[r * H + k] / double(gu.divisors[expert])) *
                       bf16(inputs[pattern][k]);
            projected[r] = sum;
        }
        for (int r = 0; r < I; ++r)
            activation[r] = projected[r] / (1 + std::exp(-projected[r])) * projected[I + r];
        auto& out = it->second;
        out.resize(H);
        for (int r = 0; r < H; ++r) {
            double sum = 0;
            for (int k = 0; k < I; ++k)
                sum += (down_matrix.represented[r * I + k] / double(down.divisors[expert])) *
                       activation[k];
            out[r] = sum;
        }
        return out;
    }
};

void compare(const float* actual, const std::vector<double>& expected) {
    double error2 = 0, norm2 = 0, worst = 0, scale = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(std::isfinite(actual[i]), "nonfinite CPU NVFP4 output");
        const double error = std::abs(double(actual[i]) - expected[i]);
        error2 += error * error;
        norm2 += expected[i] * expected[i];
        worst = std::max(worst, error);
        scale = std::max(scale, std::abs(expected[i]));
    }
    const double relative = std::sqrt(error2 / std::max(norm2, 1e-300));
    // FP32 contractions and full-precision SwiGLU, with no activation quantization.
    check(relative <= 5e-5 && worst <= 3e-5 * scale + 1e-37,
          "CPU NVFP4 differs from FP64 mathematics");
    std::cout << " relative_l2=" << relative << " max_error=" << worst << '\n';
}

void qualify(Fixture& f, CpuExecutor& executor, int tokens, int active, bool same,
             ops::ExpertSubsetKind kind, int inactive_prefix = 0) {
    std::vector<std::uint16_t> input(std::size_t(H) * tokens);
    std::vector<int> ids(std::size_t(K) * tokens);
    std::vector<float> coefficients(ids.size());
    std::array<int, E> mask{};
    for (int e = 0; e < E; ++e) mask[e] = (e & 1) == 0;
    if (inactive_prefix) {
        mask.fill(0);
        mask[511] = 1;
    }
    std::vector<double> expected(std::size_t(H) * tokens);
    const auto expert_ids = Bank::used_experts();
    for (int t = 0; t < tokens; ++t) {
        const int pattern = t & 1;
        std::copy(f.inputs[pattern].begin(), f.inputs[pattern].end(),
                  input.begin() + std::size_t(t) * H);
        for (int j = 0; j < K; ++j) {
            const int e        = inactive_prefix
                                     ? (t < inactive_prefix || j > 0 ? j : 511)
                                     : expert_ids[(j + (same ? 0 : t * 7)) % expert_ids.size()];
            const auto slot    = std::size_t(t) * K + j;
            ids[slot]          = e;
            coefficients[slot] = j < active ? (j & 1 ? -0.07F : 0.113F) : 0;
            if (kind == ops::ExpertSubsetKind::None ||
                (kind == ops::ExpertSubsetKind::MaskOnes && !mask[e]) ||
                (kind == ops::ExpertSubsetKind::MaskZeros && mask[e]))
                continue;
            if (coefficients[slot] == 0) continue;
            const auto& contribution = f.contribution(e, pattern);
            for (int r = 0; r < H; ++r)
                expected[std::size_t(t) * H + r] += double(coefficients[slot]) * contribution[r];
        }
    }
    Tensor x(input.data(), Device::cpu(), DType::BF16, {H, tokens});
    const auto capacity =
        ops::routed_swiglu_workspace_capacity_bytes(f.profile, kind, tokens, tokens);
    HostBuffer scratch(capacity + 512), output(std::size_t(H) * tokens * 4 + 256);
    std::memset(scratch.data(), 0x5a, scratch.size());
    std::memset(output.data(), 0x6b, output.size());
    MemoryArena arena({scratch.data(), capacity + 256, Device::cpu()});
    (void)arena.alloc_bytes(256);
    Tensor out(output.span(), DType::FP32, {H, tokens});
    const ops::ExpertRoutes routes{
        Tensor(ids.data(), Device::cpu(), DType::I32, {K, tokens}),
        Tensor(coefficients.data(), Device::cpu(), DType::FP32, {K, tokens})};
    const ops::ExpertSubset subset{kind, Tensor(mask.data(), Device::cpu(), DType::I32, {E})};
    const ops::CpuSwiGLUExperts weights{f.gu.view, f.down.view};
    ops::routed_swiglu(x, f.profile,
                       kind == ops::ExpertSubsetKind::None ? ops::CpuSwiGLUExperts{} : weights,
                       kind == ops::ExpertSubsetKind::None ? ops::ExpertRoutes{} : routes, subset,
                       out, arena, executor.execution_view());
    check(arena.used() == 256 && arena.peak_used() == capacity + 256,
          "CPU NVFP4 capacity disagrees with execution");
    for (int b = 0; b < 256; ++b) {
        check(static_cast<unsigned char*>(scratch.data())[b] == 0x5a &&
                  static_cast<unsigned char*>(scratch.data())[capacity + 256 + b] == 0x5a,
              "CPU NVFP4 scratch guard overwritten");
        check(static_cast<unsigned char*>(output.data())[std::size_t(H) * tokens * 4 + b] == 0x6b,
              "CPU NVFP4 output guard overwritten");
    }
    std::cout << "T=" << tokens << " active=" << active << " same=" << same
              << " subset=" << int(kind);
    compare(static_cast<const float*>(out.data), expected);
}

void tiny_inputs(Fixture& f, CpuExecutor& executor) {
    for (auto* bank : {&f.gu, &f.down}) {
        auto& m = bank->matrices[0];
        std::fill(m.codes.begin(), m.codes.end(), 0);
        std::fill(m.scales.begin(), m.scales.end(), 0);
        for (int row = 0; row < m.rows; ++row) {
            const int s =
                (row / 128) * (m.columns / 64) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4;
            if (bank == &f.down) {
                m.codes[row * m.columns / 2] = 2;
                m.scales[s]                  = 56;
            } else if (row < I) {
                m.codes[row * m.columns / 2] = 0x91;
                m.scales[s]                  = 1;
            } else {
                m.codes[row * m.columns / 2 + 1] = 2;
                m.scales[s]                      = 126;
            }
        }
        m.decode_reference();
        bank->divisors.fill(1);
        std::memcpy(static_cast<std::byte*>(bank->storage.data()) +
                        bank->parent.geometry.divisor_offset,
                    bank->divisors.data(), bank->divisors.size() * sizeof(float));
        for (int e : Bank::used_experts()) bank->publish(e, bank->matrices[e == 511]);
    }
    for (std::uint16_t base : {0x0001, 0x08ff, 0x0900, 0x0901}) {
        for (int p = 0; p < 2; ++p) {
            std::fill(f.inputs[p].begin(), f.inputs[p].end(), 0);
            f.inputs[p][0] = base;
            f.inputs[p][1] = base - 1;
            f.inputs[p][2] = 0x6780;
        }
        f.oracle.clear();
        qualify(f, executor, 3, 1, true, ops::ExpertSubsetKind::MaskOnes);
    }
    std::fill(f.inputs[0].begin(), f.inputs[0].end(), 0);
    f.oracle.clear();
    qualify(f, executor, 1, K, true, ops::ExpertSubsetKind::All);
}

void validation(Fixture& f, CpuExecutor& executor) {
    const auto cap =
        ops::routed_swiglu_workspace_capacity_bytes(f.profile, ops::ExpertSubsetKind::All, 1, 1);
    HostBuffer storage(cap), output(H * 4);
    Tensor x(f.inputs[0].data(), Device::cpu(), DType::BF16, {H, 1}),
        out(output.span(), DType::FP32, {H, 1});
    std::array<int, K> ids{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<float, K> coefficients{};
    coefficients.fill(0.1F);
    ops::ExpertRoutes routes{Tensor(ids.data(), Device::cpu(), DType::I32, {K, 1}),
                             Tensor(coefficients.data(), Device::cpu(), DType::FP32, {K, 1})};
    ops::CpuSwiGLUExperts weights{f.gu.view, f.down.view};
    MemoryArena arena(storage.span());
    auto call = [&] {
        ops::routed_swiglu(x, f.profile, weights, routes, {}, out, arena,
                           executor.execution_view());
    };
    f.gu.divisors[0] = 0;
    rejects(call);
    f.gu.divisors[0] = 1024;
    weights.gate_up.layout.scales.expert_stride += 1;
    rejects(call);
    weights.gate_up        = f.gu.view;
    weights.gate_up.parent = nullptr;
    rejects(call);
    auto bad       = f.profile;
    bad.ffn.hidden = 2561;
    rejects([&] {
        (void)ops::routed_swiglu_workspace_capacity_bytes(bad, ops::ExpertSubsetKind::All, 1, 1);
    });
    MemoryArena small({storage.data(), cap - 1, Device::cpu()});
    weights.gate_up = f.gu.view;
    try {
        ops::routed_swiglu(x, f.profile, weights, routes, {}, out, small,
                           executor.execution_view());
        check(false, "accepted insufficient workspace");
    } catch (const std::bad_alloc&) {}
    // Empty masks must not resolve even missing banks or their divisor metadata.
    std::array<int, E> zeros{};
    ops::routed_swiglu(
        x, f.profile, {}, routes,
        {ops::ExpertSubsetKind::MaskOnes, Tensor(zeros.data(), Device::cpu(), DType::I32, {E})},
        out, arena, executor.execution_view());
    check(std::all_of(static_cast<float*>(out.data), static_cast<float*>(out.data) + H,
                      [](float v) { return v == 0; }),
          "empty subset is not zero");
}
} // namespace

int main(int argc, char** argv) {
    using namespace ninfer::test;
    OpSuite suite(false);

    struct Case {
        int tokens, active;
        bool same;
        ops::ExpertSubsetKind subset;
        int workers = 8;
    };

    std::vector<Case> cases;
    using ops::ExpertSubsetKind;
    for (int t : {1, 2, 3, 4, 5, 6, 7, 8, 9, 72, 255, 256, 257})
        cases.push_back({t, K, true, ExpertSubsetKind::All});
    for (auto subset : {ExpertSubsetKind::All, ExpertSubsetKind::None, ExpertSubsetKind::MaskOnes,
                        ExpertSubsetKind::MaskZeros})
        cases.push_back({9, K, false, subset});
    cases.push_back({3, 1, false, ExpertSubsetKind::MaskOnes, 1});
    std::vector<OpCase> catalogue;
    for (auto c : cases)
        catalogue.push_back({"cpu_moe/nvfp4/t" + std::to_string(c.tokens) + "/active" +
                                 std::to_string(c.active) + "/subset" +
                                 std::to_string(int(c.subset)) + "/workers" +
                                 std::to_string(c.workers) + (c.same ? "/same" : "/mixed"),
                             {{"format", "nvfp4"}}});
    suite.add(std::move(catalogue), [=](auto selected) {
        CpuTopology topology;
        const auto count = std::min<std::size_t>(8, cpu_worker_order(topology.info()).size());
        CpuExecutor team(topology, plan_cpu_team(topology.info(), count));
        CpuExecutor single(topology, plan_cpu_team(topology.info(), 1));
        Fixture fixture;
        for (auto i : selected) {
            auto c = cases[i];
            qualify(fixture, c.workers == 1 ? single : team, c.tokens, c.active, c.same, c.subset);
        }
        return 0;
    });
    enum class Recipe { InactiveDivisors, Policies, InactivePrefix, Contract, Tiny };
    for (auto [recipe, name] : {std::pair{Recipe::InactiveDivisors, "inactive_divisors"},
                                {Recipe::Policies, "policies"},
                                {Recipe::InactivePrefix, "inactive_prefix"},
                                {Recipe::Contract, "contract"},
                                {Recipe::Tiny, "tiny"}})
        suite.add(std::string("cpu_moe/nvfp4/") + name,
                  [=] {
                      CpuTopology topology;
                      CpuExecutor executor(
                          topology,
                          plan_cpu_team(
                              topology.info(),
                              std::min<std::size_t>(8, cpu_worker_order(topology.info()).size())));
                      Fixture f;
                      if (recipe == Recipe::InactiveDivisors) {
                          for (int e = 1; e < E; e += 2)
                              f.gu.divisors[e] = std::numeric_limits<float>::quiet_NaN();
                          qualify(f, executor, 3, K, false, ExpertSubsetKind::MaskOnes);
                      } else if (recipe == Recipe::Policies) {
                          f.profile.ffn.gate_up_policy = ops::LinearPolicy::A16Only;
                          f.profile.ffn.down_policy    = ops::LinearPolicy::AllowA8;
                          qualify(f, executor, 4, K, true, ExpertSubsetKind::All);
                      } else if (recipe == Recipe::InactivePrefix)
                          qualify(f, executor, 257, K, false, ExpertSubsetKind::MaskOnes, 256);
                      else if (recipe == Recipe::Contract)
                          validation(f, executor);
                      else
                          tiny_inputs(f, executor);
                      return 0;
                  },
                  {{"format", "nvfp4"}});
    return suite.main(argc, argv);
}
