#include "models/glm5next/vision/execution.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"

#include <cmath>
#include <memory>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace glm = models::glm5next;
using Values  = std::vector<double>;

float rounded(double x) { return bf16_to_f32(f32_to_bf16(static_cast<float>(x))); }

struct Vector {
    Values values;
    DeviceBuffer storage;
    Tensor tensor;

    Vector(int n, float base, float amplitude) {
        std::vector<float> host(n);
        for (int i = 0; i < n; ++i) host[i] = rounded(base + amplitude * ((i * 7 % 23) - 11));
        values.assign(host.begin(), host.end());
        storage = to_device_bf16(host);
        tensor  = Tensor(storage.span(), DType::BF16, {n});
    }
};

// Two nonzero coefficients per row keep the independent CPU reference small while exercising
// every output feature of the actual registered dense GPU projections.
struct Matrix {
    int n, k, seed;
    DeviceBuffer storage;
    ops::SingleProjectionWeight parameter;

    Matrix(int rows, int columns, int salt) : n(rows), k(columns), seed(salt) {
        std::vector<std::uint16_t> host(std::size_t(n) * k);
        for (int r = 0; r < n; ++r)
            for (int j = 0; j < 2; ++j)
                host[std::size_t(r) * k + index(r, j)] = f32_to_bf16(value(r, j));
        storage = to_device(host);
        Weight w;
        w.device  = storage.span().device;
        w.payload = w.qdata = storage.data();
        w.payload_bytes     = storage.size();
        w.qtype             = QType::BF16;
        w.layout            = QuantLayout::Contiguous;
        w.n                 = n;
        w.k                 = k;
        w.ndim              = 2;
        w.shape[0] = w.padded_shape[0] = n;
        w.shape[1] = w.padded_shape[1] = k;
        parameter                      = {w, ops::LinearPolicy::A16Only};
    }

    int index(int r, int j) const { return (r * 13 + seed * 19 + j * 17) % k; }

    float value(int r, int j) const { return rounded((j ? -0.25 : 0.5) + (r % 5) / 64.0); }

    Values apply(const Values& x, int tokens, const Values* bias = nullptr,
                 bool round = true) const {
        Values y(std::size_t(n) * tokens);
        for (int t = 0; t < tokens; ++t)
            for (int r = 0; r < n; ++r) {
                double v = bias ? (*bias)[r] : 0;
                for (int j = 0; j < 2; ++j) v += value(r, j) * x[std::size_t(t) * k + index(r, j)];
                y[std::size_t(t) * n + r] = round ? rounded(v) : v;
            }
        return y;
    }
};

Values norm(const Values& x, const Values& gamma, int width, bool center = false,
            const Values* bias = nullptr) {
    Values y(x.size());
    for (std::size_t start = 0; start < x.size(); start += width) {
        double mean = 0, variance = 0;
        if (center) {
            for (int d = 0; d < width; ++d) mean += x[start + d];
            mean /= width;
        }
        for (int d = 0; d < width; ++d) variance += std::pow(x[start + d] - mean, 2);
        const auto inv = 1 / std::sqrt(variance / width + 1e-5);
        for (int d = 0; d < width; ++d)
            y[start + d] =
                rounded((x[start + d] - mean) * inv * gamma[d] + (bias ? (*bias)[d] : 0));
    }
    return y;
}

Values add(const Values& x, const Values& y) {
    Values out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = rounded(x[i] + y[i]);
    return out;
}

Values gu(const Matrix& w, const Values& x, int tokens, const Values* bias = nullptr) {
    const auto projected = w.apply(x, tokens, bias, false);
    const auto rows      = w.n / 2;
    Values y(std::size_t(rows) * tokens);
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
            const auto g = std::min(10.0, projected[std::size_t(t) * w.n + r]);
            const auto u = std::clamp(projected[std::size_t(t) * w.n + rows + r], -10.0, 10.0);
            y[std::size_t(t) * rows + r] = rounded(g / (1 + std::exp(-g)) * u);
        }
    return y;
}

Values attention(Values qkv, const Values& qgamma, const Values& kgamma,
                 const glm::vision::EncoderBatch& batch) {
    const auto n = batch.cu_seqlens.back();
    for (int t = 0; t < n; ++t)
        for (int component = 0; component < 2; ++component)
            for (int head = 0; head < 16; ++head) {
                const auto at = std::size_t(t) * 3072 + component * 1024 + head * 64;
                double ss     = 0;
                for (int d = 0; d < 64; ++d) ss += qkv[at + d] * qkv[at + d];
                const auto inv    = 1 / std::sqrt(ss / 64 + 1e-5);
                const auto& gamma = component ? kgamma : qgamma;
                for (int d = 0; d < 32; ++d) {
                    const auto angle =
                        batch.positions[t + (d / 16) * n] * std::pow(10000.0, -2.0 * (d % 16) / 32);
                    const auto a     = qkv[at + d] * inv * gamma[d];
                    const auto b     = qkv[at + d + 32] * inv * gamma[d + 32];
                    qkv[at + d]      = rounded(a * std::cos(angle) - b * std::sin(angle));
                    qkv[at + d + 32] = rounded(b * std::cos(angle) + a * std::sin(angle));
                }
            }
    Values out(std::size_t(n) * 1024);
    for (std::size_t segment = 1; segment < batch.cu_seqlens.size(); ++segment) {
        const auto first = batch.cu_seqlens[segment - 1], end = batch.cu_seqlens[segment];
        for (int t = first; t < end; ++t)
            for (int head = 0; head < 16; ++head) {
                Values scores(end - first);
                for (int s = first; s < end; ++s)
                    for (int d = 0; d < 64; ++d)
                        scores[s - first] += qkv[t * 3072 + head * 64 + d] *
                                             qkv[s * 3072 + 1024 + head * 64 + d] / 8;
                const auto mx = *std::max_element(scores.begin(), scores.end());
                double total  = 0;
                for (auto& v : scores) {
                    v = std::exp(v - mx);
                    total += v;
                }
                for (int d = 0; d < 64; ++d) {
                    double v = 0;
                    for (int s = first; s < end; ++s)
                        v += scores[s - first] / total * qkv[s * 3072 + 2048 + head * 64 + d];
                    out[t * 1024 + head * 64 + d] = rounded(v);
                }
            }
    }
    return out;
}

struct Block {
    Vector input{1024, 1, .005F}, post{1024, 1, .004F}, qnorm{64, 1, .003F}, knorm{64, 1, .002F};
    Vector qb{3072, 0, .02F}, ob{1024, 0, .01F}, gb{8192, 0, 1.1F}, db{1024, 0, .02F};
    Matrix qkv, output, gate_up, down;

    explicit Block(int salt)
        : qkv(3072, 1024, salt), output(1024, 1024, salt + 1), gate_up(8192, 1024, salt + 2),
          down(1024, 4096, salt + 3) {}

    glm::execution::VisionBlockParameters parameters() const {
        return {input.tensor,  post.tensor,      qnorm.tensor,      knorm.tensor,
                qkv.parameter, output.parameter, gate_up.parameter, down.parameter,
                qb.tensor,     ob.tensor,        gb.tensor,         db.tensor};
    }
};

int run() {
    auto& env = cuda_environment();
    const glm::VisionConfig c{2, 1024, 16, 4096, 3, 14, 2, 2, 4096, 10240, 1e-5F, 10};
    Matrix patch(1024, 1176, 1), downsample(4096, 4096, 3), proj(4096, 4096, 5),
        gate_up(20480, 4096, 7), down(4096, 10240, 9);
    Vector pb(1024, 0, .01F), db(4096, 0, .02F), pn(1024, 1, .003F), gamma(4096, 1, .002F),
        beta(4096, 0, .003F);
    std::array<Block, 2> blocks{Block(21), Block(31)};
    glm::execution::VisionParameters p{patch.parameter,
                                       downsample.parameter,
                                       proj.parameter,
                                       gate_up.parameter,
                                       down.parameter,
                                       pb.tensor,
                                       db.tensor,
                                       pn.tensor,
                                       gamma.tensor,
                                       beta.tensor,
                                       {}};
    glm::execution::VisionProfile profile{ops::linear_profile(p.patch),
                                          ops::linear_profile(p.downsample),
                                          ops::linear_profile(p.merger_projection),
                                          ops::linear_profile(p.merger_gate_up),
                                          ops::linear_profile(p.merger_down),
                                          {}};
    for (const auto& block : blocks) {
        p.layers.push_back(block.parameters());
        profile.layers.push_back({ops::linear_profile(block.qkv.parameter),
                                  ops::linear_profile(block.output.parameter),
                                  ops::linear_profile(block.gate_up.parameter),
                                  ops::linear_profile(block.down.parameter)});
    }
    glm::vision::EncoderBatch batch;
    batch.features   = 3;
    batch.cu_seqlens = {0, 4, 12};
    batch.positions  = {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3};
    Values input(1176 * 12);
    std::vector<std::uint16_t> bits(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = rounded(std::sin(i * .031) * .8);
        bits[i]  = f32_to_bf16(input[i]);
    }
    batch.patches = {std::span(bits).first(4 * 1176), std::span(bits).subspan(4 * 1176)};
    auto x        = patch.apply(input, 12, &pb.values);
    for (const auto& block : blocks) {
        const auto normalized = norm(x, block.input.values, 1024);
        const auto qkv        = block.qkv.apply(normalized, 12, &block.qb.values);
        const auto attended   = attention(qkv, block.qnorm.values, block.knorm.values, batch);
        x                     = add(x, block.output.apply(attended, 12, &block.ob.values));
        const auto branch =
            gu(block.gate_up, norm(x, block.post.values, 1024), 12, &block.gb.values);
        x = add(x, block.down.apply(branch, 12, &block.db.values));
    }
    x = norm(x, pn.values, 1024);
    x = downsample.apply(x, 3, &db.values);
    x = proj.apply(x, 3);
    x = norm(x, gamma.values, 4096, true, &beta.values);
    for (auto& v : x) v = rounded(.5 * v * (1 + std::erf(v / std::sqrt(2.0))));
    const auto reference = down.apply(gu(gate_up, x, 3), 3);
    const auto capacity  = glm::vision::plan_resources(c, profile, 3, 64ULL << 20);
    DeviceBuffer scratch(env.context.device(), capacity.encoder.workspace_bytes);
    models::vision::Storage storage(env.context, capacity);
    auto output = storage.output(0, 3);
    (void)glm::vision::encode(c, p, batch, output, scratch.span(), storage.staging(),
                              env.default_execution);
    CUDA_CHECK(cudaStreamSynchronize(env.default_execution.stream()));
    const auto actual = from_device_bf16(output.data, reference.size());
    double error = 0, energy = 0, maximum = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        error += std::pow(actual[i] - reference[i], 2);
        energy += std::pow(reference[i], 2);
        maximum = std::max(maximum, std::abs(actual[i] - reference[i]));
    }
    const auto relative = std::sqrt(error / energy);
    std::cout << "GLM Vision two-block segmented reference: rel_l2=" << relative
              << " max_abs=" << maximum << '\n';
    return relative <= .02 && maximum <= .05 ? 0 : 1;
}
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) return 77;
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
