#include "ops/suite.h"
#include "core/buffer.h"
#include "ninfer/ops/moe.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <map>
#include <numeric>
#include <tuple>

namespace ninfer::test::moe_block_scaled {
namespace {
constexpr int H = 2560, I = 640, E = 512, TopK = 10;
// Two activation quantizations with a nonlinear product between them. The FP64 oracle
// retains the original BF16 input and decoded weights; no private quantization is imitated.
constexpr ReductionCriterion kA4{0.25, 1e-7, 0.35};
constexpr ReductionCriterion kA8{0.08, 1e-7, 0.12};
constexpr ReductionCriterion kA16{0.012, 1e-6, 0.008};
// Same BF16/FP32 routing budget as the integer MoE suite; probabilities have unit scale.
constexpr ReductionCriterion kRoute{3e-5, 5e-6, 0};

ops::MoeProfile make_profile(bool fp8) {
    const auto type   = fp8 ? QType::FP8_E4M3FN_BLOCK128X128_BF16 : QType::NVFP4;
    const auto layout = fp8 ? QuantLayout::BlockScaleM128K128 : QuantLayout::BlockScaleK16M128x4;
    const auto policy = fp8 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::AllowA4;
    return {{{H, I, type, type, layout, layout, policy, policy}, E, TopK},
            {H, I, QType::BF16, QType::BF16, QuantLayout::Contiguous, QuantLayout::Contiguous}};
}

Weight dense(const DeviceBuffer& buffer, int rows, int columns) {
    Weight w;
    w.device  = buffer.device();
    w.payload = w.qdata = buffer.data();
    w.payload_bytes     = buffer.size();
    w.qtype             = QType::BF16;
    w.layout            = QuantLayout::Contiguous;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = columns;
    return w;
}

struct Matrix {
    int n, k;
    bool fp8;
    std::size_t code_bytes, scale_bytes;
    std::vector<std::vector<std::uint8_t>> payload;
    std::vector<std::vector<float>> decoded;
    DeviceBuffer data, offsets, divisors;
    std::vector<float> dw;
    std::size_t stride, scale_offset;

    Matrix(int rows, int columns, unsigned seed, bool is_fp8)
        : n(rows), k(columns), fp8(is_fp8), offsets(cuda_environment().context.device(), E * 8),
          divisors(cuda_environment().context.device(), is_fp8 ? 0 : E * 4) {
        for (int slot = 0; slot < 2; ++slot) {
            std::vector<std::uint8_t> bytes;
            std::vector<float> values(std::size_t(n) * k);
            if (fp8) {
                code_bytes   = std::size_t(n) * k;
                scale_bytes  = std::size_t(n / 128) * (k / 128) * 2;
                scale_offset = code_bytes;
                bytes.resize(code_bytes + scale_bytes);
                for (std::size_t b = 0; b < code_bytes; ++b) {
                    unsigned code =
                        quantized_weight::detail::mix64(b + seed + slot * 998123ULL) & 255;
                    if ((code & 127) == 127) { --code; }
                    bytes[b] = code;
                }
                for (int nr = 0; nr < n / 128; ++nr) {
                    for (int kc = 0; kc < k / 128; ++kc) {
                        const int index = nr * (k / 128) + kc;
                        const float scale =
                            (n == 2 * I && nr == 0)
                                ? 0
                                : (1 + (nr * 3 + kc * 7 + seed + slot) % 13) * 0.0000083F;
                        quantized_weight::detail::store_u16_le(bytes, scale_offset + index * 2,
                                                               f32_to_bf16(scale));
                    }
                }
                for (int r = 0; r < n; ++r) {
                    for (int c = 0; c < k; ++c) {
                        const auto index = std::size_t(r) * k + c;
                        const auto scale = quantized_weight::detail::load_u16_le(
                            bytes, scale_offset + ((r / 128) * (k / 128) + c / 128) * 2);
                        values[index] = quantized_weight::detail::decode_e4m3fn(bytes[index]) *
                                        quantized_weight::detail::bf16_to_f32(scale);
                    }
                }
            } else {
                auto p = quantized_weight::make_patterned_weight(
                    QType::NVFP4, n, k, seed + slot,
                    {.weight_scale_divisor = 1, .input_scale_divisor = 1});
                for (std::size_t b = 0; b < p.code_plane_bytes; ++b) {
                    p.payload[b] =
                        quantized_weight::detail::mix64(b + seed + slot * 998123ULL) & 255;
                }
                code_bytes   = p.code_plane_bytes;
                scale_bytes  = p.scale_plane_bytes;
                scale_offset = p.scale_plane_offset;
                for (int r = 0; r < n; ++r) {
                    for (int c = 0; c < k; ++c) {
                        values[std::size_t(r) * k + c] =
                            quantized_weight::logical_weight_fp64(p, r, c);
                    }
                }
                bytes = std::move(p.payload);
            }
            payload.push_back(std::move(bytes));
            decoded.push_back(std::move(values));
        }
        stride = (payload[0].size() + 255) / 256 * 256;
        data   = DeviceBuffer(cuda_environment().context.device(), 2 * stride);
        for (int slot = 0; slot < 2; ++slot) {
            data.copy_from_host(payload[slot].data(), payload[slot].size(), slot * stride);
        }
        dw.resize(E);
        for (int e = 0; e < E; ++e) {
            dw[e] = fp8 ? 1.0F : (seed == 101 ? 6.0F : 3.0F) + (e % 8) * 0.5F;
        }
        if (!fp8) { divisors.copy_from_host(dw.data(), divisors.size()); }
        publish({}, ops::ExpertSubsetKind::All, false);
    }

    void publish(std::span<const int> mask, ops::ExpertSubsetKind kind, bool flip) {
        std::vector<std::uint64_t> indices(E, UINT64_MAX);
        for (int e = 0; e < E; ++e) {
            if (kind == ops::ExpertSubsetKind::All ||
                (kind == ops::ExpertSubsetKind::MaskOnes && mask[e]) ||
                (kind == ops::ExpertSubsetKind::MaskZeros && !mask[e])) {
                indices[e] = ((e & 1) ^ int(flip)) * stride;
            }
        }
        offsets.copy_from_host(indices.data(), offsets.size());
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView result;
        result.codes           = static_cast<const std::byte*>(data.data());
        result.scales          = result.codes + scale_offset;
        result.offsets         = static_cast<const std::uint64_t*>(offsets.data());
        result.indexed_storage = data.span();
        result.divisors        = static_cast<const float*>(divisors.data());
        result.divisor_stride  = fp8 ? 0 : 1;
        return result;
    }
};

std::vector<float> random_values(std::size_t count, unsigned seed, float limit) {
    std::vector<float> result(count);
    fill_uniform(result, seed, -limit, limit);
    round_to_bf16(result);
    return result;
}

std::vector<double> ffn(std::span<const float> x, const std::vector<float>& gu,
                        const std::vector<float>& down, float gu_divisor, float down_divisor) {
    std::vector<double> h(I), out(H);
    for (int row = 0; row < I; ++row) {
        double gate = 0, up = 0;
        for (int k = 0; k < H; ++k) {
            gate += double(x[k]) * gu[std::size_t(row) * H + k] / gu_divisor;
            up += double(x[k]) * gu[std::size_t(row + I) * H + k] / gu_divisor;
        }
        h[row] = gate / (1 + std::exp(-gate)) * up;
    }
    for (int row = 0; row < H; ++row) {
        for (int k = 0; k < I; ++k) {
            out[row] += h[k] * down[std::size_t(row) * I + k] / down_divisor;
        }
    }
    return out;
}

struct Fixture {
    const bool fp8;
    const ops::MoeProfile profile;
    Matrix gu, down;
    std::vector<std::vector<float>> x;
    std::vector<float> control, shared_gu, shared_down;
    DeviceBuffer control_d, shared_gu_d, shared_down_d;
    ops::MoeWeights weights;
    std::map<std::tuple<int, int, bool>, std::vector<double>> cache;
    std::vector<std::vector<double>> shared_values;
    std::array<std::vector<double>, 3> logits;

    explicit Fixture(bool is_fp8)
        : fp8(is_fp8), profile(make_profile(is_fp8)), gu(2 * I, H, 101, is_fp8),
          down(H, I, 211, is_fp8), control(random_values(std::size_t(E + 1) * H, 303, .0004F)),
          shared_gu(random_values(std::size_t(2 * I) * H, 404, .012F)),
          shared_down(random_values(std::size_t(H) * I, 505, .025F)) {
        for (int p = 0; p < 3; ++p) {
            x.push_back(random_values(H, 1701 + p, .7F));
            if (p == 2) {
                std::mt19937 generator(1703);
                std::normal_distribution<float> normal(0, 1);
                for (auto& value : x.back()) {
                    value = std::clamp(normal(generator), -3.25F, 3.25F);
                }
                round_to_bf16(x.back());
            }
            for (int k = 0; k < (fp8 ? 128 : 32); ++k) { x.back()[k] = 0; }
            // Deliberate zero groups and a last-K coordinate expose stage/scale-tail errors.
            x.back()[H - 1] = p == 2 ? 3.5F : .875F;
            shared_values.push_back(ffn(x.back(), shared_gu, shared_down, 1, 1));
        }
        for (int e = 0; e <= E; ++e) { control[std::size_t(e) * H + e + 32] = 2; }
        // The highest two rows are identical dense projections: lower ID must win ties.
        control[std::size_t(E - 2) * H + E - 2 + 32] = 0;
        control[std::size_t(E - 2) * H + H - 1]      = 2;
        std::copy_n(control.begin() + std::size_t(E - 2) * H, H,
                    control.begin() + std::size_t(E - 1) * H);
        for (int p = 0; p < 3; ++p) {
            logits[p].resize(E + 1);
            for (int e = 0; e <= E; ++e) {
                for (int k = 0; k < H; ++k) {
                    logits[p][e] += double(control[std::size_t(e) * H + k]) * x[p][k];
                }
            }
        }
        control_d     = to_device_bf16(control);
        shared_gu_d   = to_device_bf16(shared_gu);
        shared_down_d = to_device_bf16(shared_down);
        weights       = {dense(control_d, E + 1, H),
                         {{gu.view(), down.view(), E, control_d.device()},
                    fp8 ? ops::ExpertActivationDivisors{}
                              : ops::ExpertActivationDivisors{719.7991333F, 313.927002F}},
                         {{dense(shared_gu_d, 2 * I, H)}, {dense(shared_down_d, H, I)}}};
    }

    const std::vector<double>& expert(int p, int e, bool flip) {
        auto key   = std::tuple{p, e % 8, flip};
        auto found = cache.find(key);
        if (found != cache.end()) { return found->second; }
        const int slot = (e & 1) ^ int(flip);
        return cache
            .emplace(key, ffn(x[p], gu.decoded[slot], down.decoded[slot], gu.dw[e], down.dw[e]))
            .first->second;
    }

    void publish(std::span<const int> mask, ops::ExpertSubsetKind kind, bool flip) {
        gu.publish(mask, kind, flip);
        down.publish(mask, kind, flip);
    }
};

int run(Fixture& f, CudaExecutionView execution, int tokens, ops::ExpertSubsetKind kind,
        bool shared, bool complete, DType dtype, ops::MoeEpilogue mode, bool graph) {
    const auto& profile = f.profile;
    std::vector<float> x(std::size_t(tokens) * H), residual = random_values(x.size(), 906, .002F);
    std::vector<int> ids(tokens * TopK), mask(E);
    std::vector<float> alpha(ids.size()), beta(tokens);
    for (int e = 0; e < E; ++e) { mask[e] = e % 3 != 0; }
    auto dx = to_device_bf16(x), di = to_device(ids), da = to_device(alpha), db = to_device(beta),
         dm = to_device(mask);
    Tensor input(dx.span(), DType::BF16, {H, tokens}),
        shared_tensor(db.span(), DType::FP32, {tokens});
    ops::ExpertRoutes routes{Tensor(di.span(), DType::I32, {TopK, tokens}),
                             Tensor(da.span(), DType::FP32, {TopK, tokens})};
    ops::ExpertSubset subset{kind, Tensor(dm.span(), DType::I32, {E})};
    const std::size_t bytes =
        complete
            ? ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(profile, mode, tokens, tokens)
        : shared ? ops::routed_swiglu_workspace_capacity_bytes(profile.routed, profile.shared, kind,
                                                               dtype, mode, tokens, tokens)
                 : ops::routed_swiglu_workspace_capacity_bytes(profile.routed, kind, dtype, mode,
                                                               tokens, tokens);
    GuardedDeviceBuffer scratch(bytes), result(x.size() * (dtype == DType::BF16 ? 2 : 4));
    MemoryArena arena(scratch.span());
    Tensor output(result.span(), dtype, {H, tokens});
    auto invoke = [&] {
        if (complete) {
            ops::topk_softmax_swiglu_moe(input, profile, f.weights, mode, output, arena, execution);
        } else {
            const auto experts =
                kind == ops::ExpertSubsetKind::None ? ops::CudaSwiGLUExperts{} : f.weights.routed;
            const auto selection =
                kind == ops::ExpertSubsetKind::None ? ops::ExpertRoutes{} : routes;
            if (shared) {
                ops::routed_swiglu(input, profile.routed, experts, selection, subset,
                                   {f.weights.shared, shared_tensor}, mode, output, arena,
                                   execution);
            } else {
                ops::routed_swiglu(input, profile.routed, experts, selection, subset, mode, output,
                                   arena, execution);
            }
        }
    };
    cudaGraph_t captured       = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        cuda_check(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeGlobal),
                   "capture");
        invoke();
        cuda_check(cudaStreamEndCapture(execution.stream(), &captured), "end capture");
        cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0), "instantiate");
    }
    int failures = 0;
    const bool masked =
        kind == ops::ExpertSubsetKind::MaskOnes || kind == ops::ExpertSubsetKind::MaskZeros;
    const int replays = graph ? (masked ? 4 : 3) : 1;
    for (int replay = 0; replay < replays; ++replay) {
        const bool flip = replay == 2;
        for (int e = 0; e < E; ++e) {
            mask[e] = replay == 3 ? int(kind == ops::ExpertSubsetKind::MaskZeros)
                                  : int((e + replay) % 3 != 0);
        }
        f.publish(mask, complete ? ops::ExpertSubsetKind::All : kind, flip);
        std::vector<double> ref(x.size());
        for (int t = 0; t < tokens; ++t) {
            const int p = (t + replay) % 3;
            std::copy(f.x[p].begin(), f.x[p].end(), x.begin() + std::size_t(t) * H);
            std::vector<int> order(E);
            std::iota(order.begin(), order.end(), 0);
            const auto& logits = f.logits[p];
            if (complete) {
                std::stable_sort(order.begin(), order.end(),
                                 [&](int a, int b) { return logits[a] > logits[b]; });
            }
            std::array<double, TopK> coefficients;
            double total = 0;
            for (int j = 0; j < TopK; ++j) {
                const int e       = complete ? order[j] : (replay == 1 ? j : (t * 13 + j * 31) % E);
                ids[t * TopK + j] = e;
                alpha[t * TopK + j] = .025F + j * .007F;
                coefficients[j] =
                    complete ? std::exp(logits[e] - logits[order[0]]) : alpha[t * TopK + j];
                total += coefficients[j];
            }
            beta[t]                         = .25F + p * .125F;
            const double shared_coefficient = complete ? 1 / (1 + std::exp(-logits[E])) : beta[t];
            for (int j = 0; j < TopK; ++j) {
                if (complete) { coefficients[j] /= total; }
                const int e        = ids[t * TopK + j];
                const bool include = complete || kind == ops::ExpertSubsetKind::All ||
                                     (kind == ops::ExpertSubsetKind::MaskOnes && mask[e]) ||
                                     (kind == ops::ExpertSubsetKind::MaskZeros && !mask[e]);
                if (include) {
                    const auto& value = f.expert(p, e, flip);
                    for (int h = 0; h < H; ++h) {
                        ref[std::size_t(t) * H + h] += coefficients[j] * value[h];
                    }
                }
            }
            for (int h = 0; h < H; ++h) {
                const auto i = std::size_t(t) * H + h;
                if (shared) { ref[i] += shared_coefficient * f.shared_values[p][h]; }
                if (mode == ops::MoeEpilogue::AddResidual) { ref[i] += residual[i]; }
            }
        }
        std::vector<std::uint16_t> bits(x.size());
        std::transform(x.begin(), x.end(), bits.begin(), f32_to_bf16);
        dx.copy_from_host(bits.data(), dx.size());
        di.copy_from_host(ids.data(), di.size());
        da.copy_from_host(alpha.data(), da.size());
        db.copy_from_host(beta.data(), db.size());
        dm.copy_from_host(mask.data(), dm.size());
        if (mode == ops::MoeEpilogue::AddResidual) {
            std::transform(residual.begin(), residual.end(), bits.begin(), f32_to_bf16);
            result.copy_from_host(bits.data(), bits.size() * 2);
        }
        if (graph) {
            cuda_check(cudaGraphLaunch(executable, execution.stream()), "replay");
        } else {
            invoke();
        }
        cuda_synchronize(execution.stream());
        std::vector<double> actual;
        if (dtype == DType::BF16) {
            actual = from_device_bf16(output.data, x.size());
        } else {
            const auto values = from_device<float>(output.data, x.size());
            actual.assign(values.begin(), values.end());
        }
        const std::string label =
            std::string(complete ? "complete" : "routed") + (f.fp8 ? " fp8 M=" : " nvfp4 M=") +
            std::to_string(tokens) + " subset=" + std::to_string(int(kind)) +
            " shared=" + std::to_string(shared) + " replay=" + std::to_string(replay);
        failures += verify_reduction(label, actual, ref,
                                     kind == ops::ExpertSubsetKind::None ||
                                             (tokens == 1 || (f.fp8 && tokens == 2))
                                         ? kA16
                                         : (f.fp8 ? kA8 : kA4));
        failures +=
            scratch.verify_guards(label + " scratch") + result.verify_guards(label + " output");
        if (arena.used() || arena.peak_used() != bytes) {
            std::cerr << label << ": workspace query differs from execution\n";
            ++failures;
        }
    }
    if (graph) {
        cuda_check(cudaGraphExecDestroy(executable), "destroy exec");
        cuda_check(cudaGraphDestroy(captured), "destroy graph");
    }
    return failures;
}

int route_cases(Fixture& f, CudaExecutionView execution) {
    int failures = 0;
    for (int m : {1, 2, 33, 4097}) {
        std::vector<float> input(std::size_t(m) * H);
        for (int t = 0; t < m; ++t) {
            std::copy(f.x[t % 3].begin(), f.x[t % 3].end(), input.begin() + std::size_t(t) * H);
        }
        auto dx = to_device_bf16(input);
        GuardedDeviceBuffer ids(std::size_t(m) * TopK * 4), alpha(std::size_t(m) * TopK * 4),
            beta(m * 4);
        GuardedDeviceBuffer scratch(
            ops::topk_softmax_route_workspace_capacity_bytes({H, E, TopK}, m, m));
        MemoryArena arena(scratch.span());
        Tensor x(dx.span(), DType::BF16, {H, m}), shared(beta.span(), DType::FP32, {m});
        ops::ExpertRoutes routes{Tensor(ids.span(), DType::I32, {TopK, m}),
                                 Tensor(alpha.span(), DType::FP32, {TopK, m})};
        int pattern_offset = 0;
        if (m == 33) {
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            cuda_check(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeGlobal),
                       "capture route");
            ops::topk_softmax_route(x, {H, E, TopK}, f.weights.control, routes, shared, arena,
                                    execution);
            cuda_check(cudaStreamEndCapture(execution.stream(), &graph), "end route capture");
            cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                       "instantiate route");
            pattern_offset = 1;
            std::vector<std::uint16_t> changed(input.size());
            for (int t = 0; t < m; ++t) {
                const auto& values = f.x[(t + pattern_offset) % 3];
                std::transform(values.begin(), values.end(), changed.begin() + std::size_t(t) * H,
                               f32_to_bf16);
            }
            dx.copy_from_host(changed.data(), dx.size());
            cuda_check(cudaGraphLaunch(executable, execution.stream()), "replay route");
            cuda_synchronize(execution.stream());
            cuda_check(cudaGraphExecDestroy(executable), "destroy route exec");
            cuda_check(cudaGraphDestroy(graph), "destroy route graph");
        } else {
            ops::topk_softmax_route(x, {H, E, TopK}, f.weights.control, routes, shared, arena,
                                    execution);
            cuda_synchronize(execution.stream());
        }
        const auto actual_ids   = from_device<int>(routes.ids.data, m * TopK);
        const auto actual_alpha = from_device<float>(routes.coefficients.data, m * TopK);
        const auto actual_beta  = from_device<float>(shared.data, m);
        std::vector<double> expected_alpha(m * TopK), expected_beta(m);
        for (int t = 0; t < m; ++t) {
            const int p        = (t + pattern_offset) % 3;
            const auto& scores = f.logits[p];
            std::vector<int> order(E);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return scores[a] > scores[b]; });
            double total = 0;
            for (int j = 0; j < TopK; ++j) {
                total += std::exp(scores[order[j]] - scores[order[0]]);
            }
            for (int j = 0; j < TopK; ++j) {
                if (actual_ids[t * TopK + j] != order[j]) {
                    std::cerr << "route M=" << m << " token=" << t << " rank=" << j
                              << ": ID mismatch\n";
                    ++failures;
                }
                expected_alpha[t * TopK + j] =
                    std::exp(scores[order[j]] - scores[order[0]]) / total;
            }
            expected_beta[t] = 1 / (1 + std::exp(-scores[E]));
        }
        const auto label = "route nvfp4 geometry M=" + std::to_string(m);
        failures += verify_reduction(label + " alpha",
                                     std::vector<double>(actual_alpha.begin(), actual_alpha.end()),
                                     expected_alpha, kRoute);
        failures += verify_reduction(label + " beta",
                                     std::vector<double>(actual_beta.begin(), actual_beta.end()),
                                     expected_beta, kRoute);
        failures += ids.verify_guards(label) + alpha.verify_guards(label) +
                    beta.verify_guards(label) + scratch.verify_guards(label);
    }
    return failures;
}

int affine_cases(Fixture& f, CudaExecutionView execution) {
    auto gu_codes    = DeviceBuffer(execution.device(), E * f.gu.code_bytes);
    auto gu_scales   = DeviceBuffer(execution.device(), E * f.gu.scale_bytes);
    auto down_codes  = DeviceBuffer(execution.device(), E * f.down.code_bytes);
    auto down_scales = DeviceBuffer(execution.device(), E * f.down.scale_bytes);
    const auto copy  = [&](Matrix& from, DeviceBuffer& codes, DeviceBuffer& scales) {
        const auto cb = from.code_bytes, sb = from.scale_bytes;
        for (int e = 0; e < E; ++e) {
            const auto* source =
                static_cast<const std::byte*>(from.data.data()) + (e & 1) * from.stride;
            cuda_check(cudaMemcpyAsync(static_cast<std::byte*>(codes.data()) + e * cb, source, cb,
                                        cudaMemcpyDeviceToDevice, execution.stream()),
                        "affine codes");
            cuda_check(cudaMemcpyAsync(static_cast<std::byte*>(scales.data()) + e * sb,
                                        source + from.scale_offset, sb, cudaMemcpyDeviceToDevice,
                                        execution.stream()),
                        "affine scales");
        }
        auto view            = from.view();
        view.offsets         = nullptr;
        view.indexed_storage = {};
        view.codes           = static_cast<const std::byte*>(codes.data());
        view.scales          = static_cast<const std::byte*>(scales.data());
        view.code_stride     = cb;
        view.scale_stride    = sb;
        return view;
    };
    auto saved                    = f.weights.routed.bank;
    f.weights.routed.bank.gate_up = copy(f.gu, gu_codes, gu_scales);
    f.weights.routed.bank.down    = copy(f.down, down_codes, down_scales);
    cuda_synchronize(execution.stream());
    int failures = 0;
    for (int m : {1, 129}) {
        failures += run(f, execution, m, ops::ExpertSubsetKind::All, false, false, DType::FP32,
                        ops::MoeEpilogue::Write, false);
    }
    f.weights.routed.bank = saved;
    return failures;
}

int zero_expert_cases(Fixture& f, CudaExecutionView execution) {
    // Keep the shared term observable at its own scale through the complete Op.
    // The existing A4/A8 criterion stays unchanged; no private route is forced.
    std::cout << "shared-only oracle (zero routed GU), format=" << (f.fp8 ? "fp8" : "nvfp4")
              << '\n';
    for (std::size_t slot = 0; slot < f.gu.payload.size(); ++slot) {
        auto& bytes = f.gu.payload[slot];
        std::fill(bytes.begin() + f.gu.scale_offset,
                  bytes.begin() + f.gu.scale_offset + f.gu.scale_bytes, 0);
        std::fill(f.gu.decoded[slot].begin(), f.gu.decoded[slot].end(), 0);
        f.gu.data.copy_from_host(bytes.data(), bytes.size(), slot * f.gu.stride);
    }
    f.cache.clear();
    int failures = 0;
    for (int m : {512, 513}) {
        failures += run(f, execution, m, ops::ExpertSubsetKind::All, true, true, DType::BF16,
                        ops::MoeEpilogue::Write, true);
    }
    return failures;
}

} // namespace
} // namespace ninfer::test::moe_block_scaled

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace ninfer::test;
    using namespace ninfer::test::moe_block_scaled;
    OpSuite suite;
    for (bool fp8 : {false, true}) {
        const std::string format = fp8 ? "fp8" : "nvfp4";
        std::vector<OpCase> catalogue;
        std::vector<std::function<int(Fixture&, CudaExecutionView)>> runners;
        const auto special = [&](std::string id, auto run,
                                 std::vector<std::pair<std::string, std::string>> tags = {}) {
            tags.emplace_back("format", format);
            catalogue.push_back({"moe/" + format + "/" + id, std::move(tags)});
            runners.push_back(run);
        };
        special("affine", affine_cases);
        if (!fp8) special("route", route_cases, {{"execution", "graph"}});
        const auto add = [&](int m, ops::ExpertSubsetKind subset, bool shared, bool complete,
                             DType dtype, ops::MoeEpilogue epilogue, bool graph) {
            const auto id = std::string(complete ? "complete" : "routed") + "/t" +
                            std::to_string(m) + "/subset" + std::to_string(int(subset)) +
                            (shared ? "/shared" : "/no_shared") + "/dtype" +
                            std::to_string(int(dtype)) + "/epilogue" +
                            std::to_string(int(epilogue)) + (graph ? "/graph" : "/eager");
            special(id,
                    [=](Fixture& f, auto e) {
                        return run(f, e, m, subset, shared, complete, dtype, epilogue, graph);
                    },
                    {{"execution", graph ? "graph" : "eager"}});
        };
        using ops::ExpertSubsetKind;
        using ops::MoeEpilogue;
        for (int m : {1,   2,   3,   7,   8,   9,    15,   16,   17,   18,   31,   32,
                      33,  34,  63,  64,  65,  66,   127,  128,  129,  130,  255,  256,
                      257, 258, 511, 512, 513, 1023, 1024, 1025, 3583, 3584, 3585, 4097}) {
            add(m, ExpertSubsetKind::All, false, false, DType::FP32, MoeEpilogue::Write, false);
        }
        for (auto subset :
             {ExpertSubsetKind::None, ExpertSubsetKind::MaskOnes, ExpertSubsetKind::MaskZeros}) {
            add(17, subset, true, false, DType::BF16, MoeEpilogue::AddResidual, true);
        }
        for (auto subset : {ExpertSubsetKind::MaskOnes, ExpertSubsetKind::MaskZeros}) {
            add(129, subset, false, false, DType::FP32, MoeEpilogue::Write, true);
            add(1, subset, true, false, DType::BF16, MoeEpilogue::AddResidual, true);
        }
        add(1, ExpertSubsetKind::None, true, false, DType::FP32, MoeEpilogue::Write, true);
        for (int m : {1, 65}) {
            add(m, ExpertSubsetKind::None, false, false, DType::FP32, MoeEpilogue::Write, true);
        }
        for (int m : {512, 1024}) {
            add(m, ExpertSubsetKind::All, false, false, DType::BF16, MoeEpilogue::Write, true);
            add(m, ExpertSubsetKind::MaskZeros, true, false, DType::FP32, MoeEpilogue::Write, true);
        }
        add(2048, ExpertSubsetKind::None, true, false, DType::FP32, MoeEpilogue::Write, false);
        if (fp8) {
            add(2, ExpertSubsetKind::MaskZeros, true, false, DType::BF16, MoeEpilogue::AddResidual,
                true);
        }
        for (int m : {1, 2, 7, 8, 9, 65, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 4096, 4097}) {
            add(m, ExpertSubsetKind::All, true, true, DType::BF16, MoeEpilogue::Write, true);
        }
        special("shared_only", zero_expert_cases, {{"execution", "graph"}});
        suite.add(std::move(catalogue), [=](auto selected) {
            DeviceContext device;
            Fixture fixture(fp8);
            int failures = 0;
            for (auto i : selected) failures += runners[i](fixture, device.execution_view());
            return failures;
        });
    }
    return suite.main(argc, argv);
}
