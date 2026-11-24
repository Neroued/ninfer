#include "support.h"
#include "cuda_fixture.h"
#include "ninfer/ops/weight_input.h"
#include <algorithm>

namespace ninfer::test::gdn_control {
std::size_t workspace_capacity(const ops::ProjectionWeights& w, bool normalize_input, int first,
                               int last) {
    return normalize_input ? ops::gdn_norm_control_proj_workspace_capacity_bytes(w, first, last)
                           : ops::gdn_control_proj_workspace_capacity_bytes(w, first, last);
}

std::string label(Profile p, Case c, bool normalize_input) {
    constexpr std::array names{"random",           "zero-input",    "zero-A",     "zero-B",
                               "sparse",           "cancellation",  "gate-edges", "tiny-norm-input",
                               "large-norm-input", "zero-norm-gain"};
    return std::string(normalize_input ? "gdn_norm_control_proj" : "gdn_control_proj") +
           " H=" + std::to_string(p.heads) + " K=" + std::to_string(p.input_rows) +
           (p.storage == Storage::Parent ? " parent" : " pair") +
           " M=" + std::to_string(c.columns) + " " + names[static_cast<std::size_t>(c.pattern)] +
           (c.execution == Execution::Graph ? " graph" : " eager") +
           (c.prefix ? " arena-prefix" : "");
}

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> outputs(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) outputs[i] = f32_to_bf16(values[i]);
    return outputs;
}

Weights::Weights(Profile p, Pattern pattern)
    : profile(p), a(std::size_t(p.heads) * p.input_rows), b(a.size()),
      first(ninfer::test::cuda_environment().context.device(),
            a.size() * 2 * (p.storage == Storage::Parent ? 2 : 1)),
      second(ninfer::test::cuda_environment().context.device(),
             p.storage == Storage::Pair ? a.size() * 2 : 0) {
    fill_uniform(a, 103, -0.015F, 0.015F);
    fill_uniform(b, 105, -0.015F, 0.015F);
    if (pattern == Pattern::ZeroA) std::fill(a.begin(), a.end(), 0);
    if (pattern == Pattern::ZeroB) std::fill(b.begin(), b.end(), 0);
    if (pattern == Pattern::Sparse) {
        std::fill(a.begin(), a.end(), 0);
        std::fill(b.begin(), b.end(), 0);
        for (int head = 0; head < p.heads; ++head)
            for (int k : {0, 31, 32, 63, 64, 319, 320, 639, 640, 1279, 1280, 2559}) {
                const auto i = std::size_t(head) * p.input_rows + k;
                a[i]         = (head + 1) * ((k & 1) ? -0.03125F : 0.03125F);
                b[i]         = (p.heads - head) * ((k & 1) ? 0.0625F : -0.0625F);
            }
    }
    if (pattern == Pattern::Cancellation)
        for (std::size_t i = 0; i < a.size(); i += 2) {
            a[i + 1] = -a[i];
            b[i + 1] = -b[i];
        }
    if (pattern == Pattern::GateEdges) {
        std::fill(a.begin(), a.end(), 0);
        std::fill(b.begin(), b.end(), 0);
        constexpr std::array edges{-40.F, -12.F, -1.e-4F, 0.F, 1.e-4F, 12.F, 40.F};
        for (int head = 0; head < p.heads; ++head) {
            a[std::size_t(head) * p.input_rows] = edges[head % 7];
            b[std::size_t(head) * p.input_rows] = edges[(head + 3) % 7];
        }
    }
    upload();
    const std::uint64_t count = a.size();
    // Retain coverage of logical projections over a differently shaped contiguous parent.
    // K=2560 profiles use their actual [2H,2560] stored parent.
    const bool reshaped_parent = p.storage == Storage::Parent && p.input_rows != 2560;
    const std::array<std::uint64_t, 2> parent_shape{
        std::uint64_t(reshaped_parent                ? 4 * p.heads
                      : p.storage == Storage::Parent ? 2 * p.heads
                                                     : p.heads),
        std::uint64_t(reshaped_parent ? p.input_rows / 2 : p.input_rows)};
    const WeightParent pa{weight_geometry(QType::BF16, QuantLayout::Contiguous, parent_shape),
                          first.device(), static_cast<const std::byte*>(first.data())};
    const WeightParent pb{pa.geometry, second.device(),
                          static_cast<const std::byte*>(second.data())};
    const WeightView av{{std::uint64_t(p.heads), std::uint64_t(p.input_rows)}, {{&pa, 0, count}}};
    WeightView bv{{std::uint64_t(p.heads), std::uint64_t(p.input_rows)}, {}};
    if (p.storage == Storage::Parent)
        bv.parts.push_back({&pa, count, 2 * count});
    else
        bv.parts.push_back({&pb, 0, count});
    prepared           = ops::prepare_gdn_control_proj_weights({av}, {bv});
    const auto* single = std::get_if<ops::SingleProjectionWeight>(&prepared);
    if ((p.storage == Storage::Parent) != bool(single) ||
        (single && single->weight.qdata != first.data()))
        throw std::runtime_error("control preparation changed the stored representation");
}

void Weights::upload() {
    round_to_bf16(a);
    round_to_bf16(b);
    first_bits_  = bits(a);
    second_bits_ = bits(b);
    if (profile.storage == Storage::Parent)
        first_bits_.insert(first_bits_.end(), second_bits_.begin(), second_bits_.end());
    first.copy_from_host(first_bits_.data(), first_bits_.size() * 2);
    if (profile.storage == Storage::Pair)
        second.copy_from_host(second_bits_.data(), second_bits_.size() * 2);
}

int Weights::check(std::string_view name) const {
    int failed = verify_exact((std::string(name) + " weight immutable").c_str(),
                              from_device<std::uint16_t>(first, first_bits_.size()), first_bits_);
    if (profile.storage == Storage::Pair)
        failed +=
            verify_exact((std::string(name) + " B immutable").c_str(),
                         from_device<std::uint16_t>(second, second_bits_.size()), second_bits_);
    return failed;
}

Fixture::Fixture(Profile p, Case c, bool n)
    : profile(p), test_case(c), normalize_input(n), weights(p, c.pattern),
      capacity(workspace_capacity(weights.prepared, n, c.columns, c.columns)),
      prefix(c.prefix ? 17 : 0), x(std::size_t(p.input_rows) * c.columns), norm_gain(p.input_rows),
      a_log(p.heads), dt_bias(p.heads),
      device_x(ninfer::test::cuda_environment().context.device(), x.size() * 2),
      device_norm_weight(ninfer::test::cuda_environment().context.device(), norm_gain.size() * 2),
      device_a_log(ninfer::test::cuda_environment().context.device(), a_log.size() * 4),
      device_dt_bias(ninfer::test::cuda_environment().context.device(), dt_bias.size() * 4),
      g(std::size_t(p.heads) * c.columns * 4), beta(std::size_t(p.heads) * c.columns * 4),
      normalized(x.size() * 2),
      scratch(std::max<std::size_t>(256, capacity + (c.prefix ? 256 : 0))),
      input(device_x.span(), DType::BF16, {p.input_rows, c.columns}),
      norm_weight(device_norm_weight.span(), DType::BF16, {p.input_rows}),
      a_log_tensor(device_a_log.span(), DType::FP32, {p.heads}),
      dt_bias_tensor(device_dt_bias.span(), DType::FP32, {p.heads}),
      normalized_tensor(normalized.span(), DType::BF16, {p.input_rows, c.columns}),
      outputs{Tensor(g.span(), DType::FP32, {p.heads, c.columns}),
              Tensor(beta.span(), DType::FP32, {p.heads, c.columns})},
      arena(
          scratch.span().subspan(0, std::max<std::size_t>(256, capacity + (c.prefix ? 256 : 0)))) {
    fill_uniform(x, 101, -1.F, 1.F);
    fill_uniform(norm_gain, 107, -0.125F, 0.125F);
    fill_uniform(a_log, 111, -2.F, 1.F);
    fill_uniform(dt_bias, 113, -1.F, 1.F);
    if (normalize_input) {
        norm_gain[0] = -1.F;
        norm_gain[1] = -1.5F;
    }
    if (c.pattern == Pattern::ZeroInput) std::fill(x.begin(), x.end(), 0);
    if (c.pattern == Pattern::TinyNormInput)
        for (auto& v : x) v *= 0.0001220703125F;
    if (c.pattern == Pattern::LargeNormInput)
        for (auto& v : x) v *= 512.F;
    if (c.pattern == Pattern::ZeroNormGain) std::fill(norm_gain.begin(), norm_gain.end(), -1.F);
    if (c.pattern == Pattern::Cancellation)
        for (std::size_t i = 0; i < x.size(); i += 2) x[i + 1] = x[i];
    if (c.pattern == Pattern::GateEdges) {
        constexpr std::array edges{-40.F, -12.F, -1.e-4F, 0.F, 1.e-4F, 12.F, 40.F};
        for (int h = 0; h < p.heads; ++h) {
            a_log[h]   = float(h % 9 - 4);
            dt_bias[h] = edges[(h + 5) % 7];
        }
        for (int t = 0; t < c.columns; ++t) x[std::size_t(t) * p.input_rows] = (t & 1) ? -1.F : 1.F;
    }
    upload();
    if (prefix) (void)arena.alloc_bytes(prefix);
    cuda_synchronize();
}

void Fixture::upload() {
    round_to_bf16(x);
    round_to_bf16(norm_gain);
    const auto xb = bits(x), nb = bits(norm_gain);
    device_x.copy_from_host(xb.data(), xb.size() * 2);
    device_norm_weight.copy_from_host(nb.data(), nb.size() * 2);
    device_a_log.copy_from_host(a_log.data(), a_log.size() * 4);
    device_dt_bias.copy_from_host(dt_bias.data(), dt_bias.size() * 4);
}

void Fixture::advance() {
    for (auto& v : x) v *= -0.5F;
    if (normalize_input)
        for (auto& v : norm_gain) v = -v;
    for (auto& v : weights.a) v = -v;
    for (auto& v : weights.b) v *= -0.5F;
    for (auto& v : a_log) v += 0.125F;
    for (auto& v : dt_bias) v = -v;
    weights.upload();
    upload();
}

void Fixture::repaint(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(g.data(), 0xff, std::size_t(profile.heads) * test_case.columns * 4,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(beta.data(), 0xff,
                               std::size_t(profile.heads) * test_case.columns * 4, stream));
    CUDA_CHECK(cudaMemsetAsync(normalized.data(), 0xff, x.size() * 2, stream));
    CUDA_CHECK(cudaMemsetAsync(scratch.data(), 0xff,
                               std::max<std::size_t>(256, capacity + (prefix ? 256 : 0)), stream));
}

void Fixture::invoke(MemoryArena& ws, CudaExecutionView execution) {
    if (normalize_input)
        ops::gdn_norm_control_proj(input, norm_weight, kEps, weights.prepared, a_log_tensor,
                                   dt_bias_tensor, {normalized_tensor, outputs}, ws, execution);
    else
        ops::gdn_control_proj(input, weights.prepared, a_log_tensor, dt_bias_tensor, outputs, ws,
                              execution);
}

int Fixture::check(std::string_view name) const {
    const std::string s(name);
    int failed = weights.check(name);
    failed += verify_exact((s + " x immutable").c_str(),
                           from_device<std::uint16_t>(device_x, x.size()), bits(x));
    failed += verify_exact((s + " norm weight immutable").c_str(),
                           from_device<std::uint16_t>(device_norm_weight, norm_gain.size()),
                           bits(norm_gain));
    failed += verify_exact((s + " a_log immutable").c_str(),
                           from_device<float>(device_a_log, a_log.size()), a_log);
    failed += verify_exact((s + " dt_bias immutable").c_str(),
                           from_device<float>(device_dt_bias, dt_bias.size()), dt_bias);
    failed += g.verify_guards(s + " g");
    failed += beta.verify_guards(s + " beta");
    failed += normalized.verify_guards(s + " normalized");
    failed += scratch.verify_guards(s + " scratch");
    if (prefix)
        failed += verify_exact((s + " arena prefix").c_str(),
                               from_device<std::uint8_t>(scratch.data(), prefix),
                               std::vector<std::uint8_t>(prefix, 0xff));
    return failed;
}

} // namespace ninfer::test::gdn_control
