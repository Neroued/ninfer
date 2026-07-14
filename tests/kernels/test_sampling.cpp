// Correctness for sample: greedy (temperature<=0) must equal argmax
// exactly (bf16-rounded logits, lowest-index tie-break), and the sampling path
// must respect top-k/top-p/min-p truncation, be reproducible under a fixed
// seed, and match the softmax distribution it claims to draw from.
#include "targets/qwen3_6_27b_rtx5090/impl/kernels/sampling/sampling.h"
#include "kernels/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct DeviceConfig {
    DBuf buf;

    explicit DeviceConfig(const kernels::SamplingConfig& cfg)
        : buf(sizeof(kernels::SamplingConfig)) {
        cudaMemcpy(buf.p, &cfg, sizeof(cfg), cudaMemcpyHostToDevice);
    }

    const kernels::SamplingConfig* ptr() const {
        return static_cast<const kernels::SamplingConfig*>(buf.p);
    }
};

DBuf device_pos(int value) {
    DBuf d(sizeof(std::int32_t));
    cudaMemcpy(d.p, &value, sizeof(value), cudaMemcpyHostToDevice);
    return d;
}

// Column-major [vocab, cols] with every column equal to `base`.
std::vector<float> broadcast_columns(const std::vector<float>& base, int cols) {
    const int vocab = static_cast<int>(base.size());
    std::vector<float> out(static_cast<std::size_t>(vocab) * cols);
    for (int t = 0; t < cols; ++t) {
        for (int v = 0; v < vocab; ++v) { out[static_cast<std::size_t>(t) * vocab + v] = base[v]; }
    }
    return out;
}

std::vector<double> softmax(const std::vector<float>& logits) {
    double m = logits[0];
    for (float x : logits) { m = std::max(m, static_cast<double>(x)); }
    std::vector<double> e(logits.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        e[i] = std::exp(static_cast<double>(logits[i]) - m);
        sum += e[i];
    }
    for (double& x : e) { x /= sum; }
    return e;
}

std::vector<int> sample_many(const std::vector<float>& base, int cols,
                             const kernels::SamplingConfig& cfg, std::int32_t purpose,
                             int pos_start) {
    const int vocab             = static_cast<int>(base.size());
    std::vector<float> logits_h = broadcast_columns(base, cols);
    DBuf dlogits                = to_device_bf16(logits_h);
    DBuf dout = to_device_i32(std::vector<int>(static_cast<std::size_t>(cols), -1));
    DBuf dpos = device_pos(pos_start);
    DeviceConfig dcfg(cfg);
    Tensor tlogits(dlogits.p, DType::BF16, {vocab, cols});
    Tensor tout(dout.p, DType::I32, {cols});
    kernels::sample(tlogits, tout, vocab, dcfg.ptr(),
                    static_cast<const std::int32_t*>(dpos.p), purpose, nullptr);
    cudaDeviceSynchronize();
    return from_device_i32(dout, static_cast<std::size_t>(cols));
}

std::vector<int> sample_many_batched(const std::vector<float>& base, int total, int cols,
                                     kernels::SamplingConfig cfg, std::int32_t purpose,
                                     int pos_start, bool counts_active) {
    const int vocab             = static_cast<int>(base.size());
    std::vector<float> logits_h = broadcast_columns(base, cols);
    DBuf dlogits                = to_device_bf16(logits_h);
    DBuf dout = to_device_i32(std::vector<int>(static_cast<std::size_t>(cols), -1));
    DBuf dpos = device_pos(pos_start);
    DBuf dcollect(static_cast<std::size_t>(total) * sizeof(std::int32_t));
    DBuf dcounts(static_cast<std::size_t>(vocab) * sizeof(std::int32_t));
    cudaMemset(dcounts.p, 0, dcounts.bytes);
    if (counts_active) { cfg.token_counts = static_cast<std::int32_t*>(dcounts.p); }
    DeviceConfig dcfg(cfg);
    Tensor tlogits(dlogits.p, DType::BF16, {vocab, cols});
    Tensor tout(dout.p, DType::I32, {cols});

    int produced = 0;
    while (produced < total) {
        const int batch = std::min(cols, total - produced);
        const int pos   = pos_start + produced;
        cudaMemcpy(dpos.p, &pos, sizeof(pos), cudaMemcpyHostToDevice);
        kernels::sample(tlogits, tout, vocab, dcfg.ptr(),
                        static_cast<const std::int32_t*>(dpos.p), purpose, nullptr);
        cudaMemcpyAsync(static_cast<std::int32_t*>(dcollect.p) + produced, dout.p,
                        static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                        cudaMemcpyDeviceToDevice, nullptr);
        produced += batch;
    }
    cudaDeviceSynchronize();
    return from_device_i32(dcollect, static_cast<std::size_t>(total));
}

// --- greedy == argmax --------------------------------------------------------
int greedy_matches_argmax(const char* tag, int vocab, int cols, std::uint32_t seed) {
    const auto n = static_cast<std::size_t>(vocab) * cols;
    std::vector<float> logits(n);
    fill_uniform(logits, seed, -9.0f, 9.0f);
    // Force ties so the lowest-index tie-break is exercised.
    for (int t = 0; t < cols && vocab > 1; ++t) {
        const int b = t * vocab;
        const int a = (5 + t * 97) % vocab;
        int bb      = vocab - 1 - ((11 + t * 131) % vocab);
        if (bb == a) { bb = (a + 1) % vocab; }
        logits[b + a]  = 24.0f + static_cast<float>(t);
        logits[b + bb] = 24.0f + static_cast<float>(t);
    }
    round_to_bf16(logits);

    std::vector<int> ref(static_cast<std::size_t>(cols));
    for (int t = 0; t < cols; ++t) {
        const int b    = t * vocab;
        int best       = 0;
        float best_val = logits[b];
        for (int v = 1; v < vocab; ++v) {
            if (logits[b + v] > best_val) {
                best_val = logits[b + v];
                best     = v;
            }
        }
        ref[t] = best;
    }

    DBuf dlogits = to_device_bf16(logits);
    DBuf dout    = to_device_i32(std::vector<int>(static_cast<std::size_t>(cols), -1));
    DBuf dpos    = device_pos(0);
    kernels::SamplingConfig cfg; // temperature 0 => greedy
    DeviceConfig dcfg(cfg);
    Tensor tlogits(dlogits.p, DType::BF16, {vocab, cols});
    Tensor tout(dout.p, DType::I32, {cols});
    kernels::sample(tlogits, tout, vocab, dcfg.ptr(),
                    static_cast<const std::int32_t*>(dpos.p), kernels::kSamplePurposeDecode,
                    nullptr);
    cudaDeviceSynchronize();
    std::vector<int> got = from_device_i32(dout, static_cast<std::size_t>(cols));

    for (int t = 0; t < cols; ++t) {
        if (got[t] != ref[t]) {
            std::cerr << tag << ": greedy mismatch at col " << t << " got=" << got[t]
                      << " ref=" << ref[t] << '\n';
            return 1;
        }
    }
    std::cout << "    " << tag << " greedy==argmax\n";
    return 0;
}

int physical_stride_and_token_domain(int cols, bool stochastic) {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * cols, -20.0f);
    std::vector<int> expected(static_cast<std::size_t>(cols));
    for (int col = 0; col < cols; ++col) {
        const int best         = (17 + col * 7919) % token_domain;
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        expected[static_cast<std::size_t>(col)] = best;
        logits[base + best]                      = 20.0f + col;
        logits[base + token_domain]              = 100.0f + col;
        logits[base + physical_rows - 1]         = 200.0f + col;
    }
    round_to_bf16(logits);

    DBuf dlogits = to_device_bf16(logits);
    DBuf dout    = to_device_i32(std::vector<int>(static_cast<std::size_t>(cols), -1));
    DBuf dpos    = device_pos(0);
    kernels::SamplingConfig cfg;
    cfg.temperature = stochastic ? 1.0f : 0.0f;
    cfg.top_k       = 1;
    DeviceConfig dcfg(cfg);
    Tensor tlogits(dlogits.p, DType::BF16, {physical_rows, cols});
    Tensor tout(dout.p, DType::I32, {cols});
    kernels::sample(tlogits, tout, token_domain, dcfg.ptr(),
                    static_cast<const std::int32_t*>(dpos.p),
                    kernels::kSamplePurposeDecode, nullptr);
    cudaDeviceSynchronize();
    const std::vector<int> got = from_device_i32(dout, static_cast<std::size_t>(cols));
    if (got != expected) {
        std::cerr << "physical_stride_and_token_domain: "
                  << (stochastic ? "top-k" : "greedy") << " mismatch for cols=" << cols
                  << '\n';
        return 1;
    }
    std::cout << "    sample physical stride + token domain cols=" << cols << ' '
              << (stochastic ? "top-k" : "greedy") << " ok\n";
    return 0;
}

// --- top-k subset ------------------------------------------------------------
int top_k_subset() {
    const int vocab = 32;
    std::vector<float> base(vocab);
    fill_uniform(base, 123u, -2.0f, 2.0f);
    // Make a clean top-4 well above the rest so the boundary is unambiguous.
    for (int i = 0; i < 4; ++i) { base[(i * 7 + 3) % vocab] = 8.0f + static_cast<float>(i); }
    round_to_bf16(base);

    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return base[a] > base[b]; });
    std::vector<char> allowed(vocab, 0);
    for (int i = 0; i < 4; ++i) { allowed[order[i]] = 1; }

    kernels::SamplingConfig cfg;
    cfg.temperature      = 1.0f;
    cfg.top_k            = 4;
    cfg.seed             = 7u;
    std::vector<int> got = sample_many(base, 4000, cfg, kernels::kSamplePurposeDecode, 0);
    for (int tok : got) {
        if (tok < 0 || tok >= vocab || !allowed[tok]) {
            std::cerr << "top_k_subset: sampled out-of-set token " << tok << '\n';
            return 1;
        }
    }
    std::cout << "    top_k subset ok\n";
    return 0;
}

// --- top-p nucleus subset ----------------------------------------------------
int top_p_subset() {
    const int vocab = 32;
    std::vector<float> base(vocab);
    fill_uniform(base, 321u, -3.0f, 3.0f);
    round_to_bf16(base);

    const std::vector<double> p = softmax(base);
    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });

    const double top_p = 0.9;
    std::vector<char> allowed(vocab, 0);
    double cum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        allowed[order[i]] = 1;
        cum += p[order[i]];
        if (cum >= top_p) { break; }
    }

    kernels::SamplingConfig cfg;
    cfg.temperature      = 1.0f;
    cfg.top_p            = static_cast<float>(top_p);
    cfg.seed             = 11u;
    std::vector<int> got = sample_many(base, 4000, cfg, kernels::kSamplePurposeDecode, 0);
    for (int tok : got) {
        if (tok < 0 || tok >= vocab || !allowed[tok]) {
            std::cerr << "top_p_subset: sampled out-of-nucleus token " << tok << '\n';
            return 1;
        }
    }
    std::cout << "    top_p nucleus subset ok\n";
    return 0;
}

// --- min-p subset ------------------------------------------------------------
int min_p_subset() {
    const int vocab = 32;
    std::vector<float> base(vocab);
    fill_uniform(base, 555u, -3.0f, 3.0f);
    round_to_bf16(base);

    const std::vector<double> p = softmax(base);
    const double pmax           = *std::max_element(p.begin(), p.end());
    const double min_p          = 0.3;
    std::vector<char> allowed(vocab, 0);
    for (int v = 0; v < vocab; ++v) {
        if (p[v] >= min_p * pmax) { allowed[v] = 1; }
    }

    kernels::SamplingConfig cfg;
    cfg.temperature      = 1.0f;
    cfg.min_p            = static_cast<float>(min_p);
    cfg.seed             = 13u;
    std::vector<int> got = sample_many(base, 4000, cfg, kernels::kSamplePurposeDecode, 0);
    for (int tok : got) {
        if (tok < 0 || tok >= vocab || !allowed[tok]) {
            std::cerr << "min_p_subset: sampled out-of-set token " << tok << '\n';
            return 1;
        }
    }
    std::cout << "    min_p subset ok\n";
    return 0;
}

// --- reproducibility ---------------------------------------------------------
int reproducible() {
    const int vocab = 16;
    std::vector<float> base(vocab);
    fill_uniform(base, 202u, -3.0f, 3.0f);
    round_to_bf16(base);

    kernels::SamplingConfig cfg;
    cfg.temperature    = 0.8f;
    cfg.seed           = 42u;
    std::vector<int> a = sample_many(base, 2000, cfg, kernels::kSamplePurposeDecode, 0);
    std::vector<int> b = sample_many(base, 2000, cfg, kernels::kSamplePurposeDecode, 0);
    if (a != b) {
        std::cerr << "reproducible: identical seed/pos produced different tokens\n";
        return 1;
    }

    kernels::SamplingConfig cfg2 = cfg;
    cfg2.seed                    = 43u;
    std::vector<int> c           = sample_many(base, 2000, cfg2, kernels::kSamplePurposeDecode, 0);
    if (a == c) {
        std::cerr << "reproducible: different seed produced identical stream\n";
        return 1;
    }
    std::cout << "    reproducibility ok\n";
    return 0;
}

// --- distribution match ------------------------------------------------------
int distribution_match() {
    const int vocab = 8;
    std::vector<float> base(vocab);
    fill_uniform(base, 909u, -2.0f, 2.0f);
    round_to_bf16(base);
    const std::vector<double> p = softmax(base);

    kernels::SamplingConfig cfg;
    cfg.temperature      = 1.0f; // full distribution: no truncation
    cfg.seed             = 2024u;
    const int N          = 60000;
    std::vector<int> got = sample_many(base, N, cfg, kernels::kSamplePurposeDecode, 0);

    std::vector<double> freq(vocab, 0.0);
    for (int tok : got) {
        if (tok < 0 || tok >= vocab) {
            std::cerr << "distribution_match: token out of range " << tok << '\n';
            return 1;
        }
        freq[tok] += 1.0;
    }
    for (double& f : freq) { f /= static_cast<double>(N); }

    double max_abs = 0.0;
    for (int v = 0; v < vocab; ++v) { max_abs = std::max(max_abs, std::abs(freq[v] - p[v])); }
    if (max_abs > 0.02) {
        std::cerr << "distribution_match: empirical/target gap " << max_abs << " too large\n";
        for (int v = 0; v < vocab; ++v) {
            std::cerr << "      v=" << v << " freq=" << freq[v] << " p=" << p[v] << '\n';
        }
        return 1;
    }
    std::cout << "    distribution match ok (max abs diff " << max_abs << ")\n";
    return 0;
}

int real_shape_distribution_match() {
    const int vocab = 248320;
    std::vector<float> base(vocab, -20.0f);
    const int ids[]    = {17, 7919, 65537, 200003};
    const float vals[] = {3.0f, 2.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) { base[ids[i]] = vals[i]; }
    round_to_bf16(base);

    std::vector<float> support(vals, vals + 4);
    const std::vector<double> p = softmax(support);

    kernels::SamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k       = 4;
    cfg.seed        = 20260706u;
    const int N     = 4096;
    std::vector<int> got =
        sample_many_batched(base, N, 8, cfg, kernels::kSamplePurposeDecode, 1000, false);

    std::vector<double> freq(4, 0.0);
    for (int tok : got) {
        int slot = -1;
        for (int i = 0; i < 4; ++i) {
            if (tok == ids[i]) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            std::cerr << "real_shape_distribution: token out of support " << tok << '\n';
            return 1;
        }
        freq[slot] += 1.0;
    }
    for (double& f : freq) { f /= static_cast<double>(N); }

    double max_abs = 0.0;
    for (int i = 0; i < 4; ++i) { max_abs = std::max(max_abs, std::abs(freq[i] - p[i])); }
    if (max_abs > 0.04) {
        std::cerr << "real_shape_distribution: empirical/target gap " << max_abs << " too large\n";
        for (int i = 0; i < 4; ++i) {
            std::cerr << "      token=" << ids[i] << " freq=" << freq[i] << " p=" << p[i] << '\n';
        }
        return 1;
    }
    std::cout << "    real-shape distribution match ok (max abs diff " << max_abs << ")\n";
    return 0;
}

int real_shape_reproducible_counts_active() {
    const int vocab = 248320;
    std::vector<float> base(vocab, -18.0f);
    for (int i = 0; i < 20; ++i) {
        base[(17 + i * 7919) % vocab] = 4.0f - 0.08f * static_cast<float>(i);
    }
    round_to_bf16(base);

    kernels::SamplingConfig cfg;
    cfg.temperature      = 0.6f;
    cfg.top_k            = 20;
    cfg.top_p            = 0.95f;
    cfg.presence_penalty = 1.0f;
    cfg.seed             = 123456u;
    const int N          = 1024;
    std::vector<int> a =
        sample_many_batched(base, N, 8, cfg, kernels::kSamplePurposeDecode, 2000, true);
    std::vector<int> b =
        sample_many_batched(base, N, 8, cfg, kernels::kSamplePurposeDecode, 2000, true);
    if (a != b) {
        std::cerr << "real_shape_reproducible: identical seed/count path diverged\n";
        return 1;
    }
    kernels::SamplingConfig cfg2 = cfg;
    cfg2.seed                    = 123457u;
    std::vector<int> c =
        sample_many_batched(base, N, 8, cfg2, kernels::kSamplePurposeDecode, 2000, true);
    if (a == c) {
        std::cerr << "real_shape_reproducible: different seed produced identical stream\n";
        return 1;
    }
    std::cout << "    real-shape reproducibility with counts ok\n";
    return 0;
}

// cols==1 stochastic decode path at real vocab. This is the single-column path
// (formerly the fused kernel, now the shared two-launch partial+group merge) that
// no earlier test exercised, which is how F1 slipped through.
int real_shape_col1_distribution_match() {
    const int vocab = 248320;
    std::vector<float> base(vocab, -20.0f);
    const int ids[]    = {17, 7919, 65537, 200003};
    const float vals[] = {3.0f, 2.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) { base[ids[i]] = vals[i]; }
    round_to_bf16(base);

    std::vector<float> support(vals, vals + 4);
    const std::vector<double> p = softmax(support);

    kernels::SamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k       = 20;
    cfg.seed        = 20260707u;
    const int N     = 4096;
    std::vector<int> got =
        sample_many_batched(base, N, 1, cfg, kernels::kSamplePurposeDecode, 1000, false);

    std::vector<double> freq(4, 0.0);
    for (int tok : got) {
        int slot = -1;
        for (int i = 0; i < 4; ++i) {
            if (tok == ids[i]) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            std::cerr << "real_shape_col1_distribution: token out of support " << tok << '\n';
            return 1;
        }
        freq[slot] += 1.0;
    }
    for (double& f : freq) { f /= static_cast<double>(N); }

    double max_abs = 0.0;
    for (int i = 0; i < 4; ++i) { max_abs = std::max(max_abs, std::abs(freq[i] - p[i])); }
    if (max_abs > 0.04) {
        std::cerr << "real_shape_col1_distribution: empirical/target gap " << max_abs
                  << " too large\n";
        for (int i = 0; i < 4; ++i) {
            std::cerr << "      token=" << ids[i] << " freq=" << freq[i] << " p=" << p[i] << '\n';
        }
        return 1;
    }
    std::cout << "    real-shape cols=1 distribution match ok (max abs diff " << max_abs << ")\n";
    return 0;
}

// F1 regression: top_k above the internal cap (or <= 0) must clamp to 20 and
// yield the identical, correct distribution rather than collapsing onto one
// token. Before the fix, top_k=64/0 overflowed the merge tile at real vocab and
// the cols==1 path collapsed entirely onto id 17.
int topk_clamp_equivalence() {
    const int vocab = 248320;
    std::vector<float> base(vocab, -20.0f);
    const int ids[]    = {17, 7919, 65537, 200003};
    const float vals[] = {3.0f, 2.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) { base[ids[i]] = vals[i]; }
    round_to_bf16(base);

    std::vector<float> support(vals, vals + 4);
    const std::vector<double> p = softmax(support);

    const int N       = 4096;
    const int topks[] = {20, 64, 0};
    std::vector<std::vector<int>> streams;
    for (int topk : topks) {
        kernels::SamplingConfig cfg;
        cfg.temperature = 1.0f;
        cfg.top_k       = topk;
        cfg.seed        = 424242u;
        streams.push_back(
            sample_many_batched(base, N, 1, cfg, kernels::kSamplePurposeDecode, 500, false));
    }

    for (std::size_t s = 1; s < streams.size(); ++s) {
        if (streams[s] != streams[0]) {
            std::cerr << "topk_clamp_equivalence: top_k=" << topks[s]
                      << " stream differs from top_k=20 (clamp not applied)\n";
            return 1;
        }
    }

    std::vector<double> freq(4, 0.0);
    for (int tok : streams[0]) {
        int slot = -1;
        for (int i = 0; i < 4; ++i) {
            if (tok == ids[i]) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            std::cerr << "topk_clamp_equivalence: token out of support " << tok << '\n';
            return 1;
        }
        freq[slot] += 1.0;
    }
    for (double& f : freq) { f /= static_cast<double>(N); }

    if (freq[0] > 0.95) {
        std::cerr << "topk_clamp_equivalence: distribution collapsed onto token " << ids[0]
                  << " (freq=" << freq[0] << ")\n";
        return 1;
    }
    double max_abs = 0.0;
    for (int i = 0; i < 4; ++i) { max_abs = std::max(max_abs, std::abs(freq[i] - p[i])); }
    if (max_abs > 0.04) {
        std::cerr << "topk_clamp_equivalence: empirical/target gap " << max_abs << " too large\n";
        return 1;
    }
    std::cout << "    top_k clamp equivalence ok (max abs diff " << max_abs << ")\n";
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int f = 0;
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += greedy_matches_argmax("sample [257,1]", 257, 1, seed);
        f += greedy_matches_argmax("sample [257,3]", 257, 3, seed);
        f += greedy_matches_argmax("sample [248320,1]", 248320, 1, seed);
    }
    f += physical_stride_and_token_domain(1, false);
    f += physical_stride_and_token_domain(1, true);
    f += physical_stride_and_token_domain(3, false);
    f += physical_stride_and_token_domain(3, true);
    f += top_k_subset();
    f += top_p_subset();
    f += min_p_subset();
    f += reproducible();
    f += distribution_match();
    f += real_shape_distribution_match();
    f += real_shape_col1_distribution_match();
    f += topk_clamp_equivalence();
    f += real_shape_reproducible_counts_active();

    std::cout << (f ? "FAIL" : "OK") << " sample correctness\n";
    return f ? 1 : 0;
}
