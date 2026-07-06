// Correctness for sample_column: greedy (temperature<=0) must equal argmax
// exactly (bf16-rounded logits, lowest-index tie-break), and the sampling path
// must respect top-k/top-p/min-p truncation, be reproducible under a fixed
// seed, and match the softmax distribution it claims to draw from.
#include "qus/kernels/sampling.h"
#include "kernels/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using namespace qus;
using namespace qus::test;

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
    DBuf dout    = to_device_i32(std::vector<int>(static_cast<std::size_t>(cols), -1));
    DBuf dpos    = device_pos(pos_start);
    DeviceConfig dcfg(cfg);
    Tensor tlogits(dlogits.p, DType::BF16, {vocab, cols});
    Tensor tout(dout.p, DType::I32, {cols});
    kernels::sample_column(tlogits, tout, dcfg.ptr(), static_cast<const std::int32_t*>(dpos.p),
                           purpose, nullptr);
    cudaDeviceSynchronize();
    return from_device_i32(dout, static_cast<std::size_t>(cols));
}

// --- greedy == argmax --------------------------------------------------------
int greedy_matches_argmax(const char* tag, int vocab, int cols, std::uint32_t seed) {
    const auto n = static_cast<std::size_t>(vocab) * cols;
    std::vector<float> logits(n);
    fill_uniform(logits, seed, -9.0f, 9.0f);
    // Force ties so the lowest-index tie-break is exercised.
    for (int t = 0; t < cols && vocab > 1; ++t) {
        const int b   = t * vocab;
        const int a   = (5 + t * 97) % vocab;
        int bb        = vocab - 1 - ((11 + t * 131) % vocab);
        if (bb == a) { bb = (a + 1) % vocab; }
        logits[b + a]  = 24.0f + static_cast<float>(t);
        logits[b + bb] = 24.0f + static_cast<float>(t);
    }
    round_to_bf16(logits);

    std::vector<int> ref(static_cast<std::size_t>(cols));
    for (int t = 0; t < cols; ++t) {
        const int b     = t * vocab;
        int best        = 0;
        float best_val  = logits[b];
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
    kernels::SamplingConfig cfg;  // temperature 0 => greedy
    DeviceConfig dcfg(cfg);
    Tensor tlogits(dlogits.p, DType::BF16, {vocab, cols});
    Tensor tout(dout.p, DType::I32, {cols});
    kernels::sample_column(tlogits, tout, dcfg.ptr(), static_cast<const std::int32_t*>(dpos.p),
                           kernels::kSamplePurposeDecode, nullptr);
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
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return base[a] > base[b]; });
    std::vector<char> allowed(vocab, 0);
    for (int i = 0; i < 4; ++i) { allowed[order[i]] = 1; }

    kernels::SamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k       = 4;
    cfg.seed        = 7u;
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
    cfg.temperature = 1.0f;
    cfg.top_p       = static_cast<float>(top_p);
    cfg.seed        = 11u;
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
    cfg.temperature = 1.0f;
    cfg.min_p       = static_cast<float>(min_p);
    cfg.seed        = 13u;
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
    cfg.temperature = 0.8f;
    cfg.seed        = 42u;
    std::vector<int> a = sample_many(base, 2000, cfg, kernels::kSamplePurposeDecode, 0);
    std::vector<int> b = sample_many(base, 2000, cfg, kernels::kSamplePurposeDecode, 0);
    if (a != b) {
        std::cerr << "reproducible: identical seed/pos produced different tokens\n";
        return 1;
    }

    kernels::SamplingConfig cfg2 = cfg;
    cfg2.seed                    = 43u;
    std::vector<int> c = sample_many(base, 2000, cfg2, kernels::kSamplePurposeDecode, 0);
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
    cfg.temperature = 1.0f;  // full distribution: no truncation
    cfg.seed        = 2024u;
    const int N        = 60000;
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
    f += top_k_subset();
    f += top_p_subset();
    f += min_p_subset();
    f += reproducible();
    f += distribution_match();

    std::cout << (f ? "FAIL" : "OK") << " sample_column correctness\n";
    return f ? 1 : 0;
}
