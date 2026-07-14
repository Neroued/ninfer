// Correctness + coverage for argmax, against the frozen op-test standard
// (docs/kernel-development.md): exact i32 index match from bf16-rounded
// logits, with lowest-index tie-break and no tolerance preset.
#include "ninfer/kernels/argmax.h"
#include "kernels/op_tester.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

static void cpu_argmax(const std::vector<float>& logits, int vocab, int t_count,
                       std::vector<int>& out) {
    for (int t = 0; t < t_count; ++t) {
        const int base = t * vocab;
        int best_idx = 0;
        float best_value = logits[base];
        for (int v = 1; v < vocab; ++v) {
            const float value = logits[base + v];
            if (value > best_value) {
                best_value = value;
                best_idx = v;
            }
        }
        out[t] = best_idx;
    }
}

static void install_ties(std::vector<float>& logits, int vocab, int t_count) {
    for (int t = 0; t < t_count; ++t) {
        const int base = t * vocab;
        if (vocab == 1) {
            logits[base] = -3.5f - static_cast<float>(t);
            continue;
        }

        int a = (5 + t * 97) % vocab;
        int b = vocab - 1 - ((11 + t * 131) % vocab);
        if (a == b) { b = (a + 1) % vocab; }
        if (b < a) {
            const int tmp = a;
            a = b;
            b = tmp;
        }

        const float tie_value = 24.0f + static_cast<float>(t);
        logits[base + a] = tie_value;
        logits[base + b] = tie_value;
    }
}

static int verify_i32(const char* label, const std::vector<int>& got,
                      const std::vector<int>& ref) {
    if (got.size() != ref.size()) {
        std::cerr << label << ": size mismatch got=" << got.size() << " ref=" << ref.size()
                  << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != ref[i]) {
            std::cerr << label << ": mismatch at output[" << i << "] got=" << got[i]
                      << " ref=" << ref[i] << '\n';
            return 1;
        }
    }
    std::cout << "    " << label << " exact match\n";
    return 0;
}

static int one_shape(const char* tag, int vocab, int t_count, std::uint32_t seed) {
    const auto n = static_cast<std::size_t>(vocab) * static_cast<std::size_t>(t_count);
    std::vector<float> logits(n);
    fill_uniform(logits, seed, -9.0f, 9.0f);
    install_ties(logits, vocab, t_count);
    round_to_bf16(logits);

    std::vector<int> ref(static_cast<std::size_t>(t_count));
    cpu_argmax(logits, vocab, t_count, ref);

    DBuf dlogits = to_device_bf16(logits);
    DBuf dout = to_device_i32(std::vector<int>(static_cast<std::size_t>(t_count), -777));
    Tensor tlogits(dlogits.p, DType::BF16, {vocab, t_count});
    Tensor tout(dout.p, DType::I32, {t_count});

    kernels::argmax(tlogits, tout, nullptr);
    cudaDeviceSynchronize();

    return verify_i32(tag, from_device_i32(dout, static_cast<std::size_t>(t_count)), ref);
}

template <typename Fn>
static int expect_invalid(const char* label, const Fn& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return 0;
    } catch (const std::exception& e) {
        std::cerr << label << ": expected invalid_argument, got " << e.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

template <typename Fn>
static int expect_overflow(const char* label, const Fn& fn) {
    try {
        fn();
    } catch (const std::overflow_error&) {
        return 0;
    } catch (const std::exception& e) {
        std::cerr << label << ": expected overflow_error, got " << e.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected overflow_error\n";
    return 1;
}

static int validation_checks() {
    int f = 0;

    std::vector<float> logits_h(4, -1.0f);
    round_to_bf16(logits_h);
    DBuf dlogits = to_device_bf16(logits_h);
    DBuf dout = to_device_i32(std::vector<int>(2, -1));
    Tensor logits(dlogits.p, DType::BF16, {2, 2});
    Tensor out(dout.p, DType::I32, {2});

    f += expect_invalid("argmax validation logits dtype", [&] {
        Tensor bad = logits;
        bad.dtype = DType::FP32;
        kernels::argmax(bad, out, nullptr);
    });
    f += expect_invalid("argmax validation out dtype", [&] {
        Tensor bad = out;
        bad.dtype = DType::BF16;
        kernels::argmax(logits, bad, nullptr);
    });
    f += expect_invalid("argmax validation out shape", [&] {
        Tensor bad(dout.p, DType::I32, {3});
        kernels::argmax(logits, bad, nullptr);
    });
    f += expect_invalid("argmax validation logits rank", [&] {
        Tensor bad(dlogits.p, DType::BF16, {2, 2, 2});
        kernels::argmax(bad, out, nullptr);
    });
    f += expect_invalid("argmax validation out rank", [&] {
        Tensor bad(dout.p, DType::I32, {2, 2});
        kernels::argmax(logits, bad, nullptr);
    });
    f += expect_invalid("argmax validation logits contiguous", [&] {
        Tensor bad = logits;
        bad.nb[0] = 4;
        kernels::argmax(bad, out, nullptr);
    });
    f += expect_invalid("argmax validation out contiguous", [&] {
        Tensor bad = out;
        bad.nb[0] = 8;
        kernels::argmax(logits, bad, nullptr);
    });
    f += expect_invalid("argmax validation null data", [&] {
        Tensor null_logits(nullptr, DType::BF16, {2, 1});
        Tensor null_out(nullptr, DType::I32, {1});
        kernels::argmax(null_logits, null_out, nullptr);
    });
    f += expect_invalid("argmax validation negative dim", [&] {
        Tensor bad = logits;
        bad.ne[0] = -1;
        kernels::argmax(bad, out, nullptr);
    });
    f += expect_invalid("argmax validation zero vocab", [&] {
        Tensor zero_logits(nullptr, DType::BF16, {1});
        Tensor zero_out(nullptr, DType::I32, {1});
        zero_logits.ne[0] = 0;
        zero_logits.ne[1] = 0;
        zero_out.ne[0] = 0;
        kernels::argmax(zero_logits, zero_out, nullptr);
    });
    f += expect_overflow("argmax validation overflow", [&] {
        Tensor huge_logits(nullptr, DType::BF16, {1});
        Tensor huge_out(nullptr, DType::I32, {1});
        for (int d = 0; d < 4; ++d) {
            huge_logits.ne[d] = std::numeric_limits<std::int32_t>::max();
            huge_out.ne[d] = std::numeric_limits<std::int32_t>::max();
        }
        kernels::argmax(huge_logits, huge_out, nullptr);
    });

    try {
        Tensor zero_logits(nullptr, DType::BF16, {1});
        Tensor zero_out(nullptr, DType::I32, {1});
        zero_logits.ne[0] = 7;
        zero_logits.ne[1] = 0;
        zero_logits.ne[2] = 1;
        zero_logits.ne[3] = 1;
        zero_out.ne[0] = 0;
        zero_out.ne[1] = 1;
        zero_out.ne[2] = 1;
        zero_out.ne[3] = 1;
        kernels::argmax(zero_logits, zero_out, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "argmax validation zero T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    return f;
}

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int f = 0;
    f += validation_checks();

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_shape("argmax [1,1]", 1, 1, seed);
        f += one_shape("argmax [1,3]", 1, 3, seed);
        f += one_shape("argmax [257,1]", 257, 1, seed);
        f += one_shape("argmax [257,3]", 257, 3, seed);
        f += one_shape("argmax [248320,1]", 248320, 1, seed);
        f += one_shape("argmax [248320,3]", 248320, 3, seed);
    }

    std::cout << (f ? "FAIL" : "OK") << " argmax correctness\n";
    return f ? 1 : 0;
}
