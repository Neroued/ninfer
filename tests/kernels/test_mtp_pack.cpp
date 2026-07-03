#include "qus/kernels/mtp_pack.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

std::vector<std::uint16_t> pattern(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> out(n);
    std::uint32_t x = seed;
    for (std::size_t i = 0; i < n; ++i) {
        x      = 1664525u * x + 1013904223u;
        out[i] = static_cast<std::uint16_t>(0x3f80u ^ (x >> 16) ^ static_cast<std::uint32_t>(i));
    }
    return out;
}

DBuf to_device_bits(const std::vector<std::uint16_t>& h) {
    DBuf d(h.size() * sizeof(std::uint16_t));
    cudaMemcpy(d.p, h.data(), h.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    return d;
}

std::vector<std::uint16_t> from_device_bits(const DBuf& d, std::size_t n) {
    std::vector<std::uint16_t> out(n);
    cudaMemcpy(out.data(), d.p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return out;
}

int expect_bits(const std::string& label, const std::vector<std::uint16_t>& got,
                const std::vector<std::uint16_t>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << label << ": size mismatch got=" << got.size()
                  << " expected=" << expected.size() << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            std::cerr << label << ": mismatch at " << i << " got=0x" << std::hex << got[i]
                      << " expected=0x" << expected[i] << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

int one_pack_case(int T) {
    constexpr int hidden = 5120;
    constexpr int out_rows = 10240;
    const std::size_t hidden_n = static_cast<std::size_t>(hidden) * T;
    const std::size_t out_n    = static_cast<std::size_t>(out_rows) * T;
    const auto emb             = pattern(hidden_n, 17u + static_cast<std::uint32_t>(T));
    const auto hid             = pattern(hidden_n, 9001u + static_cast<std::uint32_t>(T));
    std::vector<std::uint16_t> expected(out_n);
    for (int t = 0; t < T; ++t) {
        for (int row = 0; row < hidden; ++row) {
            expected[static_cast<std::size_t>(t) * out_rows + row] =
                emb[static_cast<std::size_t>(t) * hidden + row];
            expected[static_cast<std::size_t>(t) * out_rows + hidden + row] =
                hid[static_cast<std::size_t>(t) * hidden + row];
        }
    }

    DBuf demb = to_device_bits(emb);
    DBuf dhid = to_device_bits(hid);
    DBuf dout(out_n * sizeof(std::uint16_t));
    Tensor temb(demb.p, DType::BF16, {hidden, T});
    Tensor thid(dhid.p, DType::BF16, {hidden, T});
    Tensor tout(dout.p, DType::BF16, {out_rows, T});
    kernels::mtp_pack_fc_input(temb, thid, tout, nullptr);
    cudaDeviceSynchronize();
    return expect_bits("mtp_pack_fc_input T=" + std::to_string(T), from_device_bits(dout, out_n),
                       expected);
}

int one_split_case(int T) {
    constexpr int attn_rows = 14336;
    constexpr int q_rows = 6144;
    constexpr int kv_rows = 1024;
    const std::size_t attn_n = static_cast<std::size_t>(attn_rows) * T;
    const auto attn          = pattern(attn_n, 4242u + static_cast<std::uint32_t>(T));
    std::vector<std::uint16_t> expected_q(static_cast<std::size_t>(q_rows) * T);
    std::vector<std::uint16_t> expected_k(static_cast<std::size_t>(kv_rows) * T);
    std::vector<std::uint16_t> expected_gate(static_cast<std::size_t>(q_rows) * T);
    std::vector<std::uint16_t> expected_v(static_cast<std::size_t>(kv_rows) * T);
    for (int t = 0; t < T; ++t) {
        const std::size_t src_base = static_cast<std::size_t>(t) * attn_rows;
        for (int row = 0; row < q_rows; ++row) {
            expected_q[static_cast<std::size_t>(t) * q_rows + row] = attn[src_base + row];
            expected_gate[static_cast<std::size_t>(t) * q_rows + row] =
                attn[src_base + q_rows + kv_rows + row];
        }
        for (int row = 0; row < kv_rows; ++row) {
            expected_k[static_cast<std::size_t>(t) * kv_rows + row] =
                attn[src_base + q_rows + row];
            expected_v[static_cast<std::size_t>(t) * kv_rows + row] =
                attn[src_base + q_rows + kv_rows + q_rows + row];
        }
    }

    DBuf dattn = to_device_bits(attn);
    DBuf dq(expected_q.size() * sizeof(std::uint16_t));
    DBuf dk(expected_k.size() * sizeof(std::uint16_t));
    DBuf dgate(expected_gate.size() * sizeof(std::uint16_t));
    DBuf dv(expected_v.size() * sizeof(std::uint16_t));
    Tensor tattn(dattn.p, DType::BF16, {attn_rows, T});
    Tensor tq(dq.p, DType::BF16, {256, 24, T});
    Tensor tk(dk.p, DType::BF16, {256, 4, T});
    Tensor tgate(dgate.p, DType::BF16, {256, 24, T});
    Tensor tv(dv.p, DType::BF16, {256, 4, T});
    kernels::mtp_split_attn_in(tattn, tq, tk, tgate, tv, nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    failures += expect_bits("mtp_split_attn_in q T=" + std::to_string(T),
                            from_device_bits(dq, expected_q.size()), expected_q);
    failures += expect_bits("mtp_split_attn_in k T=" + std::to_string(T),
                            from_device_bits(dk, expected_k.size()), expected_k);
    failures += expect_bits("mtp_split_attn_in gate T=" + std::to_string(T),
                            from_device_bits(dgate, expected_gate.size()), expected_gate);
    failures += expect_bits("mtp_split_attn_in v T=" + std::to_string(T),
                            from_device_bits(dv, expected_v.size()), expected_v);
    return failures;
}

int validation_case() {
    try {
        DBuf d(2);
        Tensor x(d.p, DType::BF16, {1});
        kernels::mtp_pack_fc_input(x, x, x, nullptr);
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "mtp_pack validation: expected invalid_argument\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int failures = 0;
    for (int T : {1, 2, 6, 17, 1024}) {
        failures += one_pack_case(T);
        failures += one_split_case(T);
    }
    failures += validation_case();
    std::cout << (failures ? "FAIL" : "OK") << " mtp_pack correctness\n";
    return failures ? 1 : 0;
}

