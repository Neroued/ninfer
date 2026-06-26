// Correctness + coverage for embed_gather, against the frozen op-test standard
// (docs/l1-op-test-standard.md): fp64 golden from bf16-rounded dense inputs,
// exact Q6 row-grouped dequant reference, composite tolerance bf16_elementwise.
#include "qus/kernels/embed_gather.h"
#include "kernels/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

constexpr std::int32_t kVocab = 16;
constexpr std::int32_t kD     = 128;
constexpr std::int32_t kGroup = 64;
constexpr std::int32_t kBpr   = 48;
constexpr std::int32_t kRoww  = 2 + kBpr;
constexpr std::int32_t kKg    = kD / kGroup;

static std::uint16_t f32_to_f16(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::uint32_t mant       = bits & 0x007fffffu;
    int exp                  = static_cast<int>((bits >> 23) & 0xffu) - 127;

    if (((bits >> 23) & 0xffu) == 0xffu) {
        if (mant == 0) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    }
    if (exp > 15) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (exp >= -14) {
        std::uint32_t half_exp = static_cast<std::uint32_t>(exp + 15);
        std::uint32_t rounded  = mant + 0x00000fffu + ((mant >> 13) & 1u);
        if (rounded & 0x00800000u) {
            rounded = 0;
            ++half_exp;
            if (half_exp >= 31u) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
        return static_cast<std::uint16_t>(sign | (half_exp << 10) | (rounded >> 13));
    }
    if (exp < -24) { return static_cast<std::uint16_t>(sign); }

    mant |= 0x00800000u;
    const int shift = -exp - 14;
    std::uint32_t half_mant = mant >> (shift + 13);
    const std::uint32_t rem = mant & ((1u << (shift + 13)) - 1u);
    const std::uint32_t halfway = 1u << (shift + 12);
    if (rem > halfway || (rem == halfway && (half_mant & 1u) != 0u)) { ++half_mant; }
    return static_cast<std::uint16_t>(sign | half_mant);
}

static float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u)) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1fu;
    const std::uint32_t mant = h & 0x03ffu;
    std::uint32_t bits       = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            std::uint32_t m = mant;
            while ((m & 0x0400u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x03ffu;
            bits = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static std::vector<float> make_source_table() {
    std::vector<float> table(static_cast<std::size_t>(kVocab) * kD);
    for (std::int32_t row = 0; row < kVocab; ++row) {
        for (std::int32_t col = 0; col < kD; ++col) {
            const float wave = std::sin(0.17f * static_cast<float>(row * 13 + col));
            table[static_cast<std::size_t>(row) * kD + col] =
                0.125f * static_cast<float>(row - 7) + 0.03125f * static_cast<float>((col % 17) - 8) + wave;
        }
    }
    return table;
}

static std::vector<int> ids_for_T(std::int32_t T) {
    std::vector<int> ids(T);
    const int base[4] = {0, 5, 15, 0};
    for (std::int32_t t = 0; t < T; ++t) {
        ids[t] = (t < 4) ? base[t] : ((t * 7 + 3) % kVocab);
    }
    return ids;
}

static void pack_q6_group(const std::int8_t* codes, std::uint8_t* out) {
    std::fill(out, out + kBpr, std::uint8_t{0});
    for (std::int32_t c = 0; c < kGroup; ++c) {
        const std::uint32_t u = static_cast<std::uint8_t>(codes[c]) & 0x3fu;
        for (std::int32_t bit = 0; bit < 6; ++bit) {
            const std::int32_t pos = c * 6 + bit;
            out[pos / 8] |= static_cast<std::uint8_t>(((u >> bit) & 1u) << (pos % 8));
        }
    }
}

static int unpack_q6_code(const std::uint8_t* packed, std::int32_t c) {
    std::uint32_t u = 0;
    for (std::int32_t bit = 0; bit < 6; ++bit) {
        const std::int32_t pos = c * 6 + bit;
        u |= static_cast<std::uint32_t>((packed[pos / 8] >> (pos % 8)) & 1u) << bit;
    }
    return (u & 0x20u) ? static_cast<int>(u) - 64 : static_cast<int>(u);
}

static std::vector<std::uint8_t> encode_q6_row_grouped(const std::vector<float>& src,
                                                       std::vector<float>& deq) {
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(kVocab) * kKg * kRoww);
    deq.assign(src.size(), 0.0f);
    for (std::int32_t row = 0; row < kVocab; ++row) {
        for (std::int32_t g = 0; g < kKg; ++g) {
            const std::size_t base = static_cast<std::size_t>(row) * kD + g * kGroup;
            float maxabs = 0.0f;
            for (std::int32_t i = 0; i < kGroup; ++i) {
                maxabs = std::max(maxabs, std::abs(src[base + i]));
            }

            std::uint16_t scale_h = f32_to_f16(maxabs / 31.0f);
            if (scale_h == 0 && maxabs > 0.0f) { scale_h = 0x0001u; }
            const float scale = f16_to_f32(scale_h);

            std::int8_t codes[kGroup];
            for (std::int32_t i = 0; i < kGroup; ++i) {
                int q = 0;
                if (scale > 0.0f) {
                    q = static_cast<int>(std::nearbyint(src[base + i] / scale));
                    q = std::clamp(q, -32, 31);
                }
                codes[i]      = static_cast<std::int8_t>(q);
                deq[base + i] = static_cast<float>(q) * scale;
            }

            const std::size_t off = (static_cast<std::size_t>(row) * kKg + g) * kRoww;
            payload[off + 0] = static_cast<std::uint8_t>(scale_h & 0xffu);
            payload[off + 1] = static_cast<std::uint8_t>(scale_h >> 8);
            pack_q6_group(codes, payload.data() + off + 2);
        }
    }
    return payload;
}

static void cpu_gather(const std::vector<float>& table, const std::vector<int>& ids,
                       std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * ids.size(), 0.0);
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const std::int32_t row = ids[t];
        for (std::int32_t d = 0; d < kD; ++d) {
            out[static_cast<std::size_t>(t) * kD + d] =
                static_cast<double>(table[static_cast<std::size_t>(row) * kD + d]);
        }
    }
}

static Weight dense_weight(void* data) {
    Weight w{};
    w.qtype           = QType::BF16_CTRL;
    w.layout          = QuantLayout::Contiguous;
    w.q5090_scale_dtype = ScaleDType::None;
    w.payload         = data;
    w.payload_bytes   = static_cast<std::uint64_t>(kVocab) * kD * 2u;
    w.qdata           = data;
    w.scales          = nullptr;
    w.group_size      = 0;
    w.group           = 0;
    w.ndim            = 2;
    w.shape[0]        = kVocab;
    w.shape[1]        = kD;
    w.padded_shape[0] = kVocab;
    w.padded_shape[1] = kD;
    w.n               = kVocab;
    w.k               = kD;
    return w;
}

static Weight q6_weight(void* payload) {
    Weight w{};
    w.qtype           = QType::Q6G64_F16S;
    w.layout          = QuantLayout::RowGroupedG64;
    w.q5090_scale_dtype = ScaleDType::FP16;
    w.payload         = payload;
    w.payload_bytes   = static_cast<std::uint64_t>(kVocab) * kKg * kRoww;
    w.qdata           = payload;
    w.scales          = nullptr;
    w.group_size      = kGroup;
    w.group           = kGroup;
    w.ndim            = 2;
    w.shape[0]        = kVocab;
    w.shape[1]        = kD;
    w.padded_shape[0] = kVocab;
    w.padded_shape[1] = kD;
    w.n               = kVocab;
    w.k               = kD;
    return w;
}

static int one_dense_shape(std::int32_t T) {
    std::vector<float> src = make_source_table();
    round_to_bf16(src);
    const std::vector<int> ids = ids_for_T(T);

    std::vector<double> ref;
    cpu_gather(src, ids, ref);

    DBuf dtable = to_device_bf16(src);
    DBuf dids = to_device_i32(ids);
    DBuf dout(static_cast<std::size_t>(kD) * T * 2u);
    Tensor tids(dids.p, DType::I32, {T});
    Tensor tout(dout.p, DType::BF16, {kD, T});
    kernels::embed_gather(tids, dense_weight(dtable.p), tout, nullptr);
    cudaDeviceSynchronize();

    return verify((std::string("embed_gather dense T=") + std::to_string(T)).c_str(),
                  from_device_bf16(dout, static_cast<std::size_t>(kD) * T), ref,
                  Tolerance::bf16_elementwise());
}

static int one_q6_shape(std::int32_t T) {
    const std::vector<float> src = make_source_table();
    std::vector<float> deq;
    std::vector<std::uint8_t> payload = encode_q6_row_grouped(src, deq);
    const std::vector<int> ids = ids_for_T(T);

    std::vector<double> ref;
    cpu_gather(deq, ids, ref);

    DBuf dtable(payload.size());
    cudaMemcpy(dtable.p, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    DBuf dids = to_device_i32(ids);
    DBuf dout(static_cast<std::size_t>(kD) * T * 2u);
    Tensor tids(dids.p, DType::I32, {T});
    Tensor tout(dout.p, DType::BF16, {kD, T});
    kernels::embed_gather(tids, q6_weight(dtable.p), tout, nullptr);
    cudaDeviceSynchronize();

    return verify((std::string("embed_gather q6 T=") + std::to_string(T)).c_str(),
                  from_device_bf16(dout, static_cast<std::size_t>(kD) * T), ref,
                  Tolerance::bf16_elementwise());
}

static int validation_checks() {
    int f = 0;
    Tensor ids(nullptr, DType::I32, {4});
    Tensor out(nullptr, DType::BF16, {kD, 4});
    Weight dense = dense_weight(nullptr);
    Weight q6    = q6_weight(nullptr);

    try {
        Tensor empty_ids(nullptr, DType::I32, {1});
        Tensor empty_out(nullptr, DType::BF16, {kD, 1});
        empty_ids.ne[0] = 0;
        empty_out.ne[1] = 0;
        kernels::embed_gather(empty_ids, dense, empty_out, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "validation empty T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_ids = ids;
        bad_ids.ne[0] = -1;
        kernels::embed_gather(bad_ids, dense, out, nullptr);
        std::cerr << "validation negative ids dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_ids(nullptr, DType::FP32, {4});
        kernels::embed_gather(bad_ids, dense, out, nullptr);
        std::cerr << "validation ids dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(nullptr, DType::FP32, {kD, 4});
        kernels::embed_gather(ids, dense, bad_out, nullptr);
        std::cerr << "validation out dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_ids(nullptr, DType::I32, {4, 2});
        kernels::embed_gather(bad_ids, dense, out, nullptr);
        std::cerr << "validation ids shape: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(nullptr, DType::BF16, {kD + 1, 4});
        kernels::embed_gather(ids, dense, bad_out, nullptr);
        std::cerr << "validation out d: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(nullptr, DType::BF16, {kD, 5});
        kernels::embed_gather(ids, dense, bad_out, nullptr);
        std::cerr << "validation out T: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_stride = out;
        bad_stride.nb[0]  = 4;
        kernels::embed_gather(ids, dense, bad_stride, nullptr);
        std::cerr << "validation contiguous: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_qtype = dense;
        bad_qtype.qtype = QType::FP32_CTRL;
        kernels::embed_gather(ids, bad_qtype, out, nullptr);
        std::cerr << "validation unsupported qtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_dense_layout = dense;
        bad_dense_layout.layout = QuantLayout::RowGroupedG64;
        kernels::embed_gather(ids, bad_dense_layout, out, nullptr);
        std::cerr << "validation dense layout: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_dense_payload = dense;
        bad_dense_payload.payload_bytes = static_cast<std::uint64_t>(kVocab) * kD * 2u - 1u;
        kernels::embed_gather(ids, bad_dense_payload, out, nullptr);
        std::cerr << "validation dense payload size: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_q6_layout = q6;
        bad_q6_layout.layout = QuantLayout::Contiguous;
        kernels::embed_gather(ids, bad_q6_layout, out, nullptr);
        std::cerr << "validation q6 layout: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_q6_group = q6;
        bad_q6_group.group_size = 32;
        kernels::embed_gather(ids, bad_q6_group, out, nullptr);
        std::cerr << "validation q6 group: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        kernels::embed_gather(ids, dense, out, nullptr);
        std::cerr << "validation null dense data: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        kernels::embed_gather(ids, q6, out, nullptr);
        std::cerr << "validation null q6 data: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    return f;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int f = 0;
    f += validation_checks();
    for (std::int32_t T : {1, 4, 64}) {
        f += one_dense_shape(T);
        f += one_q6_shape(T);
    }

    std::cout << (f ? "FAIL" : "OK") << " embed_gather correctness\n";
    return f ? 1 : 0;
}
