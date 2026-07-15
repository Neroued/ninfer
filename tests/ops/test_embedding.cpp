// Correctness + coverage for embedding, against the frozen op-test standard
// (docs/op-development.md): fp64 golden from bf16-rounded dense inputs,
// exact Q6 ROW_SPLIT dequant reference, composite tolerance bf16_elementwise.
#include "ninfer/ops/embedding.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kVocab       = 16;
constexpr std::int32_t kD           = 128;
constexpr std::int32_t kQwenHiddenD = 5120;
constexpr std::int32_t kGroup       = 64;
constexpr std::int32_t kNibbleBpr   = 32;
constexpr std::int32_t kHighBpr     = 16;

static std::int32_t groups_for_d(std::int32_t d) {
    if (d <= 0 || d % kGroup != 0) {
        throw std::invalid_argument("embedding test d must be positive and divisible by 64");
    }
    return d / kGroup;
}

static std::size_t align_up_size(std::size_t x, std::size_t m) { return ((x + m - 1) / m) * m; }

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
    const int shift             = -exp - 14;
    std::uint32_t half_mant     = mant >> (shift + 13);
    const std::uint32_t rem     = mant & ((1u << (shift + 13)) - 1u);
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
            int e           = -14;
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

static std::vector<float> make_source_table(std::int32_t d) {
    std::vector<float> table(static_cast<std::size_t>(kVocab) * d);
    for (std::int32_t row = 0; row < kVocab; ++row) {
        for (std::int32_t col = 0; col < d; ++col) {
            const float wave = std::sin(0.17f * static_cast<float>(row * 13 + col));
            table[static_cast<std::size_t>(row) * d + col] =
                0.125f * static_cast<float>(row - 7) +
                0.03125f * static_cast<float>((col % 17) - 8) + wave;
        }
    }
    return table;
}

static std::vector<int> ids_for_T(std::int32_t T) {
    std::vector<int> ids(T);
    const int base[4] = {0, 5, 15, 0};
    for (std::int32_t t = 0; t < T; ++t) { ids[t] = (t < 4) ? base[t] : ((t * 7 + 3) % kVocab); }
    return ids;
}

static void pack_q6_group(const std::int8_t* codes, std::uint8_t* nibble, std::uint8_t* high) {
    std::fill(nibble, nibble + kNibbleBpr, std::uint8_t{0});
    std::fill(high, high + kHighBpr, std::uint8_t{0});
    for (std::int32_t c = 0; c < kGroup; ++c) {
        const std::uint32_t u   = static_cast<std::uint8_t>(codes[c]) & 0x3fu;
        const std::uint32_t low = u & 0x0fu;
        if ((c & 1) == 0) {
            nibble[c >> 1] |= static_cast<std::uint8_t>(low);
        } else {
            nibble[c >> 1] |= static_cast<std::uint8_t>(low << 4);
        }
        const std::int32_t high_pos = c * 2;
        high[high_pos >> 3] |= static_cast<std::uint8_t>(((u >> 4) & 0x03u) << (high_pos & 7));
    }
}

static int unpack_q6_code(const std::uint8_t* nibble, const std::uint8_t* high, std::int32_t c) {
    const std::uint8_t low_byte = nibble[c >> 1];
    const std::uint32_t low     = (c & 1) ? (low_byte >> 4) : (low_byte & 0x0fu);
    const std::int32_t high_pos = c * 2;
    const std::uint32_t hi      = (high[high_pos >> 3] >> (high_pos & 7)) & 0x03u;
    const std::uint32_t u       = low | (hi << 4);
    return (u & 0x20u) ? static_cast<int>(u) - 64 : static_cast<int>(u);
}

static std::vector<std::uint8_t> encode_q6_row_split(const std::vector<float>& src, std::int32_t d,
                                                     std::vector<float>& deq) {
    const std::int32_t kg = groups_for_d(d);
    const std::size_t nibble_plane_bytes =
        static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(kg) * kNibbleBpr;
    const std::size_t high_plane_offset = align_up_size(nibble_plane_bytes, 256);
    const std::size_t high_plane_bytes =
        static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(kg) * kHighBpr;
    const std::size_t scale_plane_offset = high_plane_offset + align_up_size(high_plane_bytes, 256);
    const std::size_t scale_plane_bytes =
        static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(kg) * 2u;
    std::vector<std::uint8_t> payload(scale_plane_offset + scale_plane_bytes);
    deq.assign(src.size(), 0.0f);
    for (std::int32_t row = 0; row < kVocab; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            const std::size_t base = static_cast<std::size_t>(row) * d + g * kGroup;
            float maxabs           = 0.0f;
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

            const std::size_t group_index = static_cast<std::size_t>(row) * kg + g;
            const std::size_t nibble_off  = group_index * kNibbleBpr;
            const std::size_t high_off    = high_plane_offset + group_index * kHighBpr;
            const std::size_t scale_off   = scale_plane_offset + group_index * 2;
            pack_q6_group(codes, payload.data() + nibble_off, payload.data() + high_off);
            payload[scale_off + 0] = static_cast<std::uint8_t>(scale_h & 0xffu);
            payload[scale_off + 1] = static_cast<std::uint8_t>(scale_h >> 8);
        }
    }
    return payload;
}

static void cpu_gather(const std::vector<float>& table, const std::vector<int>& ids, std::int32_t d,
                       std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(d) * ids.size(), 0.0);
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const std::int32_t row = ids[t];
        for (std::int32_t col = 0; col < d; ++col) {
            out[static_cast<std::size_t>(t) * d + col] =
                static_cast<double>(table[static_cast<std::size_t>(row) * d + col]);
        }
    }
}

static Weight dense_weight(void* data, std::int32_t d = kD) {
    Weight w{};
    w.qtype           = QType::BF16_CTRL;
    w.layout          = QuantLayout::Contiguous;
    w.payload         = data;
    w.payload_bytes   = static_cast<std::uint64_t>(kVocab) * static_cast<std::uint64_t>(d) * 2u;
    w.qdata           = data;
    w.scales          = nullptr;
    w.group_size      = 0;
    w.group           = 0;
    w.ndim            = 2;
    w.shape[0]        = kVocab;
    w.shape[1]        = d;
    w.padded_shape[0] = kVocab;
    w.padded_shape[1] = d;
    w.n               = kVocab;
    w.k               = d;
    return w;
}

static Weight q6_weight(void* payload, std::int32_t d = kD) {
    const std::int32_t kg = groups_for_d(d);
    const std::uint64_t nibble_plane_bytes =
        static_cast<std::uint64_t>(kVocab) * static_cast<std::uint64_t>(kg) * kNibbleBpr;
    const std::uint64_t high_plane_offset = ((nibble_plane_bytes + 255ULL) / 256ULL) * 256ULL;
    const std::uint64_t high_plane_bytes =
        static_cast<std::uint64_t>(kVocab) * static_cast<std::uint64_t>(kg) * kHighBpr;
    const std::uint64_t scale_plane_offset =
        high_plane_offset + ((high_plane_bytes + 255ULL) / 256ULL) * 256ULL;
    const std::uint64_t scale_plane_bytes =
        static_cast<std::uint64_t>(kVocab) * static_cast<std::uint64_t>(kg) * 2ULL;
    Weight w{};
    w.qtype            = QType::Q6G64_F16S;
    w.layout           = QuantLayout::RowSplit;
    w.scale_dtype      = DType::FP16;
    w.payload          = payload;
    w.payload_bytes    = scale_plane_offset + scale_plane_bytes;
    w.high_plane_bytes = high_plane_bytes;
    if (payload != nullptr) {
        w.qdata  = payload;
        w.qhigh  = static_cast<std::uint8_t*>(payload) + high_plane_offset;
        w.scales = static_cast<std::uint8_t*>(payload) + scale_plane_offset;
    }
    w.group_size      = kGroup;
    w.group           = kGroup;
    w.ndim            = 2;
    w.shape[0]        = kVocab;
    w.shape[1]        = d;
    w.padded_shape[0] = kVocab;
    w.padded_shape[1] = d;
    w.n               = kVocab;
    w.k               = d;
    return w;
}

static int one_dense_shape(std::int32_t T, std::int32_t d) {
    std::vector<float> src = make_source_table(d);
    round_to_bf16(src);
    const std::vector<int> ids = ids_for_T(T);

    std::vector<double> ref;
    cpu_gather(src, ids, d, ref);

    DBuf dtable = to_device_bf16(src);
    DBuf dids   = to_device_i32(ids);
    DBuf dout(static_cast<std::size_t>(d) * T * 2u);
    cudaMemset(dout.p, 0x7d, dout.bytes);
    Tensor tids(dids.p, DType::I32, {T});
    Tensor tout(dout.p, DType::BF16, {d, T});
    ops::embedding(tids, dense_weight(dtable.p, d), tout, nullptr);
    cudaDeviceSynchronize();

    return verify(
        (std::string("embedding dense d=") + std::to_string(d) + " T=" + std::to_string(T)).c_str(),
        from_device_bf16(dout, static_cast<std::size_t>(d) * T), ref,
        Tolerance::bf16_elementwise());
}

static int one_q6_shape(std::int32_t T, std::int32_t d) {
    const std::vector<float> src = make_source_table(d);
    std::vector<float> deq;
    std::vector<std::uint8_t> payload = encode_q6_row_split(src, d, deq);
    const std::vector<int> ids        = ids_for_T(T);

    std::vector<double> ref;
    cpu_gather(deq, ids, d, ref);

    DBuf dtable(payload.size());
    cudaMemcpy(dtable.p, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    DBuf dids = to_device_i32(ids);
    DBuf dout(static_cast<std::size_t>(d) * T * 2u);
    cudaMemset(dout.p, 0x7d, dout.bytes);
    Tensor tids(dids.p, DType::I32, {T});
    Tensor tout(dout.p, DType::BF16, {d, T});
    ops::embedding(tids, q6_weight(dtable.p, d), tout, nullptr);
    cudaDeviceSynchronize();

    return verify(
        (std::string("embedding q6 d=") + std::to_string(d) + " T=" + std::to_string(T)).c_str(),
        from_device_bf16(dout, static_cast<std::size_t>(d) * T), ref,
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
        ops::embedding(empty_ids, dense, empty_out, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "validation empty T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_ids = ids;
        bad_ids.ne[0]  = -1;
        ops::embedding(bad_ids, dense, out, nullptr);
        std::cerr << "validation negative ids dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_ids(nullptr, DType::FP32, {4});
        ops::embedding(bad_ids, dense, out, nullptr);
        std::cerr << "validation ids dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(nullptr, DType::FP32, {kD, 4});
        ops::embedding(ids, dense, bad_out, nullptr);
        std::cerr << "validation out dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_ids(nullptr, DType::I32, {4, 2});
        ops::embedding(bad_ids, dense, out, nullptr);
        std::cerr << "validation ids shape: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(nullptr, DType::BF16, {kD + 1, 4});
        ops::embedding(ids, dense, bad_out, nullptr);
        std::cerr << "validation out d: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(nullptr, DType::BF16, {kD, 5});
        ops::embedding(ids, dense, bad_out, nullptr);
        std::cerr << "validation out T: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_stride = out;
        bad_stride.nb[0]  = 4;
        ops::embedding(ids, dense, bad_stride, nullptr);
        std::cerr << "validation contiguous: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_qtype = dense;
        bad_qtype.qtype  = QType::FP32_CTRL;
        ops::embedding(ids, bad_qtype, out, nullptr);
        std::cerr << "validation unsupported qtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_dense_layout = dense;
        bad_dense_layout.layout = QuantLayout::RowSplit;
        ops::embedding(ids, bad_dense_layout, out, nullptr);
        std::cerr << "validation dense layout: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_dense_payload        = dense;
        bad_dense_payload.payload_bytes = static_cast<std::uint64_t>(kVocab) * kD * 2u - 1u;
        ops::embedding(ids, bad_dense_payload, out, nullptr);
        std::cerr << "validation dense payload size: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_q6_layout = q6;
        bad_q6_layout.layout = QuantLayout::Contiguous;
        ops::embedding(ids, bad_q6_layout, out, nullptr);
        std::cerr << "validation q6 layout: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_q6_group     = q6;
        bad_q6_group.group_size = 32;
        ops::embedding(ids, bad_q6_group, out, nullptr);
        std::cerr << "validation q6 group: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        ops::embedding(ids, dense, out, nullptr);
        std::cerr << "validation null dense data: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        ops::embedding(ids, q6, out, nullptr);
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
        f += one_dense_shape(T, kD);
        f += one_q6_shape(T, kD);
    }
    f += one_dense_shape(5, kQwenHiddenD);
    f += one_q6_shape(5, kQwenHiddenD);

    std::cout << (f ? "FAIL" : "OK") << " embedding correctness\n";
    return f ? 1 : 0;
}
