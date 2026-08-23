#include "ninfer/ops/embedding.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kVocab             = 248320;
constexpr std::int32_t kLastFrontendToken = 248076;
constexpr std::int32_t kMaskToken         = 248077;
constexpr std::int32_t kQ6D               = 5120;
constexpr std::int32_t kW8VisionD         = 2048;
constexpr std::int32_t kW8TextD           = 5120;
constexpr std::int32_t kFp8D              = 5120;
constexpr std::int32_t kDenseRows         = 2304;
constexpr std::int32_t kDenseD            = 1152;
constexpr std::int32_t kQ6Group           = 64;
constexpr std::int32_t kW8Group           = 32;

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

std::uint16_t f32_to_f16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    int exponent             = static_cast<int>((bits >> 23) & 0xffu) - 127;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }
    if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent >= -14) {
        std::uint32_t half_exponent = static_cast<std::uint32_t>(exponent + 15);
        std::uint32_t rounded       = mantissa + 0x00000fffu + ((mantissa >> 13) & 1u);
        if ((rounded & 0x00800000u) != 0) {
            rounded = 0;
            if (++half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        return static_cast<std::uint16_t>(sign | (half_exponent << 10) | (rounded >> 13));
    }
    if (exponent < -24) return static_cast<std::uint16_t>(sign);

    mantissa |= 0x00800000u;
    const int shift             = -exponent - 14;
    std::uint32_t half_mantissa = mantissa >> (shift + 13);
    const std::uint32_t rem     = mantissa & ((1u << (shift + 13)) - 1u);
    const std::uint32_t halfway = 1u << (shift + 12);
    if (rem > halfway || (rem == halfway && (half_mantissa & 1u) != 0)) { ++half_mantissa; }
    return static_cast<std::uint16_t>(sign | half_mantissa);
}

float f16_to_f32(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    const std::uint32_t exp  = (value >> 10) & 0x1fu;
    std::uint32_t mantissa   = value & 0x03ffu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            bits = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23) |
                   ((mantissa & 0x03ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

double decode_e4m3fn(std::uint8_t word) {
    const bool negative   = (word & 0x80u) != 0;
    const std::uint32_t e = (word >> 3) & 0x0fu;
    const std::uint32_t m = word & 0x07u;
    if (e == 0x0fu && m == 0x07u) {
        throw std::invalid_argument("embedding test E4M3FN code is NaN");
    }
    const double magnitude =
        e == 0 ? static_cast<double>(m) * std::ldexp(1.0, -9)
               : (1.0 + static_cast<double>(m) / 8.0) * std::ldexp(1.0, static_cast<int>(e) - 7);
    return negative ? -magnitude : magnitude;
}

std::uint16_t load_u16_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + 2 > bytes.size()) throw std::out_of_range("embedding test scale read");
    return static_cast<std::uint16_t>(bytes[offset]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

void store_u16_le(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset]     = static_cast<std::uint8_t>(value);
    bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

std::vector<std::int32_t> repeated_ids(std::size_t count) {
    constexpr std::int32_t values[] = {
        0, 1, 42, 12345, kLastFrontendToken, kMaskToken, kVocab - 1, 42,
    };
    std::vector<std::int32_t> result(count);
    for (std::size_t i = 0; i < count; ++i) result[i] = values[i % std::size(values)];
    return result;
}

template <typename T>
std::vector<T> guarded_to_host(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> result(count);
    buffer.copy_to_host(result.data(), count * sizeof(T));
    return result;
}

int verify_input(const char* label, const GuardedDeviceBuffer& ids,
                 const std::vector<std::int32_t>& expected) {
    int failures = ids.verify_guards(label);
    failures += verify_exact(label, guarded_to_host<std::int32_t>(ids, expected.size()), expected);
    return failures;
}

// There is no reduction in quantized embedding. Compare every finite BF16 output directly with
// decoded_code * exact_stored_scale. The FP8 fixture satisfies the existing Q6/W8 relative
// criterion without relaxing it. Half the minimum BF16 subnormal is the exact absolute rounding
// bound when a finite ideal underflows to zero.
constexpr PointwiseCriterion kQuantizedOutputTolerance{
    /*absolute=*/0x1p-134,
    /*relative=*/3.8e-3,
};

int verify_quantized(const char* label, const GuardedDeviceBuffer& output,
                     const std::vector<double>& expected) {
    const std::vector<std::uint16_t> bits = guarded_to_host<std::uint16_t>(output, expected.size());
    std::vector<double> actual(bits.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        actual[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return verify_pointwise(label, actual, expected, kQuantizedOutputTolerance);
}

struct Q6Row {
    std::int32_t id;
    std::vector<std::uint8_t> low;
    std::vector<std::uint8_t> high;
    std::vector<std::uint8_t> scales;
};

class Q6Table {
public:
    Q6Table()
        : groups_(kQ6D / kQ6Group),
          low_plane_bytes_(static_cast<std::size_t>(kVocab) * groups_ * 32),
          high_offset_(align_up(low_plane_bytes_, 256)),
          high_plane_bytes_(static_cast<std::size_t>(kVocab) * groups_ * 16),
          scale_offset_(high_offset_ + align_up(high_plane_bytes_, 256)),
          payload_(scale_offset_ + static_cast<std::size_t>(kVocab) * groups_ * 2) {
        for (const std::int32_t row : repeated_ids(8)) {
            if (find(row) == nullptr) add_row(row);
        }
    }

    Weight weight() {
        auto* base = static_cast<std::uint8_t*>(payload_.data());
        Weight result{};
        result.qtype            = QType::Q6G64_F16S;
        result.layout           = QuantLayout::RowSplit;
        result.scale_dtype      = DType::FP16;
        result.payload          = base;
        result.payload_bytes    = payload_.bytes();
        result.qdata            = base;
        result.qhigh            = base + high_offset_;
        result.scales           = base + scale_offset_;
        result.high_plane_bytes = high_plane_bytes_;
        result.group_size       = kQ6Group;
        result.group            = kQ6Group;
        result.ndim             = 2;
        result.shape[0]         = kVocab;
        result.shape[1]         = kQ6D;
        result.padded_shape[0]  = kVocab;
        result.padded_shape[1]  = kQ6D;
        result.n                = kVocab;
        result.k                = kQ6D;
        return result;
    }

    std::vector<double> oracle(const std::vector<std::int32_t>& ids) const {
        std::vector<double> result(static_cast<std::size_t>(kQ6D) * ids.size());
        for (std::size_t t = 0; t < ids.size(); ++t) {
            const Q6Row* row = find(ids[t]);
            if (row == nullptr) throw std::out_of_range("Q6 oracle row was not materialized");
            for (std::int32_t d = 0; d < kQ6D; ++d) {
                const std::int32_t group = d / kQ6Group;
                const std::int32_t lane  = d % kQ6Group;
                const std::uint32_t low =
                    (row->low[static_cast<std::size_t>(group) * 32 + lane / 2] >>
                     ((lane & 1) * 4)) &
                    0x0fu;
                const std::int32_t bit = lane * 2;
                const std::uint32_t high =
                    (row->high[static_cast<std::size_t>(group) * 16 + bit / 8] >> (bit & 7)) &
                    0x03u;
                const std::uint32_t encoded = low | (high << 4);
                const int code = (encoded & 0x20u) != 0 ? static_cast<int>(encoded) - 64
                                                        : static_cast<int>(encoded);
                const double scale =
                    static_cast<double>(f16_to_f32(load_u16_le(row->scales, group * 2)));
                result[t * static_cast<std::size_t>(kQ6D) + d] = static_cast<double>(code) * scale;
            }
        }
        return result;
    }

    int verify_unchanged(const char* label) const {
        int failures = payload_.verify_guards(label);
        for (const Q6Row& row : rows_) {
            std::vector<std::uint8_t> got(row.low.size());
            payload_.copy_to_host(got.data(), got.size(),
                                  static_cast<std::size_t>(row.id) * groups_ * 32);
            failures += verify_exact(label, got, row.low);
            got.resize(row.high.size());
            payload_.copy_to_host(got.data(), got.size(),
                                  high_offset_ + static_cast<std::size_t>(row.id) * groups_ * 16);
            failures += verify_exact(label, got, row.high);
            got.resize(row.scales.size());
            payload_.copy_to_host(got.data(), got.size(),
                                  scale_offset_ + static_cast<std::size_t>(row.id) * groups_ * 2);
            failures += verify_exact(label, got, row.scales);
        }
        return failures;
    }

private:
    const Q6Row* find(std::int32_t id) const {
        const auto it = std::find_if(rows_.begin(), rows_.end(),
                                     [id](const Q6Row& row) { return row.id == id; });
        return it == rows_.end() ? nullptr : &*it;
    }

    void add_row(std::int32_t id) {
        Q6Row row{id, std::vector<std::uint8_t>(static_cast<std::size_t>(groups_) * 32),
                  std::vector<std::uint8_t>(static_cast<std::size_t>(groups_) * 16),
                  std::vector<std::uint8_t>(static_cast<std::size_t>(groups_) * 2)};
        for (std::int32_t group = 0; group < groups_; ++group) {
            const std::uint16_t scale =
                f32_to_f16(0.0013f + 0.00037f * static_cast<float>((id + group * 3) % 11));
            store_u16_le(row.scales, static_cast<std::size_t>(group) * 2, scale);
            for (std::int32_t lane = 0; lane < kQ6Group; ++lane) {
                int code = ((id % 61 + group * 17 + lane * 11) & 63) - 32;
                if (lane == 0) code = -32;
                if (lane == 1) code = 31;
                if (lane == 2) code = 0;
                const std::uint32_t encoded =
                    static_cast<std::uint32_t>(static_cast<std::uint8_t>(code)) & 0x3fu;
                auto& low = row.low[static_cast<std::size_t>(group) * 32 + lane / 2];
                low |= static_cast<std::uint8_t>((encoded & 0x0fu) << ((lane & 1) * 4));
                const std::int32_t bit = lane * 2;
                row.high[static_cast<std::size_t>(group) * 16 + bit / 8] |=
                    static_cast<std::uint8_t>(((encoded >> 4) & 0x03u) << (bit & 7));
            }
        }
        payload_.copy_from_host(row.low.data(), row.low.size(),
                                static_cast<std::size_t>(id) * groups_ * 32);
        payload_.copy_from_host(row.high.data(), row.high.size(),
                                high_offset_ + static_cast<std::size_t>(id) * groups_ * 16);
        payload_.copy_from_host(row.scales.data(), row.scales.size(),
                                scale_offset_ + static_cast<std::size_t>(id) * groups_ * 2);
        rows_.push_back(std::move(row));
    }

    std::int32_t groups_;
    std::size_t low_plane_bytes_;
    std::size_t high_offset_;
    std::size_t high_plane_bytes_;
    std::size_t scale_offset_;
    GuardedDeviceBuffer payload_;
    std::vector<Q6Row> rows_;
};

struct W8Row {
    std::int32_t id;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> scales;
};

class W8Table {
public:
    explicit W8Table(std::int32_t d)
        : d_(d), groups_(d / kW8Group), code_plane_bytes_(static_cast<std::size_t>(kVocab) * d),
          scale_offset_(align_up(code_plane_bytes_, 256)),
          payload_(scale_offset_ + static_cast<std::size_t>(kVocab) * groups_ * 2) {
        for (const std::int32_t row : repeated_ids(8)) {
            if (find(row) == nullptr) add_row(row);
        }
    }

    Weight weight() {
        auto* base = static_cast<std::uint8_t*>(payload_.data());
        Weight result{};
        result.qtype            = QType::W8G32_F16S;
        result.layout           = QuantLayout::RowSplit;
        result.scale_dtype      = DType::FP16;
        result.payload          = base;
        result.payload_bytes    = payload_.bytes();
        result.qdata            = base;
        result.qhigh            = nullptr;
        result.scales           = base + scale_offset_;
        result.high_plane_bytes = 0;
        result.group_size       = kW8Group;
        result.group            = kW8Group;
        result.ndim             = 2;
        result.shape[0]         = kVocab;
        result.shape[1]         = d_;
        result.padded_shape[0]  = kVocab;
        result.padded_shape[1]  = d_;
        result.n                = kVocab;
        result.k                = d_;
        return result;
    }

    std::vector<double> oracle(const std::vector<std::int32_t>& ids) const {
        std::vector<double> result(static_cast<std::size_t>(d_) * ids.size());
        for (std::size_t t = 0; t < ids.size(); ++t) {
            const W8Row* row = find(ids[t]);
            if (row == nullptr) throw std::out_of_range("W8 oracle row was not materialized");
            for (std::int32_t d = 0; d < d_; ++d) {
                const std::uint8_t raw = row->codes[d];
                const int code = raw < 0x80u ? static_cast<int>(raw) : static_cast<int>(raw) - 256;
                const double scale                           = static_cast<double>(f16_to_f32(
                    load_u16_le(row->scales, static_cast<std::size_t>(d / kW8Group) * 2)));
                result[t * static_cast<std::size_t>(d_) + d] = static_cast<double>(code) * scale;
            }
        }
        return result;
    }

    int verify_unchanged(const char* label) const {
        int failures = payload_.verify_guards(label);
        for (const W8Row& row : rows_) {
            std::vector<std::uint8_t> got(row.codes.size());
            payload_.copy_to_host(got.data(), got.size(), static_cast<std::size_t>(row.id) * d_);
            failures += verify_exact(label, got, row.codes);
            got.resize(row.scales.size());
            payload_.copy_to_host(got.data(), got.size(),
                                  scale_offset_ + static_cast<std::size_t>(row.id) * groups_ * 2);
            failures += verify_exact(label, got, row.scales);
        }
        return failures;
    }

private:
    const W8Row* find(std::int32_t id) const {
        const auto it = std::find_if(rows_.begin(), rows_.end(),
                                     [id](const W8Row& row) { return row.id == id; });
        return it == rows_.end() ? nullptr : &*it;
    }

    void add_row(std::int32_t id) {
        W8Row row{id, std::vector<std::uint8_t>(d_),
                  std::vector<std::uint8_t>(static_cast<std::size_t>(groups_) * 2)};
        for (std::int32_t group = 0; group < groups_; ++group) {
            const std::uint16_t scale =
                f32_to_f16(0.00091f + 0.00023f * static_cast<float>((id + group * 5) % 13));
            store_u16_le(row.scales, static_cast<std::size_t>(group) * 2, scale);
            for (std::int32_t lane = 0; lane < kW8Group; ++lane) {
                int code = ((id % 251 + group * 29 + lane * 17) % 255) - 127;
                if (lane == 0) code = -127;
                if (lane == 1) code = 127;
                if (lane == 2) code = 0;
                if (lane == 3) code = -1;
                if (lane == 4) code = 1;
                row.codes[static_cast<std::size_t>(group) * kW8Group + lane] =
                    static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
            }
        }
        payload_.copy_from_host(row.codes.data(), row.codes.size(),
                                static_cast<std::size_t>(id) * d_);
        payload_.copy_from_host(row.scales.data(), row.scales.size(),
                                scale_offset_ + static_cast<std::size_t>(id) * groups_ * 2);
        rows_.push_back(std::move(row));
    }

    std::int32_t d_;
    std::int32_t groups_;
    std::size_t code_plane_bytes_;
    std::size_t scale_offset_;
    GuardedDeviceBuffer payload_;
    std::vector<W8Row> rows_;
};

struct Fp8Row {
    std::int32_t id;
    std::vector<std::uint8_t> codes;
    std::uint16_t scale;
};

class Fp8Table {
public:
    Fp8Table()
        : code_plane_bytes_(static_cast<std::size_t>(kVocab) * kFp8D),
          scale_offset_(align_up(code_plane_bytes_, 256)),
          payload_(scale_offset_ + static_cast<std::size_t>(kVocab) * 2) {
        for (const std::int32_t row : repeated_ids(8)) {
            if (find(row) == nullptr) add_row(row);
        }
    }

    Weight weight() {
        auto* base = static_cast<std::uint8_t*>(payload_.data());
        Weight result{};
        result.qtype            = QType::FP8_E4M3FN_ROW_BF16S;
        result.layout           = QuantLayout::RowScale;
        result.scale_dtype      = DType::BF16;
        result.payload          = base;
        result.payload_bytes    = payload_.bytes();
        result.qdata            = base;
        result.qhigh            = nullptr;
        result.scales           = base + scale_offset_;
        result.high_plane_bytes = 0;
        result.group_size       = kFp8D;
        result.group            = kFp8D;
        result.ndim             = 2;
        result.shape[0]         = kVocab;
        result.shape[1]         = kFp8D;
        result.padded_shape[0]  = kVocab;
        result.padded_shape[1]  = kFp8D;
        result.n                = kVocab;
        result.k                = kFp8D;
        result.scale_ne[0]      = kVocab;
        result.scale_nb[0]      = 2;
        result.scale_nb[1]      = static_cast<std::int64_t>(kVocab) * 2;
        result.scale_nb[2]      = result.scale_nb[1];
        result.scale_nb[3]      = result.scale_nb[1];
        return result;
    }

    std::vector<double> oracle(const std::vector<std::int32_t>& ids) const {
        std::vector<double> result(static_cast<std::size_t>(kFp8D) * ids.size());
        for (std::size_t t = 0; t < ids.size(); ++t) {
            const Fp8Row* row = find(ids[t]);
            if (row == nullptr) throw std::out_of_range("FP8 oracle row was not materialized");
            const double scale = static_cast<double>(bf16_to_f32(row->scale));
            for (std::int32_t d = 0; d < kFp8D; ++d) {
                result[t * static_cast<std::size_t>(kFp8D) + d] =
                    decode_e4m3fn(row->codes[static_cast<std::size_t>(d)]) * scale;
            }
        }
        return result;
    }

    int verify_unchanged(const char* label) const {
        int failures = payload_.verify_guards(label);
        for (const Fp8Row& row : rows_) {
            std::vector<std::uint8_t> got(row.codes.size());
            payload_.copy_to_host(got.data(), got.size(), static_cast<std::size_t>(row.id) * kFp8D);
            failures += verify_exact(label, got, row.codes);
            std::uint8_t scale_bytes[2]{};
            payload_.copy_to_host(scale_bytes, sizeof(scale_bytes),
                                  scale_offset_ + static_cast<std::size_t>(row.id) * 2);
            const std::uint16_t scale = static_cast<std::uint16_t>(scale_bytes[0]) |
                                        static_cast<std::uint16_t>(scale_bytes[1] << 8);
            if (scale != row.scale) {
                std::cerr << label << " scale changed for row " << row.id << '\n';
                ++failures;
            }
        }
        return failures;
    }

private:
    const Fp8Row* find(std::int32_t id) const {
        const auto it = std::find_if(rows_.begin(), rows_.end(),
                                     [id](const Fp8Row& row) { return row.id == id; });
        return it == rows_.end() ? nullptr : &*it;
    }

    void add_row(std::int32_t id) {
        static constexpr std::uint8_t kFiniteCodes[]{
            0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88,
            0x38, 0xb8, 0x3a, 0xba, 0x55, 0xd5, 0x7e, 0xfe,
        };
        Fp8Row row{id, std::vector<std::uint8_t>(kFp8D),
                   id == 0    ? std::uint16_t{0}
                   : id == 1  ? std::uint16_t{1}
                   : id == 42 ? std::uint16_t{0x0080}
                              : f32_to_bf16(0.0017f + 0.00031f * static_cast<float>(id % 13))};
        for (std::int32_t d = 0; d < kFp8D; ++d) {
            row.codes[static_cast<std::size_t>(d)] =
                id == 0 ? 0
                        : kFiniteCodes[(static_cast<std::size_t>(id) + d * 7) %
                                       std::size(kFiniteCodes)];
        }
        payload_.copy_from_host(row.codes.data(), row.codes.size(),
                                static_cast<std::size_t>(id) * kFp8D);
        const std::uint8_t scale_bytes[]{static_cast<std::uint8_t>(row.scale),
                                         static_cast<std::uint8_t>(row.scale >> 8)};
        payload_.copy_from_host(scale_bytes, sizeof(scale_bytes),
                                scale_offset_ + static_cast<std::size_t>(id) * 2);
        rows_.push_back(std::move(row));
    }

    std::size_t code_plane_bytes_;
    std::size_t scale_offset_;
    GuardedDeviceBuffer payload_;
    std::vector<Fp8Row> rows_;
};

template <typename Table>
int run_quantized_case(const char* label, Table& table, const std::vector<std::int32_t>& ids,
                       std::int32_t d) {
    GuardedDeviceBuffer device_ids(ids.size() * sizeof(std::int32_t));
    device_ids.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer output(static_cast<std::size_t>(d) * ids.size() * sizeof(std::uint16_t));
    output.fill(0x7d);

    Tensor input(device_ids.data(), DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor result(output.data(), DType::BF16, {d, static_cast<std::int32_t>(ids.size())});
    Weight weight = table.weight();
    ops::embedding(input, weight, result, nullptr);
    cuda_synchronize();

    int failures = verify_quantized(label, output, table.oracle(ids));
    failures += output.verify_guards(label);
    failures += verify_input(label, device_ids, ids);
    failures += table.verify_unchanged(label);
    return failures;
}

int test_q6() {
    Q6Table table;
    int failures = 0;
    failures += run_quantized_case("embedding Q6 [248320,5120] T=1", table, repeated_ids(1), kQ6D);
    failures += run_quantized_case("embedding Q6 [248320,5120] T=7", table, repeated_ids(7), kQ6D);
    failures +=
        run_quantized_case("embedding Q6 [248320,5120] T=128", table, repeated_ids(128), kQ6D);
    return failures;
}

int test_w8() {
    int failures = 0;
    for (const std::int32_t d : {kW8VisionD, kW8TextD}) {
        W8Table table(d);
        for (const std::size_t t : {1u, 6u, 16u, 1024u}) {
            const std::string label = "embedding W8 [248320," + std::to_string(d) +
                                      "] T=" + std::to_string(static_cast<unsigned long long>(t));
            failures += run_quantized_case(label.c_str(), table, repeated_ids(t), d);
        }
    }
    return failures;
}

int test_fp8() {
    Fp8Table table;
    int failures = 0;
    for (const std::size_t t :
         {1u, 4u, 5u, 8u, 9u, 13u, 16u, 17u, 19u, 32u, 33u, 48u, 49u, 81u, 1024u}) {
        const std::string label =
            "embedding FP8 [248320,5120] T=" + std::to_string(static_cast<unsigned long long>(t));
        failures += run_quantized_case(label.c_str(), table, repeated_ids(t), kFp8D);
    }
    const std::vector<std::int32_t> ids = {0};
    GuardedDeviceBuffer device_ids(sizeof(std::int32_t));
    device_ids.copy_from_host(ids.data(), sizeof(std::int32_t));
    GuardedDeviceBuffer output(static_cast<std::size_t>(kFp8D) * sizeof(std::uint16_t));
    Tensor input(device_ids.data(), DType::I32, {1});
    Tensor result(output.data(), DType::BF16, {kFp8D, 1});
    Weight invalid      = table.weight();
    invalid.scale_dtype = DType::FP16;
    try {
        ops::embedding(input, invalid, result, nullptr);
        std::cerr << "embedding FP8 accepted malformed row-scale metadata\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

int test_dense() {
    std::vector<std::uint16_t> table(static_cast<std::size_t>(kDenseRows) * kDenseD);
    for (std::int32_t row = 0; row < kDenseRows; ++row) {
        for (std::int32_t d = 0; d < kDenseD; ++d) {
            const float value = std::sin(static_cast<float>(row * 17 + d) * 0.03125f) +
                                static_cast<float>((row + d) % 13 - 6) * 0.125f;
            table[static_cast<std::size_t>(row) * kDenseD + d] = f32_to_bf16(value);
        }
    }
    const std::vector<std::int32_t> ids = {0, kDenseRows - 1, 17, 17, 1024, 1, 2299};
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(kDenseD) * ids.size());
    for (std::size_t t = 0; t < ids.size(); ++t) {
        std::copy_n(table.data() + static_cast<std::size_t>(ids[t]) * kDenseD, kDenseD,
                    expected.data() + t * kDenseD);
    }

    GuardedDeviceBuffer device_table(table.size() * sizeof(std::uint16_t));
    device_table.copy_from_host(table.data(), table.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(ids.size() * sizeof(std::int32_t));
    device_ids.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer output(expected.size() * sizeof(std::uint16_t));
    output.fill(0x7d);

    Weight weight{};
    weight.qtype           = QType::BF16_CTRL;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = device_table.data();
    weight.payload_bytes   = device_table.bytes();
    weight.qdata           = device_table.data();
    weight.ndim            = 2;
    weight.shape[0]        = kDenseRows;
    weight.shape[1]        = kDenseD;
    weight.padded_shape[0] = kDenseRows;
    weight.padded_shape[1] = kDenseD;
    weight.n               = kDenseRows;
    weight.k               = kDenseD;

    Tensor input(device_ids.data(), DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor result(output.data(), DType::BF16, {kDenseD, static_cast<std::int32_t>(ids.size())});
    ops::embedding(input, weight, result, nullptr);
    cuda_synchronize();

    int failures = verify_exact("embedding BF16 [2304,1152]",
                                guarded_to_host<std::uint16_t>(output, expected.size()), expected);
    failures += output.verify_guards("embedding BF16 output");
    failures += verify_input("embedding BF16 ids", device_ids, ids);
    failures += device_table.verify_guards("embedding BF16 table");
    failures += verify_exact("embedding BF16 table",
                             guarded_to_host<std::uint16_t>(device_table, table.size()), table);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    try {
        failures += test_dense();
        failures += test_q6();
        failures += test_w8();
        failures += test_fp8();
    } catch (const std::exception& error) {
        std::cerr << "embedding test exception: " << error.what() << '\n';
        return 1;
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " embedding correctness\n";
    return failures == 0 ? 0 : 1;
}
