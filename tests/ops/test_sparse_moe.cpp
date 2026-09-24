#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// SparseMoe is the exact qwen3_6_35b_a3b post-mixer Op. qwen3_6_27b has a dense SwiGLU
// post-mixer and therefore contributes no second SparseMoe geometry.
constexpr std::int32_t kHidden         = 2048;
constexpr std::int32_t kExperts        = 256;
constexpr std::int32_t kTopK           = 8;
constexpr std::int32_t kIntermediate   = 512;
constexpr std::int32_t kExpertGateRows = 2 * kIntermediate;
constexpr std::int32_t kRoutedGateRows = kExperts * kExpertGateRows;
constexpr std::int32_t kRoutedDownRows = kExperts * kHidden;
constexpr std::int32_t kSharedGateRows = 2 * kIntermediate;

// All registered variants consume represented BF16 activations and expose one BF16 destination.
// The bound belongs to that A16 compute profile, not to a codec, T, private route, or schedule.
// The limits retain modest headroom over the measured maxima from the complete case matrix:
// rel-L2 1.117e-2 and pointwise 3.687e-3.
constexpr ReductionCriterion kSparseMoeA16Tolerance{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 0.0,
};

// NVFP4 below its prefill frontier reads represented BF16 activations, and its weights decode into
// exactly the values the oracle is built from, so the remaining error is the BF16 rounding of the
// destination plus the FP32 accumulation of the dot product. Both terms are relative: the absolute
// term above is a number from the row-split fixture, whose references peak near 0.3 where these
// peak near 1154, so it bounds a different quantity.
//
// The two terms are bounded differently, and the second is why the gross bound is not the same
// number as the L2 one. Over a whole vector the roundings are independent and average down, which
// is what 2.5e-3 of the reference RMS describes. One element does not average: the accumulation
// can carry the exact value across a rounding boundary, and the destination then lands on the
// wrong side of it.
//
// BF16 keeps eight significand bits, so inside a binade the spacing is 2^(e-7) and the relative
// ulp runs over (2^-8, 2^-7]. Half an ulp is therefore at most 2^-8 of the value - 3.9e-3 - and
// since no element exceeds the maximum reference, 4.0e-3 of that maximum bounds the rounding of
// any one of them. That is the format's number, not this fixture's.
//
// The case that showed the old bound was below it: at T=1 the reference peaks at 665, so 2.5e-3
// set the limit at 1.66, and an element whose exact value is 637.701 lands on 636 - the nearest
// BF16 there, the grid stepping by 4 - for an error of 1.70. The destination was right and the
// bound was not.
//
// Measured over the represented cases: rel-L2 2.09e-3 against 2.5e-3, worst element 1.79 against
// the 2.66 this bound allows. The L2 term is the binding one and stays where it was.
constexpr ReductionCriterion kSparseMoeNvfp4Tolerance{
    /*relative_l2*/ 2.5e-3,
    /*gross_absolute*/ 0.0,
    /*gross_relative_to_max_reference*/ 4.0e-3,
};

// The NVFP4 prefill route quantises activations as well as weights, so the A16 bound above does
// not apply to it. Against an oracle built from the same represented weights the remaining error
// is the four-bit activation plane, and a dot product does not average it away: the sum of K terms
// carries the same cancellation as the error does, so the output keeps the per-element relative
// error of e2m1 rather than that error over the square root of K. Both terms are relative for the
// same reason -- an absolute bound would mean nothing across cases whose references differ by an
// order of magnitude. It also depends on how far the experts spread in magnitude, which is a
// property of the fixture rather than of the route: every source matrix here carries its own
// divisor over a fourfold range, narrower than the 3.3x to 11.9x a real plane spans. Measured
// maximum over the prefill cases is 1.011e-1, and the worst element is half the gross bound.
constexpr ReductionCriterion kSparseMoeA4Tolerance{
    /*relative_l2*/ 1.8e-1,
    /*gross_absolute*/ 0.0,
    /*gross_relative_to_max_reference*/ 2.4e-1,
};

constexpr std::size_t kOutputGuardBytes = 256;
constexpr std::uint8_t kOutputGuardByte = 0xa5;

struct QuantGeometry {
    std::int32_t group;
    std::size_t code_bytes_per_group;
    std::size_t high_bytes_per_group;
};

QuantGeometry quant_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return {64, 32, 0};
    case QType::Q5_G64_FP16:
        return {64, 32, 8};
    case QType::Q6_G64_FP16:
        return {64, 32, 16};
    case QType::Q8_G32_FP16:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("sparse_moe test: unsupported codec");
    }
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

class GuardedBf16Output {
public:
    explicit GuardedBf16Output(std::size_t words)
        : storage_(words * sizeof(std::uint16_t), kOutputGuardBytes, kOutputGuardByte),
          words_(words) {}

    void* data() noexcept { return storage_.data(); }

    const void* data() const noexcept { return storage_.data(); }

    std::vector<double> values() const { return from_device_bf16(data(), words_); }

    int verify_guards(const std::string& label) const { return storage_.verify_guards(label); }

private:
    GuardedDeviceBuffer storage_;
    std::size_t words_;
};

// NVFP4 keeps its two planes in one payload at a fixed geometry -- codes from the start,
// scales at the 256-aligned end of them, and after those one weight divisor for each separately
// quantised matrix stacked into the plane -- so this cannot reuse the two independent buffers a
// row-split weight has.
class DeviceNvfp4 {
public:
    DeviceNvfp4(std::int32_t rows, std::int32_t columns, std::int32_t divisor_rows)
        : rows_(rows), columns_(columns), divisor_rows_(checked_divisor_rows(rows, divisor_rows)),
          code_plane_bytes_(static_cast<std::size_t>(rows) * columns / 2),
          scale_offset_((code_plane_bytes_ + 255) / 256 * 256),
          scale_plane_bytes_(static_cast<std::size_t>(rows) * columns / 16),
          storage_(scale_offset_ + scale_plane_bytes_ +
                   static_cast<std::size_t>(rows / divisor_rows) * sizeof(float)) {
        storage_.fill(0);
    }

    // The member initialiser divides by this, so the check cannot wait for the body.
    static std::int32_t checked_divisor_rows(std::int32_t rows, std::int32_t divisor_rows) {
        if (divisor_rows <= 0 || rows % divisor_rows != 0 || divisor_rows % 128 != 0) {
            throw std::invalid_argument("sparse_moe test: invalid NVFP4 divisor stride");
        }
        return divisor_rows;
    }

    void copy_rows(const quantized_weight::PackedWeight& source, std::int32_t destination_row) {
        const std::int32_t source_rows = source.weight.n;
        if (source.weight.qtype != QType::NVFP4 || source.weight.k != columns_ || source_rows < 1 ||
            (source_rows % 128) != 0 || destination_row < 0 || (destination_row % 128) != 0 ||
            destination_row > rows_ - source_rows) {
            throw std::invalid_argument("sparse_moe test: invalid NVFP4 row copy");
        }
        const std::size_t code_bytes  = static_cast<std::size_t>(source_rows) * columns_ / 2;
        const std::size_t scale_bytes = static_cast<std::size_t>(source_rows) * columns_ / 16;
        storage_.copy_from_host(source.payload.data(), code_bytes,
                                static_cast<std::size_t>(destination_row) * columns_ / 2);
        // A scale tile spans 128 rows, and both the source and the destination row are multiples
        // of that, so the tile index scales with the row exactly.
        storage_.copy_from_host(source.payload.data() + source.scale_plane_offset, scale_bytes,
                                scale_offset_ +
                                    static_cast<std::size_t>(destination_row) * columns_ / 16);
        if (source_rows > divisor_rows_ || destination_row % divisor_rows_ != 0) {
            throw std::invalid_argument("sparse_moe test: a copy may not span two divisors");
        }
        storage_.copy_from_host(source.payload.data() + source.weight_divisor_offset, sizeof(float),
                                scale_offset_ + scale_plane_bytes_ +
                                    static_cast<std::size_t>(destination_row / divisor_rows_) *
                                        sizeof(float));
        if (destination_row == 0) { weight_scale_divisor_ = source.weight.weight_scale_divisor; }
        input_scale_divisor_ = source.weight.input_scale_divisor;
    }

    Weight weight() const {
        Weight result{};
        result.payload              = storage_.p;
        result.payload_bytes        = storage_.bytes;
        result.high_plane_bytes     = 0;
        result.qtype                = QType::NVFP4;
        result.group_size           = 16;
        result.qdata                = storage_.p;
        result.qhigh                = nullptr;
        result.scales               = static_cast<std::uint8_t*>(storage_.p) + scale_offset_;
        result.n                    = rows_;
        result.k                    = columns_;
        result.group                = 16;
        result.layout               = QuantLayout::BlockScaleK16M128x4;
        result.scale_dtype          = DType::FP8_E4M3FN;
        result.ndim                 = 2;
        result.shape[0]             = rows_;
        result.shape[1]             = columns_;
        result.padded_shape[0]      = rows_;
        result.padded_shape[1]      = columns_;
        result.weight_scale_divisor = weight_scale_divisor_;
        result.input_scale_divisor  = input_scale_divisor_;
        result.weight_divisors =
            static_cast<std::uint8_t*>(storage_.p) + scale_offset_ + scale_plane_bytes_;
        result.weight_divisor_rows = divisor_rows_;
        return result;
    }

    int verify_rows(const std::string& label, const quantized_weight::PackedWeight& source,
                    std::int32_t destination_row) const {
        const std::int32_t source_rows = source.weight.n;
        const std::size_t code_bytes   = static_cast<std::size_t>(source_rows) * columns_ / 2;
        const std::size_t scale_bytes  = static_cast<std::size_t>(source_rows) * columns_ / 16;
        int failures                   = 0;
        failures +=
            verify_span(label + " code", source.payload.data(),
                        static_cast<std::size_t>(destination_row) * columns_ / 2, code_bytes);
        failures += verify_span(
            label + " scale", source.payload.data() + source.scale_plane_offset,
            scale_offset_ + static_cast<std::size_t>(destination_row) * columns_ / 16, scale_bytes);
        failures += verify_span(
            label + " divisor", source.payload.data() + source.weight_divisor_offset,
            scale_offset_ + scale_plane_bytes_ +
                static_cast<std::size_t>(destination_row / divisor_rows_) * sizeof(float),
            sizeof(float));
        return failures;
    }

private:
    int verify_span(const std::string& label, const std::uint8_t* expected, std::size_t offset,
                    std::size_t bytes) const {
        std::vector<std::uint8_t> actual(bytes);
        storage_.copy_to_host(actual.data(), bytes, offset);
        if (std::memcmp(actual.data(), expected, bytes) == 0) { return 0; }
        std::cerr << label << ": persistent weight was modified\n";
        return 1;
    }

    std::int32_t rows_;
    std::int32_t columns_;
    std::int32_t divisor_rows_;
    std::size_t code_plane_bytes_;
    std::size_t scale_offset_;
    std::size_t scale_plane_bytes_;
    DeviceBuffer storage_;
    float weight_scale_divisor_ = 1.0F;
    float input_scale_divisor_  = 1.0F;
};

class DeviceRowSplit {
public:
    DeviceRowSplit(QType qtype, std::int32_t rows, std::int32_t columns)
        : qtype_(qtype), rows_(rows), columns_(columns), geometry_(quant_geometry(qtype)),
          groups_per_row_(columns / geometry_.group),
          code_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.code_bytes_per_group),
          high_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.high_bytes_per_group),
          scale_row_bytes_(static_cast<std::size_t>(groups_per_row_) * sizeof(std::uint16_t)),
          codes_(static_cast<std::size_t>(rows) * code_row_bytes_),
          scales_(static_cast<std::size_t>(rows) * scale_row_bytes_) {
        codes_.fill(0);
        scales_.fill(0);
        if (high_row_bytes_ != 0) {
            high_ =
                std::make_unique<DeviceBuffer>(static_cast<std::size_t>(rows) * high_row_bytes_);
            high_->fill(0);
        }
    }

    void copy_rows(const quantized_weight::PackedWeight& source, std::int32_t destination_row) {
        const std::int32_t source_rows = source.weight.n;
        if (source.weight.qtype != qtype_ || source.weight.k != columns_ || source_rows < 1 ||
            destination_row < 0 || destination_row > rows_ - source_rows) {
            throw std::invalid_argument("sparse_moe test: invalid packed row copy");
        }
        codes_.copy_from_host(source.payload.data(),
                              static_cast<std::size_t>(source_rows) * code_row_bytes_,
                              static_cast<std::size_t>(destination_row) * code_row_bytes_);
        if (high_row_bytes_ != 0) {
            high_->copy_from_host(source.payload.data() + source.high_plane_offset,
                                  static_cast<std::size_t>(source_rows) * high_row_bytes_,
                                  static_cast<std::size_t>(destination_row) * high_row_bytes_);
        }
        scales_.copy_from_host(source.payload.data() + source.scale_plane_offset,
                               static_cast<std::size_t>(source_rows) * scale_row_bytes_,
                               static_cast<std::size_t>(destination_row) * scale_row_bytes_);
    }

    Weight weight() const {
        Weight result{};
        result.payload          = codes_.p;
        result.payload_bytes    = codes_.bytes + (high_ ? high_->bytes : 0) + scales_.bytes;
        result.high_plane_bytes = high_ ? high_->bytes : 0;
        result.qtype            = qtype_;
        result.group_size       = static_cast<std::uint32_t>(geometry_.group);
        result.qdata            = codes_.p;
        result.qhigh            = high_ ? high_->p : nullptr;
        result.scales           = scales_.p;
        result.n                = rows_;
        result.k                = columns_;
        result.group            = geometry_.group;
        result.layout           = QuantLayout::RowSplit;
        result.scale_dtype      = DType::FP16;
        result.ndim             = 2;
        result.shape[0]         = rows_;
        result.shape[1]         = columns_;
        result.padded_shape[0]  = rows_;
        result.padded_shape[1]  = columns_;
        return result;
    }

    int verify_rows(const std::string& label, const quantized_weight::PackedWeight& source,
                    std::int32_t destination_row) const {
        const std::int32_t source_rows = source.weight.n;
        int failures                   = 0;
        failures += verify_plane(label + " code", codes_, source.payload.data(),
                                 static_cast<std::size_t>(destination_row) * code_row_bytes_,
                                 static_cast<std::size_t>(source_rows) * code_row_bytes_);
        if (high_row_bytes_ != 0) {
            failures += verify_plane(label + " high", *high_,
                                     source.payload.data() + source.high_plane_offset,
                                     static_cast<std::size_t>(destination_row) * high_row_bytes_,
                                     static_cast<std::size_t>(source_rows) * high_row_bytes_);
        }
        failures += verify_plane(label + " scale", scales_,
                                 source.payload.data() + source.scale_plane_offset,
                                 static_cast<std::size_t>(destination_row) * scale_row_bytes_,
                                 static_cast<std::size_t>(source_rows) * scale_row_bytes_);
        return failures;
    }

private:
    static int verify_plane(const std::string& label, const DeviceBuffer& device,
                            const std::uint8_t* expected, std::size_t offset, std::size_t bytes) {
        std::vector<std::uint8_t> actual(bytes);
        device.copy_to_host(actual.data(), bytes, offset);
        if (std::memcmp(actual.data(), expected, bytes) == 0) { return 0; }
        std::cerr << label << ": persistent weight was modified\n";
        return 1;
    }

    QType qtype_;
    std::int32_t rows_;
    std::int32_t columns_;
    QuantGeometry geometry_;
    std::int32_t groups_per_row_;
    std::size_t code_row_bytes_;
    std::size_t high_row_bytes_;
    std::size_t scale_row_bytes_;
    DeviceBuffer codes_;
    std::unique_ptr<DeviceBuffer> high_;
    DeviceBuffer scales_;
};

Weight dense_bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
    Weight result{};
    result.payload         = data;
    result.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    result.qtype           = QType::BF16;
    result.qdata           = data;
    result.n               = rows;
    result.k               = columns;
    result.layout          = QuantLayout::Contiguous;
    result.ndim            = 2;
    result.shape[0]        = rows;
    result.shape[1]        = columns;
    result.padded_shape[0] = rows;
    result.padded_shape[1] = columns;
    return result;
}

struct RoutePattern {
    std::array<int, kTopK> selected;
    int tied_excluded = -1;
};

constexpr std::array<RoutePattern, 3> kRoutePatterns{{
    {{{255, 0, 17, 31, 63, 127, 191, 223}}, -1},
    {{{0, 17, 31, 63, 127, 191, 223, 254}}, 255},
    {{{223, 191, 127, 63, 31, 17, 0, 255}}, -1},
}};

std::vector<float> make_input(int pattern) {
    std::vector<float> input(kHidden);
    input[0] = 1.0f;
    for (int column = 1; column < kHidden; ++column) {
        input[column] = 0.025f + static_cast<float>((column * 7 + pattern * 11) % 19) * 0.002f;
    }
    for (std::size_t marker = 0; marker < kRoutePatterns.size(); ++marker) {
        input[kHidden - static_cast<int>(kRoutePatterns.size()) + static_cast<int>(marker)] = 0.0f;
    }
    input[kHidden - static_cast<int>(kRoutePatterns.size()) + pattern] = 1.0f;
    round_to_bf16(input);
    return input;
}

std::vector<float> make_residual(int pattern) {
    std::vector<float> residual(kHidden);
    for (int row = 0; row < kHidden; ++row) {
        residual[row] = 0.125f + static_cast<float>((row * 5 + pattern * 13) % 23) * 0.003f;
    }
    round_to_bf16(residual);
    return residual;
}

std::vector<float> make_router() {
    std::vector<float> router(static_cast<std::size_t>(kExperts + 1) * kHidden);
    for (int row = 0; row < kExperts + 1; ++row) {
        for (int column = 0; column < kHidden; ++column) {
            const int pattern = (column * 13 + 5) % 17 - 8;
            router[static_cast<std::size_t>(row) * kHidden + column] =
                static_cast<float>(pattern) * 0.001f;
        }
    }
    for (int expert = 0; expert < kExperts; ++expert) {
        router[static_cast<std::size_t>(expert) * kHidden] -= 8.0f;
    }
    for (std::size_t pattern = 0; pattern < kRoutePatterns.size(); ++pattern) {
        const int marker =
            kHidden - static_cast<int>(kRoutePatterns.size()) + static_cast<int>(pattern);
        const RoutePattern& route = kRoutePatterns[pattern];
        for (int rank = 0; rank < kTopK; ++rank) {
            const float score =
                rank == kTopK - 1 && route.tied_excluded >= 0 ? 2.0f : 4.0f - 0.25f * rank;
            router[static_cast<std::size_t>(route.selected[rank]) * kHidden + marker] +=
                score + 8.0f;
        }
        if (route.tied_excluded >= 0) {
            router[static_cast<std::size_t>(route.tied_excluded) * kHidden + marker] += 10.0f;
        }
    }
    router[static_cast<std::size_t>(kExperts) * kHidden] += 0.375f;
    round_to_bf16(router);
    return router;
}

std::vector<float> make_gate_up(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                                float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    const std::int32_t split = rows / 2;
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias       = row < split ? 0.75f : 1.15f;
        const float row_factor = 1.0f + static_cast<float>((row + seed) % 7) * 0.025f;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 11LL + column * 5LL + seed) % 15) - 7;
            source[static_cast<std::size_t>(row) * columns + column] =
                0.008f * expert_factor * row_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

std::vector<float> make_down(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                             float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias = 0.45f + static_cast<float>((row + seed) % 5) * 0.08f;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 7LL + column * 13LL + seed) % 17) - 8;
            source[static_cast<std::size_t>(row) * columns + column] =
                0.007f * expert_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

// Gate and up are quantised apart in every published checkpoint, so each is packed on its own
// with its own divisor and the two are stacked into one plane. This takes one of them out of the
// generator's combined matrix.
std::vector<float> take_rows(const std::vector<float>& matrix, std::int32_t columns,
                             std::int32_t begin, std::int32_t rows) {
    const auto first = matrix.begin() + static_cast<std::size_t>(begin) * columns;
    return std::vector<float>(first, first + static_cast<std::size_t>(rows) * columns);
}

struct HostExpert {
    int id;
    quantized_weight::PackedWeight gate;
    quantized_weight::PackedWeight up;
    quantized_weight::PackedWeight down;
};

const HostExpert& find_expert(const std::vector<HostExpert>& experts, int id) {
    const auto found = std::find_if(experts.begin(), experts.end(),
                                    [id](const HostExpert& expert) { return expert.id == id; });
    if (found == experts.end()) {
        throw std::logic_error("sparse_moe oracle selected an unpopulated expert");
    }
    return *found;
}

template <class Function>
void parallel_rows(std::int32_t rows, Function&& function) {
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(rows, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * thread) / threads);
        const std::int32_t end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) { function(row); }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
}

double dot_fp64(const std::vector<float>& matrix, std::int32_t row, std::int32_t columns,
                const std::vector<double>& input) {
    const float* weights = matrix.data() + static_cast<std::size_t>(row) * columns;
    double result        = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
        result += static_cast<double>(weights[column]) * input[column];
    }
    return result;
}

// The one SparseMoe oracle. It directly evaluates the complete public formula from represented
// BF16 values and independently decoded logical weights, with FP64 accumulation throughout.
// It has no production route, staging cast, workspace dtype, reduction tree, or output rounding.
std::vector<double> sparse_moe_oracle(const std::vector<float>& input,
                                      const std::vector<float>& residual,
                                      const std::vector<float>& router,
                                      const std::vector<HostExpert>& experts,
                                      const quantized_weight::PackedWeight& shared_gate,
                                      const quantized_weight::PackedWeight& shared_up,
                                      const quantized_weight::PackedWeight& shared_down,
                                      const RoutePattern& intended_route) {
    const std::vector<double> x(input.begin(), input.end());
    std::vector<double> scores(kExperts + 1);
    parallel_rows(kExperts + 1,
                  [&](std::int32_t row) { scores[row] = dot_fp64(router, row, kHidden, x); });

    std::vector<int> ranked(kExperts);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::sort(ranked.begin(), ranked.end(), [&](int left, int right) {
        return scores[left] > scores[right] || (scores[left] == scores[right] && left < right);
    });

    std::array<int, kTopK> selected{};
    std::array<double, kTopK> route_weight{};
    double route_denominator = 0.0;
    for (int route = 0; route < kTopK; ++route) {
        selected[route]     = ranked[route];
        route_weight[route] = std::exp(scores[selected[route]] - scores[selected[0]]);
        route_denominator += route_weight[route];
    }
    for (double& weight : route_weight) { weight /= route_denominator; }
    if (selected != intended_route.selected) {
        throw std::logic_error("sparse_moe test router did not create its intended ordered top-8");
    }
    const double shared_scale = 1.0 / (1.0 + std::exp(-scores[kExperts]));

    std::array<std::vector<double>, kTopK> routed_activation;
    for (int route = 0; route < kTopK; ++route) {
        routed_activation[route].resize(kIntermediate);
        const HostExpert& expert = find_expert(experts, selected[route]);
        parallel_rows(kIntermediate, [&](std::int32_t row) {
            const double gate             = dot_fp64(expert.gate.dequant, row, kHidden, x);
            const double up               = dot_fp64(expert.up.dequant, row, kHidden, x);
            routed_activation[route][row] = (gate / (1.0 + std::exp(-gate))) * up;
        });
    }

    std::vector<double> shared_activation(kIntermediate);
    parallel_rows(kIntermediate, [&](std::int32_t row) {
        const double gate      = dot_fp64(shared_gate.dequant, row, kHidden, x);
        const double up        = dot_fp64(shared_up.dequant, row, kHidden, x);
        shared_activation[row] = (gate / (1.0 + std::exp(-gate))) * up;
    });

    std::vector<double> output(kHidden);
    parallel_rows(kHidden, [&](std::int32_t row) {
        double value = static_cast<double>(residual[row]);
        for (int route = 0; route < kTopK; ++route) {
            const HostExpert& expert = find_expert(experts, selected[route]);
            value += route_weight[route] *
                     dot_fp64(expert.down.dequant, row, kIntermediate, routed_activation[route]);
        }
        value +=
            shared_scale * dot_fp64(shared_down.dequant, row, kIntermediate, shared_activation);
        output[row] = value;
    });
    return output;
}

struct CodecProfile {
    const char* name;
    QType routed_gate_up;
    QType routed_down;
    QType shared;
    const ReductionCriterion* tolerance;
    // The bound from the token count at which this profile starts quantising its activations, and
    // that token count. Zero where no route of the profile does.
    const ReductionCriterion* quantized_activation_tolerance;
    std::int32_t quantized_activation_from;
    std::span<const std::int32_t> token_cases;
    bool verify_graph_replay;

    [[nodiscard]] const ReductionCriterion& criterion(std::int32_t tokens) const {
        return quantized_activation_from != 0 && tokens >= quantized_activation_from
                   ? *quantized_activation_tolerance
                   : *tolerance;
    }
};

// The four expert matrices take whichever storage the profile names; nothing else in the
// fixture depends on which.
class ExpertPlane {
public:
    ExpertPlane(QType qtype, std::int32_t rows, std::int32_t columns, std::int32_t divisor_rows) {
        if (qtype == QType::NVFP4) {
            nvfp4_.emplace(rows, columns, divisor_rows);
        } else {
            row_split_.emplace(qtype, rows, columns);
        }
    }

    void copy_rows(const quantized_weight::PackedWeight& source, std::int32_t destination_row) {
        if (nvfp4_) {
            nvfp4_->copy_rows(source, destination_row);
        } else {
            row_split_->copy_rows(source, destination_row);
        }
    }

    Weight weight() const { return nvfp4_ ? nvfp4_->weight() : row_split_->weight(); }

    int verify_rows(const std::string& label, const quantized_weight::PackedWeight& source,
                    std::int32_t destination_row) const {
        return nvfp4_ ? nvfp4_->verify_rows(label, source, destination_row)
                      : row_split_->verify_rows(label, source, destination_row);
    }

private:
    std::optional<DeviceNvfp4> nvfp4_;
    std::optional<DeviceRowSplit> row_split_;
};

float decode_e4m3_scale(std::uint8_t bits) {
    const int exponent = (bits >> 3) & 15;
    const int mantissa = bits & 7;
    const float value  = exponent == 0
                             ? std::ldexp(static_cast<float>(mantissa) / 8.0F, -6)
                             : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    return ((bits >> 7) & 1) != 0 ? -value : value;
}

// The oracle needs the values the weight represents. Rather than quantise a second time and risk
// disagreeing with the packer, decode the payload the packer produced: by construction those are
// the values, and the decode is the format's own definition.
std::vector<float> decode_nvfp4_payload(const quantized_weight::PackedWeight& packed,
                                        std::int32_t n, std::int32_t k) {
    static constexpr float kMagnitude[8]{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    std::vector<float> out(static_cast<std::size_t>(n) * k);
    const std::int32_t k_tiles = k / 64;
    const float inverse        = 1.0F / packed.weight.weight_scale_divisor;
    for (std::int32_t row = 0; row < n; ++row) {
        const std::int32_t row_tile  = row / 128;
        const std::int32_t row_inner = row % 128;
        for (std::int32_t column = 0; column < k; ++column) {
            const std::uint8_t byte =
                packed.payload[static_cast<std::size_t>(row) * k / 2 + column / 2];
            const unsigned nibble    = (column & 1) != 0 ? (byte >> 4) : (byte & 0x0fU);
            const std::int32_t group = column / 16;
            const std::size_t scale_at =
                static_cast<std::size_t>(packed.scale_plane_offset) +
                static_cast<std::size_t>(row_tile * k_tiles + group / 4) * 512U +
                static_cast<std::size_t>(row_inner % 32) * 16U +
                static_cast<std::size_t>(row_inner / 32) * 4U + static_cast<std::size_t>(group % 4);
            const float magnitude = kMagnitude[nibble & 7U];
            out[static_cast<std::size_t>(row) * k + column] =
                ((nibble & 8U) != 0 ? -magnitude : magnitude) *
                decode_e4m3_scale(packed.payload[scale_at]) * inverse;
        }
    }
    return out;
}

// A patterned NVFP4 matrix carries its own values; the groupwise path quantises the fixture's.
// Either way the oracle reads the represented values this returns.
quantized_weight::PackedWeight pack_expert(QType qtype, const std::vector<float>& values,
                                           std::int32_t n, std::int32_t k, std::uint32_t seed,
                                           float weight_divisor = 0.125F) {
    if (qtype == QType::NVFP4) {
        quantized_weight::PatternedWeightOptions options;
        options.weight_scale_divisor = weight_divisor;
        options.input_scale_divisor  = 3.5F;
        quantized_weight::PackedWeight packed =
            quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
        packed.dequant = decode_nvfp4_payload(packed, n, k);
        return packed;
    }
    return quantized_weight::pack_row_split_lowbit(values, n, k, qtype);
}

class SparseMoeFixture {
public:
    explicit SparseMoeFixture(const CodecProfile& profile)
        : profile_(profile), router_(make_router()), router_bits_(bf16_bits(router_)),
          device_router_(to_device(router_bits_)),
          routed_gate_(profile.routed_gate_up, kRoutedGateRows, kHidden, kIntermediate),
          routed_down_(profile.routed_down, kRoutedDownRows, kIntermediate, kHidden),
          shared_gate_(profile.shared, kSharedGateRows, kHidden, kIntermediate),
          shared_down_device_(profile.shared, kHidden, kIntermediate, kHidden) {
        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            inputs_.push_back(make_input(pattern));
            residuals_.push_back(make_residual(pattern));
        }

        std::vector<int> expert_ids;
        for (const RoutePattern& route : kRoutePatterns) {
            for (int expert : route.selected) {
                if (std::find(expert_ids.begin(), expert_ids.end(), expert) == expert_ids.end()) {
                    expert_ids.push_back(expert);
                }
            }
        }
        std::sort(expert_ids.begin(), expert_ids.end());
        // A published checkpoint quantises every expert matrix on its own, so no two of them share
        // a divisor. Here they differ by construction, which is what makes a lookup that ignores
        // the row visible to the oracle.
        const auto expert_divisor = [](int expert, int matrix) {
            return 0.0625F * (1.0F + static_cast<float>((expert * 7 + matrix * 3) % 13) * 0.25F);
        };
        for (int expert : expert_ids) {
            const float factor       = 0.8f + static_cast<float>((expert * 3) % 11) * 0.045f;
            const std::uint32_t seed = 100U + static_cast<std::uint32_t>(expert);
            const std::vector<float> gate_up_source =
                make_gate_up(kExpertGateRows, kHidden, seed, factor);
            auto gate = pack_expert(profile.routed_gate_up,
                                    take_rows(gate_up_source, kHidden, 0, kIntermediate),
                                    kIntermediate, kHidden, seed, expert_divisor(expert, 0));
            // A different seed, not just a different divisor: the NVFP4 packer derives its codes
            // from the seed, so one seed for both halves would leave them bit-identical and a route
            // that read the wrong half would look right.
            auto up =
                pack_expert(profile.routed_gate_up,
                            take_rows(gate_up_source, kHidden, kIntermediate, kIntermediate),
                            kIntermediate, kHidden, seed ^ 0x9e3779b9U, expert_divisor(expert, 1));
            auto down =
                pack_expert(profile.routed_down,
                            make_down(kHidden, kIntermediate,
                                      300U + static_cast<std::uint32_t>(expert), factor),
                            kHidden, kIntermediate, 300U + static_cast<std::uint32_t>(expert),
                            expert_divisor(expert, 2));
            routed_gate_.copy_rows(gate, expert * kExpertGateRows);
            routed_gate_.copy_rows(up, expert * kExpertGateRows + kIntermediate);
            routed_down_.copy_rows(down, expert * kHidden);
            experts_.push_back({expert, std::move(gate), std::move(up), std::move(down)});
        }

        const std::vector<float> shared_source =
            make_gate_up(kSharedGateRows, kHidden, 0x512U, 0.93f);
        shared_gate_host_ =
            pack_expert(profile.shared, take_rows(shared_source, kHidden, 0, kIntermediate),
                        kIntermediate, kHidden, 0x512U, 0.3125F);
        shared_up_host_ = pack_expert(
            profile.shared, take_rows(shared_source, kHidden, kIntermediate, kIntermediate),
            kIntermediate, kHidden, 0x513U, 0.4375F);
        shared_down_host_ =
            pack_expert(profile.shared, make_down(kHidden, kIntermediate, 0x731U, 0.87f), kHidden,
                        kIntermediate, 0x731U);
        shared_gate_.copy_rows(shared_gate_host_, 0);
        shared_gate_.copy_rows(shared_up_host_, kIntermediate);
        shared_down_device_.copy_rows(shared_down_host_, 0);

        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            references_.push_back(sparse_moe_oracle(inputs_[pattern], residuals_[pattern], router_,
                                                    experts_, shared_gate_host_, shared_up_host_,
                                                    shared_down_host_, kRoutePatterns[pattern]));
        }
    }

    int run(std::int32_t tokens, int first_pattern, bool graph_replay) {
        const std::string label = std::string(profile_.name) + " T=" + std::to_string(tokens);
        std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<float> residual(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<double> reference(static_cast<std::size_t>(kHidden) * tokens);
        for (std::int32_t token = 0; token < tokens; ++token) {
            const int pattern = (first_pattern + token) % static_cast<int>(kRoutePatterns.size());
            std::copy(inputs_[pattern].begin(), inputs_[pattern].end(),
                      input.begin() + static_cast<std::size_t>(token) * kHidden);
            std::copy(residuals_[pattern].begin(), residuals_[pattern].end(),
                      residual.begin() + static_cast<std::size_t>(token) * kHidden);
            std::copy(references_[pattern].begin(), references_[pattern].end(),
                      reference.begin() + static_cast<std::size_t>(token) * kHidden);
        }

        const std::vector<std::uint16_t> input_bits    = bf16_bits(input);
        const std::vector<std::uint16_t> residual_bits = bf16_bits(residual);
        DeviceBuffer device_input                      = to_device(input_bits);
        DeviceBuffer residual_seed                     = to_device(residual_bits);
        GuardedBf16Output destination_storage(residual.size());
        cuda_check(cudaMemcpy(destination_storage.data(), residual_seed.p, residual_seed.bytes,
                              cudaMemcpyDeviceToDevice),
                   "seed SparseMoe destination");

        const ops::SparseMoeWeights weights{
            dense_bf16_weight(device_router_.p, kExperts + 1, kHidden),
            routed_gate_.weight(),
            routed_down_.weight(),
            shared_gate_.weight(),
            shared_down_device_.weight(),
        };
        Tensor x(device_input.p, DType::BF16, {kHidden, tokens});
        Tensor destination(destination_storage.data(), DType::BF16, {kHidden, tokens});
        const std::size_t workspace_bytes = ops::sparse_moe_workspace_capacity_bytes(
            weights.routed_gate_up.qtype, weights.routed_down.qtype, tokens, tokens);
        WorkspaceArena workspace(workspace_bytes);

        if (graph_replay) {
            cudaStream_t stream  = nullptr;
            cudaGraph_t graph    = nullptr;
            cudaGraphExec_t exec = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "create SparseMoe graph stream");
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                       "begin SparseMoe graph capture");
            cuda_check(cudaMemcpyAsync(destination.data, residual_seed.p, destination.bytes(),
                                       cudaMemcpyDeviceToDevice, stream),
                       "capture SparseMoe residual seed");
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            stream);
            cuda_check(cudaStreamEndCapture(stream, &graph), "end SparseMoe graph capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                       "instantiate SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "launch SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "replay SparseMoe graph");
            cuda_check(cudaStreamSynchronize(stream), "synchronize SparseMoe graph");
            cuda_check(cudaGraphExecDestroy(exec), "destroy SparseMoe graph executable");
            cuda_check(cudaGraphDestroy(graph), "destroy SparseMoe graph");
            cuda_check(cudaStreamDestroy(stream), "destroy SparseMoe graph stream");
        } else {
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            nullptr);
            cuda_synchronize();
        }

        int failures = verify_reduction(label, destination_storage.values(), reference,
                                        profile_.criterion(tokens));
        failures += destination_storage.verify_guards(label);
        failures +=
            verify_exact((label + " input preservation").c_str(),
                         from_device<std::uint16_t>(device_input, input_bits.size()), input_bits);
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        return failures;
    }

    int verify_persistent_inputs() const {
        int failures = 0;
        failures += verify_exact((std::string(profile_.name) + " router preservation").c_str(),
                                 from_device<std::uint16_t>(device_router_, router_bits_.size()),
                                 router_bits_);
        for (const HostExpert& expert : experts_) {
            failures += routed_gate_.verify_rows(
                std::string(profile_.name) + " routed gate expert " + std::to_string(expert.id),
                expert.gate, expert.id * kExpertGateRows);
            failures += routed_gate_.verify_rows(
                std::string(profile_.name) + " routed up expert " + std::to_string(expert.id),
                expert.up, expert.id * kExpertGateRows + kIntermediate);
            failures += routed_down_.verify_rows(
                std::string(profile_.name) + " routed down expert " + std::to_string(expert.id),
                expert.down, expert.id * kHidden);
        }
        failures += shared_gate_.verify_rows(std::string(profile_.name) + " shared gate",
                                             shared_gate_host_, 0);
        failures += shared_gate_.verify_rows(std::string(profile_.name) + " shared up",
                                             shared_up_host_, kIntermediate);
        failures += shared_down_device_.verify_rows(std::string(profile_.name) + " shared down",
                                                    shared_down_host_, 0);
        return failures;
    }

private:
    const CodecProfile& profile_;
    std::vector<float> router_;
    std::vector<std::uint16_t> router_bits_;
    DeviceBuffer device_router_;
    ExpertPlane routed_gate_;
    ExpertPlane routed_down_;
    ExpertPlane shared_gate_;
    ExpertPlane shared_down_device_;
    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> residuals_;
    std::vector<HostExpert> experts_;
    quantized_weight::PackedWeight shared_gate_host_;
    quantized_weight::PackedWeight shared_up_host_;
    quantized_weight::PackedWeight shared_down_host_;
    std::vector<std::vector<double>> references_;
};

int run_profile(const CodecProfile& profile) {
    SparseMoeFixture fixture(profile);
    int failures        = 0;
    std::size_t witness = 0;
    for (std::size_t index = 0; index < profile.token_cases.size(); ++index) {
        const std::int32_t tokens = profile.token_cases[index];
        witness =
            std::max(witness, ops::sparse_moe_workspace_capacity_bytes(
                                  profile.routed_gate_up, profile.routed_down, tokens, tokens));
        // Decode starts with the exact top-8 boundary tie; multi-token cases cycle the tie,
        // high/low expert ids, and a different ordering of the same experts.
        failures +=
            fixture.run(tokens, index == 0 ? 1 : 0, profile.verify_graph_replay && index == 1);
    }
    const std::size_t interval = ops::sparse_moe_workspace_capacity_bytes(
        profile.routed_gate_up, profile.routed_down, 1, profile.token_cases.back());
    if (interval != witness) {
        std::cerr << profile.name << ": interval workspace capacity missed a route witness\n";
        ++failures;
    }
    failures += fixture.verify_persistent_inputs();
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // These are public-behavior cases, not route assertions. They exercise decode (T=1), the
    // Small-T supported-domain edges, each profile's first prefill T, the wide-prefill boundary,
    // and one call crossing the 4096-token internal slice without observing any private plan.
    constexpr std::array<std::int32_t, 6> kQ4Q5Tokens{{1, 2, 46, 47, 768, 4097}};
    constexpr std::array<std::int32_t, 5> kQ4Q6Tokens{{1, 2, 46, 47, 768}};
    // 4097 ends on a one-token slice, which is the only way a profile whose prefill starts at
    // twenty reaches the route's small-token branch.
    constexpr std::array<std::int32_t, 6> kQ8Q8Tokens{{1, 2, 19, 20, 768, 4097}};
    // NVFP4 walks both frontiers: its Small-T edge and first prefill T, where the activation stops
    // being represented, and then the prefill tile boundary and the internal slice.
    constexpr std::array<std::int32_t, 7> kNvfp4Tokens{{1, 2, 12, 13, 64, 768, 4097}};
    const std::array<CodecProfile, 4> profiles{{
        {"sparse_moe q4+q5 a16", QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q8_G32_FP16,
         &kSparseMoeA16Tolerance, nullptr, 0, kQ4Q5Tokens, true},
        {"sparse_moe q4+q6 a16", QType::Q4_G64_FP16, QType::Q6_G64_FP16, QType::Q8_G32_FP16,
         &kSparseMoeA16Tolerance, nullptr, 0, kQ4Q6Tokens, false},
        {"sparse_moe q8+q8 a16", QType::Q8_G32_FP16, QType::Q8_G32_FP16, QType::Q8_G32_FP16,
         &kSparseMoeA16Tolerance, nullptr, 0, kQ8Q8Tokens, false},
        {"sparse_moe nvfp4", QType::NVFP4, QType::NVFP4, QType::NVFP4, &kSparseMoeNvfp4Tolerance,
         &kSparseMoeA4Tolerance, 13, kNvfp4Tokens, true},
    }};

    int failures = 0;
    for (const CodecProfile& profile : profiles) { failures += run_profile(profile); }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe correctness\n";
    return failures == 0 ? 0 : 1;
}
