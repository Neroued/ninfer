#include "fixture.h"
#include "cuda_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include "ops/suite.h"

namespace ninfer::test::moe {
namespace {
constexpr int kExpertGateRows = 2 * kIntermediate, kRoutedGateRows = kExperts * kExpertGateRows;
constexpr int kRoutedDownRows = kExperts * kHidden, kSharedGateRows = 2 * kIntermediate;

ops::MoeProfile profile(QType gate_up, QType down) {
    const ops::SwiGLUProfile shared{kHidden,
                                    kIntermediate,
                                    QType::Q8_G32_FP16,
                                    QType::Q8_G32_FP16,
                                    QuantLayout::RowSplit,
                                    QuantLayout::RowSplit};
    return {{{kHidden, kIntermediate, gate_up, down, QuantLayout::RowSplit, QuantLayout::RowSplit},
             kExperts,
             kTopK},
            shared};
}

// Test banks deliberately permit independent allocations for their encoded planes.
ExpertMatrixDeviceView affine(const Weight& weight) {
    const auto groups     = std::uint64_t(weight.n / kExperts) * (weight.k / weight.group);
    const auto high_bytes = weight.qtype == QType::Q5_G64_FP16   ? 8
                            : weight.qtype == QType::Q6_G64_FP16 ? 16
                                                                 : 0;
    return {static_cast<const std::byte*>(weight.qdata),
            static_cast<const std::byte*>(weight.qhigh),
            static_cast<const std::byte*>(weight.scales),
            groups * 32,
            groups * high_bytes,
            groups * 2};
}

ops::MoeWeights weights(const Weight& control, const Weight& gate_up, const Weight& down,
                        const Weight& shared_gate_up, const Weight& shared_down) {
    return {control,
            {{affine(gate_up), affine(down), kExperts, gate_up.device}, {}},
            {{shared_gate_up}, {shared_down}}};
}

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
        throw std::invalid_argument("moe fixture: unsupported codec");
    }
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

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
          codes_(ninfer::test::cuda_environment().context.device(),
                 static_cast<std::size_t>(rows) * code_row_bytes_),
          scales_(ninfer::test::cuda_environment().context.device(),
                  static_cast<std::size_t>(rows) * scale_row_bytes_) {
        codes_.fill(0);
        scales_.fill(0);
        if (high_row_bytes_ != 0) {
            high_ = std::make_unique<DeviceBuffer>(codes_.device(), static_cast<std::size_t>(rows) *
                                                                        high_row_bytes_);
            high_->fill(0);
        }
    }

    void copy_rows(const quantized_weight::PackedWeight& source, std::int32_t destination_row) {
        const std::int32_t source_rows = source.weight.n;
        if (source.weight.qtype != qtype_ || source.weight.k != columns_ || source_rows < 1 ||
            destination_row < 0 || destination_row > rows_ - source_rows) {
            throw std::invalid_argument("moe fixture: invalid packed row copy");
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
        result.device           = codes_.device();
        result.payload          = codes_.data();
        result.payload_bytes    = codes_.size() + (high_ ? high_->size() : 0) + scales_.size();
        result.high_plane_bytes = high_ ? high_->size() : 0;
        result.qtype            = qtype_;
        result.group_size       = static_cast<std::uint32_t>(geometry_.group);
        result.qdata            = codes_.data();
        result.qhigh            = high_ ? high_->data() : nullptr;
        result.scales           = scales_.data();
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

Weight dense_bf16_weight(ConstMemorySpan storage, std::int32_t rows, std::int32_t columns) {
    Weight result{};
    result.device          = storage.device;
    result.payload         = storage.data;
    result.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    result.qtype           = QType::BF16;
    result.qdata           = storage.data;
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

template <class Function>
void for_oracle_rows(std::int32_t rows, Function&& function) {
    ninfer::test::parallel_rows(rows, std::uint64_t(rows) * kHidden, [&](int begin, int end) {
        for (int row = begin; row < end; ++row) function(row);
    });
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

RouteReference route_reference(const std::vector<float>& input, const std::vector<float>& weights) {
    const std::vector<double> x(input.begin(), input.end());
    std::vector<double> scores(kExperts + 1);
    for_oracle_rows(kExperts + 1,
                    [&](int row) { scores[row] = dot_fp64(weights, row, kHidden, x); });
    std::vector<int> ids(kExperts);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    });
    RouteReference out;
    double denominator = 0;
    for (int j = 0; j < kTopK; ++j) {
        out.ids[j]          = ids[j];
        out.coefficients[j] = std::exp(scores[ids[j]] - scores[ids[0]]);
        denominator += out.coefficients[j];
    }
    for (auto& v : out.coefficients) { v /= denominator; }
    out.shared = 1 / (1 + std::exp(-scores[kExperts]));
    return out;
}

std::vector<double> ffn_reference(const std::vector<float>& input,
                                  const quantized_weight::PackedWeight& gu,
                                  const quantized_weight::PackedWeight& down) {
    const std::vector<double> x(input.begin(), input.end());
    std::vector<double> activation(kIntermediate), out(kHidden);
    for_oracle_rows(kIntermediate, [&](int row) {
        const double gate = dot_fp64(gu.dequant, row, kHidden, x);
        const double up   = dot_fp64(gu.dequant, kIntermediate + row, kHidden, x);
        activation[row]   = gate / (1 + std::exp(-gate)) * up;
    });
    for_oracle_rows(kHidden, [&](int row) {
        out[row] = dot_fp64(down.dequant, row, kIntermediate, activation);
    });
    return out;
}
} // namespace

const std::vector<double>& LogicalCase::expert(int id) const {
    const auto pos = std::find(expert_ids.begin(), expert_ids.end(), id);
    if (pos == expert_ids.end()) { throw std::logic_error("MoE oracle has no represented expert"); }
    return expert_values[pos - expert_ids.begin()];
}

struct Fixture::Impl {
    ops::MoeProfile description;
    std::vector<float> router;
    std::vector<std::uint16_t> router_bits;
    DeviceBuffer device_router;
    DeviceRowSplit gate_up, down, shared_gu, shared_down;
    std::vector<HostExpert> experts;
    quantized_weight::PackedWeight shared_gu_host, shared_down_host;
    std::vector<LogicalCase> cases;
    ops::MoeWeights operands;

    Impl(QType gu_type, QType down_type)
        : description(moe::profile(gu_type, down_type)), router(make_router()),
          router_bits(bf16_bits(router)), device_router(to_device(router_bits)),
          gate_up(gu_type, kRoutedGateRows, kHidden),
          down(down_type, kRoutedDownRows, kIntermediate),
          shared_gu(QType::Q8_G32_FP16, kSharedGateRows, kHidden),
          shared_down(QType::Q8_G32_FP16, kHidden, kIntermediate) {
        std::vector<int> ids;
        for (const auto& route : kRoutePatterns) {
            for (int e : route.selected) {
                if (std::find(ids.begin(), ids.end(), e) == ids.end()) { ids.push_back(e); }
            }
        }
        std::sort(ids.begin(), ids.end());
        for (int e : ids) {
            const float factor = 0.8f + float((e * 3) % 11) * 0.045f;
            auto gu            = quantized_weight::pack_row_split_lowbit(
                make_gate_up(kExpertGateRows, kHidden, 100U + e, factor), kExpertGateRows, kHidden,
                gu_type);
            auto dn = quantized_weight::pack_row_split_lowbit(
                make_down(kHidden, kIntermediate, 300U + e, factor), kHidden, kIntermediate,
                down_type);
            gate_up.copy_rows(gu, e * kExpertGateRows);
            down.copy_rows(dn, e * kHidden);
            experts.push_back({e, std::move(gu), std::move(dn)});
        }
        shared_gu_host = quantized_weight::pack_q8_g32_row_split(
            make_gate_up(kSharedGateRows, kHidden, 0x512U, 0.93f), kSharedGateRows, kHidden);
        shared_down_host = quantized_weight::pack_q8_g32_row_split(
            make_down(kHidden, kIntermediate, 0x731U, 0.87f), kHidden, kIntermediate);
        shared_gu.copy_rows(shared_gu_host, 0);
        shared_down.copy_rows(shared_down_host, 0);
        operands =
            moe::weights(dense_bf16_weight(device_router.span(), kExperts + 1, kHidden),
                         gate_up.weight(), down.weight(), shared_gu.weight(), shared_down.weight());
        for (int p = 0; p < int(kRoutePatterns.size()); ++p) {
            LogicalCase c;
            c.input    = make_input(p);
            c.residual = make_residual(p);
            c.route    = route_reference(c.input, router);
            if (c.route.ids != kRoutePatterns[p].selected) {
                throw std::logic_error("unexpected fixture route");
            }
            c.expert_ids = ids;
            for (const auto& e : experts) {
                c.expert_values.push_back(ffn_reference(c.input, e.gate_up, e.down));
            }
            c.shared = ffn_reference(c.input, shared_gu_host, shared_down_host);
            c.complete.resize(kHidden);
            for (int h = 0; h < kHidden; ++h) {
                double value = c.residual[h];
                for (int j = 0; j < kTopK; ++j) {
                    value += c.route.coefficients[j] * c.expert(c.route.ids[j])[h];
                }
                c.complete[h] = value + c.route.shared * c.shared[h];
            }
            cases.push_back(std::move(c));
        }
    }
};

Fixture::Fixture(QType gu, QType down) : impl_(std::make_unique<Impl>(gu, down)) {}

Fixture::~Fixture() = default;

const ops::MoeProfile& Fixture::profile() const { return impl_->description; }

const ops::MoeWeights& Fixture::weights() const { return impl_->operands; }

struct HostBank::Impl {
    struct Matrix {
        HostBuffer storage;
        WeightParent parent;
        WeightBankView view;
        CudaHostRegistration registration;

        Matrix(const Fixture& fixture, bool gate_up) {
            const auto& profile = fixture.profile().routed.ffn;
            const int rows      = gate_up ? 2 * kIntermediate : kHidden;
            const int columns   = gate_up ? kHidden : kIntermediate;
            const auto type     = gate_up ? profile.gate_up_type : profile.down_type;
            const std::array<std::uint64_t, 2> shape{std::uint64_t(kExperts) * rows,
                                                     std::uint64_t(columns)};
            parent.geometry = weight_geometry(type, QuantLayout::RowSplit, shape);
            storage         = HostBuffer(parent.geometry.bytes, 4096);
            std::memset(storage.data(), 0, storage.size());
            parent.data   = static_cast<const std::byte*>(storage.data());
            parent.device = Device::cpu();
            view          = bind_weight_bank(
                parent, describe_weight_bank(parent.geometry, {0, kExperts, std::uint64_t(rows),
                                                                        std::uint64_t(rows)}));
            for (const auto& expert : fixture.experts()) {
                const auto& packed = gate_up ? expert.gate_up : expert.down;
                for (const auto [plane, offset] :
                     {std::pair{view.layout.codes, std::uint64_t{0}},
                      std::pair{view.layout.high, packed.high_plane_offset},
                      std::pair{view.layout.scales, packed.scale_plane_offset}}) {
                    if (!plane.expert_bytes) { continue; }
                    const auto destination = view.plane(plane, expert.id);
                    std::memcpy(const_cast<void*>(destination.data), packed.payload.data() + offset,
                                destination.bytes);
                }
            }
        }
    };

    Matrix gate_up, down;

    explicit Impl(const Fixture& fixture) : gate_up(fixture, true), down(fixture, false) {}
};

HostBank::HostBank(const Fixture& fixture) : impl_(std::make_unique<Impl>(fixture)) {}

HostBank::~HostBank() = default;

void HostBank::register_upload(Device device) {
    for (auto* matrix : {&impl_->gate_up, &impl_->down}) {
        if (!matrix->registration.view().host.bytes) {
            matrix->registration = CudaHostRegistration(matrix->storage.span(), device);
        }
    }
}

ExpertBankSource HostBank::source() const {
    return {impl_->gate_up.view, impl_->down.view, impl_->gate_up.registration.view(),
            impl_->down.registration.view()};
}

std::span<const HostExpert> Fixture::experts() const { return impl_->experts; }

const LogicalCase& Fixture::logical(int pattern) const { return impl_->cases.at(pattern); }

Batch Fixture::batch(int tokens, int first_pattern) const {
    Batch out;
    out.input.resize(std::size_t(kHidden) * tokens);
    out.residual.resize(out.input.size());
    out.ids.resize(std::size_t(kTopK) * tokens);
    out.coefficients.resize(out.ids.size());
    out.shared_coefficients.resize(tokens);
    out.complete.resize(out.input.size());
    out.patterns.resize(tokens);
    for (int t = 0; t < tokens; ++t) {
        const int p     = (first_pattern + t) % impl_->cases.size();
        const auto& c   = logical(p);
        out.patterns[t] = p;
        for (int h = 0; h < kHidden; ++h) {
            out.input[std::size_t(t) * kHidden + h]    = f32_to_bf16(c.input[h]);
            out.residual[std::size_t(t) * kHidden + h] = f32_to_bf16(c.residual[h]);
            out.complete[std::size_t(t) * kHidden + h] = c.complete[h];
        }
        for (int j = 0; j < kTopK; ++j) {
            out.ids[t * kTopK + j]          = c.route.ids[j];
            out.coefficients[t * kTopK + j] = float(c.route.coefficients[j]);
        }
        out.shared_coefficients[t] = float(c.route.shared);
    }
    return out;
}

std::vector<double> Fixture::contribution(const Batch& batch, ops::ExpertSubsetKind subset,
                                          std::span<const int> mask, bool shared, bool residual,
                                          std::span<const int> aliases) const {
    const int tokens = batch.patterns.size();
    std::vector<double> out(std::size_t(tokens) * kHidden);
    for (int t = 0; t < tokens; ++t) {
        const auto& c = logical(batch.patterns[t]);
        for (int h = 0; h < kHidden; ++h) {
            double value = residual ? c.residual[h] : 0;
            for (int j = 0; j < kTopK; ++j) {
                const int id    = batch.ids[t * kTopK + j];
                const bool keep = subset == ops::ExpertSubsetKind::All ||
                                  (subset == ops::ExpertSubsetKind::MaskOnes && mask[id] == 1) ||
                                  (subset == ops::ExpertSubsetKind::MaskZeros && mask[id] == 0);
                if (keep) {
                    value += double(batch.coefficients[t * kTopK + j]) *
                             c.expert(aliases.empty() ? id : aliases[id])[h];
                }
            }
            if (shared) { value += double(batch.shared_coefficients[t]) * c.shared[h]; }
            out[std::size_t(t) * kHidden + h] = value;
        }
    }
    return out;
}

int Fixture::verify_preserved() const {
    const auto& p = *impl_;
    int errors    = 0;
    errors += verify_exact("MoE control preservation",
                           from_device<std::uint16_t>(p.device_router, p.router_bits.size()),
                           p.router_bits);
    for (const auto& e : p.experts) {
        errors +=
            p.gate_up.verify_rows("MoE gate/up preservation", e.gate_up, e.id * kExpertGateRows);
        errors += p.down.verify_rows("MoE down preservation", e.down, e.id * kHidden);
    }
    errors += p.shared_gu.verify_rows("shared gate/up preservation", p.shared_gu_host, 0);
    errors += p.shared_down.verify_rows("shared down preservation", p.shared_down_host, 0);
    return errors;
}

DeviceBatch::DeviceBatch(const Batch& batch)
    : input_(to_device(batch.input)), residual_(to_device(batch.residual)),
      ids_(to_device(batch.ids)), coefficients_(to_device(batch.coefficients)),
      shared_(to_device(batch.shared_coefficients)) {
    const int tokens = batch.patterns.size();
    x                = Tensor(input_.span(), DType::BF16, {kHidden, tokens});
    residual         = Tensor(residual_.span(), DType::BF16, {kHidden, tokens});
    routes           = {Tensor(ids_.span(), DType::I32, {kTopK, tokens}),
                        Tensor(coefficients_.span(), DType::FP32, {kTopK, tokens})};
    shared           = Tensor(shared_.span(), DType::FP32, {tokens});
}
} // namespace ninfer::test::moe
