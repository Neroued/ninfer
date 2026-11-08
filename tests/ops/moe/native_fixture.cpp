#include "ops/moe/native_fixture.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <array>
#include <map>
#include <numeric>

namespace ninfer::test::native_moe {
namespace qw = quantized_weight;

namespace {
DeviceBuffer upload(Device device, const void* data, std::size_t bytes) {
    DeviceBuffer result(device, bytes);
    result.copy_from_host(data, bytes);
    return result;
}

std::vector<std::uint32_t> logical_ids(bool ep, int rank) {
    std::vector<std::uint32_t> ids;
    for (int e = 0; e < Fixture::experts_count; ++e)
        if (!ep || e % 2 == rank) ids.push_back(e);
    if (ep) std::reverse(ids.begin(), ids.end());
    return ids;
}

qw::PatternedWeightOptions bank_options(bool main, ScalePattern pattern) {
    qw::PatternedWeightOptions o;
    if (main) {
        o.layout = QuantLayout::Csf32;
        if (pattern == ScalePattern::Stress) {
            o.csf_dense_exception_block = true;
            o.outlier_expert_period     = 67;
        } else {
            // The source checkpoint's scale-count-weighted exception rate is 0.2832%.
            o.csf_exception_period = 353;
        }
    }
    return o;
}

struct Bank {
    qw::PackedWeightBank host;
    DeviceBuffer storage, indices, calibration, offsets, multipliers;
    ExpertMatrixDeviceView view;

    Bank(Device device, bool main, bool indexed, ScalePattern pattern, int n, int k,
         std::span<const std::uint32_t> ids, std::uint32_t seed, qw::PackedWeightBank* source)
        : host(source ? std::move(*source)
                      : qw::make_patterned_bank(QType::NVFP4_MUL, n, k, ids, Fixture::experts_count,
                                                seed, bank_options(main, pattern))) {
        if (indexed) {
            const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
            const auto matrix =
                weight_geometry(QType::NVFP4_MUL, QuantLayout::BlockScaleK16M128x4, shape);
            const auto stride =
                (matrix.bytes + matrix.alignment - 1) / matrix.alignment * matrix.alignment;
            std::vector<std::uint8_t> bytes(stride * ids.size());
            std::vector<std::uint64_t> address(Fixture::experts_count, UINT64_MAX);
            std::vector<float> scales(Fixture::experts_count, 1);
            for (std::size_t slot = 0; slot < ids.size(); ++slot) {
                auto* target = bytes.data() + slot * stride;
                std::memcpy(target, host.payload.data() + slot * matrix.code_bytes,
                            matrix.code_bytes);
                std::memcpy(target + matrix.scale_offset,
                            host.payload.data() + host.geometry.scale_offset +
                                slot * matrix.scale_bytes,
                            matrix.scale_bytes);
                std::memcpy(target + matrix.matrix_scale_offset, &host.matrix_scales[slot], 4);
                address[ids[slot]] = slot * stride;
                scales[ids[slot]]  = host.matrix_scales[slot];
            }
            storage     = upload(device, bytes.data(), bytes.size());
            offsets     = upload(device, address.data(), address.size() * 8);
            multipliers = upload(device, scales.data(), scales.size() * 4);
            const WeightParent parent{
                matrix, device, static_cast<const std::byte*>(storage.data()), {}};
            view = indexed_weight_matrix(parent, static_cast<const std::uint64_t*>(offsets.data()),
                                         static_cast<const float*>(multipliers.data()),
                                         storage.span());
        } else {
            storage = upload(device, host.payload.data(), host.payload.size());
            if (ids.size() != Fixture::experts_count)
                indices =
                    upload(device, host.expert_indices.data(), host.expert_indices.size() * 4);
            view = host.device_view(storage.span(), indices.span());
        }
        if (main)
            calibration =
                upload(device, host.activation_scales.data(), host.activation_scales.size() * 4);
    }
};

struct SharedMatrix {
    qw::PackedWeight host;
    DeviceBuffer storage;

    SharedMatrix(Device d, QType type, int n, int k, std::uint32_t seed)
        : host(qw::make_patterned_weight(type, n, k, seed, {.native_extent = .03f})),
          storage(upload(d, host.payload.data(), host.payload.size())) {}
};
} // namespace

Profile parse_profile(std::string_view name) {
    if (name == "main-tp") return Profile::MainTp;
    if (name == "main-ep") return Profile::MainEp;
    if (name == "mtp-tp") return Profile::MtpTp;
    if (name == "mtp-ep") return Profile::MtpEp;
    throw std::invalid_argument("unknown native MoE profile");
}

std::string_view profile_name(Profile p) {
    switch (p) {
    case Profile::MainTp:
        return "main-tp";
    case Profile::MainEp:
        return "main-ep";
    case Profile::MtpTp:
        return "mtp-tp";
    case Profile::MtpEp:
        return "mtp-ep";
    }
    throw std::invalid_argument("unknown native MoE profile");
}

bool Fixture::main() const { return profile == Profile::MainTp || profile == Profile::MainEp; }

bool Fixture::ep() const { return profile == Profile::MainEp || profile == Profile::MtpEp; }

struct Fixture::Impl {
    Device device;
    std::vector<std::uint32_t> logical;
    Bank gu, down;
    SharedMatrix shared_gu, shared_down;
    std::vector<std::uint16_t> words;
    std::vector<int> ids, resident;
    std::vector<float> coefficients, bias;
    std::vector<double> router_coefficients;
    DeviceBuffer input;
    DeviceBuffer route_ids, route_coefficients, mask, control, bias_device;

    Impl(CudaExecutionView cuda, const Fixture& f, std::optional<WeightBanks>& source)
        : device(cuda.device()), logical(logical_ids(f.ep(), f.rank)),
          gu(device, f.main(), f.ep() && !f.main(), f.scale_pattern, 2 * f.intermediate, hidden,
             logical, 113, source ? &source->gate_up : nullptr),
          down(device, f.main(), f.ep() && !f.main(), f.scale_pattern, hidden, f.intermediate,
               logical, 719, source ? &source->down : nullptr),
          shared_gu(device, f.main() ? QType::FP8_E4M3FN_BLOCK128X128_FP32 : QType::MXFP8,
                    2 * shared_intermediate, hidden, 1711),
          shared_down(device, f.main() ? QType::FP8_E4M3FN_BLOCK128X128_FP32 : QType::MXFP8, hidden,
                      shared_intermediate, 1719),
          resident(experts_count), bias(experts_count),
          input(device, std::size_t(hidden) * f.max_tokens * 2),
          route_ids(device, std::size_t(top_k) * f.max_tokens * 4),
          route_coefficients(device, std::size_t(top_k) * f.max_tokens * 4) {
        for (auto e : logical) resident[e] = 1;
        mask = upload(device, resident.data(), resident.size() * 4);
        std::vector<std::uint16_t> router(std::size_t(hidden) * experts_count);
        for (int e = 0; e < experts_count; ++e) {
            router[std::size_t(e) * hidden + e] = f32_to_bf16(8.f);
            bias[e]                             = float((e * 13) % 17 - 8) * .00001f;
        }
        control     = upload(device, router.data(), router.size() * 2);
        bias_device = upload(device, bias.data(), bias.size() * 4);
    }
};

Fixture::Fixture(CudaExecutionView cuda, Profile p, int maximum, int r, ScalePattern pattern,
                 std::optional<WeightBanks> source)
    : profile(p), scale_pattern(pattern),
      intermediate((p == Profile::MainEp || p == Profile::MtpEp) ? 2048 : 1024), rank(r),
      max_tokens(maximum) {
    if (maximum <= 0 || r < 0 || r > 1)
        throw std::invalid_argument("invalid native MoE fixture extent");
    impl_ = std::make_unique<Impl>(cuda, *this, source);
}

Fixture::~Fixture() = default;

void Fixture::input(int tokens, std::string_view distribution, int phase, bool dense,
                    std::string_view locality, int expert_offset, float input_scale) {
    if (tokens < 1 || tokens > max_tokens)
        throw std::out_of_range("native MoE input exceeds fixture");
    if (distribution != "reuse" && distribution != "pool" && distribution != "tail" &&
        distribution != "spread" && distribution != "skewpool")
        throw std::invalid_argument("unknown expert distribution");
    if (locality != "mixed" && locality != "local" && locality != "remote")
        throw std::invalid_argument("unknown expert locality");
    auto& v = *impl_;
    v.words.assign(std::size_t(tokens) * hidden, 0);
    v.ids.resize(std::size_t(tokens) * top_k);
    v.coefficients.resize(v.ids.size());
    v.router_coefficients.resize(v.ids.size());
    const int domain = ep() && locality != "mixed" ? experts_count / 2 : experts_count;
    for (int t = 0; t < tokens; ++t) {
        auto* x = v.words.data() + std::size_t(t) * hidden;
        for (int j = 0; j < (dense ? hidden - experts_count : 24); ++j) {
            const auto bits = qw::detail::mix64(std::uint64_t(t + phase * 71) * hidden + j + 7331);
            const int k     = experts_count + (dense ? j : int(bits % (hidden - experts_count)));
            int word        = int((bits >> 32) % 31) - 15;
            if (!word) word = 1;
            x[k] = f32_to_bf16(word * (dense ? .015625f : .125f));
        }
        for (int j = 0; j < top_k; ++j) {
            int e = distribution == "reuse"      ? j
                    : distribution == "skewpool" ? (j < 7   ? j
                                                    : t < 9 ? 8 + (t + phase) % 8
                                                            : 7)
                    : distribution == "pool"     ? (t * top_k + phase * 3 + j) % 16
                    : distribution == "tail"
                        ? (j < 4 ? j : 4 + (t * 4 + phase * 11 + j - 4) % (domain - 4))
                        : (t * top_k + phase * 17 + j) % domain;
            e     = (107 * ((e + expert_offset) % domain) + 19) % domain;
            if (ep() && locality != "mixed") e = 2 * e + (locality == "local" ? rank : 1 - rank);
            x[e] = f32_to_bf16(.32f - j * .02f);
        }
        if (input_scale != 1.F)
            for (int k = 0; k < hidden; ++k) x[k] = f32_to_bf16(bf16_to_f32(x[k]) * input_scale);
        std::vector<int> candidates(experts_count);
        std::iota(candidates.begin(), candidates.end(), 0);
        const auto score = [&](int e) { return 1.0 / (1 + std::exp(-8.0 * bf16_to_f32(x[e]))); };
        std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                          [&](int a, int b) {
                              const auto sa = score(a) + v.bias[a], sb = score(b) + v.bias[b];
                              return sa == sb ? a < b : sa > sb;
                          });
        double total = 1e-20;
        for (int j = 0; j < top_k; ++j) total += score(candidates[j]);
        for (int j = 0; j < top_k; ++j) {
            const auto index             = std::size_t(t) * top_k + j;
            v.ids[index]                 = candidates[j];
            v.router_coefficients[index] = 2.5 * score(candidates[j]) / total;
            v.coefficients[index]        = float(v.router_coefficients[index]);
        }
    }
    v.input.copy_from_host(v.words.data(), v.words.size() * 2);
}

void Fixture::publish_routes(bool signed_coefficients) {
    auto& v = *impl_;
    if (signed_coefficients)
        for (std::size_t j = 0; j < v.coefficients.size(); ++j)
            if (j % 3 == 1) v.coefficients[j] = -v.coefficients[j];
    v.route_ids.copy_from_host(v.ids.data(), v.ids.size() * 4);
    v.route_coefficients.copy_from_host(v.coefficients.data(), v.coefficients.size() * 4);
}

void Fixture::release_host_weights() {
    impl_->gu.host.release_payload();
    impl_->down.host.release_payload();
}

Tensor Fixture::x(int tokens) const { return {impl_->input.span(), DType::BF16, {hidden, tokens}}; }

ops::ExpertRoutes Fixture::routes(int tokens) const {
    return {{impl_->route_ids.span(), DType::I32, {top_k, tokens}},
            {impl_->route_coefficients.span(), DType::FP32, {top_k, tokens}}};
}

ops::ExpertSubset Fixture::subset() const {
    return ep() ? ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes,
                                    Tensor(impl_->mask.span(), DType::I32, {experts_count})}
                : ops::ExpertSubset{};
}

ops::RoutedSwiGLUProfile Fixture::routed_profile() const {
    const auto layout = main() ? QuantLayout::Csf32 : QuantLayout::BlockScaleK16M128x4;
    const auto policy = main() ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::A16Only;
    return {{hidden, intermediate, QType::NVFP4_MUL, QType::NVFP4_MUL, layout, layout, policy,
             policy, 10.f},
            experts_count,
            top_k};
}

ops::SigmoidMoeProfile Fixture::complete_profile() const {
    const auto type   = main() ? QType::FP8_E4M3FN_BLOCK128X128_FP32 : QType::MXFP8;
    const auto layout = main() ? QuantLayout::BlockScaleM128K128 : QuantLayout::RowGroupScale;
    const auto policy = main() ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA8;
    return {{{hidden, experts_count, top_k}, 2.5f, 1e-20f},
            routed_profile(),
            {hidden, shared_intermediate, type, type, layout, layout, policy, policy, 10.f}};
}

ops::CudaSwiGLUExperts Fixture::expert_weights() const {
    const auto& v = *impl_;
    ops::ExpertActivationCalibration calibration;
    if (main()) {
        calibration.gate_up = ops::ActivationCalibrationView::per_matrix(
            ops::ActivationScaleOperation::Multiply, v.gu.host.activation_scales,
            v.gu.calibration.span());
        calibration.down = ops::ActivationCalibrationView::per_matrix(
            ops::ActivationScaleOperation::Multiply, v.down.host.activation_scales,
            v.down.calibration.span());
    }
    return {{v.gu.view, v.down.view, experts_count, v.device}, calibration};
}

ops::SigmoidMoeWeights Fixture::complete_weights() const {
    auto& v = *impl_;
    const WeightParent parent{weight_geometry(QType::BF16, QuantLayout::Contiguous,
                                              std::array<std::uint64_t, 2>{experts_count, hidden}),
                              v.device, static_cast<const std::byte*>(v.control.data())};
    const auto control =
        native_weight({parent.geometry.shape, {{&parent, 0, parent.geometry.elements}}});
    const auto policy = main() ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA8;
    return {control,
            Tensor(v.bias_device.span(), DType::FP32, {experts_count}),
            expert_weights(),
            {{v.shared_gu.host.device_weight(v.shared_gu.storage.span()), policy},
             {v.shared_down.host.device_weight(v.shared_down.storage.span()), policy}}};
}

const std::vector<std::uint16_t>& Fixture::input_words() const { return impl_->words; }

const std::vector<int>& Fixture::route_ids() const { return impl_->ids; }

const std::vector<float>& Fixture::route_coefficients() const { return impl_->coefficients; }

bool Fixture::resident(int expert) const { return impl_->resident.at(expert) != 0; }

std::vector<int> Fixture::local_counts() const {
    std::vector<int> counts(experts_count);
    for (int id : impl_->ids)
        if (resident(id)) ++counts[id];
    return counts;
}

std::uint64_t Fixture::active_weight_bytes() const {
    const auto counts   = local_counts();
    std::uint64_t bytes = 0;
    for (int e = 0; e < experts_count; ++e)
        if (counts[e])
            bytes +=
                impl_->gu.host.expert_weight_bytes(e) + impl_->down.host.expert_weight_bytes(e);
    return bytes;
}

std::uint64_t Fixture::router_bytes() const {
    return std::uint64_t(hidden) * experts_count * 2 + experts_count * 4;
}

std::uint64_t Fixture::shared_bytes() const {
    return impl_->shared_gu.host.model_weight_bytes() +
           impl_->shared_down.host.model_weight_bytes();
}

int Fixture::verify_input() const {
    std::vector<std::uint16_t> actual(impl_->words.size());
    impl_->input.copy_to_host(actual.data(), actual.size() * 2);
    return actual != impl_->words;
}

struct Oracle::Impl {
    const Fixture& f;
    std::map<std::pair<int, int>, std::vector<double>> products;

    explicit Impl(const Fixture& fixture) : f(fixture) {}

    const std::vector<double>& product(int token, int expert) {
        const auto key = std::pair{token, expert};
        if (auto it = products.find(key); it != products.end()) return it->second;
        std::vector<std::pair<int, double>> active;
        for (int k = 0; k < Fixture::hidden; ++k) {
            const double x = bf16_to_f32(f.impl_->words[std::size_t(token) * Fixture::hidden + k]);
            if (x) active.emplace_back(k, x);
        }
        const bool shared = expert == -1;
        const int size    = shared ? Fixture::shared_intermediate : f.intermediate;
        std::vector<double> value(size);
        const auto weight = [&](int r, int k) {
            return shared ? qw::logical_weight_fp64(f.impl_->shared_gu.host, r, k)
                          : qw::logical_weight_fp64(f.impl_->gu.host, expert, r, k);
        };
        for (int r = 0; r < size; ++r) {
            double gate = 0, up = 0;
            for (auto [k, x] : active) {
                gate += weight(r, k) * x;
                up += weight(r + size, k) * x;
            }
            gate     = std::min(gate, 10.0);
            up       = std::clamp(up, -10.0, 10.0);
            value[r] = gate / (1 + std::exp(-gate)) * up;
        }
        return products.emplace(key, std::move(value)).first->second;
    }

    double contribution(int row, int token, int expert) {
        const auto& p = product(token, expert);
        double value  = 0;
        for (int k = 0; k < int(p.size()); ++k)
            value +=
                p[k] * (expert == -1 ? qw::logical_weight_fp64(f.impl_->shared_down.host, row, k)
                                     : qw::logical_weight_fp64(f.impl_->down.host, expert, row, k));
        return value;
    }
};

Oracle::Oracle(const Fixture& f) : impl_(std::make_unique<Impl>(f)) {}

Oracle::~Oracle() = default;

double Oracle::output(int row, int token, bool complete, std::span<const int> mask) {
    auto& v    = *impl_;
    double out = complete ? v.contribution(row, token, -1) : 0;
    for (int j = 0; j < Fixture::top_k; ++j) {
        const auto index = std::size_t(token) * Fixture::top_k + j;
        const int expert = v.f.impl_->ids[index];
        if (!v.f.resident(expert) || (!mask.empty() && !mask[expert])) continue;
        const double coefficient =
            complete ? v.f.impl_->router_coefficients[index] : v.f.impl_->coefficients[index];
        out += coefficient * v.contribution(row, token, expert);
    }
    return out;
}
} // namespace ninfer::test::native_moe
