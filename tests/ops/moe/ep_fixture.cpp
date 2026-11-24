#include "ops/moe/ep_fixture.h"
#include "ops/quantized_weight.h"
#include "core/buffer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <string>

namespace ninfer::test::moe_ep {
namespace {
namespace q                 = quantized_weight;
constexpr std::size_t Guard = 256;

struct Matrix {
    int n, k;
    Device device;
    bool fp8;
    DeviceBuffer storage, indices, divisors;
    std::vector<q::PackedWeight> packed;
    std::vector<std::vector<float>> decoded;
    std::array<std::uint64_t, E> offsets{};
    std::array<float, E> dw{};
    std::size_t stride = 0, scale_offset = 0, encoded = 0;

    Matrix(Device dev, int rows, int columns, unsigned seed, const std::vector<int>& ids,
           bool is_fp8)
        : n(rows), k(columns), device(dev), fp8(is_fp8), indices(dev, E * 8),
          divisors(dev, is_fp8 ? 0 : E * 4), packed(Local), decoded(Local) {
        q::PackedWeight base;
        if (fp8) {
            base.code_plane_bytes   = std::size_t(n) * k;
            base.scale_plane_offset = base.code_plane_bytes;
            base.scale_plane_bytes  = std::size_t(n / 128) * (k / 128) * 2;
            base.payload.resize(base.scale_plane_offset + base.scale_plane_bytes);
        } else {
            base = q::make_patterned_weight(QType::NVFP4, n, k, seed,
                                            {.weight_scale_divisor = 1, .input_scale_divisor = 1});
            base.dequant.clear();
            base.dequant.shrink_to_fit();
            // Indexed banks address divisors separately from native code/scale planes.
            base.payload.resize(base.scale_plane_offset + base.scale_plane_bytes);
            base.weight.payload_bytes = base.payload.size();
        }
        stride       = (base.payload.size() + 255) / 256 * 256;
        scale_offset = base.scale_plane_offset;
        encoded      = base.code_plane_bytes + base.scale_plane_bytes;
        storage      = DeviceBuffer(dev, Local * stride + 2 * Guard);
        storage.fill(0xa5);
        for (int ordinal = 0; ordinal < Local; ++ordinal) {
            const int e = ids[ordinal], slot = (73 * ordinal + 19) & 255;
            packed[slot] = base;
            auto& p      = packed[slot];
            for (std::size_t b = 0; b < p.code_plane_bytes; ++b) {
                unsigned code = q::detail::mix64(b + seed + std::uint64_t(e) * 998123ULL) & 255;
                if (fp8 && (code & 127) == 127) --code;
                p.payload[b] = code;
            }
            // Fingerprints are finite, represented values and unique for every global ID.
            const std::size_t fingerprint = fp8 && n == 2 * I ? std::size_t(128) * k : 0;
            p.payload[fingerprint]        = fp8 ? std::uint8_t(e & 63) : std::uint8_t(e);
            p.payload[fingerprint + 1] =
                fp8 ? std::uint8_t(0x18 + (e >> 6)) : (e < 256 ? 0x5a : 0xa5);
            if (fp8)
                for (int nr = 0; nr < n / 128; ++nr)
                    for (int kc = 0; kc < k / 128; ++kc) {
                        const float scale =
                            n == 2 * I && nr == 0
                                ? 0
                                : (1 + (nr * 3 + kc * 7 + seed + e) % 13) * .0000083F;
                        const auto bits = q::detail::float_bits(scale);
                        const auto bf16 = std::uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
                        q::detail::store_u16_le(p.payload,
                                                scale_offset + ((nr * (k / 128) + kc) * 2), bf16);
                    }
            CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(storage.data()) + Guard + slot * stride,
                                  p.payload.data(), p.payload.size(), cudaMemcpyHostToDevice));
        }
        for (int e = 0; e < E; ++e) dw[e] = fp8 ? 1.F : (n == 2 * I ? 6.F : 3.F) + .5F * (e % 8);
        if (!fp8) CUDA_CHECK(cudaMemcpy(divisors.data(), dw.data(), E * 4, cudaMemcpyHostToDevice));
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView v;
        v.codes           = static_cast<const std::byte*>(storage.data()) + Guard;
        v.scales          = v.codes + scale_offset;
        v.offsets         = static_cast<const std::uint64_t*>(indices.data());
        v.divisors        = static_cast<const float*>(divisors.data());
        v.divisor_stride  = fp8 ? 0 : 1;
        v.indexed_storage = storage.span().subspan(Guard, Local * stride);
        return v;
    }

    const std::vector<float>& values(int slot) {
        if (packed.empty()) throw std::logic_error("oracle storage was released");
        auto& values = decoded[slot];
        if (values.empty()) {
            values.resize(std::size_t(n) * k);
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < k; ++c)
                    if (fp8) {
                        const auto& p   = packed[slot];
                        const auto bits = q::detail::load_u16_le(
                            p.payload, scale_offset + ((r / 128) * (k / 128) + c / 128) * 2);
                        values[std::size_t(r) * k + c] =
                            q::detail::decode_e4m3fn(p.payload[std::size_t(r) * k + c]) *
                            q::detail::bf16_to_f32(bits);
                    } else
                        values[std::size_t(r) * k + c] = q::logical_weight_fp64(packed[slot], r, c);
        }
        return values;
    }

    int controls() const {
        std::array<std::uint64_t, E> got{};
        std::array<float, E> gotdw{};
        CUDA_CHECK(cudaMemcpy(got.data(), indices.data(), E * 8, cudaMemcpyDeviceToHost));
        if (fp8)
            gotdw = dw;
        else
            CUDA_CHECK(cudaMemcpy(gotdw.data(), divisors.data(), E * 4, cudaMemcpyDeviceToHost));
        std::array<std::uint8_t, Guard> before{}, after{};
        CUDA_CHECK(cudaMemcpy(before.data(), storage.data(), Guard, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            after.data(), static_cast<const std::byte*>(storage.data()) + Guard + Local * stride,
            Guard, cudaMemcpyDeviceToHost));
        return got != offsets || gotdw != dw ||
               !std::all_of(before.begin(), before.end(), [](auto x) { return x == 0xa5; }) ||
               !std::all_of(after.begin(), after.end(), [](auto x) { return x == 0xa5; });
    }

    int payloads() const {
        if (packed.empty()) return controls();
        std::vector<std::uint8_t> got(stride);
        for (int s = 0; s < Local; ++s) {
            CUDA_CHECK(cudaMemcpy(
                got.data(), static_cast<const std::byte*>(storage.data()) + Guard + s * stride,
                stride, cudaMemcpyDeviceToHost));
            if (std::memcmp(got.data(), packed[s].payload.data(), packed[s].payload.size()) != 0)
                return 1;
            if (!std::all_of(got.begin() + packed[s].payload.size(), got.end(),
                             [](auto x) { return x == 0xa5; }))
                return 1;
        }
        return controls();
    }

    void release() {
        packed.clear();
        packed.shrink_to_fit();
        decoded.clear();
        decoded.shrink_to_fit();
    }
};
} // namespace

struct Weights::Impl {
    Device device;
    Placement place;
    bool fp8;
    int rank, phase = 0;
    std::vector<int> ids;
    DeviceBuffer mask;
    std::array<int, E> mask_host{};
    Matrix gu, down;
    std::map<std::tuple<int, int, int>, std::vector<double>> cache;

    static std::vector<int> assigned(Placement p, int side) {
        if (side < 0 || side > 1) throw std::invalid_argument("EP side");
        std::vector<int> result(Local);
        for (int l = 0; l < Local; ++l)
            result[l] = p == Placement::Contiguous ? side * Local + l : 2 * l + side;
        return result;
    }

    Impl(Device dev, Placement p, int side, bool is_fp8)
        : device(dev), place(p), fp8(is_fp8), rank(side), ids(assigned(p, side)), mask(dev, E * 4),
          gu(dev, 2 * I, H, 101, ids, is_fp8), down(dev, H, I, 211, ids, is_fp8) {}
};

Weights::Weights(Device d, Placement p, int side, bool fp8)
    : impl_(std::make_unique<Impl>(d, p, side, fp8)) {
    if (side < 0 || side > 1) throw std::invalid_argument("EP side");
    publish();
}

Weights::~Weights() = default;

ops::RoutedSwiGLUProfile Weights::profile() const {
    const auto type = impl_->fp8 ? QType::FP8_E4M3FN_BLOCK128X128_BF16 : QType::NVFP4;
    const auto layout =
        impl_->fp8 ? QuantLayout::BlockScaleM128K128 : QuantLayout::BlockScaleK16M128x4;
    const auto policy = impl_->fp8 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::AllowA4;
    return {{H, I, type, type, layout, layout, policy, policy}, E, K};
}

ops::CudaSwiGLUExperts Weights::experts() const {
    return {{impl_->gu.view(), impl_->down.view(), E, impl_->device},
            impl_->fp8 ? ops::ExpertActivationDivisors{}
                       : ops::ExpertActivationDivisors{719.7991333F, 313.927002F}};
}

ops::ExpertSubset Weights::subset() const {
    return {impl_->rank == 0 ? ops::ExpertSubsetKind::MaskOnes : ops::ExpertSubsetKind::MaskZeros,
            Tensor(impl_->mask.span(), DType::I32, {E})};
}

void Weights::publish(int phase, bool drop_half) {
    auto& f = *impl_;
    f.phase = phase;
    f.gu.offsets.fill(UINT64_MAX);
    f.down.offsets.fill(UINT64_MAX);
    for (int e = 0; e < E; ++e)
        f.mask_host[e] = f.place == Placement::Contiguous ? int(e < Local) : int(e % 2 == 0);
    for (int l = 0; l < Local; ++l) {
        const int e = f.ids[l], slot = (73 * l + 19 + 17 * phase) & 255;
        f.gu.offsets[e]   = slot * f.gu.stride;
        f.down.offsets[e] = slot * f.down.stride;
        if (drop_half && l % 2 == 0) f.mask_host[e] = f.rank == 0 ? 0 : 1;
    }
    CUDA_CHECK(cudaMemcpy(f.gu.indices.data(), f.gu.offsets.data(), E * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(f.down.indices.data(), f.down.offsets.data(), E * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(f.mask.data(), f.mask_host.data(), E * 4, cudaMemcpyHostToDevice));
}

bool Weights::active(int e) const {
    if (e < 0 || e >= E) return false;
    return impl_->rank == 0 ? impl_->mask_host[e] != 0 : impl_->mask_host[e] == 0;
}

int Weights::local_id(int l) const { return impl_->ids[l % Local]; }

int Weights::remote_id(int l) const {
    return impl_->place == Placement::Contiguous ? (1 - impl_->rank) * Local + l % Local
                                                 : 2 * (l % Local) + 1 - impl_->rank;
}

std::size_t Weights::encoded_expert_bytes() const {
    return impl_->gu.encoded + impl_->down.encoded;
}

std::size_t Weights::backing_bytes() const {
    return Local * (impl_->gu.stride + impl_->down.stride);
}

const std::vector<double>& Weights::oracle(int pattern, int e, std::span<const float> x) {
    auto& f  = *impl_;
    auto key = std::tuple{pattern, e, f.phase};
    if (auto it = f.cache.find(key); it != f.cache.end()) return it->second;
    if (f.gu.offsets[e] == UINT64_MAX) throw std::logic_error("oracle requested remote payload");
    const auto& gu   = f.gu.values(f.gu.offsets[e] / f.gu.stride);
    const auto& down = f.down.values(f.down.offsets[e] / f.down.stride);
    std::vector<double> h(I), out(H);
    const double gd = f.gu.dw[e], dd = f.down.dw[e];
    for (int row = 0; row < I; ++row) {
        double g = 0, u = 0;
        for (int c = 0; c < H; ++c) {
            g += double(x[c]) * gu[std::size_t(row) * H + c] / gd;
            u += double(x[c]) * gu[std::size_t(row + I) * H + c] / gd;
        }
        h[row] = g / (1 + std::exp(-g)) * u;
    }
    for (int row = 0; row < H; ++row)
        for (int c = 0; c < I; ++c) out[row] += h[c] * down[std::size_t(row) * I + c] / dd;
    return f.cache.emplace(key, std::move(out)).first->second;
}

int Weights::verify_controls() const {
    std::array<int, E> got{};
    CUDA_CHECK(cudaMemcpy(got.data(), impl_->mask.data(), E * 4, cudaMemcpyDeviceToHost));
    return (got != impl_->mask_host) + impl_->gu.controls() + impl_->down.controls();
}

int Weights::verify_payloads() const {
    return impl_->gu.payloads() + impl_->down.payloads() + verify_controls();
}

void Weights::release_host_oracle() {
    impl_->gu.release();
    impl_->down.release();
    impl_->cache.clear();
}

Routes make_routes(const Weights& w, int tokens, const char* distribution, int fixed_hits) {
    Routes result;
    result.ids.resize(std::size_t(tokens) * K);
    result.coefficients.resize(result.ids.size());
    result.per_expert.assign(E, 0);
    const std::string kind = distribution;
    std::mt19937 rng(1729);
    std::array<int, E> pool;
    for (int t = 0; t < tokens; ++t) {
        if (kind == "independent") {
            std::iota(pool.begin(), pool.end(), 0);
            if (fixed_hits < 0) {
                std::shuffle(pool.begin(), pool.end(), rng);
            } else {
                std::shuffle(pool.begin(), pool.begin() + Local, rng);
                std::shuffle(pool.begin() + Local, pool.end(), rng);
            }
        }
        const int hits  = fixed_hits >= 0                   ? fixed_hits
                          : kind == "zero"                  ? 0
                          : kind == "all" || kind == "skew" ? K
                                                            : (t * 7 + 3) % 11;
        int actual_hits = 0;
        for (int j = 0; j < K; ++j) {
            int e;
            if (kind == "independent") {
                e = fixed_hits < 0 ? pool[j]
                    : j < hits     ? w.local_id(pool[j])
                                   : w.remote_id(pool[Local + j - hits] - Local);
            } else {
                const int ordinal = kind == "skew" ? j : (t * 13 + j * 97) % Local;
                e                 = j < hits ? w.local_id(ordinal) : w.remote_id(ordinal);
            }
            result.ids[std::size_t(t) * K + j]          = e;
            result.coefficients[std::size_t(t) * K + j] = .01F * (j + 1) + .003F * (t % 3);
            if (w.active(e)) {
                ++result.per_expert[e];
                ++actual_hits;
            }
        }
        result.hits += actual_hits;
        result.active_tokens += actual_hits > 0;
    }
    return result;
}
} // namespace ninfer::test::moe_ep
