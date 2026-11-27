#include "distributed/moe/fixture.h"
#include "ops/op_tester.h"

namespace ninfer::test::exchange {
namespace {
constexpr int Patterns = 3;

float divisor(int e, bool gate, bool fp8) {
    return fp8 ? 1.F : (gate ? 2.F : 1.F) + (e % 7) * .25F;
}

Weight bf16(DeviceBuffer& buffer, int n, int k, const std::vector<std::uint16_t>& bits) {
    buffer.copy_from_host(bits.data(), buffer.size());
    Weight w;
    w.device  = buffer.device();
    w.payload = w.qdata = buffer.data();
    w.payload_bytes     = buffer.size();
    w.qtype             = QType::BF16;
    w.layout            = QuantLayout::Contiguous;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}

double silu_ref(double x) { return x / (1 + std::exp(-x)); }
} // namespace

Matrix::Matrix(Device d, bool gate, bool fp8, bool affine, std::span<const std::uint32_t> owned,
               models::IndexRange slice) {
    const int local = slice.size;
    const int n = gate ? 2 * local : H, k = gate ? H : local;
    const auto type = fp8 ? (local % 128 ? QType::FP8_E4M3FN_BLOCK64X64_BF16
                                         : QType::FP8_E4M3FN_BLOCK128X128_BF16)
                          : QType::NVFP4;
    const auto layout =
        fp8 ? (local % 128 ? QuantLayout::BlockScaleM64K64 : QuantLayout::BlockScaleM128K128)
            : QuantLayout::BlockScaleK16M128x4;
    const auto g      = weight_storage_geometry(type, layout,
                                                std::array<std::uint64_t, 2>{unsigned(n), unsigned(k)});
    const auto stride = ((g.scale_offset + g.scale_bytes + 255) / 256) * 256;
    data              = DeviceBuffer(d, stride * Patterns);
    std::vector<unsigned char> payload(stride * Patterns);
    for (int pattern = 0; pattern < Patterns; ++pattern) {
        auto* codes  = payload.data() + stride * pattern;
        auto* scales = codes + g.scale_offset;
        if (fp8) {
            for (std::size_t i = 0; i < g.scale_bytes; i += 2) {
                scales[i]     = 0x80;
                scales[i + 1] = 0x3f;
            }
        } else
            std::fill_n(scales, g.scale_bytes, 0x38);
        const auto put = [&](int r, int c, int nv, int fp) {
            const auto pos = std::size_t(r) * k + c;
            if (fp8)
                codes[pos] = fp;
            else
                codes[pos / 2] |= nv << ((pos % 2) * 4);
        };
        // The two nonzero units deliberately lie in different intermediate halves.
        for (unsigned unit : {0U, unsigned(I / 2)}) {
            if (unit < slice.begin || unit >= slice.begin + slice.size) continue;
            const int local_unit = unit - slice.begin;
            if (gate) {
                put(local_unit, (unit ? 1008 : 1000) + pattern, unit ? 9 : 2, unit ? 0xb0 : 0x38);
                put(local + local_unit, (unit ? 1012 : 1004) + pattern, unit ? 3 : 4,
                    unit ? 0x3c : 0x40);
            } else {
                for (int r = (unit ? 1 : 0); r < n; r += 2)
                    put(r, local_unit, r % 3 ? 2 : 9, r % 3 ? 0x38 : 0xb0);
            }
        }
    }
    data.copy_from_host(payload.data(), data.size());
    view.codes  = static_cast<const std::byte*>(data.data());
    view.scales = view.codes + g.scale_offset;
    view.format = type;
    view.layout = layout;
    if (affine) {
        std::vector<unsigned> indices(E, UINT_MAX);
        for (auto e : owned) indices[e] = e % Patterns;
        index = DeviceBuffer(d, E * sizeof(unsigned));
        index.copy_from_host(indices.data(), index.size());
        view.expert_indices = static_cast<const unsigned*>(index.data());
        view.code_stride = view.scale_stride = stride;
        view.physical_count                  = Patterns;
    } else {
        std::vector<std::uint64_t> offsets(E, UINT64_MAX);
        for (auto e : owned) offsets[e] = (e % Patterns) * stride;
        index = DeviceBuffer(d, E * sizeof(std::uint64_t));
        index.copy_from_host(offsets.data(), index.size());
        view.offsets         = static_cast<const std::uint64_t*>(index.data());
        view.indexed_storage = data.span();
    }
    if (!fp8) {
        std::vector<float> scales(affine ? Patterns : E);
        for (unsigned e = 0; e < scales.size(); ++e) scales[e] = divisor(e, gate, false);
        divisors = DeviceBuffer(d, scales.size() * 4);
        divisors.copy_from_host(scales.data(), divisors.size());
        view.matrix_scales       = static_cast<const float*>(divisors.data());
        view.matrix_scale_stride = 1;
    }
}

Fixture::Fixture(Device d, std::span<const std::uint32_t> owned, ExpertBankId bank, bool fp8,
                 bool affine, models::IndexRange slice)
    : gu(d, true, fp8, affine, owned, slice), down(d, false, fp8, affine, owned, slice),
      router(d, (E + 1) * H * 2), shared_gu(d, 2 * I * H * 2), shared_down(d, H * I * 2) {
    const auto policy         = fp8 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::AllowA4;
    parameters.profile.routed = {{H, static_cast<int>(slice.size), gu.view.format, down.view.format,
                                  gu.view.layout, down.view.layout, policy, policy},
                                 E,
                                 K};
    parameters.profile.shared = {H, I};
    parameters.bank           = bank;
    parameters.resident       = {gu.view, down.view, E, d};
    if (!fp8)
        parameters.activation_calibration = {
            ops::ActivationCalibrationView::broadcast(ops::ActivationScaleOperation::Divide, 17),
            ops::ActivationCalibrationView::broadcast(ops::ActivationScaleOperation::Divide, 11)};
    std::vector<std::uint16_t> c((E + 1) * H), g(2 * I * H), v(H * I);
    for (int e = 0; e <= E; ++e) c[e * H + e] = f32_to_bf16(1);
    g[1100]         = f32_to_bf16(1);
    g[I * H + 1101] = f32_to_bf16(-.5F);
    for (int r = 0; r < H; ++r) v[r * I] = f32_to_bf16(r % 2 ? .25F : -.5F);
    parameters.control = bf16(router, E + 1, H, c);
    parameters.shared  = {{bf16(shared_gu, 2 * I, H, g)}, {bf16(shared_down, H, I, v)}};
}

Column column(const std::vector<models::ExpertPartition>& peers, int source, int index,
              int local_hits) {
    Column out;
    out.input.resize(H, f32_to_bf16(-8));
    for (int f = 1000; f < 1020; ++f)
        out.input[f] = f32_to_bf16(.375F * (1 + (f + index + source * 3) % 9));
    out.input[1100] = f32_to_bf16(.75F + source);
    out.input[1101] = f32_to_bf16(1.5F + .25F * index);
    out.input[E]    = f32_to_bf16(.25F + source);
    double sum      = 0;
    for (int j = 0; j < K; ++j) {
        const auto& ids           = peers.at(j < local_hits ? source : 1 - source).ids;
        out.experts[j]            = ids.at((index * 13 + j * 3 + 1) % ids.size());
        const float logit         = 2 + j * .125F;
        out.input[out.experts[j]] = f32_to_bf16(logit);
        sum += (out.weights[j] = std::exp(double(logit)));
    }
    for (auto& w : out.weights) w /= sum;
    return out;
}

double Column::expert(int e, int r, bool fp8, bool affine) const {
    const int pattern = e % Patterns, scalar = affine ? pattern : e;
    const double gu = divisor(scalar, true, fp8), dn = divisor(scalar, false, fp8);
    const double gate =
        r % 2 ? -.5 * bf16_to_f32(input[1008 + pattern]) : bf16_to_f32(input[1000 + pattern]);
    const double up =
        r % 2 ? 1.5 * bf16_to_f32(input[1012 + pattern]) : 2 * bf16_to_f32(input[1004 + pattern]);
    return silu_ref(gate / gu) * (up / gu) * (r % 3 ? 1 : -.5) / dn;
}

double Column::shared(int r) const {
    return silu_ref(bf16_to_f32(input[1100])) * (-.5 * bf16_to_f32(input[1101])) *
           (r % 2 ? .25 : -.5) / (1 + std::exp(-double(bf16_to_f32(input[E]))));
}

void local_reference(DeviceContext& device, const models::moe::ExpertDomain& domain,
                     ExpertBankId bank) {
    const auto& peers = domain.banks.at(bank.index);
    const auto& ids   = peers.at(1).ids;
    for (bool fp8 : {false, true})
        for (bool affine : {false, true}) {
            Fixture fixture(device.device(), ids, bank, fp8, affine);
            auto c            = column(peers, 0, 0, 7);
            constexpr int cap = 7;
            const int slots   = ids.size();
            std::vector<int> counts{1, 3}, offsets(slots + 1), rows(cap);
            std::vector<float> coefficients(cap);
            int m = 0;
            for (int s = 0; s < slots; ++s) {
                for (int j = 7; j < K; ++j)
                    if (int(ids[s]) == c.experts[j]) coefficients[m++] = c.weights[j];
                offsets[s + 1] = m;
            }
            if (m != 3) throw std::runtime_error("invalid compact fixture");
            const auto d = device.device();
            DeviceBuffer yb(d, H * 2 * 4), cb(d, 8), ob(d, offsets.size() * 4),
                lb(d, ids.size() * 4), rb(d, cap * 4), wb(d, cap * 4);
            cb.copy_from_host(counts.data(), 8);
            ob.copy_from_host(offsets.data(), ob.size());
            lb.copy_from_host(ids.data(), lb.size());
            rb.copy_from_host(rows.data(), rb.size());
            wb.copy_from_host(coefficients.data(), wb.size());
            DeviceBuffer input(d, 2 * H * 2);
            input.copy_from_host(c.input.data(), H * 2);
            Tensor x(input.span(), DType::BF16, {H, 2}), y(yb.span(), DType::FP32, {H, 2});
            ops::CompactExpertAssignments a{{cb.span(), DType::I32, {2}},
                                            {ob.span(), DType::I32, {slots + 1}},
                                            {lb.span(), DType::I32, {slots}},
                                            {rb.span(), DType::I32, {cap}},
                                            {wb.span(), DType::FP32, {cap}}};
            const auto& par = fixture.parameters;
            DeviceBuffer scratch(
                d, ops::compact_routed_swiglu_workspace_bytes(par.profile.routed, cap));
            MemoryArena arena(scratch.span());
            for (bool empty : {false, true}) {
                if (empty) {
                    cb.fill();
                    ob.fill();
                }
                std::vector<float> actual(2 * H, -31.F);
                yb.copy_from_host(actual.data(), yb.size());
                ops::compact_routed_swiglu(x, par.profile.routed,
                                           {par.resident, par.activation_calibration}, a, y, arena,
                                           device.execution_view());
                device.synchronize();
                yb.copy_to_host(actual.data(), yb.size());
                for (int r = 0; r < 2 * H; ++r) {
                    double expected = -31;
                    if (!empty && r < H) {
                        expected = 0;
                        for (int j = 7; j < K; ++j)
                            expected += c.weights[j] * c.expert(c.experts[j], r, fp8, affine);
                    }
                    if (!std::isfinite(actual[r]) ||
                        std::abs(expected - actual[r]) > 2e-6 * (1 + std::abs(expected)))
                        throw std::runtime_error(
                            "compact partial differs from represented-weight FP64 reference");
                }
            }
        }
}
} // namespace ninfer::test::exchange
