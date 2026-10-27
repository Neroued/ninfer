#include "ops/gated_residual/mix/fp8/operands.h"
#include "ops/common/host/operands.h"
#include "core/platform.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail::hc::mix::fp8 {
#if !NINFER_TARGET_RTX5090
void launch(Weights, const Tensor&, Tensor&, Tensor*, Scratch, Plan, CudaExecutionView) {
    throw std::invalid_argument("hc_mix FP8: unsupported backend");
}

Plan plan(LinearPolicy, LinearPolicy, int) { return {}; }

int up_token_tile(UpSchedule) { return 0; }

Layout layout(int, Plan) { return {}; }
#endif
namespace {
constexpr const char* kOp = "hc_mix FP8";

void require(bool x, const char* why) {
    if (!x) throw std::invalid_argument(std::string(kOp) + ": " + why);
}

void profile(LinearProfile d, LinearProfile u, const LinearProfile* j, int first, int last) {
    require(NINFER_TARGET_RTX5090, "unsupported backend");
    require(first > 0 && last >= first, "invalid column interval");
    const auto fp = [](LinearProfile p, int n, int k) {
        return p.qtype == QType::FP8_E4M3FN_ROW_BF16 && p.layout == QuantLayout::RowScale &&
               p.n == n && p.k == k &&
               (p.policy == LinearPolicy::A16Only || p.policy == LinearPolicy::AllowA8);
    };
    require(fp(d, kRank, kTotal) && fp(u, kTotal, kRank), "unsupported geometry, format or policy");
    if (j)
        require(j->qtype == QType::BF16 && j->layout == QuantLayout::Contiguous &&
                    j->n == kStreams && j->k == kTotal && j->policy == LinearPolicy::A16Only,
                "invalid injection profile");
}

LinearProfile injection_profile(const Weight& j) {
    return {j.qtype, j.n, j.k, LinearPolicy::A16Only, j.layout};
}

std::array<common::operand::Range, 7> inputs(Weights w, const Tensor& z) {
    std::array<common::operand::Range, 7> a{};
    a[0]  = common::operand::range(z.data, z.bytes(), kOp);
    int i = 1;
    for (const auto* p : {&w.down, &w.up}) {
        const auto g = common::validate_fp8_weight(p->weight, kOp);
        a[i++]       = common::operand::range(p->weight.qdata, g.code_plane_bytes, kOp);
        a[i++]       = common::operand::range(p->weight.scales, g.scale_plane_bytes, kOp);
    }
    if (w.injection) {
        const auto& j = *w.injection;
        require(j.ndim == 2 && j.shape[0] == kStreams && j.shape[1] == kTotal && j.shape[2] == 1 &&
                    j.shape[3] == 1 && j.padded_shape[0] == kStreams &&
                    j.padded_shape[1] == kTotal && j.padded_shape[2] == 1 &&
                    j.padded_shape[3] == 1 && !j.qhigh && !j.scales && !j.high_plane_bytes &&
                    !j.group_size && !j.group &&
                    j.payload_bytes >= std::uint64_t(kStreams) * kTotal * 2,
                "invalid compact BF16 injection");
        require(common::operand::aligned(j.qdata, 16), "injection alignment");
        a[i] = common::operand::range(j.qdata, std::uint64_t(kStreams) * kTotal * 2, kOp);
    }
    return a;
}

common::operand::Range tensor(const Tensor& t, DType type, int n, int s, int m, Device dev) {
    common::operand::device(t.device, dev, kOp, "tensor");
    require(t.dtype == type && t.ne[0] == n && t.ne[1] == s && t.ne[2] == m && t.ne[3] == 1 &&
                t.is_contiguous() && common::operand::aligned(t.data, 16),
            "invalid tensor shape, dtype, strides or alignment");
    return common::operand::range(t.data, t.bytes(), kOp);
}

std::size_t capacity(LinearProfile d, LinearProfile u, const LinearProfile* j, int first,
                     int last) {
    profile(d, u, j, first, last);
    return layout(last, plan(d.policy, u.policy, last)).bytes;
}

void execute(Weights w, const Tensor& z, MemoryArena& arena, Tensor& out, Tensor* j,
             CudaExecutionView e) {
    const auto jp = w.injection ? injection_profile(*w.injection) : LinearProfile{};
    const int m   = z.ne[2];
    profile(linear_profile(w.down), linear_profile(w.up), w.injection ? &jp : nullptr, m, m);
    common::operand::device(arena.device(), e.device(), kOp, "workspace");
    for (const auto* p : {&w.down.weight, &w.up.weight, w.injection})
        if (p) common::operand::device(p->device, e.device(), kOp, "weight");
    (void)tensor(z, DType::BF16, kWidth, kStreams, m, e.device());
    const auto x   = tensor(out, DType::BF16, kWidth, m, 1, e.device());
    const auto ins = inputs(w, z);
    for (auto r : ins) common::operand::disjoint(x, r, kOp);
    common::operand::Range jr{};
    if (j) {
        jr = tensor(*j, DType::FP32, kStreams, m, 1, e.device());
        common::operand::disjoint(x, jr, kOp);
        for (auto r : ins) common::operand::disjoint(jr, r, kOp);
    }
    const auto p    = plan(w.down.policy, w.up.policy, m);
    const auto l    = layout(m, p);
    auto scope      = arena.scope();
    auto mem        = arena.alloc_bytes(l.bytes, 256);
    const auto work = common::operand::range(mem.data, mem.bytes, kOp);
    for (auto r : ins) common::operand::disjoint(work, r, kOp);
    common::operand::disjoint(work, x, kOp);
    common::operand::disjoint(work, jr, kOp);
    Scratch s{static_cast<__nv_bfloat16*>(l.u.bind(mem).data), {}};
    if (l.codes.bytes) {
        s.encoded.codes  = static_cast<std::uint8_t*>(l.codes.bind(mem).data);
        s.encoded.scales = static_cast<float*>(l.scales.bind(mem).data);
    }
    if (l.partials.bytes) s.encoded.partials = static_cast<float*>(l.partials.bind(mem).data);
    launch(w, z, out, j, s, p, e);
}
} // namespace
} // namespace ninfer::ops::detail::hc::mix::fp8

namespace ninfer::ops {
namespace f = detail::hc::mix::fp8;

void hc_mix(const Tensor& z, const HcFp8MixWeights& w, MemoryArena& a, Tensor& x, Tensor& j,
            CudaExecutionView e) {
    f::execute({w.down, w.up, &w.injection}, z, a, x, &j, e);
}

void hc_mix(const Tensor& z, const HcFp8FinalMixWeights& w, MemoryArena& a, Tensor& x,
            CudaExecutionView e) {
    f::execute({w.down, w.up, nullptr}, z, a, x, nullptr, e);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFp8MixProfile& p, int first, int last) {
    return f::capacity(p.down, p.up, &p.injection, first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFp8FinalMixProfile& p, int first, int last) {
    return f::capacity(p.down, p.up, nullptr, first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFp8MixWeights& w, int first, int last) {
    return hc_mix_workspace_capacity_bytes(HcFp8MixProfile{linear_profile(w.down),
                                                           linear_profile(w.up),
                                                           f::injection_profile(w.injection)},
                                           first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFp8FinalMixWeights& w, int first, int last) {
    return hc_mix_workspace_capacity_bytes(
        HcFp8FinalMixProfile{linear_profile(w.down), linear_profile(w.up)}, first, last);
}
} // namespace ninfer::ops
