#include "distributed/moe/real.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/load.h"
#include "ops/op_tester.h"
#include <numeric>

namespace ninfer::test::exchange {
namespace q   = models::qwen4_exp;
namespace moe = models::moe;

namespace {
struct SourceMatrix {
    std::vector<std::byte> codes, scales;
    int k;
    bool nv;
    double divisor = 1;

    SourceMatrix(const artifact::Reader& reader, const moe::MatrixSource& m, int expert)
        : k(m.layout.columns), nv(m.layout.format == QType::NVFP4) {
        if (!nv && m.layout.format != QType::FP8_E4M3FN_BLOCK128X128_BF16)
            throw std::runtime_error("real Qwen oracle requires NVFP4 or block128 FP8");
        const auto start = reader.directory().tensor(m.object).offset;
        const auto plane = [&](WeightBankPlane p) {
            return reader.read_range(start + p.offset + expert * p.expert_stride, p.expert_bytes);
        };
        codes  = plane(m.layout.codes);
        scales = plane(m.layout.scales);
        if (nv) {
            const auto i = m.layout.matrix_scale_begin + expert * m.layout.matrix_scale_stride;
            const auto bytes =
                reader.read_range(start + reader.geometry(m.object).matrix_scale_offset + i * 4, 4);
            float value;
            std::memcpy(&value, bytes.data(), 4);
            divisor = value;
        }
    }

    static double e4(unsigned n) {
        const int exp      = (n >> 3) & 15;
        const double value = exp ? std::ldexp(1. + (n & 7) / 8., exp - 7) : (n & 7) / 512.;
        return n & 128 ? -value : value;
    }

    double at(int r, int c) const {
        const auto pos = std::size_t(r) * k + c;
        if (nv) {
            constexpr double mag[]{0, .5, 1, 1.5, 2, 3, 4, 6};
            const int bits   = (std::to_integer<int>(codes[pos / 2]) >> (4 * (pos % 2))) & 15;
            const int group  = c / 16;
            const auto scale = ((r / 128) * (k / 64) + group / 4) * 512 + (r % 32) * 16 +
                               ((r % 128) / 32) * 4 + group % 4;
            return (bits & 8 ? -1 : 1) * mag[bits & 7] *
                   e4(std::to_integer<unsigned>(scales[scale])) / divisor;
        }
        std::uint16_t scale;
        std::memcpy(&scale, scales.data() + 2 * ((r / 128) * (k / 128) + c / 128), 2);
        return e4(std::to_integer<unsigned>(codes[pos])) * bf16_to_f32(scale);
    }
};

std::vector<std::uint16_t> host_bf16(const Weight& w) {
    if (w.qtype != QType::BF16) throw std::runtime_error("real layer control/shared must be BF16");
    std::vector<std::uint16_t> v(std::size_t(w.n) * w.k);
    CUDA_CHECK(cudaMemcpy(v.data(), w.qdata, v.size() * 2, cudaMemcpyDeviceToHost));
    return v;
}

constexpr std::array<int, 8> outputs{0, 1, 31, 127, 128, 509, 1537, 2559};
constexpr std::array<int, 4> active{0, 17, 703, 1051};

double sigmoid(double x) { return 1. / (1. + std::exp(-x)); }

std::vector<double> reference(const artifact::Reader& reader, const moe::BankDescription& bank,
                              const moe::Parameters& p, const std::vector<std::uint16_t>& x) {
    const auto control = host_bf16(p.control);
    const int h = p.profile.routed.ffn.hidden, experts = p.profile.routed.experts,
              k    = p.profile.routed.top_k;
    const auto dot = [&](const std::vector<std::uint16_t>& w, int r) {
        double sum = 0;
        for (int c : active)
            sum += bf16_to_f32(x[c]) * double(bf16_to_f32(w[std::size_t(r) * h + c]));
        return sum;
    };
    std::vector<double> logits(experts);
    std::vector<int> ids(experts);
    std::iota(ids.begin(), ids.end(), 0);
    for (int e : ids) logits[e] = dot(control, e);
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
        return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
    });
    ids.resize(k);
    double max = logits[ids[0]], den = 0;
    for (int e : ids) den += std::exp(logits[e] - max);
    auto gu = host_bf16(p.shared.gate_up.weight), down = host_bf16(p.shared.down.weight);
    std::vector<double> out(outputs.size());
    const int si = p.profile.shared.intermediate, mi = p.profile.routed.ffn.intermediate;
    for (int i = 0; i < si; ++i) {
        const double gate = dot(gu, i),
                     v    = gate * sigmoid(gate) * dot(gu, i + si) * sigmoid(dot(control, experts));
        for (std::size_t j = 0; j < out.size(); ++j)
            out[j] += v * bf16_to_f32(down[std::size_t(outputs[j]) * si + i]);
    }
    for (int e : ids) {
        SourceMatrix g(reader, bank.gate_up, e), d(reader, bank.down, e);
        const double coef = std::exp(logits[e] - max) / den;
        for (int i = 0; i < mi; ++i) {
            double gate = 0, up = 0;
            for (int c : active) {
                gate += g.at(i, c) * bf16_to_f32(x[c]);
                up += g.at(i + mi, c) * bf16_to_f32(x[c]);
            }
            const double v = coef * gate * sigmoid(gate) * up;
            for (std::size_t j = 0; j < out.size(); ++j) out[j] += v * d.at(outputs[j], i);
        }
    }
    return out;
}
} // namespace

void real_layer(distributed::WorkerConnection& c, const moe::ExpertExchangeStage& stage,
                const q::RankLoadRequest& request, const artifact::Reader& reader,
                std::shared_ptr<const q::ModelDefinition> def) {
    auto& device       = c.device();
    const auto& group  = c.communication().group(stage.group);
    const auto& domain = *request.rank.expert_domain;
    auto load          = q::resolve_rank_load(reader, def, request.rank, device.device());
    auto profile       = q::execution::describe_model(*def, load.weight_data(), load.expert_data(),
                                                      load.rank_plan());
    auto model         = q::materialize_model(std::move(load), device);
    q::execution::Parameters parameters(*model, profile);
    const auto d = device.device();
    std::vector<std::uint16_t> inputs(stage.hidden);
    for (std::size_t i = 0; i < active.size(); ++i)
        inputs[active[i]] = f32_to_bf16((i + 1) * (group.rank() + 1) * .625F);
    DeviceBuffer x(d, inputs.size() * 2), y(d, x.size()),
        obs(d, group.size() * 4 * sizeof(std::uint64_t));
    x.copy_from_host(inputs.data(), x.size());
    const std::array layers{&parameters.text.layers.front().moe, &parameters.mtp.value().layer.moe};
    // Main and MTP have different ownership/weight formats but identical physical envelopes.
    // Bind both before enqueue and reuse one frame and one caller workspace across layers.
    const auto& description = c.topology().role_group("expert", c.identity().rank);
    std::vector<moe::DeviceExpertPlacement> placements;
    std::size_t peak = 0;
    for (const auto* p : layers) {
        const auto st = moe::prepare_expert_exchange(description, domain, p->bank, stage.hidden,
                                                     stage.top_k, 1);
        placements.emplace_back(domain, st, group.rank(), d);
        peak = std::max(peak, st.softmax_workspace_bytes(group.rank(), p->profile));
    }
    moe::ExpertExchange execution(placements.front().stage, group);
    DeviceBuffer storage(d, peak);
    MemoryArena arena(storage.span());
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto* p         = layers[index];
        const auto& placement = placements[index];
        Tensor totals(obs.span(), DType::U64, {4, int(group.size())});
        const moe::LayerSource source{{x.span(), DType::BF16, {stage.hidden, 1}},
                                      {y.span(), DType::BF16, {stage.hidden, 1}},
                                      {}};
        const moe::ExpertExchangeWork work{execution, placement, 1, &totals};
        moe::resident_layer(*p, &source, &work, arena, device.execution_view());
        device.synchronize();
        std::vector<std::uint16_t> actual(inputs.size());
        y.copy_to_host(actual.data(), y.size());
        std::vector<double> got;
        for (int r : outputs) got.push_back(bf16_to_f32(actual[r]));
        const auto ref = reference(reader, def->expert_banks()[p->bank.index], *p, inputs);
        if (verify_reduction("artifact sampled layer", got, ref, {.02, 1e-6, .05}))
            throw std::runtime_error("real Qwen layer differs from original expert weights");
        std::array<std::uint64_t, 8> counts{};
        obs.copy_to_host(counts.data(), obs.size());
        if (counts[1] + counts[5] != stage.top_k)
            throw std::runtime_error("natural routes lost an assignment");
        std::cerr << "real bank=" << p->bank.index << " rank=" << group.rank()
                  << " peer_hits=" << counts[4 * (1 - group.rank()) + 1]
                  << " sampled_outputs=" << got.size() << '\n';
    }
}
} // namespace ninfer::test::exchange
