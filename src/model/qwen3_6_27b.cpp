#include "qus/model/model.h"

#include "model/position.h"
#include "qus/core/device.h"
#include "qus/core/weight_store_parser.h"
#include "qus/kernels/causal_conv1d.h"
#include "qus/kernels/argmax.h"
#include "qus/kernels/gated_delta_rule.h"
#include "qus/kernels/gdn_in_ab.h"
#include "qus/kernels/gdn_gating.h"
#include "qus/kernels/gdn_in_vz.h"
#include "qus/kernels/gqa_attention.h"
#include "qus/kernels/l2norm.h"
#include "qus/kernels/linear.h"
#include "qus/kernels/linear_residual.h"
#include "qus/kernels/mlp_gate_up_silu.h"
#include "qus/kernels/embed_gather.h"
#include "qus/kernels/residual_add.h"
#include "qus/kernels/rmsnorm.h"
#include "qus/kernels/rope.h"
#include "qus/kernels/sigmoid_gate_mul.h"
#include "qus/kernels/silu_and_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qus::model {
namespace {

constexpr ModuleKind kText               = ModuleKind::TextCore;
constexpr std::uint16_t kFusionAttnIn    = 1;
constexpr std::uint16_t kFusionGdnIn     = 2;
constexpr std::uint16_t kFusionMlpGateUp = 3;

std::uint32_t sk(SourceKind kind) { return static_cast<std::uint32_t>(kind); }

std::string source_label(const char* field, SourceKind kind, std::uint32_t layer) {
    return std::string(field) + " source_kind=" + std::to_string(sk(kind)) +
           " layer=" + (layer == kQ5090NoLayer ? std::string("NO_LAYER") : std::to_string(layer));
}

std::string fused_label(const char* field, std::uint16_t group_id, std::uint16_t fusion_index,
                        std::uint32_t layer) {
    return std::string(field) + " fusion_group=" + std::to_string(group_id) +
           " fusion_index=" + std::to_string(fusion_index) +
           " layer=" + (layer == kQ5090NoLayer ? std::string("NO_LAYER") : std::to_string(layer));
}

const Weight* require_weight(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                             const char* field) {
    const Weight* weight = store.qweight(kText, sk(kind), layer);
    if (weight == nullptr) {
        throw std::runtime_error("missing q5090 weight: " + source_label(field, kind, layer));
    }
    return weight;
}

const Weight* require_weight_fused(const WeightStore& store, std::uint16_t group_id,
                                   std::uint16_t fusion_index, std::uint32_t layer,
                                   const char* field) {
    const Weight* weight = store.qfused(kText, group_id, fusion_index, layer);
    if (weight == nullptr) {
        throw std::runtime_error("missing q5090 fused weight: " +
                                 fused_label(field, group_id, fusion_index, layer));
    }
    return weight;
}

const Tensor* require_tensor(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                             const char* field) {
    const Tensor* tensor = store.tensor(kText, sk(kind), layer);
    if (tensor == nullptr) {
        throw std::runtime_error("missing q5090 tensor: " + source_label(field, kind, layer));
    }
    return tensor;
}

Weight bind_dense_weight(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                         const char* field) {
    const Tensor* tensor = require_tensor(store, kind, layer, field);
    Weight weight        = weight_from_dense(*tensor);
    weight.module        = kText;
    weight.source_kind   = sk(kind);
    weight.source_layer  = layer;
    return weight;
}

Tensor bind_conv1d_view(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                        const char* field) {
    const Tensor* tensor = require_tensor(store, kind, layer, field);
    if (tensor->dtype == DType::BF16 && tensor->ne[0] == kCfg.conv_dim &&
        tensor->ne[1] == kCfg.gdn_conv_k && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
        tensor->is_contiguous()) {
        return tensor->view({tensor->ne[0], kCfg.gdn_conv_k});
    }
    throw std::runtime_error("expected q5090 canonical conv1d shape [10240,4,1]: " +
                             source_label(field, kind, layer));
}

MlpW bind_mlp(const WeightStore& store, std::uint32_t layer) {
    return MlpW{require_weight(store, SourceKind::MlpGate, layer, "mlp.gate"),
                require_weight(store, SourceKind::MlpUp, layer, "mlp.up"),
                require_weight_fused(store, kFusionMlpGateUp, 0, layer, "mlp.gate_up"),
                require_weight(store, SourceKind::MlpDown, layer, "mlp.down")};
}

void copy_bf16_block(const Tensor& src, Tensor& dst, int dst_channel, cudaStream_t stream) {
    const std::size_t elem_size = dtype_size(DType::BF16);
    const std::size_t width     = static_cast<std::size_t>(src.ne[0]) * elem_size;
    const std::size_t src_pitch = static_cast<std::size_t>(src.ne[0]) * elem_size;
    const std::size_t dst_pitch = static_cast<std::size_t>(dst.ne[0]) * elem_size;
    auto* dst_ptr =
        static_cast<unsigned char*>(dst.data) + static_cast<std::size_t>(dst_channel) * elem_size;
    CUDA_CHECK(cudaMemcpy2DAsync(dst_ptr, dst_pitch, src.data, src_pitch, width,
                                 static_cast<std::size_t>(src.ne[1]), cudaMemcpyDeviceToDevice,
                                 stream));
}

void extract_bf16_block(const Tensor& src, int src_channel, Tensor& dst, cudaStream_t stream) {
    const std::size_t elem_size = dtype_size(DType::BF16);
    const std::size_t width     = static_cast<std::size_t>(dst.ne[0]) * elem_size;
    const std::size_t src_pitch = static_cast<std::size_t>(src.ne[0]) * elem_size;
    const std::size_t dst_pitch = static_cast<std::size_t>(dst.ne[0]) * elem_size;
    const auto* src_ptr         = static_cast<const unsigned char*>(src.data) +
                          static_cast<std::size_t>(src_channel) * elem_size;
    CUDA_CHECK(cudaMemcpy2DAsync(dst.data, dst_pitch, src_ptr, src_pitch, width,
                                 static_cast<std::size_t>(dst.ne[1]), cudaMemcpyDeviceToDevice,
                                 stream));
}

void copy_i32_to_device(const int* src, Tensor& dst, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, static_cast<std::size_t>(dst.ne[0]) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

struct ErasedTap {
    static constexpr bool enabled = true;

    void* ctx            = nullptr;
    TapCallback callback = nullptr;

    void operator()(TapId id, int layer, Phase phase, const Tensor& x, cudaStream_t stream) {
        callback(ctx, id, layer, phase, x, stream);
    }
};

float bf16_bits_to_f32(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float out             = 0.0f;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

std::vector<float> tensor_to_f32(const Tensor& x, cudaStream_t stream) {
    if (x.data == nullptr) { throw std::invalid_argument("FileTap: tensor data must be non-null"); }
    if (!x.is_contiguous()) { throw std::invalid_argument("FileTap: tensor must be contiguous"); }
    const std::int64_t n_i64 = x.numel();
    if (n_i64 < 0) { throw std::overflow_error("FileTap: negative tensor size"); }
    const auto n = static_cast<std::size_t>(n_i64);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> out(n);
    if (x.dtype == DType::BF16) {
        std::vector<std::uint16_t> bits(n);
        CUDA_CHECK(cudaMemcpy(bits.data(), x.data, bits.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < bits.size(); ++i) { out[i] = bf16_bits_to_f32(bits[i]); }
        return out;
    }
    if (x.dtype == DType::FP32) {
        CUDA_CHECK(
            cudaMemcpy(out.data(), x.data, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return out;
    }
    if (x.dtype == DType::I32) {
        std::vector<std::int32_t> values(n);
        CUDA_CHECK(cudaMemcpy(values.data(), x.data, values.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < values.size(); ++i) { out[i] = static_cast<float>(values[i]); }
        return out;
    }
    if (x.dtype == DType::U8) {
        std::vector<std::uint8_t> values(n);
        CUDA_CHECK(cudaMemcpy(values.data(), x.data, values.size() * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < values.size(); ++i) { out[i] = static_cast<float>(values[i]); }
        return out;
    }
    throw std::invalid_argument("FileTap: unsupported tensor dtype");
}

void write_f32_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("FileTap: failed to open " + path.string()); }
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) { throw std::runtime_error("FileTap: failed to write " + path.string()); }
}

std::string tap_file_name(TapId id, int layer) {
    char name[64];
    switch (id) {
    case TapId::AfterEmbed:
        return "embed.f32";
    case TapId::AfterMixer:
        std::snprintf(name, sizeof(name), "layer_%02d_mixer.f32", layer);
        return name;
    case TapId::AfterMlp:
        std::snprintf(name, sizeof(name), "layer_%02d_mlp.f32", layer);
        return name;
    case TapId::AfterFinalNorm:
        return "final_norm.f32";
    case TapId::AfterLogits:
        return "logits.f32";
    }
    throw std::invalid_argument("FileTap: unknown tap id");
}

} // namespace

FileTap::FileTap(std::filesystem::path out_dir) : out_dir_(std::move(out_dir)) {
    std::filesystem::create_directories(out_dir_);
}

void FileTap::operator()(TapId id, int layer, Phase phase, const Tensor& x, cudaStream_t stream) {
    (void)phase;
    const std::vector<float> values = tensor_to_f32(x, stream);
    write_f32_file(out_dir_ / tap_file_name(id, layer), values);
}

Qwen3_6_27B::Qwen3_6_27B(DeviceContext& ctx, WeightStore& weights, WorkspaceArena& work,
                         KVCache& kv, GdnState& state, StepState& io, std::uint32_t prefill_chunk)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), state_(state), io_(io),
      prefill_chunk_(prefill_chunk) {
    if (prefill_chunk_ == 0 || prefill_chunk_ % kPrefillChunkAlignment != 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Qwen3_6_27B prefill_chunk must be a nonzero multiple of 128");
    }
    bind();
}

Qwen3_6_27B::~Qwen3_6_27B() = default;

void Qwen3_6_27B::bind() {
    embed_      = require_weight(weights_, SourceKind::Embed, kQ5090NoLayer, "embed");
    final_norm_ = require_tensor(weights_, SourceKind::FinalNorm, kQ5090NoLayer, "final_norm");
    lm_head_    = require_weight(weights_, SourceKind::LmHead, kQ5090NoLayer, "lm_head");

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        const auto source_layer = static_cast<std::uint32_t>(layer);
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm  = require_tensor(weights_, SourceKind::InputLayernorm, source_layer,
                                             "full.input_norm");
            out.q_proj = require_weight(weights_, SourceKind::AttnQ, source_layer, "full.q_proj");
            out.gate_proj =
                require_weight(weights_, SourceKind::AttnGate, source_layer, "full.gate_proj");
            out.k_proj = require_weight(weights_, SourceKind::AttnK, source_layer, "full.k_proj");
            out.v_proj = require_weight(weights_, SourceKind::AttnV, source_layer, "full.v_proj");
            out.qkv_q4 =
                require_weight_fused(weights_, kFusionAttnIn, 0, source_layer, "attn.qkv.q4");
            out.gatev_q5 =
                require_weight_fused(weights_, kFusionAttnIn, 1, source_layer, "attn.gatev.q5");
            out.o_proj = require_weight(weights_, SourceKind::AttnO, source_layer, "full.o_proj");
            out.q_norm =
                require_tensor(weights_, SourceKind::AttnQNorm, source_layer, "full.q_norm");
            out.k_norm =
                require_tensor(weights_, SourceKind::AttnKNorm, source_layer, "full.k_norm");
            out.post_attn_norm = require_tensor(weights_, SourceKind::PostAttnLayernorm,
                                                source_layer, "full.post_attn_norm");
            out.mlp            = bind_mlp(weights_, source_layer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            out.input_norm = require_tensor(weights_, SourceKind::InputLayernorm, source_layer,
                                            "gdn.input_norm");
            out.in_q = require_weight(weights_, SourceKind::GdnInProjQ, source_layer, "gdn.in_q");
            out.in_k = require_weight(weights_, SourceKind::GdnInProjK, source_layer, "gdn.in_k");
            out.in_qk_q4 =
                require_weight_fused(weights_, kFusionGdnIn, 0, source_layer, "gdn.in_qk.q4");
            out.in_v = require_weight(weights_, SourceKind::GdnInProjV, source_layer, "gdn.in_v");
            out.in_z = require_weight(weights_, SourceKind::GdnInProjZ, source_layer, "gdn.in_z");
            gdn_in_a_[gidx] =
                bind_dense_weight(weights_, SourceKind::GdnInProjA, source_layer, "gdn.in_a");
            gdn_in_b_[gidx] =
                bind_dense_weight(weights_, SourceKind::GdnInProjB, source_layer, "gdn.in_b");
            out.in_a = &gdn_in_a_[gidx];
            out.in_b = &gdn_in_b_[gidx];
            gdn_conv1d_views_[gidx] =
                bind_conv1d_view(weights_, SourceKind::GdnConv1d, source_layer, "gdn.conv1d");
            out.conv1d = &gdn_conv1d_views_[gidx];
            out.a_log  = require_tensor(weights_, SourceKind::GdnALog, source_layer, "gdn.a_log");
            out.dt_bias =
                require_tensor(weights_, SourceKind::GdnDtBias, source_layer, "gdn.dt_bias");
            out.gdn_norm =
                require_tensor(weights_, SourceKind::GdnNorm, source_layer, "gdn.gdn_norm");
            out.out_proj =
                require_weight(weights_, SourceKind::GdnOutProj, source_layer, "gdn.out_proj");
            out.post_attn_norm = require_tensor(weights_, SourceKind::PostAttnLayernorm,
                                                source_layer, "gdn.post_attn_norm");
            out.mlp            = bind_mlp(weights_, source_layer);
        }
    }
}

void Qwen3_6_27B::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph,
                           std::uint32_t cache_offset) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, nullptr, h, s);

    if (ph == Phase::Decode) {
        Tensor qk    = work_.alloc(DType::BF16, {kCfg.q_size + kCfg.kv_size, 1});
        Tensor gatev = work_.alloc(DType::BF16, {kCfg.q_size + kCfg.kv_size, 1});
        kernels::linear(h, *w.qkv_q4, qk, work_, s);
        kernels::linear(h, *w.gatev_q5, gatev, work_, s);

        Tensor q    = qk.slice(0, 0, kCfg.q_size).view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor k    = qk.slice(0, kCfg.q_size, kCfg.kv_size).view({kCfg.head_dim, kCfg.n_kv, 1});
        Tensor gate = gatev.slice(0, 0, kCfg.q_size).view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor v    = gatev.slice(0, kCfg.q_size, kCfg.kv_size).view({kCfg.head_dim, kCfg.n_kv, 1});

        Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, 1});
        kernels::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, nullptr, qn, s);
        kernels::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, nullptr, kn, s);
        const Tensor& positions = active_positions_ != nullptr ? *active_positions_ : io_.pos;
        kernels::rope(positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        kernels::gqa_attention_decode(qn, kn, v, io_.pos, kAttnScale, kv_, fidx, work_, a, s);
        kernels::sigmoid_gate_mul(gate, a, s);

        kernels::linear_residual_add(a.view({kCfg.q_size, 1}), *w.o_proj, x, work_, s);
        return;
    }

    Tensor q         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    kernels::linear(h, *w.q_proj, q_flat, work_, s);
    kernels::linear(h, *w.gate_proj, gate_flat, work_, s);
    kernels::linear(h, *w.k_proj, k_flat, work_, s);
    kernels::linear(h, *w.v_proj, v_flat, work_, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    kernels::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, nullptr, qn, s);
    kernels::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, nullptr, kn, s);
    const Tensor& positions = active_positions_ != nullptr ? *active_positions_ : io_.pos;
    kernels::rope(positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    kernels::gqa_attention_prefill(qn, kn, v, kAttnScale, kv_, fidx, cache_offset, a, s);
    kernels::sigmoid_gate_mul(gate, a, s);

    Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(a.view({kCfg.q_size, T}), *w.o_proj, o, work_, s);
    kernels::residual_add(o, x, s);
}

void Qwen3_6_27B::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, nullptr, h, s);

    if (ph == Phase::Decode) {
        Tensor qkv    = work_.alloc(DType::BF16, {kCfg.conv_dim, 1});
        Tensor qk_out = qkv.slice(0, 0, 2 * kCfg.key_dim);
        Tensor v_out  = qkv.slice(0, 2 * kCfg.key_dim, kCfg.value_dim);
        Tensor z      = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        Tensor z_flat = z.view({kCfg.value_dim, 1});
        kernels::linear(h, *w.in_qk_q4, qk_out, work_, s);
        kernels::gdn_in_vz_decode(h, *w.in_v, *w.in_z, v_out, z_flat, s);

        Tensor qkv_c       = work_.alloc(DType::BF16, {kCfg.conv_dim, 1});
        Tensor& conv_state = state_.conv.at(static_cast<std::size_t>(gidx));
        kernels::causal_conv1d_decode(qkv, *w.conv1d, conv_state, qkv_c, s);

        Tensor g    = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, 1});
        Tensor beta = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, 1});
        kernels::gdn_in_ab_gated_decode(h, *w.in_a, *w.in_b, *w.a_log, *w.dt_bias, g, beta, s);

        Tensor qc = qkv_c.slice(0, 0, kCfg.key_dim);
        Tensor kc = qkv_c.slice(0, kCfg.key_dim, kCfg.key_dim);
        Tensor vc = qkv_c.slice(0, 2 * kCfg.key_dim, kCfg.value_dim);

        Tensor qn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1});
        kernels::l2norm(qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1}), 1.0e-6f, qn, s);
        kernels::l2norm(kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1}), 1.0e-6f, kn, s);

        Tensor vv         = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        Tensor o          = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        Tensor& ssm_state = state_.ssm.at(static_cast<std::size_t>(gidx));
        kernels::gated_delta_rule_recurrent(qn, kn, vv, g, beta, kGdnScale, work_, ssm_state, o, s);

        Tensor on = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        kernels::rmsnorm(o, *w.gdn_norm, kCfg.rms_eps, false, &z, on, s);

        kernels::linear_residual_add(on.view({kCfg.value_dim, 1}), *w.out_proj, x, work_, s);
        return;
    }

    Tensor q = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor k = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor v = work_.alloc(DType::BF16, {kCfg.value_dim, T});
    kernels::linear(h, *w.in_q, q, work_, s);
    kernels::linear(h, *w.in_k, k, work_, s);
    kernels::linear(h, *w.in_v, v, work_, s);

    Tensor qkv = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
    copy_bf16_block(q, qkv, 0, s);
    copy_bf16_block(k, qkv, kCfg.key_dim, s);
    copy_bf16_block(v, qkv, 2 * kCfg.key_dim, s);

    Tensor qkv_c       = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
    Tensor& conv_state = state_.conv.at(static_cast<std::size_t>(gidx));
    kernels::causal_conv1d_prefill(qkv, *w.conv1d, conv_state, qkv_c, s);

    Tensor g    = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    Tensor beta = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    kernels::gdn_in_ab_gated_prefill(h, *w.in_a, *w.in_b, *w.a_log, *w.dt_bias, g, beta, s);

    Tensor qc = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor kc = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor vc = work_.alloc(DType::BF16, {kCfg.value_dim, T});
    extract_bf16_block(qkv_c, 0, qc, s);
    extract_bf16_block(qkv_c, kCfg.key_dim, kc, s);
    extract_bf16_block(qkv_c, 2 * kCfg.key_dim, vc, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    kernels::l2norm(qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T}), 1.0e-6f, qn, s);
    kernels::l2norm(kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T}), 1.0e-6f, kn, s);

    Tensor vv         = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o          = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor& ssm_state = state_.ssm.at(static_cast<std::size_t>(gidx));
    kernels::gated_delta_rule_chunked(qn, kn, vv, g, beta, kGdnScale, 64, work_, ssm_state, o, s);

    Tensor z      = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor z_flat = z.view({kCfg.value_dim, T});
    kernels::linear(h, *w.in_z, z_flat, work_, s);

    Tensor on = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    kernels::rmsnorm(o, *w.gdn_norm, kCfg.rms_eps, false, &z, on, s);

    Tensor out = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(on.view({kCfg.value_dim, T}), *w.out_proj, out, work_, s);
    kernels::residual_add(out, x, s);
}

void Qwen3_6_27B::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *post_norm, kCfg.rms_eps, true, nullptr, h, s);

    if (ph == Phase::Decode) {
        Tensor a = work_.alloc(DType::BF16, {kCfg.intermediate, 1});
        kernels::mlp_gate_up_silu_decode(h, *m.gate_up, a, s);

        kernels::linear_residual_add(a, *m.down, x, work_, s);
        return;
    }

    Tensor gate_up = work_.alloc(DType::BF16, {2 * kCfg.intermediate, T});
    kernels::linear(h, *m.gate_up, gate_up, work_, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.intermediate, T});
    kernels::silu_and_mul(gate_up.slice(0, 0, kCfg.intermediate),
                          gate_up.slice(0, kCfg.intermediate, kCfg.intermediate), a, s);

    Tensor d = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(a, *m.down, d, work_, s);
    kernels::residual_add(d, x, s);
}

template <class Tap>
void Qwen3_6_27B::run_layers(Tensor& x, Phase ph, Tap& tap, std::uint32_t cache_offset) {
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx               = ModelConfig::full_idx(layer);
            const FullLayerW& full       = full_.at(static_cast<std::size_t>(fidx));
            const std::size_t mixer_mark = work_.mark();
            attn_mix(full, x, fidx, ph, cache_offset);
            if constexpr (Tap::enabled) { tap(TapId::AfterMixer, layer, ph, x, ctx_.stream); }
            work_.rewind(mixer_mark);
            const std::size_t mlp_mark = work_.mark();
            mlp_tail(full.post_attn_norm, full.mlp, x, ph);
            if constexpr (Tap::enabled) { tap(TapId::AfterMlp, layer, ph, x, ctx_.stream); }
            work_.rewind(mlp_mark);
        } else {
            const int gidx               = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn         = gdn_.at(static_cast<std::size_t>(gidx));
            const std::size_t mixer_mark = work_.mark();
            gdn_mix(gdn, x, gidx, ph);
            if constexpr (Tap::enabled) { tap(TapId::AfterMixer, layer, ph, x, ctx_.stream); }
            work_.rewind(mixer_mark);
            const std::size_t mlp_mark = work_.mark();
            mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
            if constexpr (Tap::enabled) { tap(TapId::AfterMlp, layer, ph, x, ctx_.stream); }
            work_.rewind(mlp_mark);
        }
    }
}

void Qwen3_6_27B::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap, 0);
}

template <class Tap>
void Qwen3_6_27B::prefill_impl(std::span<const int> ids, Tap& tap) {
    if (ids.empty()) { throw std::invalid_argument("Qwen3_6_27B::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Qwen3_6_27B::prefill token count exceeds int32");
    }
    cudaStream_t s  = ctx_.stream;
    const int T     = static_cast<int>(ids.size());
    const int chunk = static_cast<int>(prefill_chunk_);

    for (int t0 = 0; t0 < T; t0 += chunk) {
        const int len      = std::min(chunk, T - t0);
        const bool is_last = (t0 + len == T);
        work_.reset();

        {
            Tensor ids_device = work_.alloc(DType::I32, {len});
            copy_i32_to_device(ids.data() + t0, ids_device, s);

            Tensor positions = work_.alloc(DType::I32, {len});
            detail::fill_positions(positions, t0, s);
            ScopedPositions scoped_positions(active_positions_, positions);

            Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, len});
            kernels::embed_gather(ids_device, *embed_, x, s);
            if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Prefill, x, s); }
            run_layers(x, Phase::Prefill, tap, static_cast<std::uint32_t>(t0));

            if (is_last) {
                Tensor last_x = x.slice(1, len - 1, 1);
                Tensor xf     = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                kernels::rmsnorm(last_x, *final_norm_, kCfg.rms_eps, true, nullptr, xf, s);
                if constexpr (Tap::enabled) {
                    tap(TapId::AfterFinalNorm, -1, Phase::Prefill, xf, s);
                }
                kernels::linear(xf, *lm_head_, io_.logits, work_, s);
                if constexpr (Tap::enabled) {
                    tap(TapId::AfterLogits, -1, Phase::Prefill, io_.logits, s);
                }
                kernels::argmax(io_.logits, io_.token, s);
            }
        }

        kv_.pos = static_cast<std::uint32_t>(t0 + len);
    }

    detail::set_pos(io_.pos, T, s);
    ctx_.synchronize();
    work_.reset();
}

void Qwen3_6_27B::prefill(std::span<const int> ids) {
    NullTap tap;
    prefill_impl(ids, tap);
}

void Qwen3_6_27B::prefill_erased(std::span<const int> ids, void* tap, TapCallback callback) {
    if (callback == nullptr) { throw std::invalid_argument("Qwen3_6_27B::prefill tap is null"); }
    ErasedTap erased{tap, callback};
    prefill_impl(ids, erased);
}

template <class Tap>
void Qwen3_6_27B::decode_step_impl(Tap& tap) {
    cudaStream_t s = ctx_.stream;
    work_.reset();

    Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    kernels::embed_gather(io_.token, *embed_, x, s);
    if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Decode, x, s); }
    run_layers(x, Phase::Decode, tap, 0);

    Tensor xf = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    kernels::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, nullptr, xf, s);
    if constexpr (Tap::enabled) { tap(TapId::AfterFinalNorm, -1, Phase::Decode, xf, s); }
    kernels::linear(xf, *lm_head_, io_.logits, work_, s);
    if constexpr (Tap::enabled) { tap(TapId::AfterLogits, -1, Phase::Decode, io_.logits, s); }
    kernels::argmax(io_.logits, io_.token, s);

    detail::advance_pos(io_.pos, s);
    work_.reset();
}

void Qwen3_6_27B::decode_step() {
    NullTap tap;
    decode_step_impl(tap);
    kv_.advance();
}

void Qwen3_6_27B::decode_step_record() {
    NullTap tap;
    decode_step_impl(tap);
}

void Qwen3_6_27B::decode_step_erased(void* tap, TapCallback callback) {
    if (callback == nullptr) { throw std::invalid_argument("Qwen3_6_27B::decode tap is null"); }
    ErasedTap erased{tap, callback};
    decode_step_impl(erased);
    kv_.advance();
}

} // namespace qus::model
