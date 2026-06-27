#include "qus/model/model.h"

#include "qus/core/device.h"
#include "qus/core/weight_store_parser.h"
#include "qus/kernels/causal_conv1d.h"
#include "qus/kernels/gated_delta_rule.h"
#include "qus/kernels/gdn_gating.h"
#include "qus/kernels/gqa_attention.h"
#include "qus/kernels/l2norm.h"
#include "qus/kernels/linear.h"
#include "qus/kernels/residual_add.h"
#include "qus/kernels/rmsnorm.h"
#include "qus/kernels/rope.h"
#include "qus/kernels/sigmoid_gate_mul.h"
#include "qus/kernels/silu_and_mul.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::model {
namespace {

constexpr ModuleKind kText = ModuleKind::TextCore;

std::uint32_t sk(SourceKind kind) { return static_cast<std::uint32_t>(kind); }

std::string source_label(const char* field, SourceKind kind, std::uint32_t layer) {
    return std::string(field) + " source_kind=" + std::to_string(sk(kind)) +
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
    if (tensor->ne[1] == 1 && tensor->ne[2] == kCfg.gdn_conv_k && tensor->is_contiguous()) {
        return tensor->view({tensor->ne[0], kCfg.gdn_conv_k});
    }
    if (tensor->ne[1] == kCfg.gdn_conv_k && tensor->ne[2] == 1) { return *tensor; }
    throw std::runtime_error("unexpected q5090 conv1d shape: " + source_label(field, kind, layer));
}

MlpW bind_mlp(const WeightStore& store, std::uint32_t layer) {
    return MlpW{require_weight(store, SourceKind::MlpGate, layer, "mlp.gate"),
                require_weight(store, SourceKind::MlpUp, layer, "mlp.up"),
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

Tensor conv_state_for_kernel(Tensor& state) {
    if (state.ne[1] == kCfg.gdn_conv_k) { return state.slice(1, 0, kCfg.gdn_conv_k - 1); }
    return state;
}

} // namespace

Qwen3_6_27B::Qwen3_6_27B(DeviceContext& ctx, WeightStore& weights, WorkspaceArena& work,
                         KVCache& kv, GdnState& state, StepState& io)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), state_(state), io_(io) {
    bind();
}

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

void Qwen3_6_27B::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, nullptr, h, s);

    Tensor q         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    kernels::linear(h, *w.q_proj, q_flat, s);
    kernels::linear(h, *w.gate_proj, gate_flat, s);
    kernels::linear(h, *w.k_proj, k_flat, s);
    kernels::linear(h, *w.v_proj, v_flat, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    kernels::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, nullptr, qn, s);
    kernels::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, nullptr, kn, s);
    kernels::rope(io_.pos, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    if (ph == Phase::Prefill) {
        kernels::gqa_attention_prefill(qn, kn, v, kAttnScale, kv_, fidx, a, s);
    } else {
        kernels::gqa_attention_decode(qn, kn, v, io_.pos, kAttnScale, kv_, fidx, a, s);
    }
    kernels::sigmoid_gate_mul(gate, a, s);

    Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(a.view({kCfg.q_size, T}), *w.o_proj, o, s);
    kernels::residual_add(o, x, s);
}

void Qwen3_6_27B::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, nullptr, h, s);

    Tensor q = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor k = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor v = work_.alloc(DType::BF16, {kCfg.value_dim, T});
    kernels::linear(h, *w.in_q, q, s);
    kernels::linear(h, *w.in_k, k, s);
    kernels::linear(h, *w.in_v, v, s);

    Tensor qkv = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
    copy_bf16_block(q, qkv, 0, s);
    copy_bf16_block(k, qkv, kCfg.key_dim, s);
    copy_bf16_block(v, qkv, 2 * kCfg.key_dim, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.gdn_v_heads, T});
    Tensor b = work_.alloc(DType::BF16, {kCfg.gdn_v_heads, T});
    kernels::linear(h, *w.in_a, a, s);
    kernels::linear(h, *w.in_b, b, s);

    Tensor qkv_c      = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
    Tensor conv_state = conv_state_for_kernel(state_.conv.at(static_cast<std::size_t>(gidx)));
    if (ph == Phase::Prefill) {
        kernels::causal_conv1d_prefill(qkv, *w.conv1d, conv_state, qkv_c, s);
    } else {
        kernels::causal_conv1d_decode(qkv, *w.conv1d, conv_state, qkv_c, s);
    }

    Tensor g    = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    Tensor beta = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    kernels::gdn_gating(a, b, *w.a_log, *w.dt_bias, g, beta, s);

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
    if (ph == Phase::Prefill) {
        kernels::gated_delta_rule_chunked(qn, kn, vv, g, beta, kGdnScale, 64, work_, ssm_state, o,
                                          s);
    } else {
        kernels::gated_delta_rule_recurrent(qn, kn, vv, g, beta, kGdnScale, ssm_state, o, s);
    }

    Tensor z      = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor z_flat = z.view({kCfg.value_dim, T});
    kernels::linear(h, *w.in_z, z_flat, s);

    Tensor on = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    kernels::rmsnorm(o, *w.gdn_norm, kCfg.rms_eps, false, &z, on, s);

    Tensor out = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(on.view({kCfg.value_dim, T}), *w.out_proj, out, s);
    kernels::residual_add(out, x, s);
}

void Qwen3_6_27B::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    (void)ph;
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *post_norm, kCfg.rms_eps, true, nullptr, h, s);

    Tensor g = work_.alloc(DType::BF16, {kCfg.intermediate, T});
    Tensor u = work_.alloc(DType::BF16, {kCfg.intermediate, T});
    kernels::linear(h, *m.gate, g, s);
    kernels::linear(h, *m.up, u, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.intermediate, T});
    kernels::silu_and_mul(g, u, a, s);

    Tensor d = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(a, *m.down, d, s);
    kernels::residual_add(d, x, s);
}

void Qwen3_6_27B::run_layers(Tensor& x, Phase ph) {
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            attn_mix(full, x, fidx, ph);
            mlp_tail(full.post_attn_norm, full.mlp, x, ph);
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            gdn_mix(gdn, x, gidx, ph);
            mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
        }
    }
}

void Qwen3_6_27B::prefill(std::span<const int> ids) {
    (void)ids;
    throw std::runtime_error("Qwen3_6_27B::prefill is implemented in m2-4");
}

void Qwen3_6_27B::decode_step() {
    throw std::runtime_error("Qwen3_6_27B::decode_step is implemented in m2-4");
}

} // namespace qus::model
