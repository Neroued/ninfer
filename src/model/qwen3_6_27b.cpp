#include "qus/model/model.h"

#include "model/position.h"
#include "model/gqa_prompt_ops.h"
#include "model/mtp_ops.h"
#include "qus/core/device.h"
#include "qus/core/weight_store_parser.h"
#include "qus/kernels/argmax.h"
#include "qus/kernels/attn_input_proj.h"
#include "qus/kernels/causal_conv1d_silu.h"
#include "qus/kernels/embedding.h"
#include "qus/kernels/gated_delta_rule.h"
#include "qus/kernels/gated_rmsnorm.h"
#include "qus/kernels/gdn_gating.h"
#include "qus/kernels/gdn_gating_proj.h"
#include "qus/kernels/gdn_input_proj.h"
#include "qus/kernels/gqa_attention.h"
#include "qus/kernels/l2norm.h"
#include "qus/kernels/linear.h"
#include "qus/kernels/linear_add.h"
#include "qus/kernels/linear_pair.h"
#include "qus/kernels/linear_swiglu.h"
#include "qus/kernels/residual_add.h"
#include "qus/kernels/rmsnorm.h"
#include "qus/kernels/rope.h"
#include "qus/kernels/sampling.h"
#include "qus/kernels/sigmoid_mul.h"
#include "qus/kernels/silu_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qus::model {
namespace {

constexpr ModuleKind kText               = ModuleKind::TextCore;
constexpr ModuleKind kMtp                = ModuleKind::MtpDraft;
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

const Weight* require_mtp_weight(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                                 const char* field) {
    const Weight* weight = store.qweight(kMtp, sk(kind), layer);
    if (weight == nullptr) {
        throw std::runtime_error("missing q5090 MTP weight: " + source_label(field, kind, layer));
    }
    return weight;
}

const Weight* require_mtp_weight_fused(const WeightStore& store, std::uint16_t group_id,
                                       std::uint16_t fusion_index, std::uint32_t layer,
                                       const char* field) {
    const Weight* weight = store.qfused(kMtp, group_id, fusion_index, layer);
    if (weight == nullptr) {
        throw std::runtime_error("missing q5090 MTP fused weight: " +
                                 fused_label(field, group_id, fusion_index, layer));
    }
    return weight;
}

const Tensor* require_mtp_tensor(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                                 const char* field) {
    const Tensor* tensor = store.tensor(kMtp, sk(kind), layer);
    if (tensor == nullptr) {
        throw std::runtime_error("missing q5090 MTP tensor: " + source_label(field, kind, layer));
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

MtpW bind_mtp(const WeightStore& store) {
    return MtpW{
        require_mtp_weight(store, SourceKind::MtpFc, kQ5090NoLayer, "mtp.fc"),
        require_mtp_tensor(store, SourceKind::MtpPreFcNormEmb, kQ5090NoLayer,
                           "mtp.pre_fc_norm_embedding"),
        require_mtp_tensor(store, SourceKind::MtpPreFcNormHid, kQ5090NoLayer,
                           "mtp.pre_fc_norm_hidden"),
        require_mtp_tensor(store, SourceKind::InputLayernorm, 0, "mtp.input_norm"),
        require_mtp_weight_fused(store, kFusionAttnIn, 0, 0, "mtp.attn_in"),
        require_mtp_weight(store, SourceKind::AttnQ, 0, "mtp.q_proj"),
        require_mtp_weight(store, SourceKind::AttnGate, 0, "mtp.gate_proj"),
        require_mtp_weight(store, SourceKind::AttnK, 0, "mtp.k_proj"),
        require_mtp_weight(store, SourceKind::AttnV, 0, "mtp.v_proj"),
        require_mtp_tensor(store, SourceKind::AttnQNorm, 0, "mtp.q_norm"),
        require_mtp_tensor(store, SourceKind::AttnKNorm, 0, "mtp.k_norm"),
        require_mtp_weight(store, SourceKind::AttnO, 0, "mtp.o_proj"),
        require_mtp_tensor(store, SourceKind::PostAttnLayernorm, 0, "mtp.post_attn_norm"),
        require_mtp_weight_fused(store, kFusionMlpGateUp, 0, 0, "mtp.gate_up"),
        require_mtp_weight(store, SourceKind::MlpDown, 0, "mtp.down"),
        require_mtp_tensor(store, SourceKind::MtpNorm, kQ5090NoLayer, "mtp.norm"),
    };
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

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_vector_window(const Tensor& t, DType dtype, std::int32_t cols, const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] < cols || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

Tensor vector_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("vector_window cols must be positive"); }
    if (t.ne[0] < cols || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("vector_window shape mismatch");
    }
    return t.slice(0, 0, cols);
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
    if (x.dtype == DType::I64) {
        std::vector<std::int64_t> values(n);
        CUDA_CHECK(cudaMemcpy(values.data(), x.data, values.size() * sizeof(std::int64_t),
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
                         KVCache& kv, GdnState& state, StepState& io, std::uint32_t prefill_chunk,
                         KVCache* mtp_kv)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
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

    // Optional independent LM_HEAD_DRAFT module. Non-resident modules publish no descriptors.
    const auto draft_module = ModuleKind::LmHeadDraft;
    const Weight* draft_head =
        weights_.qweight(draft_module, sk(SourceKind::LmHeadDraft), kQ5090NoLayer);
    const Tensor* draft_ids =
        weights_.tensor(draft_module, sk(SourceKind::LmHeadDraftIdmap), kQ5090NoLayer);
    if (draft_head != nullptr && draft_ids != nullptr) {
        set_lm_head_draft(draft_head, static_cast<const std::int32_t*>(draft_ids->data),
                          draft_head->n);
    }

    if (mtp_kv_ != nullptr) { mtp_ = bind_mtp(weights_); }

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

const MtpW& Qwen3_6_27B::mtp_weights() const {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void Qwen3_6_27B::mtp_forward_stem(const Tensor& ids, const Tensor& hidden, Tensor& x, Tensor& ah) {
    cudaStream_t s = ctx_.stream;
    const int T    = ids.ne[0];

    Tensor emb = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::embedding(ids, *embed_, emb, s);

    Tensor e = work_.alloc(DType::BF16, {kCfg.hidden, T});
    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    kernels::rmsnorm(hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = work_.alloc(DType::BF16, {kCfg.mtp_fc_in, T});
    kernels::mtp_pack_fc_input(e, h, fc_in, s);

    x = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(fc_in, *mtp_.fc, x, work_, s);

    ah = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void Qwen3_6_27B::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor attn_in = work_.alloc(DType::BF16, {kCfg.mtp_attn_in, T});
    kernels::linear(ah, *mtp_.attn_in, attn_in, work_, s);

    Tensor q    = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor k    = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor v    = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    kernels::mtp_split_attn_in(attn_in, q, k, gate, v, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    kernels::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    kernels::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    kernels::rope(positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    kernels::gqa_attention(qn, kn, v, positions, kAttnScale, *mtp_kv_, 0, work_, a, s);
    kernels::sigmoid_mul(gate, a, s);

    Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, work_, s);
    kernels::residual_add(o, x, s);

    Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    Tensor gate_up = work_.alloc(DType::BF16, {kCfg.mtp_mlp_gateup_rows, T});
    kernels::linear(mh, *mtp_.gate_up, gate_up, work_, s);

    Tensor act = work_.alloc(DType::BF16, {kCfg.intermediate, T});
    kernels::silu_mul(gate_up.slice(0, 0, kCfg.intermediate),
                      gate_up.slice(0, kCfg.intermediate, kCfg.intermediate), act, s);

    Tensor d = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::linear(act, *mtp_.down, d, work_, s);
    kernels::residual_add(d, x, s);

    kernels::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, mtp_hidden, s);
}

void Qwen3_6_27B::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   Tensor& mtp_hidden) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, x, ah);
    mtp_forward_tail(x, ah, positions, mtp_hidden);
}

void Qwen3_6_27B::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, bool final_chunk, Tensor* final_hidden,
                                    Tensor* logits, Tensor* draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        kernels::linear_pair(ah, *mtp_.k_proj, *mtp_.v_proj, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        kernels::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        kernels::rope(positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
        kernels::gqa_kv_append(kn, v, positions, *mtp_kv_, 0, s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        kernels::linear(ah_last, *mtp_.q_proj, q_flat, work_, s);
        kernels::linear(ah_last, *mtp_.gate_proj, gate_flat, work_, s);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        kernels::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        kernels::rope(last_position, kCfg.rotary_dim, kCfg.rope_theta, qn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        kernels::gqa_attention_cached(qn, last_position, kAttnScale, *mtp_kv_, 0, a, s);
        kernels::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        kernels::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, work_, s);
        kernels::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        kernels::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        Tensor gate_up = work_.alloc(DType::BF16, {kCfg.mtp_mlp_gateup_rows, 1});
        kernels::linear(mh, *mtp_.gate_up, gate_up, work_, s);
        Tensor act = work_.alloc(DType::BF16, {kCfg.intermediate, 1});
        kernels::silu_mul(gate_up.slice(0, 0, kCfg.intermediate),
                          gate_up.slice(0, kCfg.intermediate, kCfg.intermediate), act, s);
        Tensor d = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        kernels::linear(act, *mtp_.down, d, work_, s);
        kernels::residual_add(d, x_last, s);
        kernels::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        mtp_draft_argmax(*final_hidden, *logits, *draft_token);
    }
}

void Qwen3_6_27B::mtp_draft_argmax(const Tensor& hidden_col, Tensor& logits, Tensor& draft_token) {
    if (lm_head_draft_ != nullptr) {
        Tensor draft_logits = work_.alloc(DType::BF16, {lm_head_draft_n_, 1});
        kernels::linear(hidden_col, *lm_head_draft_, draft_logits, work_, ctx_.stream);
        kernels::argmax(draft_logits, draft_token, ctx_.stream);
        kernels::mtp_remap_draft_token(draft_token, lm_head_draft_ids_, lm_head_draft_n_,
                                       ctx_.stream);
    } else {
        kernels::linear(hidden_col, *lm_head_, logits, work_, ctx_.stream);
        kernels::argmax(logits, draft_token, ctx_.stream);
    }
}

void Qwen3_6_27B::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, Tensor& mtp_hidden, int logits_column,
                                    Tensor* logits, Tensor* draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    mtp_forward_core(ids, hidden, positions, mtp_hidden);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        mtp_draft_argmax(col, *logits, *draft_token);
    }
}

void Qwen3_6_27B::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position, Tensor& mtp_hidden, Tensor& logits,
                                      Tensor& draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    mtp_forward_core(token, previous_hidden, position, mtp_hidden);
    auto logits_scope = work_.scope();
    mtp_draft_argmax(mtp_hidden, logits, draft_token);
}

void Qwen3_6_27B::mtp_sample_from_hidden_row(const Tensor& mtp_hidden, const Tensor& row,
                                             Tensor& out_hidden, Tensor& logits,
                                             Tensor& draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    if (mtp_hidden.dtype != DType::BF16 || mtp_hidden.ne[0] != kCfg.hidden ||
        mtp_hidden.ne[1] <= 0 || mtp_hidden.ne[2] != 1 || mtp_hidden.ne[3] != 1 ||
        !mtp_hidden.is_contiguous() || mtp_hidden.data == nullptr) {
        throw std::invalid_argument("MTP sample hidden shape mismatch");
    }
    require_tensor_shape(row, DType::I32, {1}, "MTP sample row");
    require_tensor_shape(out_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP sample hidden out");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP sample logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP sample draft token");

    auto scratch_scope = work_.scope();
    kernels::mtp_gather_hidden_row(mtp_hidden, row, out_hidden, ctx_.stream);
    mtp_draft_argmax(out_hidden, logits, draft_token);
}

void Qwen3_6_27B::target_verify(const Tensor& ids, const Tensor& positions) {
    const int T = ids.ne[0];
    if (T <= 0) { throw std::invalid_argument("target_verify T must be positive"); }
    require_tensor_shape(ids, DType::I32, {T}, "target_verify ids");
    require_tensor_shape(positions, DType::I32, {T}, "target_verify positions");
    require_tensor_window(io_.verify_hidden, DType::BF16, kCfg.hidden, T, "target_verify hidden");
    require_tensor_window(io_.logits, DType::BF16, kCfg.vocab, T, "target_verify logits");
    require_vector_window(io_.target_tokens, DType::I32, T, "target_verify target_tokens");

    cudaStream_t s = ctx_.stream;
    work_.reset();

    {
        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, T});
        kernels::embedding(ids, *embed_, x, s);
        ScopedPositions scoped_positions(active_positions_, positions);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);

        Tensor hidden = matrix_window(io_.verify_hidden, T);
        Tensor logits = matrix_window(io_.logits, T);
        Tensor target = vector_window(io_.target_tokens, T);
        kernels::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, s);
        kernels::linear(hidden, *lm_head_, logits, work_, s);
        kernels::argmax(logits, target, s);
    }

    work_.reset();
}

void Qwen3_6_27B::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

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
        kernels::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
        kernels::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
        const Tensor& positions = active_positions_ != nullptr ? *active_positions_ : io_.pos;
        kernels::rope(positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        kernels::gqa_attention(qn, kn, v, positions, kAttnScale, kv_, fidx, work_, a, s);
        kernels::sigmoid_mul(gate, a, s);

        kernels::linear_add(a.view({kCfg.q_size, 1}), *w.o_proj, x, work_, s);
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
    kernels::attn_input_proj(h, *w.q_proj, *w.gate_proj, *w.k_proj, *w.v_proj, q_flat, gate_flat,
                             k_flat, v_flat, work_, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    kernels::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    kernels::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& positions = active_positions_ != nullptr ? *active_positions_ : io_.pos;
    kernels::rope(positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    kernels::gqa_attention(qn, kn, v, positions, kAttnScale, kv_, fidx, work_, a, s);
    kernels::sigmoid_mul(gate, a, s);

    kernels::linear_add(a.view({kCfg.q_size, T}), *w.o_proj, x, work_, s);
}

void Qwen3_6_27B::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    if (ph == Phase::Decode) {
        Tensor qkv    = work_.alloc(DType::BF16, {kCfg.conv_dim, 1});
        Tensor qk_out = qkv.slice(0, 0, 2 * kCfg.key_dim);
        Tensor v_out  = qkv.slice(0, 2 * kCfg.key_dim, kCfg.value_dim);
        Tensor z      = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        Tensor z_flat = z.view({kCfg.value_dim, 1});
        kernels::linear(h, *w.in_qk_q4, qk_out, work_, s);
        kernels::linear_pair(h, *w.in_v, *w.in_z, v_out, z_flat, work_, s);

        Tensor qkv_c = work_.alloc(DType::BF16, {kCfg.conv_dim, 1});
        if (mtp_enabled()) {
            Tensor& conv_states = state_.conv.at(static_cast<std::size_t>(gidx));
            kernels::causal_conv1d_silu_snapshot(qkv, *w.conv1d, conv_states, io_.gdn_initial_slot,
                                                 qkv_c, s);
        } else {
            Tensor conv_state = state_.conv_slot(static_cast<std::uint32_t>(gidx), 0);
            kernels::causal_conv1d_silu(qkv, *w.conv1d, conv_state, qkv_c, s);
        }

        Tensor g    = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, 1});
        Tensor beta = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, 1});
        kernels::gdn_gating_proj(h, *w.in_a, *w.in_b, *w.a_log, *w.dt_bias, work_, g, beta, s);

        Tensor qc = qkv_c.slice(0, 0, kCfg.key_dim);
        Tensor kc = qkv_c.slice(0, kCfg.key_dim, kCfg.key_dim);
        Tensor vc = qkv_c.slice(0, 2 * kCfg.key_dim, kCfg.value_dim);

        Tensor qn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1});
        kernels::l2norm(qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1}), 1.0e-6f, qn, s);
        kernels::l2norm(kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1}), 1.0e-6f, kn, s);

        Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        Tensor o  = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        if (mtp_enabled()) {
            Tensor& ssm_states = state_.ssm.at(static_cast<std::size_t>(gidx));
            kernels::gated_delta_rule_snapshot(qn, kn, vv, g, beta, kGdnScale, work_, ssm_states,
                                               io_.gdn_initial_slot, o, s);
        } else {
            Tensor ssm_state = state_.ssm_slot(static_cast<std::uint32_t>(gidx), 0);
            kernels::gated_delta_rule(qn, kn, vv, g, beta, kGdnScale, work_, ssm_state, o, s);
        }

        Tensor on = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        kernels::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

        kernels::linear_add(on.view({kCfg.value_dim, 1}), *w.out_proj, x, work_, s);
        return;
    }

    Tensor qkv = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
    kernels::gdn_input_proj(h, *w.in_qk_q4, *w.in_v, qkv, work_, s);

    Tensor qkv_c = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
    if (ph == Phase::Verify) {
        Tensor& conv_states = state_.conv.at(static_cast<std::size_t>(gidx));
        kernels::causal_conv1d_silu_snapshot(qkv, *w.conv1d, conv_states, io_.gdn_initial_slot,
                                             qkv_c, s);
    } else {
        // Prefill reads the committed conv window from gdn_prefill_read_slot_ and writes the
        // running window to slot 0 (in-place when the read slot is 0).
        Tensor conv_in = state_.conv_slot(static_cast<std::uint32_t>(gidx), gdn_prefill_read_slot_);
        Tensor conv_out = state_.conv_slot(static_cast<std::uint32_t>(gidx), 0);
        kernels::causal_conv1d_silu(qkv, *w.conv1d, conv_in, conv_out, qkv_c, s);
    }

    Tensor g    = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    Tensor beta = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    kernels::gdn_gating_proj(h, *w.in_a, *w.in_b, *w.a_log, *w.dt_bias, work_, g, beta, s);

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

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor& ssm_states = state_.ssm.at(static_cast<std::size_t>(gidx));
        kernels::gated_delta_rule_snapshot(qn, kn, vv, g, beta, kGdnScale, work_, ssm_states,
                                           io_.gdn_initial_slot, o, s);
    } else {
        // Prefill reads the committed recurrent state from gdn_prefill_read_slot_ and writes the
        // running state to slot 0 (in-place when the read slot is 0).
        Tensor ssm_in  = state_.ssm_slot(static_cast<std::uint32_t>(gidx), gdn_prefill_read_slot_);
        Tensor ssm_out = state_.ssm_slot(static_cast<std::uint32_t>(gidx), 0);
        kernels::gated_delta_rule(qn, kn, vv, g, beta, kGdnScale, work_, ssm_in, ssm_out, o, s);
    }

    Tensor z      = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor z_flat = z.view({kCfg.value_dim, T});
    kernels::linear(h, *w.in_z, z_flat, work_, s);

    Tensor on = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    kernels::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    kernels::linear_add(on.view({kCfg.value_dim, T}), *w.out_proj, x, work_, s);
}

void Qwen3_6_27B::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    (void)ph;

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    kernels::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.intermediate, T});
    kernels::linear_swiglu(h, *m.gate_up, a, work_, s);
    kernels::linear_add(a, *m.down, x, work_, s);
}

template <class Tap>
void Qwen3_6_27B::run_layers(Tensor& x, Phase ph, Tap& tap) {
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            {
                auto mixer_scope = work_.scope();
                attn_mix(full, x, fidx, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMixer, layer, ph, x, ctx_.stream); }
            }
            {
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMlp, layer, ph, x, ctx_.stream); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            {
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMixer, layer, ph, x, ctx_.stream); }
            }
            {
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMlp, layer, ph, x, ctx_.stream); }
            }
        }
    }
}

void Qwen3_6_27B::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
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

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    const std::uint32_t base = kv_.pos;
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Qwen3_6_27B::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    // The committed GDN state for the resident prefix lives in snapshot slot gdn_initial_slot
    // (slot 0 for a reset). Chunk 0 reads from it; every later chunk reads slot 0 (the running
    // state chunk 0 produced). Resolved on the host because prefill is eager.
    std::int32_t gdn_read_slot0 = 0;
    if (mtp_enabled()) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(&gdn_read_slot0, io_.gdn_initial_slot.data, sizeof(gdn_read_slot0),
                              cudaMemcpyDeviceToHost));
        if (gdn_read_slot0 < 0 || gdn_read_slot0 >= state_.snapshot_slots) { gdn_read_slot0 = 0; }
    }

    // Turn-boundary GDN snapshot: when a boundary is requested, publish the running conv/SSM state
    // into the dedicated last snapshot slot exactly at absolute position
    // prefill_snapshot_boundary_. Chunks are capped so one ends precisely at the boundary, so slot
    // 0 holds the state at that position when we copy it out; the next turn continues the
    // recurrence from there for partial prefix reuse. The boundary must lie strictly inside (base,
    // base+T].
    const std::int64_t base64   = static_cast<std::int64_t>(base);
    const std::int64_t snap_abs = prefill_snapshot_boundary_;
    const bool has_snapshot =
        snap_abs > base64 && snap_abs <= base64 + static_cast<std::int64_t>(T);
    const int snap_rel               = has_snapshot ? static_cast<int>(snap_abs - base64) : -1;
    const std::int32_t boundary_slot = state_.snapshot_slots - 1;

    bool prepare_mtp_prompt = false;
    if (mtp_enabled() && io_.drafts.data != nullptr) {
        const std::uint64_t mtp_required_end =
            static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) +
            static_cast<std::uint64_t>(std::max(0, io_.drafts.ne[0] - 1));
        // Engine::decode_round consumes these drafts only when the following target verify and
        // proposal window fits. Avoid preparing a prompt/draft tail that the scheduler will
        // immediately discard in favor of its one-token capacity fallback.
        const std::uint64_t target_round_required_end =
            static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) +
            2ULL * static_cast<std::uint64_t>(io_.drafts.ne[0]);
        prepare_mtp_prompt =
            mtp_required_end <= static_cast<std::uint64_t>(mtp_kv_->max_context) &&
            target_round_required_end <= static_cast<std::uint64_t>(kv_.max_context);
    }

    for (int t0 = 0; t0 < T;) {
        int len = std::min(chunk, T - t0);
        // Cap the chunk so it ends exactly at the snapshot boundary. Because a capped chunk is
        // shorter than `chunk`, the loop must advance by the processed `len` (see the t0 += len at
        // the end of the body), not by `chunk`, or it would skip [t0+len, t0+chunk) and drop the
        // tail.
        if (snap_rel > 0 && t0 < snap_rel && t0 + len > snap_rel) { len = snap_rel - t0; }
        const bool is_last = (t0 + len == T);
        work_.reset();

        gdn_prefill_read_slot_ = (t0 == 0) ? gdn_read_slot0 : 0;

        {
            Tensor ids_device = work_.alloc(DType::I32, {len});
            copy_i32_to_device(ids.data() + t0, ids_device, s);

            Tensor positions = work_.alloc(DType::I32, {len});
            detail::fill_positions(positions, base_i + t0, s);
            ScopedPositions scoped_positions(active_positions_, positions);

            Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, len});
            kernels::embedding(ids_device, *embed_, x, s);
            if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Prefill, x, s); }
            run_layers(x, Phase::Prefill, tap);

            Tensor xf = io_.prefill_hidden.data != nullptr
                            ? matrix_window(io_.prefill_hidden, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            kernels::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);
            if constexpr (Tap::enabled) { tap(TapId::AfterFinalNorm, -1, Phase::Prefill, xf, s); }

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                kernels::linear(last_xf, *lm_head_, logits, work_, s);
                if constexpr (Tap::enabled) {
                    tap(TapId::AfterLogits, -1, Phase::Prefill, logits, s);
                }
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                detail::set_pos(io_.pos, base_i + T, s);
                if (sampling_config_ != nullptr) {
                    kernels::sample(logits, io_.token, sampling_config_,
                                    static_cast<const std::int32_t*>(io_.pos.data),
                                    kernels::kSamplePurposePrefill, s);
                } else {
                    kernels::argmax(logits, io_.token, s);
                }
            }

            if (prepare_mtp_prompt) {
                std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
                if (is_last) {
                    for (int j = 0; j + 1 < len; ++j) {
                        mtp_ids_host[static_cast<std::size_t>(j)] = ids[t0 + j + 1];
                    }
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    CUDA_CHECK(cudaMemcpy(&next_token, io_.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                } else {
                    for (int j = 0; j < len; ++j) {
                        mtp_ids_host[static_cast<std::size_t>(j)] = ids[t0 + j + 1];
                    }
                }

                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                copy_i32_to_device(mtp_ids_host.data(), mtp_ids, s);
                if (is_last) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.drafts.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, positions, true, &io_.mtp_ar_hidden, &logits,
                                      &draft0);

                    detail::set_pos(io_.ar_pos, base_i + T, s);
                    for (int i = 1; i < io_.drafts.ne[0]; ++i) {
                        Tensor prev_token  = io_.drafts.slice(0, i - 1, 1);
                        Tensor next_token  = io_.drafts.slice(0, i, 1);
                        Tensor next_hidden = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        mtp_forward_ar_step(prev_token, io_.mtp_ar_hidden, io_.ar_pos, next_hidden,
                                            logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp_ar_hidden.data, next_hidden.data,
                                                   io_.mtp_ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        kernels::mtp_increment_i32(io_.ar_pos, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, positions, false, nullptr, nullptr, nullptr);
                }
            }
        }

        kv_.pos = base + static_cast<std::uint32_t>(t0 + len);

        // Snapshot the running GDN state at the requested boundary into the dedicated slot. This
        // chunk ended exactly at the boundary (see the len cap above), so slot 0 is the state
        // there.
        if (snap_rel > 0 && t0 + len == snap_rel) { state_.copy_slot(0, boundary_slot, s); }

        t0 += len;
    }

    // Consume the one-shot boundary request so a subsequent boundary-less prefill does not
    // snapshot.
    prefill_snapshot_boundary_ = -1;

    // Prefill wrote the running GDN state to slot 0; the first decode round must read slot 0.
    if (mtp_enabled()) { kernels::mtp_reset_gdn_initial_slot(io_.gdn_initial_slot, s); }

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
    kernels::embedding(io_.token, *embed_, x, s);
    if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Decode, x, s); }
    run_layers(x, Phase::Decode, tap);

    Tensor xf = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    kernels::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);
    if constexpr (Tap::enabled) { tap(TapId::AfterFinalNorm, -1, Phase::Decode, xf, s); }
    Tensor logits = matrix_window(io_.logits, 1);
    kernels::linear(xf, *lm_head_, logits, work_, s);
    if constexpr (Tap::enabled) { tap(TapId::AfterLogits, -1, Phase::Decode, logits, s); }
    // io_.pos holds the input token's absolute position here; the decode purpose
    // keeps this draw distinct from the prefill bonus token that shares io_.pos.
    if (sampling_config_ != nullptr) {
        kernels::sample(logits, io_.token, sampling_config_,
                        static_cast<const std::int32_t*>(io_.pos.data),
                        kernels::kSamplePurposeDecode, s);
    } else {
        kernels::argmax(logits, io_.token, s);
    }

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
