#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/visual_scatter.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_rule.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
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

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::GqaExecutionEnvelope*& slot,
                   const ops::GqaExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::GqaExecutionEnvelope*& slot_;
};

struct CallbackTap {
    static constexpr bool enabled = true;

    void* context            = nullptr;
    TextTapCallback callback = nullptr;

    void operator()(TapId id, int layer, Phase phase, const Tensor& value,
                    cudaStream_t stream) const {
        callback(context, id, layer, phase, value, stream);
    }
};

} // namespace

TextContext::TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                         KVCache& kv, qwen3_6::GdnStateStore& state, qwen3_6::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, KVCache* mtp_kv)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base) {
    if (prefill_chunk_ == 0 || prefill_chunk_ % kPrefillChunkAlignment != 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext prefill_chunk must be a nonzero multiple of 128");
    }
    bind();
}

TextContext::~TextContext() = default;

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    set_lm_head_draft(&weights_.draft_head,
                      static_cast<const std::int32_t*>(weights_.draft_head_token_ids.data),
                      weights_.draft_head.n);

    if (mtp_kv_ != nullptr) {
        const auto& source = weights_.mtp;
        mtp_               = MtpW{&source,
                    &source.input_projection,
                    &source.embedding_norm,
                    &source.hidden_norm,
                    &source.input_norm,
                    &source.query_norm,
                    &source.key_norm,
                    &source.output,
                    &source.post_attention_norm,
                    &source.final_norm};
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s = ctx_.stream;
    const int T    = ids.ne[0];

    Tensor emb;
    if (input_embeddings != nullptr) {
        require_tensor_shape(*input_embeddings, DType::BF16, {kCfg.hidden, T},
                             "MTP input embeddings");
        emb = *input_embeddings;
    } else {
        emb = work_.alloc(DType::BF16, {kCfg.hidden, T});
        ops::embedding(ids, *embed_, emb, s);
    }

    Tensor e = work_.alloc(DType::BF16, {kCfg.hidden, T});
    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = work_.alloc(DType::BF16, {kCfg.mtp_fc_in, T});
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::linear(fc_in, *mtp_.fc, x, work_, s);

    ah = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor q         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate      = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor v         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    ops::gqa_attention(qn, kn, v, positions, kAttnScale, mtp_kv_->layer_view(0), envelope, work_, a,
                       s);
    ops::sigmoid_mul(gate, a, s);

    Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, nullptr, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::GqaExecutionEnvelope envelope, bool final_chunk,
                                    Tensor* final_hidden, Tensor* logits, Tensor* draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
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
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
        ops::gqa_kv_append(kn, v, positions, mtp_kv_->layer_view(0), s);

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
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::gqa_attention_cached(qn, last_position, kAttnScale, mtp_kv_->layer_view(0), envelope,
                                  work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, work_, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        mtp_draft_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::mtp_draft_argmax(const Tensor& hidden_col, Tensor& logits, Tensor& draft_token) {
    if (lm_head_draft_ != nullptr) {
        Tensor draft_logits = work_.alloc(DType::BF16, {lm_head_draft_n_, 1});
        ops::linear(hidden_col, *lm_head_draft_, draft_logits, work_, ctx_.stream);
        ops::argmax(draft_logits, draft_token, lm_head_draft_n_, ctx_.stream);
        ops::mtp_remap_draft_token(draft_token, lm_head_draft_ids_, lm_head_draft_n_, ctx_.stream);
    } else {
        ops::linear(hidden_col, *lm_head_, logits, work_, ctx_.stream);
        ops::argmax(logits, draft_token, kCfg.token_domain, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, ops::GqaExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions) {
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

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        mtp_draft_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position, ops::GqaExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden);
    auto logits_scope = work_.scope();
    mtp_draft_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::mtp_sample_from_hidden_row(const Tensor& mtp_hidden, const Tensor& row,
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
    ops::mtp_gather_hidden_row(mtp_hidden, row, out_hidden, ctx_.stream);
    mtp_draft_argmax(out_hidden, logits, draft_token);
}

template <class Tap>
void TextContext::target_verify_impl(const Tensor& ids, const Tensor& positions,
                                     ops::GqaExecutionEnvelope envelope, Tap& tap) {
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
        Tensor rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, ctx_.stream);
        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, T});
        ops::embedding(ids, *embed_, x, s);
        if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Verify, x, s); }
        ScopedPositions scoped_cache(active_cache_positions_, positions);
        ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
        ScopedEnvelope scoped_envelope(active_gqa_envelope_, envelope);
        run_layers(x, Phase::Verify, tap);

        Tensor hidden = matrix_window(io_.verify_hidden, T);
        Tensor logits = matrix_window(io_.logits, T);
        Tensor target = vector_window(io_.target_tokens, T);
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, s);
        if constexpr (Tap::enabled) { tap(TapId::AfterFinalNorm, -1, Phase::Verify, hidden, s); }
        ops::linear(hidden, *lm_head_, logits, work_, s);
        if constexpr (Tap::enabled) { tap(TapId::AfterLogits, -1, Phase::Verify, logits, s); }
        ops::argmax(logits, target, kCfg.token_domain, s);
    }

    work_.reset();
}

void TextContext::target_verify(const Tensor& ids, const Tensor& positions,
                                ops::GqaExecutionEnvelope envelope) {
    NullTap tap;
    target_verify_impl(ids, positions, envelope, tap);
}

void TextContext::diagnostic_target_verify(const Tensor& ids, const Tensor& positions,
                                           ops::GqaExecutionEnvelope envelope, void* context,
                                           TextTapCallback callback) {
    if (callback == nullptr) {
        throw std::invalid_argument("diagnostic target verify callback is null");
    }
    CallbackTap tap{context, callback};
    target_verify_impl(ids, positions, envelope, tap);
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, work_, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    ops::gqa_attention(qn, kn, v, cache_positions, kAttnScale, kv_.layer_view(fidx),
                       *active_gqa_envelope_, work_, a, s);
    ops::sigmoid_mul(gate, a, s);

    ops::linear_add(a.view({kCfg.q_size, T}), *w.o_proj, x, work_, s);
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor z  = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor kc = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor vc = work_.alloc(DType::BF16, {kCfg.value_dim, T});
    if (ph == Phase::Verify) {
        Tensor& conv_states = state_.conv.at(static_cast<std::size_t>(gidx));
        Variant::gdn_input_projection_snapshot(h, *w.projection, *w.conv1d, conv_states,
                                               io_.gdn_initial_slot, qc, kc, vc, z, work_, s);
    } else {
        Tensor qkv = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
        Variant::gdn_input_projection(h, *w.projection, qkv, z, work_, s);
        Tensor qkv_c = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
        // Prefill reads the committed conv window from gdn_prefill_read_slot_ and writes the
        // running window to slot 0 (in-place when the read slot is 0).
        Tensor conv_in = state_.conv_slot(static_cast<std::uint32_t>(gidx), gdn_prefill_read_slot_);
        Tensor conv_out = state_.conv_slot(static_cast<std::uint32_t>(gidx), 0);
        ops::causal_conv1d_silu(qkv, *w.conv1d, conv_in, conv_out, qkv_c, s);
        ops::extract_bf16_columns(qkv_c, 0, qc, s);
        ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
        ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);
    }

    Tensor g    = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    Tensor beta = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    Variant::gdn_control_projection(h, *w.projection, g, beta, work_, s);

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor& ssm_states = state_.ssm.at(static_cast<std::size_t>(gidx));
        ops::gated_delta_rule_snapshot(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                                       /*normalize_qk=*/true, work_, ssm_states,
                                       io_.gdn_initial_slot, o, s);
    } else {
        // Prefill reads the committed recurrent state from gdn_prefill_read_slot_ and writes the
        // running state to slot 0 (in-place when the read slot is 0).
        Tensor ssm_in  = state_.ssm_slot(static_cast<std::uint32_t>(gidx), gdn_prefill_read_slot_);
        Tensor ssm_out = state_.ssm_slot(static_cast<std::uint32_t>(gidx), 0);
        ops::gated_delta_rule(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                              /*normalize_qk=*/true, work_, ssm_in, ssm_out, o, s);
    }

    Variant::gdn_output_gate_projection(h, *w.projection, z, work_, s);

    Tensor on = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    ops::linear_add(on.view({kCfg.value_dim, T}), *w.out_proj, x, work_, s);
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    (void)ph;

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);

    Variant::post_mixer(h, *m.payload, x, work_, s);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(full, x, fidx, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMixer, layer, ph, x, ctx_.stream); }
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMlp, layer, ph, x, ctx_.stream); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMixer, layer, ph, x, ctx_.stream); }
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMlp, layer, ph, x, ctx_.stream); }
            }
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
void TextContext::prefill_impl(std::span<const int> ids, const MultimodalPrefill* multimodal,
                               Tap& tap) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    mtp_prompt_prepared_     = false;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() != static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    // The committed GDN state for the resident prefix lives in snapshot slot gdn_initial_slot
    // (slot 0 for a reset). Chunk 0 reads from it; every later chunk reads slot 0 (the running
    // state chunk 0 produced). Resolved on the host because prefill is eager.
    std::int32_t gdn_read_slot0 = 0;
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(&gdn_read_slot0, io_.gdn_initial_slot.data, sizeof(gdn_read_slot0),
                          cudaMemcpyDeviceToHost));
    if (gdn_read_slot0 < 0 || gdn_read_slot0 >= state_.spec.snapshot_slots) { gdn_read_slot0 = 0; }

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
    const std::int32_t boundary_slot = state_.spec.snapshot_slots - 1;

    bool prepare_mtp_prompt = false;
    if (mtp_enabled() && io_.drafts.data != nullptr) {
        const std::uint64_t mtp_required_end =
            static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) +
            static_cast<std::uint64_t>(std::max(0, io_.drafts.ne[0] - 1));
        // Program::decode_round consumes these drafts only when the following target verify and
        // proposal window fits. Avoid preparing a prompt/draft tail that the scheduler will
        // immediately discard in favor of its one-token capacity fallback.
        const std::uint64_t target_round_required_end =
            static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) +
            2ULL * static_cast<std::uint64_t>(io_.drafts.ne[0]);
        prepare_mtp_prompt =
            mtp_required_end <= static_cast<std::uint64_t>(mtp_kv_->max_context) &&
            target_round_required_end <= static_cast<std::uint64_t>(kv_.max_context);
    }
    mtp_prompt_prepared_       = prepare_mtp_prompt;
    last_prefill_chunk_length_ = 0;

    for (int t0 = 0; t0 < T;) {
        int len = std::min(chunk, T - t0);
        // Cap the chunk so it ends exactly at the snapshot boundary. Because a capped chunk is
        // shorter than `chunk`, the loop must advance by the processed `len` (see the t0 += len at
        // the end of the body), not by `chunk`, or it would skip [t0+len, t0+chunk) and drop the
        // tail.
        if (snap_rel > 0 && t0 < snap_rel && t0 + len > snap_rel) { len = snap_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = (t0 + len == T);
        if (is_last) { last_prefill_chunk_length_ = static_cast<std::uint32_t>(len); }
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        gdn_prefill_read_slot_ = (t0 == 0) ? gdn_read_slot0 : 0;

        {
            Tensor ids_device = work_.alloc(DType::I32, {len});
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = work_.alloc(DType::I32, {len});
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = work_.alloc(DType::I32, {len, 3});
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = work_.alloc(DType::I32, {len});
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_gqa_envelope_, chunk_envelope);

            Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::embedding(ids_device, *embed_, x, s);
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                if (count > 0) {
                    const auto visual_begin = static_cast<std::int32_t>(begin - scatter.begin());
                    std::vector<std::int32_t> local_indices(static_cast<std::size_t>(count));
                    for (std::int32_t i = 0; i < count; ++i) {
                        local_indices[static_cast<std::size_t>(i)] =
                            begin[i] - static_cast<std::int32_t>(prompt_t0);
                    }
                    Tensor indices_device = work_.alloc(DType::I32, {count});
                    copy_i32(local_indices.data(), indices_device, s);
                    Tensor embeddings = vision_chunk.embeddings.slice(1, visual_begin, count);
                    ops::scatter(embeddings, indices_device, x, s);
                }
            }
            if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Prefill, x, s); }
            run_layers(x, Phase::Prefill, tap);

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);
            if constexpr (Tap::enabled) { tap(TapId::AfterFinalNorm, -1, Phase::Prefill, xf, s); }

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, work_, s);
                if constexpr (Tap::enabled) {
                    tap(TapId::AfterLogits, -1, Phase::Prefill, logits, s);
                }
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_,
                                static_cast<const std::int32_t*>(io_.pos.data),
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                                          : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr ? prompt_t0 : static_cast<std::uint32_t>(t0);
                const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr ? multimodal->token_ids : ids;
                std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                for (int j = 0; j < prompt_columns; ++j) {
                    mtp_ids_host[static_cast<std::size_t>(j)] =
                        alignment_ids[static_cast<std::size_t>(mtp_window.shifted_embedding_begin) +
                                      static_cast<std::size_t>(j)];
                }
                if (mtp_window.final_column_uses_generated_token) {
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    CUDA_CHECK(cudaMemcpy(&next_token, io_.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                }

                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                copy_i32(mtp_ids_host.data(), mtp_ids, s);
                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        qwen3_6::detail::scatter_shifted_visual_embeddings(
                            mtp_input_embeddings, vision_chunk.embeddings,
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window,
                            work_, s);
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.drafts.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp_ar_hidden,
                                      &logits, &draft0);

                    ops::set_i32_scalar(io_.ar_pos, base_i + T, s);
                    for (int i = 1; i < io_.drafts.ne[0]; ++i) {
                        Tensor prev_token     = io_.drafts.slice(0, i - 1, 1);
                        Tensor next_token     = io_.drafts.slice(0, i, 1);
                        Tensor next_hidden    = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::GqaExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp_ar_hidden, io_.ar_pos, ar_envelope,
                                            next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp_ar_hidden.data, next_hidden.data,
                                                   io_.mtp_ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(io_.ar_pos, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (snap_rel > 0 && t0 + len == snap_rel && boundary_hidden_output_ != nullptr) {
                require_tensor_shape(*boundary_hidden_output_, DType::BF16, {kCfg.hidden, 1},
                                     "boundary hidden output");
                const Tensor boundary_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(boundary_hidden_output_->data, boundary_hidden.data,
                                           boundary_hidden.bytes(), cudaMemcpyDeviceToDevice, s));
            }
        }

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
    ops::set_i32_scalar(io_.gdn_initial_slot, 0, s);

    ctx_.synchronize();
    work_.reset();
}

void TextContext::prefill(std::span<const int> ids) {
    NullTap tap;
    prefill_impl(ids, nullptr, tap);
}

void TextContext::diagnostic_prefill(std::span<const int> ids, void* context,
                                     TextTapCallback callback) {
    if (callback == nullptr) { throw std::invalid_argument("diagnostic prefill callback is null"); }
    CallbackTap tap{context, callback};
    prefill_impl(ids, nullptr, tap);
}

void TextContext::prefill(const qwen3_6::PreparedPromptData& input, std::uint32_t begin,
                          VisionPrefillSession& vision) {
    if (begin >= input.token_ids.size()) {
        throw std::invalid_argument("multimodal prefill suffix is empty");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    prefill_impl(tokens.subspan(begin), &multimodal, tap);
}

void TextContext::diagnostic_prefill(const qwen3_6::PreparedPromptData& input, std::uint32_t begin,
                                     VisionPrefillSession& vision, void* context,
                                     TextTapCallback callback) {
    if (callback == nullptr) { throw std::invalid_argument("diagnostic prefill callback is null"); }
    if (begin >= input.token_ids.size()) {
        throw std::invalid_argument("diagnostic multimodal prefill suffix is empty");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    CallbackTap tap{context, callback};
    prefill_impl(tokens.subspan(begin), &multimodal, tap);
}


} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
