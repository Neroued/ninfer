#include "targets/qwen3_6_27b_rtx5090/impl/schedule/text_context.h"

#include "targets/qwen3_6_27b_rtx5090/impl/schedule/visual_scatter.h"
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
#include "ninfer/ops/l2norm.h"
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

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule {
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

TextContext::TextContext(DeviceContext& ctx,
                         const targets::qwen3_6_27b_rtx5090::detail::LoadedModelData& weights,
                         WorkspaceArena& work, KVCache& kv, GdnState& state, StepState& io,
                         std::uint32_t prefill_chunk, KVCache* mtp_kv)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_chunk_(prefill_chunk) {
    if (prefill_chunk_ == 0 || prefill_chunk_ % kPrefillChunkAlignment != 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext prefill_chunk must be a nonzero multiple of 128");
    }
    bind();
}

TextContext::~TextContext() = default;

void TextContext::bind() {
    using TargetBindings = targets::qwen3_6_27b_rtx5090::detail::LoadedModelData;
    using TargetMlp      = targets::qwen3_6_27b_rtx5090::detail::MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) {
        return MlpW{&source.gate, &source.up, &source.gate_up, &source.down};
    };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    set_lm_head_draft(&weights_.draft_head,
                      static_cast<const std::int32_t*>(weights_.draft_head_token_ids.data),
                      weights_.draft_head.n);

    if (mtp_kv_ != nullptr) {
        const auto& source = weights_.mtp;
        mtp_               = MtpW{&source.input_projection,
                    &source.embedding_norm,
                    &source.hidden_norm,
                    &source.input_norm,
                    &source.query_key_gate_value,
                    &source.query,
                    &source.output_gate,
                    &source.key,
                    &source.value,
                    &source.query_norm,
                    &source.key_norm,
                    &source.output,
                    &source.post_attention_norm,
                    &source.mlp.gate_up,
                    &source.mlp.down,
                    &source.final_norm};
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.q_proj         = &source.query;
            out.gate_proj      = &source.output_gate;
            out.k_proj         = &source.key;
            out.v_proj         = &source.value;
            out.qkv_q4         = &source.query_key;
            out.gatev_q5       = &source.gate_value;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.mlp);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.in_q               = &source.query;
            out.in_k               = &source.key;
            out.in_qk_q4           = &source.query_key;
            out.in_v               = &source.value;
            out.in_z               = &source.z;
            out.in_a               = &source.a_projection;
            out.in_b               = &source.b_projection;
            out.conv1d             = &source.convolution;
            out.a_log              = &source.a_log;
            out.dt_bias            = &source.dt_bias;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.mlp);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void detail::scatter_shifted_visual_embeddings(Tensor& input_embeddings,
                                               const Tensor& visual_embeddings,
                                               std::span<const std::int32_t> scatter_indices,
                                               std::int32_t shifted_begin,
                                               std::int32_t prompt_tokens, WorkspaceArena& work,
                                               cudaStream_t stream) {
    if (shifted_begin >= prompt_tokens || scatter_indices.empty()) { return; }

    const std::int32_t shifted_count =
        std::min(input_embeddings.ne[1], prompt_tokens - shifted_begin);
    const std::int32_t shifted_end = shifted_begin + shifted_count;
    const auto begin =
        std::lower_bound(scatter_indices.begin(), scatter_indices.end(), shifted_begin);
    const auto end   = std::lower_bound(begin, scatter_indices.end(), shifted_end);
    const auto count = static_cast<std::int32_t>(end - begin);
    if (count == 0) { return; }

    const auto visual_begin = static_cast<std::int32_t>(begin - scatter_indices.begin());
    std::vector<std::int32_t> local_indices(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        local_indices[static_cast<std::size_t>(i)] = begin[i] - shifted_begin;
    }
    Tensor indices_device = work.alloc(DType::I32, {count});
    copy_i32(local_indices.data(), indices_device, stream);
    Tensor embeddings = visual_embeddings.slice(1, visual_begin, count);
    ops::scatter(embeddings, indices_device, input_embeddings, stream);
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
                                   const Tensor& rope_positions, Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor attn_in = work_.alloc(DType::BF16, {kCfg.mtp_attn_in, T});
    ops::linear(ah, *mtp_.attn_in, attn_in, work_, s);

    Tensor q    = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor k    = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor v    = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    ops::mtp_split_attn_in(attn_in, q, k, gate, v, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, T});
    ops::gqa_attention(qn, kn, v, positions, kAttnScale, *mtp_kv_, 0, work_, a, s);
    ops::sigmoid_mul(gate, a, s);

    Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    Tensor gate_up = work_.alloc(DType::BF16, {kCfg.mtp_mlp_gateup_rows, T});
    ops::linear(mh, *mtp_.gate_up, gate_up, work_, s);

    Tensor act = work_.alloc(DType::BF16, {kCfg.intermediate, T});
    ops::silu_mul(gate_up.slice(0, 0, kCfg.intermediate),
                  gate_up.slice(0, kCfg.intermediate, kCfg.intermediate), act, s);

    Tensor d = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::linear(act, *mtp_.down, d, work_, s);
    ops::residual_add(d, x, s);

    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions, Tensor& mtp_hidden) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, nullptr, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions, bool final_chunk,
                                    Tensor* final_hidden, Tensor* logits, Tensor* draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
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
        ops::linear_pair(ah, *mtp_.k_proj, *mtp_.v_proj, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
        ops::gqa_kv_append(kn, v, positions, *mtp_kv_, 0, s);

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
        ops::linear(ah_last, *mtp_.q_proj, q_flat, work_, s);
        ops::linear(ah_last, *mtp_.gate_proj, gate_flat, work_, s);
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
        ops::gqa_attention_cached(qn, last_position, kAttnScale, *mtp_kv_, 0, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, work_, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        Tensor gate_up = work_.alloc(DType::BF16, {kCfg.mtp_mlp_gateup_rows, 1});
        ops::linear(mh, *mtp_.gate_up, gate_up, work_, s);
        Tensor act = work_.alloc(DType::BF16, {kCfg.intermediate, 1});
        ops::silu_mul(gate_up.slice(0, 0, kCfg.intermediate),
                      gate_up.slice(0, kCfg.intermediate, kCfg.intermediate), act, s);
        Tensor d = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(act, *mtp_.down, d, work_, s);
        ops::residual_add(d, x_last, s);
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

    auto position_scope   = work_.scope();
    Tensor rope_positions = work_.alloc(DType::I32, {T});
    ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, ctx_.stream);
    mtp_forward_core(ids, hidden, positions, rope_positions, mtp_hidden);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        mtp_draft_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position, Tensor& mtp_hidden, Tensor& logits,
                                      Tensor& draft_token) {
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
    mtp_forward_core(token, previous_hidden, position, rope_position, mtp_hidden);
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
void TextContext::target_verify_impl(const Tensor& ids, const Tensor& positions, Tap& tap) {
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

void TextContext::target_verify(const Tensor& ids, const Tensor& positions) {
    NullTap tap;
    target_verify_impl(ids, positions, tap);
}

void TextContext::diagnostic_target_verify(const Tensor& ids, const Tensor& positions,
                                           void* context, TextTapCallback callback) {
    if (callback == nullptr) {
        throw std::invalid_argument("diagnostic target verify callback is null");
    }
    CallbackTap tap{context, callback};
    target_verify_impl(ids, positions, tap);
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    if (ph == Phase::Decode) {
        Tensor qk    = work_.alloc(DType::BF16, {kCfg.q_size + kCfg.kv_size, 1});
        Tensor gatev = work_.alloc(DType::BF16, {kCfg.q_size + kCfg.kv_size, 1});
        ops::linear(h, *w.qkv_q4, qk, work_, s);
        ops::linear(h, *w.gatev_q5, gatev, work_, s);

        Tensor q    = qk.slice(0, 0, kCfg.q_size).view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor k    = qk.slice(0, kCfg.q_size, kCfg.kv_size).view({kCfg.head_dim, kCfg.n_kv, 1});
        Tensor gate = gatev.slice(0, 0, kCfg.q_size).view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor v    = gatev.slice(0, kCfg.q_size, kCfg.kv_size).view({kCfg.head_dim, kCfg.n_kv, 1});

        Tensor qn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, 1});
        ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
        ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
        const Tensor& cache_positions =
            active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
        const Tensor& rope_positions =
            active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::gqa_attention(qn, kn, v, cache_positions, kAttnScale, kv_, fidx, work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        ops::linear_add(a.view({kCfg.q_size, 1}), *w.o_proj, x, work_, s);
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
    ops::attn_input_proj(h, *w.q_proj, *w.gate_proj, *w.k_proj, *w.v_proj, q_flat, gate_flat,
                         k_flat, v_flat, work_, s);

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
    ops::gqa_attention(qn, kn, v, cache_positions, kAttnScale, kv_, fidx, work_, a, s);
    ops::sigmoid_mul(gate, a, s);

    ops::linear_add(a.view({kCfg.q_size, T}), *w.o_proj, x, work_, s);
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    Tensor h = work_.alloc(DType::BF16, {kCfg.hidden, T});
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    if (ph == Phase::Decode) {
        Tensor qkv    = work_.alloc(DType::BF16, {kCfg.conv_dim, 1});
        Tensor qk_out = qkv.slice(0, 0, 2 * kCfg.key_dim);
        Tensor v_out  = qkv.slice(0, 2 * kCfg.key_dim, kCfg.value_dim);
        Tensor z      = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        Tensor z_flat = z.view({kCfg.value_dim, 1});
        ops::linear(h, *w.in_qk_q4, qk_out, work_, s);
        ops::linear_pair(h, *w.in_v, *w.in_z, v_out, z_flat, work_, s);

        Tensor qkv_c = work_.alloc(DType::BF16, {kCfg.conv_dim, 1});
        if (mtp_enabled()) {
            Tensor& conv_states = state_.conv.at(static_cast<std::size_t>(gidx));
            ops::causal_conv1d_silu_snapshot(qkv, *w.conv1d, conv_states, io_.gdn_initial_slot,
                                             qkv_c, s);
        } else {
            Tensor conv_state = state_.conv_slot(static_cast<std::uint32_t>(gidx), 0);
            ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, qkv_c, s);
        }

        Tensor g    = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, 1});
        Tensor beta = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, 1});
        ops::gdn_gating_proj(h, *w.in_a, *w.in_b, *w.a_log, *w.dt_bias, work_, g, beta, s);

        Tensor qc = qkv_c.slice(0, 0, kCfg.key_dim);
        Tensor kc = qkv_c.slice(0, kCfg.key_dim, kCfg.key_dim);
        Tensor vc = qkv_c.slice(0, 2 * kCfg.key_dim, kCfg.value_dim);

        Tensor qn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1});
        ops::l2norm(qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1}), 1.0e-6f, qn, s);
        ops::l2norm(kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1}), 1.0e-6f, kn, s);

        Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        Tensor o  = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        if (mtp_enabled()) {
            Tensor& ssm_states = state_.ssm.at(static_cast<std::size_t>(gidx));
            ops::gated_delta_rule_snapshot(qn, kn, vv, g, beta, kGdnScale, work_, ssm_states,
                                           io_.gdn_initial_slot, o, s);
        } else {
            Tensor ssm_state = state_.ssm_slot(static_cast<std::uint32_t>(gidx), 0);
            ops::gated_delta_rule(qn, kn, vv, g, beta, kGdnScale, work_, ssm_state, o, s);
        }

        Tensor on = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1});
        ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

        ops::linear_add(on.view({kCfg.value_dim, 1}), *w.out_proj, x, work_, s);
        return;
    }

    Tensor qkv = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
    ops::gdn_input_proj(h, *w.in_qk_q4, *w.in_v, qkv, work_, s);

    Tensor qkv_c = work_.alloc(DType::BF16, {kCfg.conv_dim, T});
    if (ph == Phase::Verify) {
        Tensor& conv_states = state_.conv.at(static_cast<std::size_t>(gidx));
        ops::causal_conv1d_silu_snapshot(qkv, *w.conv1d, conv_states, io_.gdn_initial_slot, qkv_c,
                                         s);
    } else {
        // Prefill reads the committed conv window from gdn_prefill_read_slot_ and writes the
        // running window to slot 0 (in-place when the read slot is 0).
        Tensor conv_in = state_.conv_slot(static_cast<std::uint32_t>(gidx), gdn_prefill_read_slot_);
        Tensor conv_out = state_.conv_slot(static_cast<std::uint32_t>(gidx), 0);
        ops::causal_conv1d_silu(qkv, *w.conv1d, conv_in, conv_out, qkv_c, s);
    }

    Tensor g    = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    Tensor beta = work_.alloc(DType::FP32, {kCfg.gdn_v_heads, T});
    ops::gdn_gating_proj(h, *w.in_a, *w.in_b, *w.a_log, *w.dt_bias, work_, g, beta, s);

    Tensor qc = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor kc = work_.alloc(DType::BF16, {kCfg.key_dim, T});
    Tensor vc = work_.alloc(DType::BF16, {kCfg.value_dim, T});
    ops::extract_bf16_columns(qkv_c, 0, qc, s);
    ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
    ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);

    Tensor qn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor kn = work_.alloc(DType::BF16, {kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    ops::l2norm(qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T}), 1.0e-6f, qn, s);
    ops::l2norm(kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T}), 1.0e-6f, kn, s);

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor& ssm_states = state_.ssm.at(static_cast<std::size_t>(gidx));
        ops::gated_delta_rule_snapshot(qn, kn, vv, g, beta, kGdnScale, work_, ssm_states,
                                       io_.gdn_initial_slot, o, s);
    } else {
        // Prefill reads the committed recurrent state from gdn_prefill_read_slot_ and writes the
        // running state to slot 0 (in-place when the read slot is 0).
        Tensor ssm_in  = state_.ssm_slot(static_cast<std::uint32_t>(gidx), gdn_prefill_read_slot_);
        Tensor ssm_out = state_.ssm_slot(static_cast<std::uint32_t>(gidx), 0);
        ops::gated_delta_rule(qn, kn, vv, g, beta, kGdnScale, work_, ssm_in, ssm_out, o, s);
    }

    Tensor z      = work_.alloc(DType::BF16, {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor z_flat = z.view({kCfg.value_dim, T});
    ops::linear(h, *w.in_z, z_flat, work_, s);

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

    Tensor a = work_.alloc(DType::BF16, {kCfg.intermediate, T});
    ops::linear_swiglu(h, *m.gate_up, a, work_, s);
    ops::linear_add(a, *m.down, x, work_, s);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
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
    cudaStream_t s       = ctx_.stream;
    mtp_prompt_prepared_ = false;
    const int T          = static_cast<int>(ids.size());
    const int chunk      = static_cast<int>(prefill_chunk_);

    if (multimodal != nullptr) {
        if (kv_.pos != 0) {
            throw std::invalid_argument("multimodal prefill must start from an empty cache");
        }
        if (multimodal->positions.size() != static_cast<std::size_t>(3) * ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->embeddings == nullptr || multimodal->embeddings->dtype != DType::BF16 ||
            multimodal->embeddings->ne[0] != kCfg.hidden ||
            multimodal->embeddings->ne[1] !=
                static_cast<std::int32_t>(multimodal->scatter_indices.size()) ||
            multimodal->embeddings->ne[2] != 1 || multimodal->embeddings->ne[3] != 1 ||
            !multimodal->embeddings->is_contiguous() || multimodal->embeddings->data == nullptr) {
            throw std::invalid_argument("multimodal embeddings must be contiguous BF16 [5120,V]");
        }
        if (!std::is_sorted(multimodal->scatter_indices.begin(),
                            multimodal->scatter_indices.end()) ||
            std::adjacent_find(multimodal->scatter_indices.begin(),
                               multimodal->scatter_indices.end()) !=
                multimodal->scatter_indices.end() ||
            (!multimodal->scatter_indices.empty() && (multimodal->scatter_indices.front() < 0 ||
                                                      multimodal->scatter_indices.back() >= T))) {
            throw std::invalid_argument(
                "multimodal scatter indices must be unique, sorted, and inside prompt");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (kv_.pos == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    const std::uint32_t base = kv_.pos;
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
    if (gdn_read_slot0 < 0 || gdn_read_slot0 >= state_.snapshot_slots) { gdn_read_slot0 = 0; }

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
    mtp_prompt_prepared_ = prepare_mtp_prompt;

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
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = work_.alloc(DType::I32, {len});
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = work_.alloc(DType::I32, {len, 3});
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src =
                        multimodal->positions.data() + static_cast<std::size_t>(axis) * T + t0;
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

            Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::embedding(ids_device, *embed_, x, s);
            if (multimodal != nullptr && !multimodal->scatter_indices.empty()) {
                const auto begin = std::lower_bound(multimodal->scatter_indices.begin(),
                                                    multimodal->scatter_indices.end(), t0);
                const auto end =
                    std::lower_bound(begin, multimodal->scatter_indices.end(), t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                if (count > 0) {
                    const auto visual_begin =
                        static_cast<std::int32_t>(begin - multimodal->scatter_indices.begin());
                    std::vector<std::int32_t> local_indices(static_cast<std::size_t>(count));
                    for (std::int32_t i = 0; i < count; ++i) {
                        local_indices[static_cast<std::size_t>(i)] = begin[i] - t0;
                    }
                    Tensor indices_device = work_.alloc(DType::I32, {count});
                    copy_i32(local_indices.data(), indices_device, s);
                    Tensor embeddings = multimodal->embeddings->slice(1, visual_begin, count);
                    ops::scatter(embeddings, indices_device, x, s);
                }
            }
            if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Prefill, x, s); }
            run_layers(x, Phase::Prefill, tap);

            Tensor xf = io_.prefill_hidden.data != nullptr
                            ? matrix_window(io_.prefill_hidden, len)
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
                                ops::kSamplePurposePrefill, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
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
                copy_i32(mtp_ids_host.data(), mtp_ids, s);
                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    detail::scatter_shifted_visual_embeddings(
                        mtp_input_embeddings, *multimodal->embeddings, multimodal->scatter_indices,
                        t0 + 1, T, work_, s);
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.drafts.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, true, &io_.mtp_ar_hidden, &logits, &draft0);

                    ops::set_i32_scalar(io_.ar_pos, base_i + T, s);
                    for (int i = 1; i < io_.drafts.ne[0]; ++i) {
                        Tensor prev_token  = io_.drafts.slice(0, i - 1, 1);
                        Tensor next_token  = io_.drafts.slice(0, i, 1);
                        Tensor next_hidden = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        mtp_forward_ar_step(prev_token, io_.mtp_ar_hidden, io_.ar_pos, next_hidden,
                                            logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp_ar_hidden.data, next_hidden.data,
                                                   io_.mtp_ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(io_.ar_pos, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, false, nullptr, nullptr, nullptr);
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

void TextContext::prefill(const ProcessedInput& input, const Tensor& visual_embeddings) {
    const detail::VisionControl control = detail::build_vision_control(input);
    const MultimodalPrefill multimodal{input.positions, control.scatter_indices, &visual_embeddings,
                                       input.rope_delta};
    NullTap tap;
    prefill_impl(input.input_ids, &multimodal, tap);
}

void TextContext::diagnostic_prefill(const ProcessedInput& input, const Tensor& visual_embeddings,
                                     void* context, TextTapCallback callback) {
    if (callback == nullptr) { throw std::invalid_argument("diagnostic prefill callback is null"); }
    const detail::VisionControl control = detail::build_vision_control(input);
    const MultimodalPrefill multimodal{input.positions, control.scatter_indices, &visual_embeddings,
                                       input.rope_delta};
    CallbackTap tap{context, callback};
    prefill_impl(input.input_ids, &multimodal, tap);
}


} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule
