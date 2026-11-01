// Translation-unit fragment of src/models/qwen3_5/execution/text.cpp, included once at its end
// (after its closing namespace): TextContext at tensor-parallel width 2 apart from the Text layers,
// which the Qwen4 forward runs on both ranks (execution/qwen4_text.cpp, qwen4_forward.h): the
// binding checks, the rank accessors, the rank-pair entry points that hand over to it, the
// vocabulary-split logits and the Qwen3.5 MTP head over both ranks. It shares text.cpp's private
// helpers (require_tensor_shape, ScopedValue, ...) without copying them. Not a header. Our file;
// see docs/maintainer/upstream-merge.md.

#include "core/device_scope.h"
#include "models/qwen3_5/execution/tp2/split.h"

#include <array>

namespace ninfer::models::qwen3_5::execution {
namespace {

// Issues `body(rank)` for rank 0 then rank 1 with that rank's device current. The calls only
// enqueue, so the two ranks' work overlaps on the devices.
template <class Body>
void for_each_rank(const ExecutionContext& execution, Body&& body) {
    const ScopedCurrentDevice restore;
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        ScopedCurrentDevice::select(execution.dev[static_cast<std::size_t>(rank)]->device);
        body(rank);
    }
}

} // namespace

void TextContext::validate_tensor_parallel() {
    const TpExecution& tp = *tp_;
    if (!tp.complete() || tp.execution->tp != kTensorParallelWidth || !tp.execution->dev[0] ||
        !tp.execution->dev[1] || !tp.events->live()) {
        throw std::invalid_argument("tensor-parallel TextContext binding is incomplete");
    }
    if (tp.execution->dev[0]->device != ctx_.device ||
        tp.execution->dev[0]->stream != ctx_.stream) {
        throw std::invalid_argument(
            "tensor-parallel TextContext must execute on rank 0 of its ExecutionContext");
    }
    if (&tp.parameters->model != &parameters_.model || parameters_.device != 0 ||
        tp.parameters->device != 1 || parameters_.model.device_count() != kTensorParallelWidth) {
        throw std::invalid_argument(
            "tensor-parallel TextContext requires the rank 0 and rank 1 Parameters of one "
            "two-device Model");
    }
    if (batch_text_kv_ == nullptr) {
        throw std::invalid_argument("tensor-parallel TextContext requires the text KV caches");
    }
    if (tp.linear_attention->slot_count() != state_.slot_count()) {
        throw std::invalid_argument(
            "tensor-parallel Linear Attention state pools disagree on their slots");
    }
    validate_tensor_parallel_qwen4();
}

void TextContext::validate_tensor_parallel_mtp() {
    const TpExecution& tp = *tp_;
    if (!tp.mtp_complete() || batch_mtp_kv_ == nullptr || !tp.parameters->mtp ||
        mtp_kv_.valid() != tp.mtp_kv.valid()) {
        throw std::invalid_argument("tensor-parallel MTP binding is incomplete");
    }
    if (!io_.mtp || tp.mtp->ar_hidden.ne[0] != io_.mtp->ar_hidden.ne[0] ||
        tp.prefill_hidden.ne[0] != dimension(config_.hidden_size)) {
        throw std::invalid_argument("tensor-parallel MTP frames do not match rank 0's");
    }
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        const MtpParameters& mtp         = rank_mtp(rank);
        const PagedKVBatchLayerView view = rank_mtp_cache(rank).batch_layer_view(0);
        if (view.storage != KvCacheStorage::BFloat16 &&
            view.storage != KvCacheStorage::Int8Group64 &&
            view.storage != KvCacheStorage::Nvfp4Group16) {
            throw std::invalid_argument("tensor-parallel MTP attention supports only BF16, "
                                        "INT8-G64 and NVFP4-G16 KV caches");
        }
        if (view.num_kv_heads != dimension(shard_config_->attention->num_key_value_heads) ||
            view.head_dim != dimension(shard_config_->attention->head_dim)) {
            throw std::invalid_argument(
                "tensor-parallel MTP KV cache does not hold one rank's KV heads");
        }
        const auto packed_rows = static_cast<std::uint64_t>(mtp.projection.packed.weight.n);
        if (packed_rows != 2 * shard_config_->attention->query_width() +
                               2 * shard_config_->attention->key_width()) {
            throw std::invalid_argument(
                "tensor-parallel MTP attention projection is not one rank's Q|K|Gate|V shard");
        }
    }
}

void TextContext::require_single_device(const char* operation) const {
    if (tp_ != nullptr) {
        throw std::invalid_argument(std::string(operation) +
                                    " is not implemented at tensor-parallel width 2");
    }
}

const Parameters& TextContext::rank_parameters(int rank) const noexcept {
    return rank == 0 ? parameters_ : *tp_->parameters;
}

std::array<WorkspaceArena*, kTensorParallelWidth> TextContext::workspaces() const noexcept {
    return {&work_, tp_->work};
}

void TextContext::peer_tap_begin(const Tensor& x1) {
    if (peer_feature_sink_ != nullptr) { peer_feature_sink_->begin(x1); }
}

void TextContext::peer_tap_capture_layer(int layer, const Tensor& x1) {
    if (peer_feature_sink_ == nullptr) { return; }
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    peer_feature_sink_->capture_layer(layer, x1, rank_stream(1));
}

void TextContext::peer_tap_capture_layer_columns(int layer, const Tensor& x1, std::int32_t first) {
    if (peer_feature_sink_ == nullptr) { return; }
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    peer_feature_sink_->capture_layer_columns(layer, x1, first, rank_stream(1));
}

void TextContext::peer_tap_capture_positions(const Tensor& positions1) {
    if (peer_feature_sink_ == nullptr) { return; }
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    peer_feature_sink_->capture_positions(positions1, rank_stream(1));
}

void TextContext::peer_tap_consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (peer_feature_sink_ == nullptr) { return; }
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    tp_->work->reset();
    peer_feature_sink_->consume_prefill_chunk(tokens, rewrite_checkpoint);
}

cudaStream_t TextContext::rank_stream(int rank) const noexcept {
    return rank == 0 ? ctx_.stream : tp_->execution->dev[1]->stream;
}

DeviceExecutionView TextContext::rank_execution_view(int rank) const noexcept {
    return rank == 0 ? ctx_.execution_view() : tp_->execution->dev[1]->execution_view();
}

LinearAttentionStatePool& TextContext::rank_state(int rank) const noexcept {
    return rank == 0 ? state_ : *tp_->linear_attention;
}

const qwen3_5::PagedKVCache& TextContext::rank_text_cache(int rank) const {
    const qwen3_5::PagedKVCache* cache = rank == 0 ? batch_text_kv_ : tp_->text_cache;
    if (cache == nullptr) { throw std::logic_error("tensor-parallel text KV cache is unbound"); }
    return *cache;
}

Tensor TextContext::rank_valid_columns(int rank) const {
    if ((active_valid_columns_ == nullptr) != (peer_valid_columns_ == nullptr)) {
        throw std::logic_error("tensor-parallel valid columns are bound on one rank only");
    }
    const Tensor* valid = rank == 0 ? active_valid_columns_ : peer_valid_columns_;
    return valid != nullptr ? *valid : Tensor{};
}

const Tensor& TextContext::rank_backend_kv_table_rows(int rank) const {
    if ((active_backend_kv_table_rows_ == nullptr) != (peer_backend_kv_table_rows_ == nullptr)) {
        throw std::logic_error("tensor-parallel MTP KV rows are bound on one rank only");
    }
    if (rank == 0) {
        return active_backend_kv_table_rows_ != nullptr ? *active_backend_kv_table_rows_
                                                        : io_.backend_kv_table_row;
    }
    return peer_backend_kv_table_rows_ != nullptr ? *peer_backend_kv_table_rows_
                                                  : tp_->backend_kv_table_row;
}

const qwen3_5::PagedKVCache& TextContext::rank_mtp_cache(int rank) const {
    const qwen3_5::PagedKVCache* cache = rank == 0 ? batch_mtp_kv_ : tp_->mtp_cache;
    if (cache == nullptr) { throw std::logic_error("tensor-parallel MTP KV cache is unbound"); }
    return *cache;
}

qwen3_5::PagedKVCacheView TextContext::rank_mtp_kv(int rank) const {
    const qwen3_5::PagedKVCacheView view = rank == 0 ? mtp_kv_ : tp_->mtp_kv;
    if (!view.valid()) { throw std::logic_error("tensor-parallel MTP prefill KV row is unbound"); }
    return view;
}

const MtpParameters& TextContext::rank_mtp(int rank) const {
    const auto& mtp = rank_parameters(rank).mtp;
    if (!mtp) { throw std::logic_error("tensor-parallel MTP parameters are unavailable"); }
    return *mtp;
}

const GdnReplayRecords& TextContext::rank_replay_records(int rank) const {
    const GdnReplayRecords* records = rank == 0 ? replay_records_ : tp_->replay_records;
    if (records == nullptr) { throw std::logic_error("Replay-record GDN has no record storage"); }
    return *records;
}

void TextContext::logits_tp2(const RankTensors& hidden, Tensor& logits) {
    logits_tp2(hidden, {&parameters_.text.output_head, &tp_->parameters->text.output_head}, logits);
}

void TextContext::logits_tp2(const RankTensors& hidden,
                             const std::array<const LinearParameters*, 2>& head, Tensor& logits) {
    const auto ws              = workspaces();
    const std::int32_t columns = hidden[0].ne[1];
    const std::int32_t rows0   = head[0]->weight.n;
    const std::int32_t rows1   = head[1]->weight.n;
    auto scope0                = work_.scope();
    auto scope1                = tp_->work->scope();
    const auto part0           = workspace::tp_logits(*ws[0], rows0, rows1, columns, true);
    const auto part1           = workspace::tp_logits(*ws[1], rows1, rows0, columns, false);
    output_logits_split_rank0(hidden, head, {part0.partial, part1.partial}, logits, part0.staging,
                              ws, *tp_->execution, *tp_->events);
}

// The Qwen3.5 MTP head's prompt alignment of one prefill chunk over both ranks (as
// mtp_prompt_chunk at one device): `mtp_input` is each rank's final-normed chunk.
void TextContext::mtp_prompt_chunk_tp2(std::span<const int> alignment_ids, std::uint32_t base,
                                       int len, int T, int base_i, bool is_last,
                                       const RankTensors& mtp_input, const RankTensors& positions,
                                       const RankTensors& rope_positions,
                                       ops::CausalAttentionExecutionEnvelope chunk_envelope,
                                       bool multimodal, const VisionChunk& vision_chunk) {
    const ExecutionContext& execution = *tp_->execution;
    const auto ws                     = workspaces();
    cudaStream_t s                    = ctx_.stream;
    // The MTP head consumes the shifted token stream against the text model's own final
    // hidden, one column behind the target, exactly as on one device. Only rank 0 needs
    // the shifted ids: its input projection half contracts the token embedding.
    const auto alignment_tokens = static_cast<std::uint32_t>(alignment_ids.size());
    const qwen3_5::MtpAlignmentWindow mtp_window =
        qwen3_5::plan_mtp_alignment_window(alignment_tokens, base, static_cast<std::uint32_t>(len));
    const int prompt_columns = len - static_cast<int>(mtp_window.final_column_uses_generated_token);
    Tensor mtp_ids           = work_.alloc(DType::I32, {len});
    if (prompt_columns != 0) {
        Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
        copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin, prompt_mtp_ids, s);
    }
    if (mtp_window.final_column_uses_generated_token) {
        Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
        CUDA_CHECK(::ninfer::memcpy_device_async(generated_mtp_id.data, io_.token.data,
                                                 sizeof(std::int32_t), s));
    }
    // A multimodal prompt's shifted stream carries visual columns: rank 0 embeds the
    // shifted ids and scatters the item's merged embeddings over the shifted visual
    // overlap, as on one device. Rank 1's stem half contracts the hidden alone.
    Tensor mtp_input_embeddings;
    const Tensor* mtp_input_embeddings_ptr = nullptr;
    if (multimodal) {
        mtp_input_embeddings = work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
        ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
        if (vision_chunk.control != nullptr) {
            const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
            if (!overlap.empty()) {
                Tensor shifted_indices = workspace::visual_scatter_indices(
                    work_, static_cast<std::int32_t>(overlap.size()));
                qwen3_5::detail::scatter_shifted_visual_embeddings(
                    mtp_input_embeddings, vision_chunk.embeddings, overlap, shifted_indices, s);
            }
        }
        mtp_input_embeddings_ptr = &mtp_input_embeddings;
    }
    if (is_last && mtp_proposal_extent_ != 0) {
        Tensor logits = matrix_window(io_.logits, 1);
        Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
        const RankTensors ar_hidden{io_.mtp->ar_hidden, tp_->mtp->ar_hidden};
        mtp_prefill_chunk_tp2(mtp_ids, mtp_input, mtp_input_embeddings_ptr, positions,
                              rope_positions, chunk_envelope, true, &ar_hidden, &logits, &draft0);

        const RankTensors ar_position{io_.mtp->position.slice(0, 0, 1),
                                      tp_->mtp->position.slice(0, 0, 1)};
        for_each_rank(execution, [&](int rank) {
            Tensor position = ar_position[static_cast<std::size_t>(rank)];
            ops::set_i32_scalar(position, base_i + T, rank_stream(rank));
        });
        for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
            auto step_scope0      = work_.scope();
            auto step_scope1      = tp_->work->scope();
            Tensor previous_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
            Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
            RankTensors next_hidden;
            for (std::size_t r = 0; r < 2; ++r) {
                next_hidden[r] = ws[r]->alloc(DType::BF16, {dimension(config_.hidden_size), 1});
            }
            const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
            const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible, ar_visible};
            mtp_forward_ar_step(previous_token, ar_hidden, ar_position, ar_envelope, next_hidden,
                                logits, next_token);
            for_each_rank(execution, [&](int rank) {
                const auto r        = static_cast<std::size_t>(rank);
                cudaStream_t stream = rank_stream(rank);
                CUDA_CHECK(::ninfer::memcpy_device_async(ar_hidden[r].data, next_hidden[r].data,
                                                         ar_hidden[r].bytes(), stream));
                Tensor position = ar_position[r];
                ops::increment_i32_scalar(position, stream);
            });
        }
    } else {
        mtp_prefill_chunk_tp2(mtp_ids, mtp_input, mtp_input_embeddings_ptr, positions,
                              rope_positions, chunk_envelope, false, nullptr, nullptr, nullptr);
    }
}

// =================================================================================================
// Tensor-parallel (tp == 2) speculative verification and MTP
// =================================================================================================
//
// Verification runs the Text forward (qwen4_text.cpp) on both ranks over K+1 columns per row, with
// masked columns and ReplaySSM records on each rank. The Qwen3.5 MTP head is one decoder layer, so
// its split form is a text layer's with the MTP's operands:
//
//   rank 0: e = rmsnorm(embed(ids), embedding_norm)   rank 1: h = rmsnorm(hidden, hidden_norm)
//   x         = all_reduce(fc_0 e + fc_1 h)          row-parallel over the packed [e; h] input
//   q|k|g|v_r = split(packed_r rmsnorm(x))           column-parallel, rank r's heads
//   a_r       = gated attention over rank r's MTP KV heads and pages
//   x        += all_reduce(o_r a_r)                  row-parallel, then the residual add
//   x        += all_reduce(down_r swiglu(gate|up_r)) column- then row-parallel
//   out       = rmsnorm(x, final_norm)               replicated
//
// Rank 0's packed-input half is the normalized embedding and rank 1's the normalized hidden, so
// neither rank forms the packed [2 * hidden, T] input and rank 1 never embeds a token. Each
// all-reduce leaves the identical BF16 residual on both ranks, which keeps both ranks' MTP KV pages
// and the next call's inputs in agreement.

void TextContext::target_verify_batch(
    const RankTensors& ids, const RankTensors& cache_positions, const RankTensors& rope_positions,
    const RankTensors& valid_columns, const RankTensors& kv_table_rows,
    const RankTensors& linear_state_source_slots, ops::CausalAttentionExecutionEnvelope envelope,
    const RankTensors& hidden, Tensor& logits, Tensor& target_tokens) {
    qwen4_target_verify(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                            linear_state_source_slots, envelope, hidden, logits, target_tokens);
}

void TextContext::target_verify_batch(
    const RankTensors& ids, const RankTensors& cache_positions, const RankTensors& rope_positions,
    const RankTensors& valid_columns, const RankTensors& kv_table_rows,
    const RankTensors& linear_state_source_slots, ops::CausalAttentionExecutionEnvelope envelope,
    const RankTensors& hidden, Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink) {
    qwen4_target_verify(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                            linear_state_source_slots, envelope, hidden, logits, target_tokens,
                            &sink);
}

void TextContext::mtp_forward_stem_tp2(const Tensor& ids, const RankTensors& hidden,
                                       const Tensor* input_embeddings, RankTensors& x,
                                       RankTensors& ah, const RankTensors& staging) {
    const ExecutionContext& execution = *tp_->execution;
    const auto ws                     = workspaces();
    const std::int32_t H              = dimension(config_.hidden_size);
    const std::int32_t T              = ids.ne[0] * ids.ne[1];
    // The aggregate column count is what must match: the prompt and proposal calls pass [H,T],
    // the decode batch [H,W,B].
    for (std::size_t r = 0; r < 2; ++r) {
        if (hidden[r].dtype != DType::BF16 || !hidden[r].is_contiguous() ||
            hidden[r].data == nullptr || hidden[r].ne[0] != H ||
            hidden[r].numel() != static_cast<std::int64_t>(H) * T) {
            throw std::invalid_argument(
                "tensor-parallel MTP stem hidden does not match the token count");
        }
    }
    // Only rank 0 embeds tokens, or takes their composed embeddings (a multimodal prompt's, with
    // its visual columns scattered in); the plan sizes both ranks for rank 0's roots.
    if (input_embeddings != nullptr &&
        (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != H ||
         input_embeddings->numel() != static_cast<std::int64_t>(H) * T ||
         !input_embeddings->is_contiguous() || input_embeddings->data == nullptr)) {
        throw std::invalid_argument("tensor-parallel MTP input embeddings shape mismatch");
    }
    std::array<workspace::MtpStemSplitRoots, 2> roots{
        workspace::mtp_stem_split(*ws[0], config_, T, input_embeddings == nullptr),
        workspace::mtp_stem_split(*ws[1], config_, T, false)};
    for (std::size_t r = 0; r < 2; ++r) {
        x[r]  = roots[r].residual;
        ah[r] = roots[r].attention_hidden;
    }
    const Tensor flat_ids = ids.view({T});
    for_each_rank(execution, [&](int rank) {
        cudaStream_t stream      = rank_stream(rank);
        const MtpParameters& mtp = rank_mtp(rank);
        if (rank == 0) {
            Tensor embedding;
            if (input_embeddings != nullptr) {
                embedding = input_embeddings->view({H, T});
            } else {
                embedding = roots[0].embedding;
                ops::embedding(flat_ids, *embed_, embedding, stream);
            }
            ops::rmsnorm(embedding, mtp.embedding_norm, config_.rms_norm_eps, true,
                         roots[0].normalized_input, stream);
        } else {
            ops::rmsnorm(hidden[1].view({H, T}), mtp.hidden_norm, config_.rms_norm_eps, true,
                         roots[1].normalized_input, stream);
        }
    });
    mtp_input_projection_split({roots[0].normalized_input, roots[1].normalized_input},
                               {&rank_mtp(0), &rank_mtp(1)}, x, staging, ws, execution,
                               *tp_->events);
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], rank_mtp(rank).input_norm, config_.rms_norm_eps, true, ah[r],
                     rank_stream(rank));
    });
}

void TextContext::mtp_forward_tail_tp2(RankTensors& x, const RankTensors& ah,
                                       const RankTensors& positions,
                                       const RankTensors& rope_positions,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       const RankTensors& mtp_hidden, const RankTensors& staging) {
    const ExecutionContext& execution = *tp_->execution;
    const AttentionConfig& shard      = *shard_config_->attention;
    const auto ws                     = workspaces();
    const std::int32_t T              = x[0].ne[1];
    const std::int32_t H              = dimension(config_.hidden_size);
    const std::int32_t head_dim       = dimension(shard.head_dim);
    const std::int32_t q_heads        = dimension(shard.num_attention_heads);
    const std::int32_t kv_heads       = dimension(shard.num_key_value_heads);
    const std::int32_t q_width        = dimension(shard.query_width());

    RankTensors q_flat;
    RankTensors gate_flat;
    RankTensors k_flat;
    RankTensors v_flat;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto projection = workspace::mtp_attention_projection(*ws[r], *shard_config_, T);
        q_flat[r]             = projection.query;
        gate_flat[r]          = projection.gate;
        k_flat[r]             = projection.key;
        v_flat[r]             = projection.value;
    }
    mtp_projection_split(ah, {&rank_mtp(0).projection, &rank_mtp(1).projection}, shard, q_flat,
                         gate_flat, k_flat, v_flat, ws, execution);

    RankTensors qn;
    RankTensors kn;
    RankTensors a;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto results = workspace::mtp_attention_results(*ws[r], *shard_config_, T);
        qn[r]              = results.normalized_query.view({head_dim, q_heads, T});
        kn[r]              = results.normalized_key.view({head_dim, kv_heads, T});
        a[r]               = results.attention.view({head_dim, q_heads, T});
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    for_each_rank(execution, [&](int rank) {
        const auto r             = static_cast<std::size_t>(rank);
        cudaStream_t stream      = rank_stream(rank);
        const MtpParameters& mtp = rank_mtp(rank);
        Tensor q                 = q_flat[r].view({head_dim, q_heads, T});
        Tensor k                 = k_flat[r].view({head_dim, kv_heads, T});
        Tensor v                 = v_flat[r].view({head_dim, kv_heads, T});
        Tensor gate              = gate_flat[r].view({head_dim, q_heads, T});
        Tensor rope_for_op =
            active_sequence_batch_ != 0 ? rope_positions[r].view({T}) : rope_positions[r];
        text_qk_norm_rope(rope_for_op, *config_.rope_parameters, mtp.query_norm, mtp.key_norm,
                          config_.rms_norm_eps, q, k, qn[r], kn[r], stream);

        const PagedKVBatchLayerView cache = rank_mtp_cache(rank).batch_layer_view(0);
        const Tensor& rows                = rank_backend_kv_table_rows(rank);
        if (active_sequence_batch_ != 0) {
            const std::int32_t width = active_sequence_width_;
            const std::int32_t batch = active_sequence_batch_;
            if (width <= 0 || width * batch != T) {
                throw std::logic_error("MTP sequence batch binding is incomplete");
            }
            Tensor q_batch        = qn[r].view({head_dim, q_heads, width, batch});
            Tensor k_batch        = kn[r].view({head_dim, kv_heads, width, batch});
            Tensor v_batch        = v.view({head_dim, kv_heads, width, batch});
            Tensor a_batch        = a[r].view({head_dim, q_heads, width, batch});
            Tensor position_batch = positions[r].view({width, batch});
            ops::causal_softmax_attention(
                q_batch, k_batch, v_batch, position_batch, rank_valid_columns(rank), rows,
                {head_dim, q_heads, kv_heads}, scale, cache, envelope, *ws[r], a_batch, stream);
        } else {
            ops::causal_softmax_attention(qn[r], kn[r], v, positions[r], Tensor{}, rows,
                                          {head_dim, q_heads, kv_heads}, scale, cache, envelope,
                                          *ws[r], a[r], stream);
        }
        ops::sigmoid_mul(gate, a[r], stream);
    });

    // The MTP output projection is a plain linear plus the residual add, as on one device.
    RankTensors o;
    RankTensors mh;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto post = workspace::mtp_post_attention(*ws[r], config_, T);
        o[r]            = post.output;
        mh[r]           = post.post_mixer_hidden;
    }
    mtp_output_split({a[0].view({q_width, T}), a[1].view({q_width, T})},
                     {&rank_mtp(0), &rank_mtp(1)}, o, staging, ws, execution, *tp_->events);
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t stream = rank_stream(rank);
        ops::residual_add(o[r], x[r], stream);
        ops::rmsnorm(x[r], rank_mtp(rank).post_attention_norm, config_.rms_norm_eps, true, mh[r],
                     stream);
    });
    {
        auto post_mixer_scope0 = work_.scope();
        auto post_mixer_scope1 = tp_->work->scope();
        mtp_ffn_split(mh, {&rank_mtp(0).ffn, &rank_mtp(1).ffn}, x, staging, ws, execution,
                      *tp_->events);
    }
    for_each_rank(execution, [&](int rank) {
        const auto r           = static_cast<std::size_t>(rank);
        Tensor flat_mtp_hidden = mtp_hidden[r].view({H, T});
        ops::rmsnorm(x[r], rank_mtp(rank).final_norm, config_.rms_norm_eps, true, flat_mtp_hidden,
                     rank_stream(rank));
    });
}

void TextContext::mtp_forward_core_tp2(const Tensor& ids, const RankTensors& hidden,
                                       const RankTensors& positions,
                                       const RankTensors& rope_positions,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       const RankTensors& mtp_hidden,
                                       const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr || tp_->mtp_cache == nullptr) {
        throw std::runtime_error("MTP forward is not enabled");
    }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    const auto ws        = workspaces();
    auto scratch_scope0  = work_.scope();
    auto scratch_scope1  = tp_->work->scope();
    const std::int32_t T = ids.ne[0] * ids.ne[1];
    // One staging per rank serves the three all-reduces, each of which leaves it free on return.
    RankTensors staging;
    for (std::size_t r = 0; r < 2; ++r) {
        staging[r] = workspace::tp_call_roots(*ws[r], config_, T).staging;
    }
    RankTensors x;
    RankTensors ah;
    mtp_forward_stem_tp2(ids, hidden, input_embeddings, x, ah, staging);
    mtp_forward_tail_tp2(x, ah, positions, rope_positions, envelope, mtp_hidden, staging);
}

void TextContext::mtp_prefill_chunk_tp2(const Tensor& ids, const RankTensors& hidden,
                                        const Tensor* input_embeddings,
                                        const RankTensors& positions,
                                        const RankTensors& rope_positions,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        bool final_chunk, const RankTensors* final_hidden,
                                        Tensor* logits, Tensor* draft_token) {
    if (!mtp_kv_.valid() || !tp_->mtp_kv.valid()) {
        throw std::runtime_error("MTP prefill is not enabled");
    }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    const std::int32_t H = dimension(config_.hidden_size);
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {H, T}, "MTP prefill hidden");
        require_tensor_shape(positions[r], DType::I32, {T}, "MTP prefill positions");
        if (rope_positions[r].dtype != DType::I32 || rope_positions[r].ne[0] != T ||
            rope_positions[r].ne[1] != rope_positions[0].ne[1] ||
            (rope_positions[r].ne[1] != 1 && rope_positions[r].ne[1] != 3) ||
            rope_positions[r].ne[2] != 1 || rope_positions[r].ne[3] != 1 ||
            !rope_positions[r].is_contiguous() || rope_positions[r].data == nullptr) {
            throw std::invalid_argument(
                "tensor-parallel MTP prefill rope positions must be [T] or [T,3] on both ranks");
        }
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        for (std::size_t r = 0; r < 2; ++r) {
            require_tensor_shape((*final_hidden)[r], DType::BF16, {H, 1},
                                 "MTP final prefill hidden");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    const ExecutionContext& execution = *tp_->execution;
    const AttentionConfig& shard      = *shard_config_->attention;
    const auto ws                     = workspaces();
    const std::int32_t head_dim       = dimension(shard.head_dim);
    const std::int32_t q_heads        = dimension(shard.num_attention_heads);
    const std::int32_t kv_heads       = dimension(shard.num_key_value_heads);
    const std::array<const MtpProjectionParameters*, 2> projection{&rank_mtp(0).projection,
                                                                   &rank_mtp(1).projection};
    auto scratch_scope0 = work_.scope();
    auto scratch_scope1 = tp_->work->scope();
    RankTensors staging;
    RankTensors last_staging;
    RankTensors x_last;
    RankTensors ah_last;
    for (std::size_t r = 0; r < 2; ++r) {
        staging[r] = workspace::tp_call_roots(*ws[r], config_, T).staging;
        if (final_chunk) {
            // The final column's all-reduces reduce one column; their staging matches it.
            last_staging[r] = ws[r]->alloc(DType::BF16, {H, 1});
            x_last[r]       = ws[r]->alloc(DType::BF16, {H, 1});
            ah_last[r]      = ws[r]->alloc(DType::BF16, {H, 1});
        }
    }

    {
        auto bulk_scope0 = work_.scope();
        auto bulk_scope1 = tp_->work->scope();
        RankTensors x;
        RankTensors ah;
        mtp_forward_stem_tp2(ids, hidden, input_embeddings, x, ah, staging);

        RankTensors k_flat;
        RankTensors v_flat;
        RankTensors kn;
        for (std::size_t r = 0; r < 2; ++r) {
            k_flat[r] = ws[r]->alloc(DType::BF16, {dimension(shard.key_width()), T});
            v_flat[r] = ws[r]->alloc(DType::BF16, {dimension(shard.key_width()), T});
        }
        mtp_kv_projection_split(ah, projection, shard, k_flat, v_flat, ws, execution);
        for (std::size_t r = 0; r < 2; ++r) {
            kn[r] = ws[r]->alloc(DType::BF16, {head_dim, kv_heads, T});
        }
        const std::size_t column_bytes = static_cast<std::size_t>(H) * dtype_size(DType::BF16);
        for_each_rank(execution, [&](int rank) {
            const auto r        = static_cast<std::size_t>(rank);
            cudaStream_t stream = rank_stream(rank);
            Tensor k            = k_flat[r].view({head_dim, kv_heads, T});
            Tensor v            = v_flat[r].view({head_dim, kv_heads, T});
            ops::rmsnorm(k, rank_mtp(rank).key_norm, config_.rms_norm_eps, true, kn[r], stream);
            text_rope(rope_positions[r], *config_.rope_parameters, kn[r], stream);
            ops::kv_cache_append(kn[r], v, positions[r], rank_mtp_kv(rank).layer_view(0), stream);
            if (final_chunk) {
                const auto* x_src = static_cast<const unsigned char*>(x[r].data) +
                                    static_cast<std::size_t>(T - 1) * column_bytes;
                const auto* ah_src = static_cast<const unsigned char*>(ah[r].data) +
                                     static_cast<std::size_t>(T - 1) * column_bytes;
                CUDA_CHECK(
                    ::ninfer::memcpy_device_async(x_last[r].data, x_src, column_bytes, stream));
                CUDA_CHECK(
                    ::ninfer::memcpy_device_async(ah_last[r].data, ah_src, column_bytes, stream));
            }
        });
    }

    if (!final_chunk) { return; }

    RankTensors q_flat;
    RankTensors gate_flat;
    RankTensors qn;
    RankTensors a;
    RankTensors o;
    RankTensors mh;
    for (std::size_t r = 0; r < 2; ++r) {
        q_flat[r]    = ws[r]->alloc(DType::BF16, {dimension(shard.query_width()), 1});
        gate_flat[r] = ws[r]->alloc(DType::BF16, {dimension(shard.query_width()), 1});
    }
    mtp_query_gate_projection_split(ah_last, projection, shard, q_flat, gate_flat, ws, execution);
    // A multimodal chunk's three-axis positions are axis-major, so the last column's three values
    // are gathered into [1,3], as on one device.
    const bool three_axis_rope = rope_positions[0].ne[1] == 3;
    RankTensors last_rope;
    for (std::size_t r = 0; r < 2; ++r) {
        qn[r]        = ws[r]->alloc(DType::BF16, {head_dim, q_heads, 1});
        a[r]         = ws[r]->alloc(DType::BF16, {head_dim, q_heads, 1});
        o[r]         = ws[r]->alloc(DType::BF16, {H, 1});
        mh[r]        = ws[r]->alloc(DType::BF16, {H, 1});
        last_rope[r] = three_axis_rope ? ws[r]->alloc(DType::I32, {1, 3})
                                       : rope_positions[r].slice(0, T - 1, 1);
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t stream = rank_stream(rank);
        Tensor q            = q_flat[r].view({head_dim, q_heads, 1});
        Tensor gate         = gate_flat[r].view({head_dim, q_heads, 1});
        ops::rmsnorm(q, rank_mtp(rank).query_norm, config_.rms_norm_eps, true, qn[r], stream);
        const Tensor last_position = positions[r].slice(0, T - 1, 1);
        if (three_axis_rope) {
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions[r].data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope[r].data) + axis;
                CUDA_CHECK(::ninfer::memcpy_device_async(dst, src, sizeof(std::int32_t), stream));
            }
        }
        text_rope(last_rope[r], *config_.rope_parameters, qn[r], stream);
        ops::causal_softmax_attention_cached(qn[r], last_position, {head_dim, q_heads, kv_heads},
                                             scale, rank_mtp_kv(rank).layer_view(0), envelope,
                                             *ws[r], a[r], stream);
        ops::sigmoid_mul(gate, a[r], stream);
    });
    const std::int32_t q_width = dimension(shard.query_width());
    mtp_output_split({a[0].view({q_width, 1}), a[1].view({q_width, 1})},
                     {&rank_mtp(0), &rank_mtp(1)}, o, last_staging, ws, execution, *tp_->events);
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t stream = rank_stream(rank);
        ops::residual_add(o[r], x_last[r], stream);
        ops::rmsnorm(x_last[r], rank_mtp(rank).post_attention_norm, config_.rms_norm_eps, true,
                     mh[r], stream);
    });
    {
        auto post_mixer_scope0 = work_.scope();
        auto post_mixer_scope1 = tp_->work->scope();
        mtp_ffn_split(mh, {&rank_mtp(0).ffn, &rank_mtp(1).ffn}, x_last, last_staging, ws, execution,
                      *tp_->events);
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        Tensor out   = (*final_hidden)[r];
        ops::rmsnorm(x_last[r], rank_mtp(rank).final_norm, config_.rms_norm_eps, true, out,
                     rank_stream(rank));
    });
    proposal_argmax_tp2(*final_hidden, *logits, *draft_token);
}

void TextContext::proposal_argmax_tp2(const RankTensors& hidden, Tensor& logits,
                                      Tensor& proposal_tokens) {
    const std::int32_t T = hidden[0].ne[1];
    const std::int32_t H = dimension(config_.hidden_size);
    const std::int32_t V = dimension(config_.vocab_size);
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {H, T}, "proposal hidden");
    }
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, V, T, "proposal logits");
    if (proposal_head_ != nullptr) {
        const std::optional<ProposalParameters>& peer = tp_->parameters->proposal;
        if (!peer) {
            // A PrimaryOnly optimized head: rank 0 projects, picks and maps alone.
            proposal_argmax(hidden[0], logits, proposal_tokens);
            return;
        }
        // Split by vocabulary: each rank projects and picks within its rows, rank 0 combines the
        // two candidates and maps the selected row to its token.
        nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                         static_cast<std::uint64_t>(T));
        const std::int32_t rows = proposal_head_->weight.n;
        if (proposal_head_ids_ == nullptr || peer->head.weight.n != rows ||
            2 * rows != proposal_head_n_) {
            throw std::logic_error("tensor-parallel proposal head split is inconsistent");
        }
        const auto ws = workspaces();
        auto scope0   = work_.scope();
        auto scope1   = tp_->work->scope();
        const std::array<workspace::TensorParallelProposalRoots, 2> roots{
            workspace::tp_proposal_argmax(*ws[0], rows, T),
            workspace::tp_proposal_argmax(*ws[1], rows, T)};
        proposal_argmax_split(
            hidden, {proposal_head_, &peer->head}, {roots[0].partial, roots[1].partial},
            {roots[0].local, roots[1].local}, {roots[0].candidates, roots[1].candidates},
            {roots[0].staging, roots[1].staging}, proposal_tokens, ws, *tp_->execution,
            *tp_->events);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
        return;
    }
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    Tensor output_logits = matrix_window(logits, T);
    // The full head is the vocabulary-split output head: the argmax runs over the gathered
    // complete logits, on rank 0.
    logits_tp2(hidden, {&rank_mtp(0).output_head, &rank_mtp(1).output_head}, output_logits);
    ops::argmax(output_logits, proposal_tokens,
                dimension(parameters_.model.resources().public_token_count), ctx_.stream);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const RankTensors& hidden,
                                           const RankTensors& cache_positions,
                                           const RankTensors& rope_positions,
                                           const RankTensors& valid_columns,
                                           const RankTensors& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           const RankTensors& mtp_hidden) {
    if (!tp2()) { throw std::invalid_argument("rank-pair MTP decode requires tensor parallelism"); }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    if (qwen4_mtp_layer()) {
        qwen4_mtp_forward_decode(ids, hidden, cache_positions, rope_positions, valid_columns,
                                     kv_table_rows, envelope, mtp_hidden);
        return;
    }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    const std::int32_t H = dimension(config_.hidden_size);
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {H, width, batch},
                             "MTP decode batch target hidden");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "MTP decode batch cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "MTP decode batch RoPE positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "MTP decode batch valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch}, "MTP decode batch KV rows");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {H, width, batch},
                             "MTP decode batch hidden");
    }
    const ScopedCurrentDevice device(ctx_.device);
    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows[0]);
    ScopedValue<const Tensor*> peer_backend_binding(peer_backend_kv_table_rows_, &kv_table_rows[1]);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
    ScopedValue<const Tensor*> peer_valid_binding(peer_valid_columns_, &valid_columns[1]);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core_tp2(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden,
                         nullptr);
}

void TextContext::mtp_propose_batch(const RankTensors& hidden, Tensor& logits, Tensor& draft_tokens,
                                    Tensor* draft_logprobs) {
    if (!tp2()) {
        throw std::invalid_argument("rank-pair MTP proposal requires tensor parallelism");
    }
    if (qwen4_mtp_layer()) {
        qwen4_mtp_propose(hidden, logits, draft_tokens, draft_logprobs);
        return;
    }
    if (draft_logprobs != nullptr && proposal_head_ != nullptr) {
        throw std::invalid_argument("MTP draft log-probabilities need the full proposal head");
    }
    const std::int32_t batch = hidden[0].ne[1];
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {dimension(config_.hidden_size), batch},
                             "MTP proposal batch hidden");
    }
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    const ScopedCurrentDevice device(ctx_.device);
    auto proposal_scope0 = work_.scope();
    auto proposal_scope1 = tp_->work->scope();
    proposal_argmax_tp2(hidden, logits, draft_tokens);
    if (draft_logprobs != nullptr) {
        ops::target_logprobs(logits, draft_tokens,
                             dimension(parameters_.model.resources().public_token_count),
                             *draft_logprobs, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const RankTensors& hidden,
                                    const RankTensors& positions, const RankTensors& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    const RankTensors& mtp_hidden, int logits_column,
                                    Tensor* logits, Tensor* draft_token,
                                    const Tensor* input_embeddings) {
    if (!tp2()) { throw std::invalid_argument("rank-pair MTP bridge requires tensor parallelism"); }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    if (qwen4_mtp_layer()) {
        const ScopedCurrentDevice device(ctx_.device);
        qwen4_mtp_forward_batch(ids, hidden, positions, rope_positions, envelope, mtp_hidden,
                                    logits_column, logits, draft_token, input_embeddings);
        return;
    }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    const std::int32_t H = dimension(config_.hidden_size);
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(positions[r], DType::I32, {T}, "MTP positions");
        if (rope_positions[r].ne[1] == 3) {
            require_tensor_shape(rope_positions[r], DType::I32, {T, 3}, "MTP RoPE positions");
        } else {
            require_tensor_shape(rope_positions[r], DType::I32, {T}, "MTP RoPE positions");
        }
        require_tensor_shape(hidden[r], DType::BF16, {H, T}, "MTP hidden");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {H, T}, "MTP output hidden");
    }
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }
    const ScopedCurrentDevice device(ctx_.device);
    // Unbatched: each rank appends and attends through its own prefill MTP row (io_ on rank 0,
    // TpExecution::backend_kv_table_row on rank 1), as the single-device bridge does.
    mtp_forward_core_tp2(ids, hidden, positions, rope_positions, envelope, mtp_hidden,
                         input_embeddings);
    if (logits_column >= 0) {
        auto logits_scope0 = work_.scope();
        auto logits_scope1 = tp_->work->scope();
        const RankTensors column{mtp_hidden[0].slice(1, logits_column, 1),
                                 mtp_hidden[1].slice(1, logits_column, 1)};
        proposal_argmax_tp2(column, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const RankTensors& previous_hidden,
                                      const RankTensors& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      const RankTensors& mtp_hidden, Tensor& logits,
                                      Tensor& draft_token) {
    if (!tp2()) {
        throw std::invalid_argument("rank-pair MTP proposal step requires tensor parallelism");
    }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    if (qwen4_mtp_layer()) {
        // The step's carried hidden is the wide residual; each rank offsets its own position copy
        // by its own copy of the sequence's RoPE delta (as qwen4_mtp_prefill's drafts).
        const ExecutionContext& execution = *tp_->execution;
        const ScopedCurrentDevice device(ctx_.device);
        auto rope_scope0 = work_.scope();
        auto rope_scope1 = tp_->work->scope();
        RankTensors rope;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            require_tensor_shape(position[r], DType::I32, {1}, "MTP AR position");
            const ScopedCurrentDevice current(rank == 0 ? ctx_.device : execution.dev[1]->device);
            rope[r] = (rank == 0 ? work_ : *tp_->work).alloc(DType::I32, {1});
            ops::offset_i32_positions(position[r], rank == 0 ? io_.rope_delta : tp_->rope_delta,
                                      rope[r], rank_stream(rank));
        }
        qwen4_mtp_forward_batch(token, previous_hidden, position, rope, envelope, mtp_hidden, 0,
                                    &logits, &draft_token, nullptr);
        return;
    }
    const std::int32_t H = dimension(config_.hidden_size);
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(position[r], DType::I32, {1}, "MTP AR position");
        require_tensor_shape(previous_hidden[r], DType::BF16, {H, 1}, "MTP AR previous hidden");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {H, 1}, "MTP AR output hidden");
    }
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws        = workspaces();
    auto position_scope0 = work_.scope();
    auto position_scope1 = tp_->work->scope();
    RankTensors rope_position;
    for (std::size_t r = 0; r < 2; ++r) { rope_position[r] = ws[r]->alloc(DType::I32, {1}); }
    // Each rank offsets its own copy of the position by its own copy of the sequence's RoPE delta,
    // which the Program and the prompt chunks keep equal on both ranks.
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::offset_i32_positions(position[r], rank == 0 ? io_.rope_delta : tp_->rope_delta,
                                  rope_position[r], rank_stream(rank));
    });
    mtp_forward_core_tp2(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                         nullptr);
    auto logits_scope0 = work_.scope();
    auto logits_scope1 = tp_->work->scope();
    proposal_argmax_tp2(mtp_hidden, logits, draft_token);
}

} // namespace ninfer::models::qwen3_5::execution
