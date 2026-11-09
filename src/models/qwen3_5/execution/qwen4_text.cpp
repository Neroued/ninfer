// TextContext's Text entry points for every model (the forward is execution/qwen4_forward.h):
// chunked Prefill (with the MTP prompt KV and first proposals), ordinary decode, speculative
// target verification and the Qwen4 MTP layer's decode, bridge and proposal calls. Each is one
// function over the ranks of the forward: one device, or both ranks of tensor-parallel width 2
// (qwen4_width()); the one-device entries that take single tensors validate them and hand them
// over as rank 0's. The round schedule is program/speculative/mtp.cpp.
//
// With hyper-connections every call carries the wide residual (S*H) as its hidden: the output
// head reads it through the Text output mixer, the MTP stem consumes it whole. Without them the
// carried hidden is the final-normed [H] one, which the output head reads directly.
//
// At tp 2 both ranks run the same Qwen4Forward schedule in lockstep: the replicated
// hyper-connection residual, router, indexer and PLE on each rank's own copy of the inputs, the
// rank's QSA/GDN heads and expert partition, and two all-reduces per block. Rank 0 owns sampling
// and the logits (vocabulary-split head gathered to rank 0, logits_tp2); rank 1 reads its own
// copies of the control tensors (prefill: uploaded here; decode: the OrdinaryPeerFrame).
#include "models/qwen3_5/execution/text.h"

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/qwen4_draft_head.h"
#include "models/qwen3_5/execution/qwen4_experts.h"
#include "models/qwen3_5/execution/reasoning_eos_guard.h"
#include "models/qwen3_5/execution/tp.h"
#include "models/qwen3_5/execution/tp2/workspace_split.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/speculative/mtp_alignment.h"

#include "core/device_scope.h"
#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <utility>

namespace ninfer::models::qwen3_5::execution {

namespace {

// The next prefill chunk's PLE rows are read from storage on a helper thread while the GPUs run
// this chunk (T165, from T164: a gather of rows that are not in the page cache otherwise stalls
// both GPUs before every chunk). prefetch() only issues MADV_WILLNEED for the rows and touches no
// shared state, so the next chunk's gather() decodes the same bytes, sooner. Joined on scope exit.
// Diagnostics (T166): NINFER_QWEN4_PLE_TIMING=1 prints the duration of every prefill PLE gather and
// prefetch to stderr; NINFER_QWEN4_PLE_NEXT_PREFETCH=0 turns the next-chunk prefetch off (A/B).
bool ple_timing() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_QWEN4_PLE_TIMING");
        return v != nullptr && *v != 0 && *v != '0';
    }();
    return on;
}

bool ple_next_prefetch_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_QWEN4_PLE_NEXT_PREFETCH");
        return v == nullptr || *v != '0';
    }();
    return on;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

class NextChunkPlePrefetch {
public:
    NextChunkPlePrefetch()                                       = default;
    NextChunkPlePrefetch(const NextChunkPlePrefetch&)            = delete;
    NextChunkPlePrefetch& operator=(const NextChunkPlePrefetch&) = delete;

    ~NextChunkPlePrefetch() { join(); }

    void join() {
        if (thread_.joinable()) { thread_.join(); }
    }

    void start(PleGather& gather, std::span<const std::int32_t> history, std::size_t first,
               std::size_t count) {
        if (!ple_next_prefetch_enabled() || count == 0 || first >= history.size()) { return; }
        count   = std::min(count, history.size() - first);
        thread_ = std::thread([&gather, history, first, count] {
            try {
                const auto t0 = std::chrono::steady_clock::now();
                gather.prefetch(history, first, count);
                if (ple_timing()) {
                    std::fprintf(stderr, "ple_timing prefetch first=%zu count=%zu ms=%.1f\n", first,
                                 count, ms_since(t0));
                }
            } catch (...) {
                // A hint: a failed prefetch leaves the next gather to read the rows itself.
            }
        });
    }

private:
    std::thread thread_;
};

} // namespace

namespace {

void require_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                   const char* label) {
    bool ok = t.dtype == dtype && t.is_contiguous() && t.data != nullptr;
    int i   = 0;
    for (const std::int32_t dim : shape) { ok = ok && t.ne[i++] == dim; }
    for (; i < 4; ++i) { ok = ok && t.ne[i] == 1; }
    if (!ok) {
        throw std::invalid_argument(std::string("Text forward ") + label + " shape mismatch");
    }
}

// T42: one prompt chunk wider than the Program's arena runs on workspace memory the expert
// backend lends (Qwen4RoutedExperts::prompt_workspace). The TextContext's arena object belongs to
// the Program; for the chunk it is moved aside and replaced by a non-owning arena over the lent
// span, and restored afterwards (also on unwinding, after the window helper waited for the device).
class ArenaLoan {
public:
    ArenaLoan() = default;

    ~ArenaLoan() { restore(); }

    ArenaLoan(const ArenaLoan&)            = delete;
    ArenaLoan& operator=(const ArenaLoan&) = delete;

    void lend(WorkspaceArena& arena, DeviceSpan span) {
        if (target_ != nullptr) { throw std::logic_error("Text forward arena loan already taken"); }
        saved_.emplace(std::move(arena));
        target_ = &arena;
        arena   = WorkspaceArena(span);
    }

    void restore() noexcept {
        if (target_ == nullptr) { return; }
        *target_ = std::move(*saved_);
        saved_.reset();
        target_ = nullptr;
    }

private:
    WorkspaceArena* target_ = nullptr;
    std::optional<WorkspaceArena> saved_;
};

// The carried hidden of a prompt chunk: the persistent prefill_hidden when it holds the chunk,
// else (a chunk on lent workspace, T42) a matrix of the chunk's arena.
Tensor chunk_hidden(const Tensor& prefill_hidden, WorkspaceArena& work, std::int32_t rows,
                    std::int32_t len) {
    if (prefill_hidden.data != nullptr && len <= prefill_hidden.ne[1]) {
        return prefill_hidden.slice(1, 0, len);
    }
    return work.alloc(DType::BF16, {rows, len});
}

// A chunk whose hidden is not prefill_hidden leaves its last column in prefill_hidden's last
// column: the Program's copy_tail reads the final chunk's last column there (T42).
void keep_last_column(const Tensor& xf, const Tensor& prefill_hidden, cudaStream_t stream) {
    if (prefill_hidden.data == nullptr || xf.data == prefill_hidden.data) { return; }
    const Tensor last   = xf.slice(1, xf.ne[1] - 1, 1);
    const Tensor target = prefill_hidden.slice(1, prefill_hidden.ne[1] - 1, 1);
    CUDA_CHECK(::ninfer::memcpy_device_async(target.data, last.data, last.bytes(), stream));
}

// Columns [first, first + count) of a chunk's RoPE positions: a view of one-axis [T] positions,
// a contiguous copy (the RoPE Ops read contiguous positions) of axis-major [T,3] ones.
Tensor rope_columns(const Tensor& rope, std::int32_t first, std::int32_t count,
                    WorkspaceArena& work, cudaStream_t stream) {
    if (rope.ne[1] != 3) { return rope.slice(0, first, count); }
    Tensor out           = work.alloc(DType::I32, {count, 3});
    const std::int32_t T = rope.ne[0];
    for (std::int32_t axis = 0; axis < 3; ++axis) {
        CUDA_CHECK(::ninfer::memcpy_device_async(
            static_cast<std::int32_t*>(out.data) + static_cast<std::ptrdiff_t>(axis) * count,
            static_cast<const std::int32_t*>(rope.data) + static_cast<std::ptrdiff_t>(axis) * T +
                first,
            static_cast<std::size_t>(count) * sizeof(std::int32_t), stream));
    }
    return out;
}

// A multimodal chunk's host inputs: its three-axis RoPE positions (axis-major [3][len], read
// from the prompt's positions) and the visual columns of its Vision item (chunk-relative, with
// the item's first merged token among them).
struct Qwen4VisualChunk {
    std::vector<std::int32_t> rope_rows;
    std::vector<std::int32_t> columns;
    std::int32_t merged_begin = 0;
};

Qwen4VisualChunk visual_chunk(std::span<const std::int32_t> positions, std::size_t prompt_tokens,
                              const VisionChunk& vision, std::uint32_t base, std::int32_t len) {
    Qwen4VisualChunk out;
    out.rope_rows.resize(static_cast<std::size_t>(3) * static_cast<std::size_t>(len));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        std::copy_n(positions.data() + axis * prompt_tokens + base, len,
                    out.rope_rows.data() + axis * static_cast<std::size_t>(len));
    }
    if (vision.control != nullptr) {
        const auto scatter = std::span<const std::int32_t>(vision.control->scatter_indices);
        const auto begin =
            std::lower_bound(scatter.begin(), scatter.end(), static_cast<std::int32_t>(base));
        const auto end =
            std::lower_bound(begin, scatter.end(), static_cast<std::int32_t>(base) + len);
        out.merged_begin = static_cast<std::int32_t>(begin - scatter.begin());
        out.columns.reserve(static_cast<std::size_t>(end - begin));
        for (auto index = begin; index != end; ++index) {
            out.columns.push_back(*index - static_cast<std::int32_t>(base));
        }
    }
    return out;
}

// The MTP shifted input of rank `r` for `count` columns starting at `first`: the token
// embeddings of the shifted ids with the shifted visual overlap scattered over them.
void compose_mtp_embeddings(const Tensor& ids, const Weight& token_embedding, const Tensor& merged,
                            std::span<const std::int32_t> columns, std::size_t source_begin,
                            WorkspaceArena& work, Tensor& out, cudaStream_t stream) {
    ops::embedding(ids, token_embedding, out, stream);
    if (columns.empty()) { return; }
    const auto count = static_cast<std::int32_t>(columns.size());
    Tensor indices   = work.alloc(DType::I32, {count});
    CUDA_CHECK(cudaMemcpyAsync(indices.data, columns.data(), columns.size_bytes(),
                               cudaMemcpyHostToDevice, stream));
    const Tensor rows = merged.slice(1, static_cast<std::int32_t>(source_begin), count);
    ops::scatter(rows, indices, out, stream);
}

// The Arena scopes of one execution on every rank of the Text forward (null arena: no rank).
class RankScopes {
public:
    explicit RankScopes(const std::array<WorkspaceArena*, 2>& arenas) {
        for (std::size_t r = 0; r < arenas.size(); ++r) {
            if (arenas[r] != nullptr) { held_[r].emplace(arenas[r]->scope()); }
        }
    }

private:
    std::array<std::optional<WorkspaceArena::Scope>, 2> held_;
};

} // namespace

// A DFlash drafter's feature sink as the Qwen4 forward's tap: rank 0's residual into `sink` on
// rank 0's stream, at tp 2 rank 1's into the peer sink (peer_tap_*).
class TextContext::Qwen4SinkTap final : public Qwen4FeatureTap {
public:
    Qwen4SinkTap(TextContext& text, DFlashFeatureSink& sink) : text_(text), sink_(sink) {}

    void begin(const std::array<Tensor, 2>& x) override {
        sink_.begin(x[0]);
        if (text_.tp2()) { text_.peer_tap_begin(x[1]); }
    }

    void capture_layer(int layer, const std::array<Tensor, 2>& x) override {
        sink_.capture_layer(layer, x[0], text_.ctx_.stream);
        if (text_.tp2()) { text_.peer_tap_capture_layer(layer, x[1]); }
    }

    [[nodiscard]] bool captures_columns() const override { return sink_.batch_features == nullptr; }

    void capture_layer_columns(int layer, const std::array<Tensor, 2>& x,
                               std::int32_t first) override {
        sink_.capture_layer_columns(layer, x[0], first, text_.ctx_.stream);
        if (text_.tp2()) { text_.peer_tap_capture_layer_columns(layer, x[1], first); }
    }

private:
    TextContext& text_;
    DFlashFeatureSink& sink_;
};

const Qwen4Runtime& TextContext::qwen4_runtime() const {
    if (qwen4_runtime_ == nullptr ||
        (config_.qwen4_routed_experts() && qwen4_runtime_->experts == nullptr)) {
        throw std::logic_error("Text forward execution has no bound Program runtime");
    }
    return *qwen4_runtime_;
}

// The whole prompt of a Qwen4Exp chunk (the n-gram embedding hashes the tokens before each column)
// after checking that exactly one text or multimodal prompt names it from the chunk's base. A
// multimodal prompt carries its three-axis positions and its Vision session.
std::span<const int> TextContext::qwen4_prompt(std::span<const int> ids,
                                               const TextPrefill* text_prefill,
                                               const MultimodalPrefill* multimodal) const {
    if ((text_prefill == nullptr) == (multimodal == nullptr)) {
        throw std::invalid_argument("Text forward prefill requires its full prompt");
    }
    const std::span<const int> prompt =
        text_prefill != nullptr ? text_prefill->token_ids : multimodal->token_ids;
    const std::uint32_t begin = text_prefill != nullptr ? text_prefill->begin : multimodal->begin;
    if (begin != text_kv_base_ || prompt.size() < static_cast<std::size_t>(begin) + ids.size()) {
        throw std::invalid_argument("Text forward prefill requires its full prompt");
    }
    if (multimodal != nullptr &&
        (multimodal->vision == nullptr || multimodal->positions.size() != 3 * prompt.size())) {
        throw std::invalid_argument(
            "Text forward multimodal prefill requires a Vision session and [3,T] positions");
    }
    return prompt;
}

void TextContext::validate_tensor_parallel_qwen4() {
    const TpExecution& tp = *tp_;
    shard_config_         = shard_text_config(config_, kTensorParallelWidth);
    if (mtp_enabled() && qwen4_mtp_ == nullptr && mtp_ != nullptr) {
        // A dense Qwen3.5 model's own MTP head (text_tp2.inc).
        validate_tensor_parallel_mtp();
    } else if (mtp_enabled()) {
        // T33: MTP over both ranks (each rank's MTP QSA heads, expert partition, MTP KV).
        if (qwen4_mtp_ == nullptr || !tp.parameters->qwen4_mtp || !tp.mtp_complete() ||
            batch_mtp_kv_ == nullptr || !batch_mtp_kv_->has_qsa_index() ||
            !tp.mtp_cache->has_qsa_index()) {
            throw std::invalid_argument("tensor-parallel Qwen4Exp MTP binding is incomplete");
        }
    }
    if (!tp.parameters->qwen4 || !parameters_.qwen4 ||
        tp.parameters->qwen4->layers.size() != parameters_.qwen4->layers.size() ||
        parameters_.qwen4->layers.size() != config_.layer_types.size()) {
        throw std::invalid_argument(
            "tensor-parallel Qwen4Exp requires both ranks' Qwen4Exp Parameters");
    }
    if (tp.qwen4 == nullptr || (config_.qwen4_routed_experts() && tp.qwen4->experts == nullptr)) {
        throw std::invalid_argument("tensor-parallel Qwen4Exp has no rank 1 runtime");
    }
    if (config_.full_attention_layers != 0) {
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const qwen3_5::PagedKVCache& cache = rank_text_cache(rank);
            const PagedKVBatchLayerView view   = cache.batch_layer_view(0);
            // Dense gated attention runs head-local over each rank's KV heads: its [256,12,2]
            // geometry is registered for these caches only (softmax_attention.h).
            if (!config_.sparse_attention && view.storage != KvCacheStorage::BFloat16 &&
                view.storage != KvCacheStorage::Int8Group64 &&
                view.storage != KvCacheStorage::Nvfp4Group16) {
                throw std::invalid_argument("tensor-parallel attention supports only BF16, "
                                            "INT8-G64 and NVFP4-G16 KV caches");
            }
            if ((config_.sparse_attention && !cache.has_qsa_index()) ||
                view.num_kv_heads != dimension(shard_config_->attention->num_key_value_heads) ||
                view.head_dim != dimension(shard_config_->attention->head_dim)) {
                throw std::invalid_argument("tensor-parallel KV cache does not hold one rank's KV "
                                            "heads (and, with QSA, their index planes)");
            }
        }
    }
}

const Qwen4Runtime& TextContext::qwen4_peer_runtime() const {
    if (tp_ == nullptr || tp_->qwen4 == nullptr ||
        (config_.qwen4_routed_experts() && tp_->qwen4->experts == nullptr)) {
        throw std::logic_error("Text forward at tp 2 has no bound rank 1 runtime");
    }
    return *tp_->qwen4;
}

Tensor TextContext::qwen4_ids_on_rank1(const Tensor& ids) {
    Tensor out = tp_->work->alloc(DType::I32, {static_cast<std::int32_t>(ids.numel())});
    const ops::PeerEvents& events = *tp_->events;
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), ctx_.stream));
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    CUDA_CHECK(cudaStreamWaitEvent(rank_stream(1), events.inputs_ready(0), 0));
    CUDA_CHECK(::ninfer::memcpy_device_async(out.data, ids.data, ids.bytes(), rank_stream(1)));
    return out;
}

TextContext::RankTensors TextContext::qwen4_rank_ids(const Tensor& ids) {
    return {ids, tp2() ? qwen4_ids_on_rank1(ids) : Tensor{}};
}

std::array<const Qwen4MtpParameters*, 2> TextContext::qwen4_mtp_layers(const char* call,
                                                                       bool cache) const {
    const Qwen4MtpParameters* peer = nullptr;
    if (tp2() && rank_parameters(1).qwen4_mtp) { peer = &*rank_parameters(1).qwen4_mtp; }
    if (qwen4_mtp_ == nullptr || (tp2() && peer == nullptr) ||
        (cache && (batch_mtp_kv_ == nullptr || !batch_mtp_kv_->has_qsa_index()))) {
        throw std::logic_error(std::string("Qwen4Exp ") + call + " requires the MTP layer" +
                               (cache ? " and its QSA cache" : "") +
                               (tp2() ? " on both ranks" : ""));
    }
    return {qwen4_mtp_, peer};
}

int TextContext::qwen4_width() const noexcept { return tp2() ? kTensorParallelWidth : 1; }

std::array<WorkspaceArena*, 2> TextContext::qwen4_arenas() const noexcept {
    return {&work_, tp2() ? tp_->work : nullptr};
}

void TextContext::reset_qwen4_arenas() {
    for (WorkspaceArena* arena : qwen4_arenas()) {
        if (arena != nullptr) { arena->reset(); }
    }
}

std::array<const Qwen4Runtime*, 2> TextContext::qwen4_runtimes() const {
    return {&qwen4_runtime(), tp2() ? &qwen4_peer_runtime() : nullptr};
}

int TextContext::qwen4_device(int rank) const noexcept {
    return rank == 0 ? ctx_.device : tp_->execution->dev[1]->device;
}

Qwen4Rank<WorkspaceArena> TextContext::qwen4_rank(int rank, const Qwen4Columns& columns,
                                                  const Qwen4State& state) const {
    return {.config     = tp2() ? &*shard_config_ : &config_,
            .parameters = &rank_parameters(rank),
            .work       = qwen4_arenas()[static_cast<std::size_t>(rank)],
            .stream     = rank_stream(rank),
            .execution  = rank_execution_view(rank),
            .device     = tp2() ? qwen4_device(rank) : 0,
            .columns    = columns,
            .state      = state};
}

Qwen4Forward<WorkspaceArena>
TextContext::qwen4_forward(const std::array<Qwen4Rank<WorkspaceArena>, 2>& ranks,
                           std::int32_t columns) const {
    const Qwen4Collectives collectives =
        tp2() ? Qwen4Collectives{tp_->execution, tp_->events} : Qwen4Collectives{};
    return Qwen4Forward<WorkspaceArena>(config_, ranks, qwen4_width(), collectives, columns,
                                        columns);
}

const qwen3_5::PagedKVCache& TextContext::qwen4_text_cache(int rank, const char* call) const {
    const qwen3_5::PagedKVCache* cache = rank == 0 ? batch_text_kv_ : tp_->text_cache;
    if (cache == nullptr || (config_.sparse_attention && !cache->has_qsa_index())) {
        throw std::logic_error(std::string("Text forward ") + call +
                               " requires the Main KV cache (with QSA planes for QSA)");
    }
    return *cache;
}

void TextContext::qwen4_logits(const RankTensors& x, Tensor& logits) {
    if (tp2()) {
        logits_tp2(x, logits);
        return;
    }
    ops::linear(x[0], lm_head_->weight, logits, lm_head_->policy, work_, ctx_.stream);
}

// The device inputs of one prompt chunk on every rank (qwen4_chunk_inputs).
struct TextContext::Qwen4ChunkInputs {
    RankTensors ids, positions, rope, ple;
    std::optional<Qwen4Forward<WorkspaceArena>::VisualColumns> visual;
};

// Every rank's device inputs of a prompt chunk: its ids, its cache and RoPE positions (a multimodal
// chunk's three-axis rows, else offset by the RoPE delta), the visual columns of its Vision item
// (merged embeddings from `merged_begin`) and the PLE rows rank 0 gathered into its staging.
auto TextContext::qwen4_chunk_inputs(std::span<const int> ids, int len, int base_i,
                                     std::span<const std::int32_t> rope_rows,
                                     std::span<const std::int32_t> visual_columns,
                                     std::int32_t merged_begin, const VisionChunk& vision_chunk,
                                     std::uint32_t ple_width) -> Qwen4ChunkInputs {
    const int width   = qwen4_width();
    const auto arenas = qwen4_arenas();
    Qwen4ChunkInputs in;
    if (!visual_columns.empty()) { in.visual.emplace(); }
    const auto visual_count = static_cast<std::int32_t>(visual_columns.size());
    for (int rank = 0; rank < width; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        std::optional<ScopedCurrentDevice> current;
        if (width == 2) { current.emplace(qwen4_device(rank)); }
        cudaStream_t stream  = rank_stream(rank);
        WorkspaceArena& work = *arenas[r];
        in.ids[r]            = work.alloc(DType::I32, {len});
        CUDA_CHECK(cudaMemcpyAsync(in.ids[r].data, ids.data(), in.ids[r].bytes(),
                                   cudaMemcpyHostToDevice, stream));
        in.positions[r] = work.alloc(DType::I32, {len});
        ops::fill_i32_positions(in.positions[r], base_i, stream);
        in.rope[r] = in.positions[r];
        if (!rope_rows.empty()) {
            in.rope[r] = work.alloc(DType::I32, {len, 3});
            CUDA_CHECK(cudaMemcpyAsync(in.rope[r].data, rope_rows.data(), in.rope[r].bytes(),
                                       cudaMemcpyHostToDevice, stream));
        } else if (rope_delta_ != 0) {
            in.rope[r] = work.alloc(DType::I32, {len});
            ops::offset_i32_positions(in.positions[r], rank == 0 ? io_.rope_delta : tp_->rope_delta,
                                      in.rope[r], stream);
        }
        if (in.visual) {
            // The hidden axis is replicated: each rank scatters its own copy.
            Tensor indices = work.alloc(DType::I32, {visual_count});
            CUDA_CHECK(cudaMemcpyAsync(indices.data, visual_columns.data(), indices.bytes(),
                                       cudaMemcpyHostToDevice, stream));
            const Tensor& merged =
                rank == 0 ? vision_chunk.embeddings : vision_chunk.peer_embeddings;
            in.visual->embeddings[r] = merged.slice(1, merged_begin, visual_count);
            in.visual->indices[r]    = indices;
        }
        if (config_.ple && width == 2) { upload_chunk_ple(in, rank, len, ple_width); }
    }
    return in;
}

// Rank `rank`'s copy of the chunk's PLE rows from rank 0's host staging.
void TextContext::upload_chunk_ple(Qwen4ChunkInputs& in, int rank, int len,
                                   std::uint32_t ple_width) {
    const auto r = static_cast<std::size_t>(rank);
    in.ple[r] = qwen4_arenas()[r]->alloc(DType::BF16, {static_cast<std::int32_t>(ple_width), len});
    CUDA_CHECK(cudaMemcpyAsync(in.ple[r].data, qwen4_runtimes()[0]->ple_host, in.ple[r].bytes(),
                               cudaMemcpyHostToDevice, rank_stream(rank)));
}

// One prompt chunk at one device or over both ranks. At tp 2 each rank uploads its own copies of
// the chunk's inputs and rank 0's host gather of the PLE rows; rank 0 keeps the chunk's carried
// hidden in prefill_hidden (causal scoring's staged columns) and rank 1 its copy for the last
// column's vocabulary-split head (and, under MTP, its own prefill_hidden for the prefix-reuse
// bridge, T66b). Rank 0 samples.
PrefillChunkResult TextContext::qwen4_prefill(std::span<const int> ids,
                                              const TextPrefill* text_prefill,
                                              const MultimodalPrefill* multimodal,
                                              bool finalize_at_end, DFlashFeatureSink* sink) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    const int width    = qwen4_width();
    const auto runtime = qwen4_runtimes();
    const auto arenas  = qwen4_arenas();
    std::optional<ScopedCurrentDevice> device;
    if (width == 2) { device.emplace(ctx_.device); }
    cudaStream_t s                    = ctx_.stream;
    const auto T                      = static_cast<std::int64_t>(ids.size());
    const std::uint32_t base          = text_kv_base_;
    const std::span<const int> prompt = qwen4_prompt(ids, text_prefill, multimodal);
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        ((qwen4_mtp_ == nullptr && mtp_ == nullptr) ||
         mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0]))) {
        throw std::logic_error("Qwen4Exp MTP prefill has no MTP layer or exceeds the draft window");
    }
    // A multimodal prompt sets its RoPE delta; a text chunk continuing a cached multimodal prefix
    // keeps the sequence's (T209); a chunk at base 0 resets it. Each rank offsets its own
    // positions by its own copy of the delta.
    if (multimodal != nullptr) {
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);
    if (width == 2) {
        const ScopedCurrentDevice peer(qwen4_device(1));
        Tensor peer_rope_delta = tp_->rope_delta;
        ops::set_i32_scalar(peer_rope_delta, rope_delta_, rank_stream(1));
    }
    const int base_i             = static_cast<int>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split         = split_abs > base_i && split_abs <= base_i + T;
    const int split_rel          = has_split ? static_cast<int>(split_abs - base_i) : -1;

    int len = static_cast<int>(std::min<std::int64_t>(prefill_chunk_, T));
    if (split_rel > 0 && len > split_rel) { len = split_rel; }
    // The Vision session caps the chunk at the next item and, on an item's first chunk, encodes it
    // (on the tower's rank, copying the merged embeddings to the other rank's handoff at tp 2),
    // outside the arenas: the encode and the merged embeddings live past their general capacity,
    // so a chunk on lent workspace reads them in place.
    VisionChunk vision_chunk;
    Qwen4VisualChunk visual;
    if (multimodal != nullptr) {
        vision_chunk = multimodal->vision->prepare_chunk(base, static_cast<std::uint32_t>(len));
        if (vision_chunk.length <= 0 || vision_chunk.length > len) {
            throw std::logic_error("Vision chunk length is outside the prefill chunk");
        }
        if (width == 2 && vision_chunk.control != nullptr &&
            vision_chunk.peer_embeddings.data == nullptr) {
            throw std::logic_error("tensor-parallel Vision chunk has no rank 1 embeddings");
        }
        len    = vision_chunk.length;
        visual = visual_chunk(multimodal->positions, prompt.size(), vision_chunk, base, len);
    }
    const bool is_last = finalize_at_end && len == T;
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    reset_qwen4_arenas();
    // One expert window per chunk (with its MTP prompt calls; rank 0's backend drives both GPUs'
    // slot tables): the backend observes the chunk's routing once it completed. A chunk wider
    // than the Program's arenas runs on each GPU's lent memory (T42). A dense FFN has no backend:
    // the chunk runs on the Program's arenas, outside any window.
    const std::array<Qwen4RoutedExperts*, 2> experts{runtime[0]->experts,
                                                     width == 2 ? runtime[1]->experts : nullptr};
    std::array<ArenaLoan, 2> loans;
    const bool lent_arena = experts[0] != nullptr && len > experts[0]->prompt_arena_columns();
    if (experts[0] != nullptr) {
        experts[0]->prepare_prompt(len, base, static_cast<std::int64_t>(prompt.size()));
    }
    const auto abandon = [&] {
        for (int rank = width - 1; rank >= 0; --rank) {
            if (Qwen4RoutedExperts* e = experts[static_cast<std::size_t>(rank)]) {
                e->abandon_window(rank_stream(rank));
            }
        }
    };
    const auto synchronize = [&] {
        ctx_.synchronize();
        if (width == 2) { tp_->execution->dev[1]->synchronize(); }
    };
    if (experts[0] != nullptr) {
        experts[0]->begin_window(s, Qwen4Window::Prompt);
        if (width == 2) {
            try {
                const ScopedCurrentDevice peer(qwen4_device(1));
                experts[1]->begin_window(rank_stream(1), Qwen4Window::Prompt);
            } catch (...) {
                experts[0]->abandon_window(s);
                throw;
            }
        }
    }
    try {
        if (lent_arena) {
            const std::array<cudaStream_t, 2> streams{s, width == 2 ? rank_stream(1) : nullptr};
            const auto spans = experts[0]->prompt_workspace(
                std::span<const cudaStream_t>(streams.data(), static_cast<std::size_t>(width)));
            for (int rank = 0; rank < width; ++rank) {
                const auto r = static_cast<std::size_t>(rank);
                loans[r].lend(*arenas[r], spans[r]);
            }
        }
        std::uint32_t ple_width = 0;
        NextChunkPlePrefetch next_ple;
        // One host gather of the chunk's n-gram rows (rank 0's staging), uploaded to every rank.
        // The staging is reused per chunk; the previous chunk ended in a synchronization of the
        // devices, so its uploads have completed.
        const auto gather_ple = [&] {
            if (runtime[0]->ple_gather == nullptr || runtime[0]->ple_host == nullptr) {
                throw std::logic_error("Qwen4Exp prefill has no PLE gather staging");
            }
            ple_width = runtime[0]->ple_gather->width();
            const std::span<const std::int32_t> history(prompt.data(), prompt.size());
            const auto gather_t0 = std::chrono::steady_clock::now();
            runtime[0]->ple_gather->gather(
                history, base, static_cast<std::size_t>(len),
                std::span<std::uint16_t>(runtime[0]->ple_host,
                                         static_cast<std::size_t>(ple_width) * len));
            if (ple_timing()) {
                std::fprintf(stderr, "ple_timing gather first=%u count=%d ms=%.1f\n", base, len,
                             ms_since(gather_t0));
            }
            // The prompt continues in `history` beyond this call's `ids` (one chunk per call).
            next_ple.start(*runtime[0]->ple_gather, history, static_cast<std::size_t>(base) + len,
                           prefill_chunk_);
        };
        // Each width keeps its own host order: at tp 2 the gather precedes every rank's uploads,
        // at one device it follows the chunk's other uploads.
        if (config_.ple && width == 2) { gather_ple(); }
        Qwen4ChunkInputs in = qwen4_chunk_inputs(ids, len, base_i, visual.rope_rows, visual.columns,
                                                 visual.merged_begin, vision_chunk, ple_width);
        if (config_.ple && width == 1) {
            gather_ple();
            upload_chunk_ple(in, 0, len, ple_width);
        }
        // The carried hidden of every prompt column: the committed wide residual (without
        // hyper-connections, the final-normed hidden).
        RankTensors xf;
        const std::int32_t wide_rows = dimension(config_.residual_width());
        xf[0]                        = chunk_hidden(prefill_hidden_, work_, wide_rows, len);
        if (width == 2) { xf[1] = chunk_hidden(tp_->prefill_hidden, *tp_->work, wide_rows, len); }
        const auto visible = static_cast<std::uint32_t>(base_i + len);
        const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
        std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            ranks[r]     = qwen4_rank(
                rank,
                {.width            = len,
                     .batch            = 1,
                     .cache_positions  = &in.positions[r],
                     .rope_positions   = &in.rope[r],
                     .valid_columns    = nullptr,
                     .table_rows       = rank == 0 ? &io_.text_kv_table_row : &tp_->text_kv_table_row,
                     .max_visible      = visible,
                     .prefill          = true,
                     .source_slot      = linear_state_source_slot_,
                     .destination_slot = linear_state_destination_slot_,
                     .first_position   = base_i,
                     .envelope         = &chunk_envelope},
                {.kv      = &qwen4_text_cache(rank, "prefill"),
                     .linear  = &rank_state(rank),
                     .runtime = runtime[r]});
        }
        Qwen4Forward<WorkspaceArena> forward = qwen4_forward(ranks, len);
        std::optional<Qwen4SinkTap> feature_tap;
        if (sink != nullptr) { forward.set_feature_tap(&feature_tap.emplace(*this, *sink)); }
        forward.run(in.ids, config_.ple ? &in.ple : nullptr, xf, in.visual ? &*in.visual : nullptr);
        if (sink != nullptr) {
            sink->capture_positions(in.positions[0], s);
            if (width == 2) { peer_tap_capture_positions(in.positions[1]); }
        }

        if (is_last) { qwen4_prompt_token(forward, xf, len, base_i + static_cast<int>(T), prompt); }

        if (prepare_mtp_prompt && runtime[0]->taps != nullptr &&
            runtime[0]->taps->wants_mtp_logits() && base == 0) {
            runtime[0]->taps->capture_tokens(
                std::span<const std::int32_t>(prompt.data(), prompt.size()));
        }
        if (prepare_mtp_prompt && qwen4_mtp_ == nullptr) {
            // A dense Qwen3.5 model keeps its own MTP head (text.cpp, tp2/text_tp2.inc).
            if (width == 1) {
                mtp_prompt_chunk(prompt, base, len, static_cast<int>(T), base_i, is_last, xf[0],
                                 in.positions[0], in.rope[0], chunk_envelope, multimodal != nullptr,
                                 vision_chunk);
            } else {
                mtp_prompt_chunk_tp2(prompt, base, len, static_cast<int>(T), base_i, is_last, xf,
                                     in.positions, in.rope, chunk_envelope, multimodal != nullptr,
                                     vision_chunk);
            }
        } else if (prepare_mtp_prompt) {
            qwen4_mtp_prompt_chunk(prompt, base, len, is_last, xf, in.positions, in.rope,
                                   vision_chunk);
        }

        qwen4_retain_chunk(xf, len, split_rel > 0 && len == split_rel, sink);
        // Observe, wait and close inside the try (T36 M2): a failure here must not leave any
        // rank's window open for the next execution.
        prefill_split_frontier_ = -1;
        timing.begin_wait();
        // At one device the next chunk's PLE prefetch completes before the window is observed
        // (at tp 2 after it has closed), as each width was measured with.
        if (width == 1) { next_ple.join(); }
        if (experts[0] != nullptr) { experts[0]->observe_window(s); }
        synchronize();
        timing.end_wait();
        if (experts[0] != nullptr) {
            experts[0]->end_window(s);
            if (width == 2) {
                const ScopedCurrentDevice peer(qwen4_device(1));
                experts[1]->end_window(rank_stream(1));
            }
        }
    } catch (...) {
        try {
            synchronize();
        } catch (...) {}
        abandon();
        throw;
    }
    for (int rank = width - 1; rank >= 0; --rank) {
        loans[static_cast<std::size_t>(rank)].restore();
    }
    reset_qwen4_arenas();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(len),
                              .finalized        = finalize_at_end && len == T,
                              .timing           = timing.finish()};
}

// The last prompt column's output head on every rank and rank 0's sampling of the first token
// (`total` = the prompt's length: the next position).
void TextContext::qwen4_prompt_token(Qwen4Forward<WorkspaceArena>& forward, const RankTensors& xf,
                                     int len, int total, std::span<const int> prompt) {
    const int width   = qwen4_width();
    const auto arenas = qwen4_arenas();
    cudaStream_t s    = ctx_.stream;
    const RankScopes head_scope(arenas);
    RankTensors last, x;
    for (int rank = 0; rank < width; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        last[r]      = xf[r].slice(1, len - 1, 1);
        x[r]         = arenas[r]->alloc(DType::BF16, {dimension(config_.hidden_size), 1});
    }
    forward.head(last, x);
    Tensor logits = io_.logits.slice(1, 0, 1);
    qwen4_logits(x, logits);
    const std::int32_t vocabulary = dimension(parameters_.model.resources().public_token_count);
    ops::set_i32_scalar(io_.pos, total, s);
    ops::set_i32_scalar(io_.rope_pos, total + rope_delta_, s);
    if (sampling_config_ != nullptr) {
        ops::sample(logits, io_.token, vocabulary, sampling_config_, io_.pos,
                    ops::kSamplePurposePrefill, work_, s);
        if (reasoning_guard_ && reasoning_eos_guard_applies(prompt)) {
            guard_first_token(logits, io_.token, vocabulary, sampling_config_, io_.pos, work_, s);
        }
    } else {
        ops::argmax(logits, io_.token, vocabulary, s);
    }
}

// What a prompt chunk leaves for later calls: at the rewrite frontier the checkpoint hidden of
// its last column (rank 1's too: the MTP bridge resumes rank 1's half of the MTP head from it,
// T66b), the last column in prefill_hidden's last column when the chunk ran on lent workspace,
// and a drafter sink's chunk.
void TextContext::qwen4_retain_chunk(const RankTensors& xf, int len, bool rewrite,
                                     DFlashFeatureSink* sink) {
    const int width = qwen4_width();
    cudaStream_t s  = ctx_.stream;
    if (rewrite && rewrite_checkpoint_hidden_output_ != nullptr) {
        const Tensor checkpoint = xf[0].slice(1, len - 1, 1);
        CUDA_CHECK(::ninfer::memcpy_device_async(rewrite_checkpoint_hidden_output_->data,
                                                 checkpoint.data, checkpoint.bytes(), s));
        if (width == 2 && peer_rewrite_checkpoint_hidden_output_ != nullptr) {
            const Tensor peer_checkpoint = xf[1].slice(1, len - 1, 1);
            require_shape(*peer_rewrite_checkpoint_hidden_output_, DType::BF16,
                          {dimension(config_.residual_width()), 1},
                          "rank 1 rewrite checkpoint hidden output");
            const ScopedCurrentDevice peer(qwen4_device(1));
            CUDA_CHECK(::ninfer::memcpy_device_async(peer_rewrite_checkpoint_hidden_output_->data,
                                                     peer_checkpoint.data, peer_checkpoint.bytes(),
                                                     rank_stream(1)));
        }
    }
    keep_last_column(xf[0], prefill_hidden_, s);
    if (width == 2 && tp_->prefill_hidden.data != nullptr) {
        const ScopedCurrentDevice peer(qwen4_device(1));
        keep_last_column(xf[1], tp_->prefill_hidden, rank_stream(1));
    }
    if (sink != nullptr) {
        reset_qwen4_arenas();
        sink->consume_prefill_chunk(len, rewrite);
        if (width == 2) { peer_tap_consume_prefill_chunk(len, rewrite); }
    }
}

// MTP alignment of one prompt chunk on every rank: hidden h_p and position p with token x_(p+1);
// the last prompt column takes the generated token (qwen3_5-model.md#prefill-decode-and-mtp). A
// shifted token inside the chunk's Vision item enters the MTP layer as its merged embedding, as
// in the target's input (the MTP embedding projection is replicated).
void TextContext::qwen4_mtp_prompt_chunk(std::span<const int> prompt, std::uint32_t base, int len,
                                         bool is_last, const RankTensors& xf,
                                         const RankTensors& positions,
                                         const RankTensors& rope_positions,
                                         const VisionChunk& vision_chunk) {
    cudaStream_t s              = ctx_.stream;
    const auto alignment_tokens = static_cast<std::uint32_t>(prompt.size());
    const qwen3_5::MtpAlignmentWindow window =
        qwen3_5::plan_mtp_alignment_window(alignment_tokens, base, static_cast<std::uint32_t>(len));
    const int prompt_columns = len - static_cast<int>(window.final_column_uses_generated_token);
    const RankScopes mtp_scope(qwen4_arenas());
    Tensor mtp_ids = work_.alloc(DType::I32, {len});
    if (prompt_columns != 0) {
        CUDA_CHECK(cudaMemcpyAsync(mtp_ids.data, prompt.data() + window.shifted_embedding_begin,
                                   static_cast<std::size_t>(prompt_columns) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, s));
    }
    if (window.final_column_uses_generated_token) {
        CUDA_CHECK(::ninfer::memcpy_device_async(mtp_ids.slice(0, len - 1, 1).data, io_.token.data,
                                                 sizeof(std::int32_t), s));
    }
    qwen3_5::MtpVisualOverlap overlap;
    if (vision_chunk.control != nullptr) {
        overlap = qwen3_5::shifted_visual_overlap(vision_chunk.control->scatter_indices,
                                                  alignment_tokens, window);
    }
    const Qwen4MtpVisual mtp_visual{
        .merged       = {vision_chunk.embeddings, vision_chunk.peer_embeddings},
        .columns      = overlap.destination_columns,
        .source_begin = overlap.source_begin};
    qwen4_mtp_prefill(mtp_ids, xf, positions, rope_positions,
                      static_cast<std::uint32_t>(static_cast<int>(base) + len),
                      is_last && mtp_proposal_extent_ != 0,
                      overlap.empty() ? nullptr : &mtp_visual);
}

void TextContext::qwen4_ordinary_decode(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& source_slots, const Tensor& destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    if (mtp_enabled() && qwen4_mtp_layer()) {
        throw std::logic_error("Qwen4Exp MTP decodes through the MTP round");
    }
    const int width          = qwen4_width();
    const std::int32_t batch = ids.ne[0];
    // Rank 0's operands are the call's; rank 1 reads its own copies from its ordinary frame.
    RankTensors rank_ids{ids, Tensor{}}, cache{cache_positions, Tensor{}},
        rope{rope_positions, Tensor{}}, rows{kv_table_rows, Tensor{}},
        sources{source_slots, Tensor{}}, destinations{destination_slots, Tensor{}};
    if (width == 2) {
        if (tp_->ordinary == nullptr) {
            throw std::logic_error("tensor-parallel decode requires rank 1's ordinary frame");
        }
        const auto peer_window = [batch](const Tensor& source, const char* label) {
            if (source.dtype != DType::I32 || source.ne[0] < batch || source.ne[1] != 1 ||
                source.ne[2] != 1 || source.ne[3] != 1 || !source.is_contiguous() ||
                source.data == nullptr) {
                throw std::invalid_argument(std::string("tensor-parallel rank 1 ordinary ") +
                                            label + " does not cover the batch");
            }
            return source.slice(0, 0, batch);
        };
        const OrdinaryPeerFrame& frame = *tp_->ordinary;
        rank_ids[1]                    = peer_window(frame.tokens, "tokens");
        cache[1]                       = peer_window(frame.cache_positions, "cache positions");
        rope[1]                        = peer_window(frame.rope_positions, "RoPE positions");
        rows[1]                        = peer_window(frame.text_kv_table_rows, "KV rows");
        sources[1]                     = peer_window(frame.state_source_slots, "source slots");
        destinations[1] = peer_window(frame.state_destination_slots, "destination slots");
    }
    const auto runtime = qwen4_runtimes();
    RankTensors ple_rows;
    if (config_.ple) {
        for (int rank = 0; rank < width; ++rank) {
            const auto r      = static_cast<std::size_t>(rank);
            const Tensor& all = runtime[r]->decode_ple_rows;
            if (all.data == nullptr || all.ne[1] < batch) {
                throw std::logic_error("Qwen4Exp decode has no PLE rows");
            }
            ple_rows[r] = all.slice(1, 0, batch);
        }
    }
    std::optional<ScopedCurrentDevice> device;
    if (width == 2) { device.emplace(ctx_.device); }
    const auto arenas = qwen4_arenas();
    reset_qwen4_arenas();
    {
        std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            ranks[r]     = qwen4_rank(rank,
                                      {.width             = 1,
                                       .batch             = batch,
                                       .cache_positions   = &cache[r],
                                       .rope_positions    = &rope[r],
                                       .valid_columns     = nullptr,
                                       .table_rows        = &rows[r],
                                       .max_visible       = envelope.max_visible_keys,
                                       .prefill           = false,
                                       .source_slots      = &sources[r],
                                       .destination_slots = &destinations[r],
                                       .envelope          = &envelope},
                                      {.kv      = &qwen4_text_cache(rank, "decode"),
                                       .linear  = &rank_state(rank),
                                       .runtime = runtime[r]});
        }
        // Rank 0's carried wide residual is the call's hidden; rank 1's only feeds its
        // vocabulary half through the replicated output mixer.
        RankTensors wide{hidden, Tensor{}};
        if (width == 2) {
            wide[1] = arenas[1]->alloc(DType::BF16, {dimension(config_.residual_width()), batch});
        }
        Qwen4Forward<WorkspaceArena> forward = qwen4_forward(ranks, batch);
        forward.run(rank_ids, config_.ple ? &ple_rows : nullptr, wide);
        RankTensors x;
        for (int rank = 0; rank < width; ++rank) {
            x[static_cast<std::size_t>(rank)] = arenas[static_cast<std::size_t>(rank)]->alloc(
                DType::BF16, {dimension(config_.hidden_size), batch});
        }
        forward.head(wide, x);
        qwen4_logits(x, logits);
    }
    reset_qwen4_arenas();
}

// Target verification of W = K+1 columns of B lanes (column 0 the anchor, then the drafts) at one
// device: the rank-pair entry with rank 0 alone.
void TextContext::qwen4_target_verify(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink* sink) {
    const std::int32_t W = ids.ne[0];
    const std::int32_t B = ids.ne[1];
    require_shape(cache_positions, DType::I32, {W, B}, "verify positions");
    require_shape(rope_positions, DType::I32, {W, B}, "verify RoPE positions");
    require_shape(valid_columns, DType::I32, {B}, "verify valid columns");
    require_shape(kv_table_rows, DType::I32, {B}, "verify KV rows");
    require_shape(source_slots, DType::I32, {B}, "verify state slots");
    qwen4_target_verify(RankTensors{ids, Tensor{}}, RankTensors{cache_positions, Tensor{}},
                        RankTensors{rope_positions, Tensor{}}, RankTensors{valid_columns, Tensor{}},
                        RankTensors{kv_table_rows, Tensor{}}, RankTensors{source_slots, Tensor{}},
                        envelope, RankTensors{hidden, Tensor{}}, logits, target_tokens, sink);
}

// Target verification on every rank: GDN records ReplaySSM transitions and PLE its convolution
// inputs (folded at commit), QSA appends in place; `hidden` receives each rank's wide residuals,
// logits/target_tokens rank 0's Text head.
void TextContext::qwen4_target_verify(
    const RankTensors& ids, const RankTensors& cache_positions, const RankTensors& rope_positions,
    const RankTensors& valid_columns, const RankTensors& kv_table_rows,
    const RankTensors& source_slots, ops::CausalAttentionExecutionEnvelope envelope,
    const RankTensors& hidden, Tensor& logits, Tensor& target_tokens, DFlashFeatureSink* sink) {
    const int width              = qwen4_width();
    const auto runtime           = qwen4_runtimes();
    const std::int32_t W         = ids[0].ne[0];
    const std::int32_t B         = ids[0].ne[1];
    const std::int32_t T         = W * B;
    const std::int32_t wide_rows = dimension(config_.residual_width());
    const std::int32_t H         = dimension(config_.hidden_size);
    if (W < 2 || B <= 0 || B > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("Text forward verification window is outside [2,K+1]x[1,8]");
    }
    if (gdn_state_action_ != GdnStateAction::RecordForReplay || replay_records_ == nullptr) {
        throw std::logic_error("Text forward verification requires ReplaySSM records");
    }
    for (int rank = 0; rank < width; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        require_shape(ids[r], DType::I32, {W, B}, "verify ids");
        require_shape(hidden[r], DType::BF16, {wide_rows, W, B}, "verify hidden");
    }
    require_shape(logits, DType::BF16, {dimension(config_.vocab_size), W, B}, "verify logits");
    require_shape(target_tokens, DType::I32, {W, B}, "verify tokens");
    RankTensors rows;
    if (config_.ple) {
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            if (runtime[r]->decode_ple_rows.data == nullptr ||
                runtime[r]->decode_ple_rows.ne[1] < T) {
                throw std::logic_error("Qwen4Exp verification has no PLE rows");
            }
            rows[r] = runtime[r]->decode_ple_rows.slice(1, 0, T);
        }
    }
    std::optional<ScopedCurrentDevice> device;
    if (width == 2) { device.emplace(ctx_.device); }
    const auto arenas = qwen4_arenas();
    reset_qwen4_arenas();
    {
        std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            ranks[r]     = qwen4_rank(rank,
                                      {.width             = W,
                                       .batch             = B,
                                       .cache_positions   = &cache_positions[r],
                                       .rope_positions    = &rope_positions[r],
                                       .valid_columns     = &valid_columns[r],
                                       .table_rows        = &kv_table_rows[r],
                                       .max_visible       = envelope.max_visible_keys,
                                       .prefill           = false,
                                       .source_slots      = &source_slots[r],
                                       .destination_slots = &source_slots[r],
                                       .replay_records    = &rank_replay_records(rank),
                                       .envelope          = &envelope},
                                      {.kv      = &qwen4_text_cache(rank, "verification"),
                                       .linear  = &rank_state(rank),
                                       .runtime = runtime[r]});
        }
        RankTensors flat_ids, wide;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            flat_ids[r]  = ids[r].view({T});
            wide[r]      = hidden[r].view({wide_rows, T});
        }
        Qwen4Forward<WorkspaceArena> forward = qwen4_forward(ranks, T);
        std::optional<Qwen4SinkTap> feature_tap;
        if (sink != nullptr) { forward.set_feature_tap(&feature_tap.emplace(*this, *sink)); }
        forward.run(flat_ids, config_.ple ? &rows : nullptr, wide);
        if (sink != nullptr) {
            sink->capture_positions(cache_positions[0], ctx_.stream);
            if (width == 2) { peer_tap_capture_positions(cache_positions[1]); }
        }
        RankTensors x;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            x[r]         = arenas[r]->alloc(DType::BF16, {H, T});
        }
        forward.head(wide, x);
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), T});
        qwen4_logits(x, flat_logits);
        Tensor flat_tokens = target_tokens.view({T});
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
    reset_qwen4_arenas();
}

// The MTP layer over the round's W alignment columns of B lanes (the verified tokens shifted by
// one with the target's wide residuals), or one autoregressive column per lane, at one device.
void TextContext::qwen4_mtp_forward_decode(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    const std::int32_t W = ids.ne[0];
    const std::int32_t B = ids.ne[1];
    require_shape(cache_positions, DType::I32, {W, B}, "MTP decode positions");
    require_shape(rope_positions, DType::I32, {W, B}, "MTP decode RoPE positions");
    require_shape(valid_columns, DType::I32, {B}, "MTP decode valid columns");
    require_shape(kv_table_rows, DType::I32, {B}, "MTP decode KV rows");
    qwen4_mtp_forward_decode(
        ids, RankTensors{hidden, Tensor{}}, RankTensors{cache_positions, Tensor{}},
        RankTensors{rope_positions, Tensor{}}, RankTensors{valid_columns, Tensor{}},
        RankTensors{kv_table_rows, Tensor{}}, envelope, RankTensors{mtp_hidden, Tensor{}});
}

// The MTP layer's round call on every rank; each rank embeds its own copy of the ids.
void TextContext::qwen4_mtp_forward_decode(const Tensor& ids, const RankTensors& hidden,
                                           const RankTensors& cache_positions,
                                           const RankTensors& rope_positions,
                                           const RankTensors& valid_columns,
                                           const RankTensors& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           const RankTensors& mtp_hidden) {
    const int width              = qwen4_width();
    const auto runtime           = qwen4_runtimes();
    const auto mtp               = qwen4_mtp_layers("MTP decode");
    const std::int32_t W         = ids.ne[0];
    const std::int32_t B         = ids.ne[1];
    const std::int32_t T         = W * B;
    const std::int32_t wide_rows = dimension(config_.residual_width());
    if (W <= 0 || W > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || B <= 0 ||
        B > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("Qwen4Exp MTP decode shape is outside the supported domain");
    }
    require_shape(ids, DType::I32, {W, B}, "MTP decode ids");
    for (int rank = 0; rank < width; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        require_shape(hidden[r], DType::BF16, {wide_rows, W, B}, "MTP decode target hidden");
        require_shape(mtp_hidden[r], DType::BF16, {wide_rows, W, B}, "MTP decode hidden");
    }
    std::optional<ScopedCurrentDevice> device;
    if (width == 2) { device.emplace(ctx_.device); }
    const RankScopes scope(qwen4_arenas());
    nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    const RankTensors rank_ids = qwen4_rank_ids(ids);
    // Only the accepted column's output is read (speculative_select_accepted_hidden: column
    // valid_columns[b] - 1 of the alignment, the single column of an AR step), so the MoE routes
    // only that column; the others still append their MTP K/V.
    std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
    RankTensors flat_ids, in, out;
    for (int rank = 0; rank < width; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        ranks[r]     = qwen4_rank(
            rank,
            {.width               = W,
                 .batch               = B,
                 .cache_positions     = &cache_positions[r],
                 .rope_positions      = &rope_positions[r],
                 .valid_columns       = &valid_columns[r],
                 .moe_last_valid_only = true,
                 .table_rows          = &kv_table_rows[r],
                 .max_visible         = envelope.max_visible_keys,
                 .prefill             = false},
            {.kv = &rank_mtp_cache(rank), .linear = &rank_state(rank), .runtime = runtime[r]});
        flat_ids[r] = rank_ids[r].view({T});
        in[r]       = hidden[r].view({wide_rows, T});
        out[r]      = mtp_hidden[r].view({wide_rows, T});
    }
    Qwen4Forward<WorkspaceArena> forward = qwen4_forward(ranks, T);
    forward.run_mtp(mtp, flat_ids, nullptr, in, out, false);
}

// One sequence's MTP forward over T consecutive positions (the prefix-reuse bridge and the
// autoregressive drafts of the prompt's last column) at one device: `explicit_rope_positions`
// [T] or, for one visual bridge column, [1,3]; null offsets `positions` by the sequence's RoPE
// delta.
void TextContext::qwen4_mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                          const Tensor& positions,
                                          ops::CausalAttentionExecutionEnvelope envelope,
                                          Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                          Tensor* draft_token,
                                          const Tensor* explicit_rope_positions,
                                          const Tensor* input_embeddings) {
    const std::int32_t T = ids.ne[0];
    auto scope           = work_.scope();
    Tensor rope;
    if (explicit_rope_positions != nullptr) {
        // The bridge passes its position as [1] when it repeats on the three M-RoPE axes (text)
        // and as [1,3] when it does not (a visual position, mtp_bridge_and_propose).
        const Tensor& given = *explicit_rope_positions;
        if (given.ne[1] == 3 && T != 1) {
            throw std::invalid_argument("Qwen4Exp MTP batch takes one three-axis position only");
        }
        rope = given;
    } else {
        rope = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, rope, ctx_.stream);
    }
    qwen4_mtp_forward_batch(ids, RankTensors{hidden, Tensor{}}, RankTensors{positions, Tensor{}},
                            RankTensors{rope, Tensor{}}, envelope,
                            RankTensors{mtp_hidden, Tensor{}}, logits_column, logits, draft_token,
                            input_embeddings);
}

// One sequence's MTP forward over T consecutive positions on every rank's prefill MTP KV row,
// each rank from its own copy of the target hidden (T66b at tp 2). With `logits_column` >= 0 rank
// 0 proposes from that column. `rope_positions` are each rank's [T] positions or, for one visual
// bridge column, its [1,3] three-axis position. `input_embeddings` (rank 0's BF16 [H,T]) replaces
// embed(ids) on every rank: rank 1 runs from its own copy.
void TextContext::qwen4_mtp_forward_batch(const Tensor& ids, const RankTensors& hidden,
                                          const RankTensors& positions,
                                          const RankTensors& rope_positions,
                                          ops::CausalAttentionExecutionEnvelope envelope,
                                          const RankTensors& mtp_hidden, int logits_column,
                                          Tensor* logits, Tensor* draft_token,
                                          const Tensor* input_embeddings) {
    const int width              = qwen4_width();
    const auto mtp               = qwen4_mtp_layers("MTP forward");
    const std::int32_t T         = ids.ne[0];
    const std::int32_t wide_rows = dimension(config_.residual_width());
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_ || logits_column >= T) {
        throw std::invalid_argument("Qwen4Exp MTP batch columns are outside [1,prefill_chunk]");
    }
    require_shape(ids, DType::I32, {T}, "MTP ids");
    if (input_embeddings != nullptr) {
        require_shape(*input_embeddings, DType::BF16, {dimension(config_.hidden_size), T},
                      "MTP input embeddings");
    }
    RankTensors rope;
    for (int rank = 0; rank < width; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        require_shape(positions[r], DType::I32, {T}, "MTP positions");
        require_shape(hidden[r], DType::BF16, {wide_rows, T}, "MTP hidden");
        require_shape(mtp_hidden[r], DType::BF16, {wide_rows, T}, "MTP output hidden");
        const Tensor& given = rope_positions[r];
        if (given.dtype != DType::I32 || given.ne[0] != T ||
            (given.ne[1] != 1 && !(given.ne[1] == 3 && T == 1)) || given.ne[2] != 1 ||
            given.ne[3] != 1 || !given.is_contiguous() || given.data == nullptr) {
            throw std::invalid_argument("MTP RoPE positions must be [T] or [1,3]");
        }
        rope[r] = given.ne[1] == 3 ? given.view({1, 3}) : given.view({T});
    }
    if (logits_column >= 0 && (logits == nullptr || draft_token == nullptr)) {
        throw std::invalid_argument("Qwen4Exp MTP logits and draft outputs are required");
    }
    std::optional<ScopedCurrentDevice> device;
    if (width == 2) { device.emplace(ctx_.device); }
    const RankScopes scope(qwen4_arenas());
    nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    const RankTensors rank_ids = qwen4_rank_ids(ids);
    RankTensors embeddings;
    if (input_embeddings != nullptr) {
        embeddings[0] = *input_embeddings;
        if (width == 2) {
            // Rank 1's copy, ordered after rank 0's stream as the ids' (staged by the driver).
            embeddings[1] = tp_->work->alloc(DType::BF16, {dimension(config_.hidden_size), T});
            const ops::PeerEvents& events = *tp_->events;
            CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), ctx_.stream));
            const ScopedCurrentDevice rank1(qwen4_device(1));
            CUDA_CHECK(cudaStreamWaitEvent(rank_stream(1), events.inputs_ready(0), 0));
            CUDA_CHECK(::ninfer::memcpy_device_async(embeddings[1].data, embeddings[0].data,
                                                     embeddings[0].bytes(), rank_stream(1)));
        }
    }
    RankTensors out = mtp_hidden;
    qwen4_mtp_columns(mtp, rank_ids, hidden, out, positions, rope, T, false,
                      envelope.max_visible_keys,
                      input_embeddings != nullptr ? &embeddings : nullptr);
    if (logits_column >= 0) {
        RankTensors column;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            column[r]    = mtp_hidden[r].slice(1, logits_column, 1);
        }
        qwen4_mtp_propose(column, *logits, *draft_token);
    }
}

// Draft proposal from MTP hiddens on every rank: the replicated MTP output mixer, then the Text
// output head over the full vocabulary (vocabulary-split at tp 2) and rank 0's argmax, or (T43)
// the draft shortlist head of Qwen4Runtime::draft_head. (The artifact proposal-head hook of
// TextContext stays refused for this architecture.)
void TextContext::qwen4_mtp_propose(const RankTensors& hidden, Tensor& logits, Tensor& draft_tokens,
                                    Tensor* draft_logprobs) {
    const int width              = qwen4_width();
    const auto mtp               = qwen4_mtp_layers("proposal", false);
    const std::int32_t B         = hidden[0].ne[1];
    const std::int32_t wide_rows = dimension(config_.residual_width());
    const std::int32_t H         = dimension(config_.hidden_size);
    for (int rank = 0; rank < width; ++rank) {
        require_shape(hidden[static_cast<std::size_t>(rank)], DType::BF16, {wide_rows, B},
                      "MTP proposal hidden");
    }
    require_shape(draft_tokens, DType::I32, {B}, "MTP proposal tokens");
    if (logits.dtype != DType::BF16 || logits.ne[0] != dimension(config_.vocab_size) ||
        logits.ne[1] < B || !logits.is_contiguous()) {
        throw std::invalid_argument("Qwen4Exp MTP proposal logits shape mismatch");
    }
    if (proposal_head_ != nullptr) {
        throw std::logic_error("Qwen4Exp has no trimmed proposal head yet");
    }
    std::optional<ScopedCurrentDevice> device;
    if (width == 2) { device.emplace(ctx_.device); }
    const auto arenas = qwen4_arenas();
    const RankScopes scope(arenas);
    nvtx::ScopedRange range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(B));
    std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
    RankTensors x;
    for (int rank = 0; rank < width; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        x[r]         = arenas[r]->alloc(DType::BF16, {H, B});
        ranks[r]     = qwen4_rank(rank, {.width = B, .batch = 1, .prefill = false},
                                  {.kv      = nullptr,
                                   .linear  = &rank_state(rank),
                                   .runtime = rank == 0 ? qwen4_runtime_ : nullptr});
    }
    Qwen4Forward<WorkspaceArena> forward = qwen4_forward(ranks, B);
    forward.head({&mtp[0]->head, width == 2 ? &mtp[1]->head : nullptr}, hidden, x);
    Tensor window = logits.slice(1, 0, B);
    if (qwen4_runtime_ != nullptr && qwen4_runtime_->draft_head != nullptr) {
        const Qwen4DraftHead& draft = *qwen4_runtime_->draft_head;
        if (width == 1 || !draft.split) {
            // The shortlist head on rank 0 alone (at tp 2 its prefix is a view of rank 0's
            // vocabulary half, its extra rows were copied there at startup); no logits leave
            // rank 1. The next MTP step reads the drafts on rank 1 through qwen4_ids_on_rank1.
            draft_head_propose(draft, x[0], window, draft_tokens, draft_logprobs, work_,
                               ctx_.stream);
            return;
        }
        // Split shortlist: each rank scores its half, one exchange of the packed candidates,
        // rank 0 selects (lower row on ties) and maps the row to its token, per column.
        if (draft_logprobs != nullptr) {
            throw std::invalid_argument(
                "the split draft shortlist has no log-probabilities (--mtp-min-p at tp 2: set "
                "NINFER_QWEN4_DRAFT_SPLIT=0)");
        }
        const auto ws = workspaces();
        for (std::int32_t b = 0; b < B; ++b) {
            auto column0 = ws[0]->scope();
            auto column1 = ws[1]->scope();
            const std::array<workspace::TensorParallelProposalRoots, 2> roots{
                workspace::tp_proposal_argmax(*ws[0], draft.prefix_rows, 1),
                workspace::tp_proposal_argmax(*ws[1], draft.peer_rows, 1)};
            const RankTensors column{x[0].slice(1, b, 1), x[1].slice(1, b, 1)};
            Tensor token = draft_tokens.slice(0, b, 1);
            proposal_argmax_split(
                column, {&draft.prefix, &draft.peer}, {roots[0].partial, roots[1].partial},
                {roots[0].local, roots[1].local}, {roots[0].candidates, roots[1].candidates},
                {roots[0].staging, roots[1].staging}, token, ws, *tp_->execution, *tp_->events);
        }
        ops::proposal_remap_token_ids(draft_tokens, draft.token_ids, draft.rows(), ctx_.stream);
        return;
    }
    qwen4_logits(x, window);
    const std::int32_t vocabulary = dimension(parameters_.model.resources().public_token_count);
    ops::argmax(window, draft_tokens, vocabulary, ctx_.stream);
    if (draft_logprobs != nullptr) {
        ops::target_logprobs(window, draft_tokens, vocabulary, *draft_logprobs, ctx_.stream);
    }
}

// The Qwen4 MTP layer over `count` columns of every rank's operands on the prefill MTP KV row
// (prompt columns, their proposal and the autoregressive drafts; the prefix-reuse bridge).
void TextContext::qwen4_mtp_columns(const std::array<const Qwen4MtpParameters*, 2>& mtp,
                                    const RankTensors& ids, const RankTensors& hidden,
                                    RankTensors& out, const RankTensors& positions,
                                    const RankTensors& rope_positions, std::int32_t count,
                                    bool append_only, std::uint32_t max_visible,
                                    const RankTensors* embeddings) {
    const auto runtime = qwen4_runtimes();
    std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
    for (int rank = 0; rank < qwen4_width(); ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        ranks[r]     = qwen4_rank(
            rank,
            {.width           = count,
                 .batch           = 1,
                 .cache_positions = &positions[r],
                 .rope_positions  = &rope_positions[r],
                 .valid_columns   = nullptr,
                 .table_rows      = rank == 0 ? &io_.backend_kv_table_row : &tp_->backend_kv_table_row,
                 .max_visible     = max_visible,
                 .prefill         = true},
            {.kv = &rank_mtp_cache(rank), .linear = &rank_state(rank), .runtime = runtime[r]});
    }
    Qwen4Forward<WorkspaceArena> forward = qwen4_forward(ranks, count);
    forward.run_mtp(mtp, ids, embeddings, hidden, out, append_only);
}

// The prompt's first proposal from the MTP hidden of its last column (`ar_hidden`, every rank's)
// and the autoregressive drafts after it: draft i at position (frontier + i) from draft i-1 and
// the previous MTP hidden, each rank on its own position copy offset by its own RoPE delta.
void TextContext::qwen4_mtp_drafts(const std::array<const Qwen4MtpParameters*, 2>& mtp,
                                   RankTensors& ar_hidden, std::uint32_t max_visible) {
    const int width              = qwen4_width();
    const auto arenas            = qwen4_arenas();
    const std::int32_t wide_rows = dimension(config_.residual_width());
    Tensor logits                = io_.logits.slice(1, 0, 1);
    Tensor draft0                = io_.mtp->draft_tokens.slice(0, 0, 1);
    qwen4_mtp_propose(ar_hidden, logits, draft0);
    const RankTensors ar_position{io_.mtp->position.slice(0, 0, 1),
                                  width == 2 ? tp_->mtp->position.slice(0, 0, 1) : Tensor{}};
    for (int rank = 0; rank < width; ++rank) {
        std::optional<ScopedCurrentDevice> current;
        if (width == 2) { current.emplace(qwen4_device(rank)); }
        Tensor p = ar_position[static_cast<std::size_t>(rank)];
        ops::set_i32_scalar(p, static_cast<std::int32_t>(max_visible), rank_stream(rank));
    }
    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
        const RankScopes step_scope(arenas);
        const Tensor previous          = io_.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next                    = io_.mtp->draft_tokens.slice(0, i, 1);
        const RankTensors previous_ids = qwen4_rank_ids(previous);
        RankTensors rope, next_hidden;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            std::optional<ScopedCurrentDevice> current;
            if (width == 2) { current.emplace(qwen4_device(rank)); }
            next_hidden[r] = arenas[r]->alloc(DType::BF16, {wide_rows, 1});
            rope[r]        = arenas[r]->alloc(DType::I32, {1});
            ops::offset_i32_positions(ar_position[r], rank == 0 ? io_.rope_delta : tp_->rope_delta,
                                      rope[r], rank_stream(rank));
        }
        const auto visible = max_visible + static_cast<std::uint32_t>(i);
        qwen4_mtp_columns(mtp, previous_ids, ar_hidden, next_hidden, ar_position, rope, 1, false,
                          visible, nullptr);
        qwen4_mtp_propose(next_hidden, logits, next);
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            std::optional<ScopedCurrentDevice> current;
            if (width == 2) { current.emplace(qwen4_device(rank)); }
            CUDA_CHECK(::ninfer::memcpy_device_async(ar_hidden[r].data, next_hidden[r].data,
                                                     ar_hidden[r].bytes(), rank_stream(rank)));
            Tensor p = ar_position[r];
            ops::increment_i32_scalar(p, rank_stream(rank));
        }
    }
}

// MTP KV of one prompt chunk on every rank: every column is appended to the MTP cache; with
// `propose` the last column (the generated token) runs the whole MTP layer and drafts
// mtp_proposal_extent_ tokens autoregressively into io_.mtp (draft_tokens, ar_hidden, position;
// rank 1's frames at tp 2, each rank on its own position copy). A multimodal chunk passes its
// three-axis RoPE positions ([T,3]) and, when a shifted token is visual, `visual`: the shifted
// input is then the composed embedding rather than embed(ids).
void TextContext::qwen4_mtp_prefill(const Tensor& ids, const RankTensors& hidden,
                                    const RankTensors& positions, const RankTensors& rope_positions,
                                    std::uint32_t max_visible, bool propose,
                                    const Qwen4MtpVisual* visual) {
    const int width    = qwen4_width();
    const auto runtime = qwen4_runtimes();
    const auto mtp     = qwen4_mtp_layers("MTP prefill");
    if (width == 2 && tp_->mtp == nullptr) {
        throw std::logic_error("Qwen4Exp MTP prefill at tp 2 has no rank 1 MTP frames");
    }
    const auto arenas            = qwen4_arenas();
    cudaStream_t s               = ctx_.stream;
    const std::int32_t T         = ids.ne[0];
    const std::int32_t append    = propose ? T - 1 : T;
    const std::int32_t wide_rows = dimension(config_.residual_width());
    nvtx::ScopedRange range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    // Each rank's columns [first, first + count) of `t` along `axis`.
    const auto columns = [&](const RankTensors& t, int axis, std::int32_t first,
                             std::int32_t count) {
        RankTensors out;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            out[r]       = t[r].slice(axis, first, count);
        }
        return out;
    };
    // Each rank's RoPE positions of those columns (contiguous copies of three-axis ones).
    const auto rope_of = [&](std::int32_t first, std::int32_t count) {
        RankTensors out;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            std::optional<ScopedCurrentDevice> current;
            if (width == 2) { current.emplace(qwen4_device(rank)); }
            out[r] = rope_columns(rope_positions[r], first, count, *arenas[r], rank_stream(rank));
        }
        return out;
    };
    const RankTensors rank_ids = qwen4_rank_ids(ids);
    // Each rank's composed shifted input (multimodal chunks whose shifted window is visual).
    const RankScopes visual_scope(arenas);
    RankTensors embeddings;
    if (visual != nullptr) {
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            std::optional<ScopedCurrentDevice> current;
            if (width == 2) { current.emplace(qwen4_device(rank)); }
            embeddings[r] = arenas[r]->alloc(DType::BF16, {dimension(config_.hidden_size), T});
            compose_mtp_embeddings(rank_ids[r], rank_parameters(rank).text.token_embedding,
                                   visual->merged[r], visual->columns, visual->source_begin,
                                   *arenas[r], embeddings[r], rank_stream(rank));
        }
    }
    const auto inputs = [&](std::int32_t first, std::int32_t count, RankTensors& slot) {
        if (visual == nullptr) { return static_cast<const RankTensors*>(nullptr); }
        slot = columns(embeddings, 1, first, count);
        return static_cast<const RankTensors*>(&slot);
    };
    if (runtime[0]->taps != nullptr && runtime[0]->taps->wants_mtp_logits()) {
        // Teacher forcing (validation, eager): every column runs the whole layer, one at a time
        // (causal, so equal to the batched append), and its draft logits are captured.
        if (width == 2) {
            throw std::invalid_argument("Qwen4Exp MTP teacher forcing runs at one device");
        }
        for (std::int32_t c = 0; c < append; ++c) {
            const RankScopes scope(arenas);
            const RankTensors pos  = columns(positions, 0, c, 1);
            const RankTensors rope = rope_of(c, 1);
            RankTensors out{work_.alloc(DType::BF16, {wide_rows, 1}), Tensor{}};
            Tensor draft = work_.alloc(DType::I32, {1});
            {
                RankTensors slot;
                qwen4_mtp_columns(mtp, columns(rank_ids, 0, c, 1), columns(hidden, 1, c, 1), out,
                                  pos, rope, 1, false, max_visible, inputs(c, 1, slot));
            }
            Tensor logits = io_.logits.slice(1, 0, 1);
            qwen4_mtp_propose(out, logits, draft);
            std::int32_t position = 0;
            CUDA_CHECK(cudaMemcpyAsync(&position, pos[0].data, sizeof(position),
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            runtime[0]->taps->capture_mtp_logits(logits, position, s);
        }
    } else if (append > 0) {
        const RankScopes scope(arenas);
        const RankTensors rope = rope_of(0, append);
        RankTensors scratch;
        for (int rank = 0; rank < width; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            scratch[r]   = arenas[r]->alloc(DType::BF16, {wide_rows, append});
        }
        RankTensors slot;
        qwen4_mtp_columns(mtp, columns(rank_ids, 0, 0, append), columns(hidden, 1, 0, append),
                          scratch, columns(positions, 0, 0, append), rope, append, true,
                          max_visible, inputs(0, append, slot));
    }
    if (!propose) { return; }
    RankTensors ar_hidden{io_.mtp->ar_hidden, width == 2 ? tp_->mtp->ar_hidden : Tensor{}};
    {
        const RankTensors rope = rope_of(T - 1, 1);
        RankTensors slot;
        qwen4_mtp_columns(mtp, columns(rank_ids, 0, T - 1, 1), columns(hidden, 1, T - 1, 1),
                          ar_hidden, columns(positions, 0, T - 1, 1), rope, 1, false, max_visible,
                          inputs(T - 1, 1, slot));
    }
    qwen4_mtp_drafts(mtp, ar_hidden, max_visible);
}

void TextContext::qwen4_logits_from_hidden(const Tensor& hidden, const Tensor* peer_hidden,
                                           Tensor& logits) {
    const int width              = qwen4_width();
    const std::int32_t wide_rows = dimension(config_.residual_width());
    const std::int32_t H         = dimension(config_.hidden_size);
    require_shape(hidden, DType::BF16, {wide_rows, 1}, "retained hidden");
    require_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "retained logits");
    const ScopedCurrentDevice device(ctx_.device);
    const auto arenas = qwen4_arenas();
    const RankScopes scope(arenas);
    RankTensors wide{hidden, Tensor{}};
    if (width == 2) {
        // The residual is replicated across the ranks: rank 1 reads its own retained copy, or a
        // copy of rank 0's ordered after rank 0's stream.
        if (peer_hidden != nullptr) {
            require_shape(*peer_hidden, DType::BF16, {wide_rows, 1}, "rank 1 retained hidden");
            wide[1] = *peer_hidden;
        } else {
            wide[1] = tp_->work->alloc(DType::BF16, {wide_rows, 1});
            CUDA_CHECK(cudaEventRecord(tp_->events->inputs_ready(0), ctx_.stream));
            const ScopedCurrentDevice rank1(qwen4_device(1));
            CUDA_CHECK(cudaStreamWaitEvent(rank_stream(1), tp_->events->inputs_ready(0), 0));
            CUDA_CHECK(::ninfer::memcpy_device_async(wide[1].data, hidden.data, wide[1].bytes(),
                                                     rank_stream(1)));
        }
    }
    std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
    RankTensors x;
    for (int rank = 0; rank < width; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        ranks[r]     = qwen4_rank(rank, {.width = 1, .batch = 1, .prefill = false},
                                  {.kv      = nullptr,
                                   .linear  = &rank_state(rank),
                                   .runtime = rank == 0 ? qwen4_runtime_ : nullptr});
        x[r]         = arenas[r]->alloc(DType::BF16, {H, 1});
    }
    Qwen4Forward<WorkspaceArena> forward = qwen4_forward(ranks, 1);
    forward.head(wide, x);
    qwen4_logits(x, logits);
}

} // namespace ninfer::models::qwen3_5::execution
