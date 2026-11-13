#pragma once

// The Text forward of every model, at one device or at tensor-parallel width 2. Qwen4Exp
// (Qwen3.8-Flash-Next, dense Qwen4): hyper-connection residual, QSA attention with its index
// planes and GDN mixers, PLE before block 1 (optional), the FFN - the router +
// shared expert + routed experts (moe_route.h, qwen4_experts.h) or a dense SwiGLU FFN - and the
// output mixer. Mathematics: docs/maintainer/qwen4-exp-model.md and, for the plain residual of a
// dense Qwen3.5 model (Qwen3.8-27B), docs/maintainer/qwen3_5-model.md.
//
// Explicit options of the one schedule, each selected by the configuration and the bound
// parameters:
//   * residual: the hyper-connection streams, or one plain [H,T] residual without them (the
//     Qwen3.5 block: input/post-attention RMSNorms, projections added into the residual, the
//     final-normed hidden carried; run_plain);
//   * full attention: QSA with its indexer, or dense gated attention (dense_attention);
//   * projection routes (Qwen4FusedRoutes): the 27B's fused Ops where the rank's geometry and
//     format have a registered route, else the generic Linear chain.
// A dense Qwen3.5 model takes the plain residual, dense gated attention and the 27B's fused Ops;
// its MTP head is the Qwen3.5 one (TextContext's mtp_* entry points).
// Execution order follows ninfer-ext's qwen4_text.cpp (giveen/ninfer-ext @9c933f01, Apache-2.0)
// with this tree's Ops.
//
// One template serves execution (Arena = WorkspaceArena: allocates and launches) and startup
// sizing (Arena = WorkspaceLayoutBuilder: the same allocations in the same scopes, each Op
// replaced by its declared scratch), so the planned workspace is the executed one by
// construction. In planning mode T is the interval's largest column count and every Linear
// reserves its capacity over [first, last].
//
// Tensor parallelism (width 2, load/sharding.h placements): every rank holds the whole wide
// residual (the hyper-connection mixers, the router, the QSA indexer and the PLE are replicated)
// and issues the same replicated work on its own stream; the per-rank shards are the QSA heads
// (12 query + 1 KV), the GDN heads (8 key + 24 value), the vocabulary half of the output head,
// the routed experts of its ID parity and the shared expert of its blocks (block % 2). Each block
// has exactly two exchanges:
//   * the mixer output: each rank's row-parallel output projection, one BF16 all-reduce
//     (ops::linear_row_parallel), so both ranks hold the identical BF16 y;
//   * the MoE: each rank's FP32 partial (its routed experts from its Qwen4RoutedExperts backend,
//     plus on the owner rank the gated shared expert, ops::moe_accumulate_shared), one FP32
//     all-reduce rounded to BF16 once (ops::allreduce_sum_f32); a dense FFN instead splits its
//     intermediate rows (gate|up rows, down columns) and its down projection is row-parallel like
//     the mixer output (output_projection).
// The router reads identical bits on both ranks (deterministic replicated kernels), so both ranks
// route every token to the same experts and the partition by ID parity is exact. At width 1 the
// schedule is the single-device one, unchanged.
//
// This header declares the forward. Its members are defined by responsibility in qwen4_forward/
// (core, residual, plain, projection, attention, qsa, gdn, ffn, moe, mtp) and instantiated once
// for execution and once for planning in qwen4_forward.cpp; callers include this header only.

#include "core/device.h"
#include "core/device_scope.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/qwen4_runtime.h"
#include "models/qwen3_5/execution/tp2/exchange_prefetch.h"
#include "models/qwen3_5/execution/tp2/prefill_slices.h"
#include "models/qwen3_5/execution/workspace.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/moe_grouped.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::models::qwen3_5::execution {

// The columns one forward processes: W columns of B sequences (prefill: W = T, B = 1).
struct Qwen4Columns {
    std::int32_t width            = 0;
    std::int32_t batch            = 0;
    const Tensor* cache_positions = nullptr; // I32 [T]
    const Tensor* rope_positions  = nullptr; // I32 [T] or [T,3] (axis-major)
    const Tensor* valid_columns   = nullptr; // null or I32 [B]
    // MTP decode (T52): the MoE routes only each lane's last valid column (the alignment of a
    // verified window is read only at the accepted column, ops::MoeRouteColumns::LastValid).
    bool moe_last_valid_only        = false;
    const Tensor* table_rows        = nullptr; // I32 [B]
    std::uint32_t max_visible       = 0;       // envelope: largest visible extent of any column
    bool prefill                    = true;    // prefill: scalar state slots; else [B] selectors
    std::int32_t source_slot        = 0;
    std::int32_t destination_slot   = 0;
    const Tensor* source_slots      = nullptr; // I32 [B]
    const Tensor* destination_slots = nullptr; // I32 [B]
    std::int32_t first_position     = 0;       // absolute position of column 0 (taps)
    // Speculative verification (W = K+1 columns per lane): GDN writes ReplaySSM records instead
    // of advancing its state and PLE copies its convolution inputs to Qwen4Runtime::ple_record;
    // the commit folds the accepted prefix. QSA K/V and index planes are appended in place and
    // rejected columns are overwritten by the next round (position-ordered re-append).
    const GdnReplayRecords* replay_records = nullptr;
    // Dense full attention (a model without QSA): the causal envelope of the call's keys (required
    // when executing). Planning sizes its scratch with {1, max_visible} and the KV storage.
    const ops::CausalAttentionExecutionEnvelope* envelope = nullptr;
    KvCacheStorage kv_storage                             = KvCacheStorage::BFloat16;
    // Planning only: the text pool may hold active KV in RAM (a decode window then allocates the
    // rewritten selection). Execution reads the pool itself (PagedKVCache::qsa_staging).
    bool kv_ram_active = false;
};

struct Qwen4State {
    const PagedKVCache* kv           = nullptr;
    LinearAttentionStatePool* linear = nullptr;
    const Qwen4Runtime* runtime      = nullptr;
};

// One rank's operands of a Qwen4Forward. `config` is the rank's geometry (shard_text_config at
// tp 2, the logical config at one device); `device` is only read at tp 2.
template <class Arena>
struct Qwen4Rank {
    const TextConfig* config     = nullptr;
    const Parameters* parameters = nullptr;
    Arena* work                  = nullptr;
    cudaStream_t stream          = nullptr;
    DeviceExecutionView execution{};
    int device = 0;
    Qwen4Columns columns;
    Qwen4State state;
};

struct GdnGatingOverlap; // tp2/gdn_gating_overlap.h

// The two-device transport of an executing tp 2 forward.
struct Qwen4Collectives {
    const ExecutionContext* execution = nullptr;
    const ops::PeerEvents* events     = nullptr;
    // Program-owned side streams of the tp 2 GDN gating (tp2/gdn_gating_overlap.h); null keeps
    // the gating on the main streams.
    GdnGatingOverlap* gating = nullptr;
};

template <class Arena>
class Qwen4Forward {
public:
    static constexpr bool kExecute = std::is_same_v<Arena, WorkspaceArena>;
    using Rank                     = Qwen4Rank<Arena>;
    using RankTensors              = std::array<Tensor, 2>;

    // One device.
    Qwen4Forward(const TextConfig& config, const Parameters& parameters, Arena& work,
                 cudaStream_t stream, DeviceExecutionView execution, Qwen4Columns columns,
                 Qwen4State state, std::int32_t first, std::int32_t last)
        : Qwen4Forward(config,
                       std::array<Rank, 2>{Rank{.config     = &config,
                                                .parameters = &parameters,
                                                .work       = &work,
                                                .stream     = stream,
                                                .execution  = execution,
                                                .device     = 0,
                                                .columns    = columns,
                                                .state      = state},
                                           Rank{}},
                       1, {}, first, last) {}

    // `width` ranks of `ranks` (rank 0 first). `config` is the logical Text config; at width 2
    // `collectives` must be complete when executing (planning ignores it).
    Qwen4Forward(const TextConfig& config, std::array<Rank, 2> ranks, int width,
                 Qwen4Collectives collectives, std::int32_t first, std::int32_t last);

    [[nodiscard]] std::int32_t wide_rows() const { return dimension(config_.residual_width()); }

    [[nodiscard]] std::int32_t hidden() const { return dimension(config_.hidden_size); }

    [[nodiscard]] std::int32_t hc_rank() const {
        return dimension(config_.hyper_connection->hc_lowrank);
    }

    // Streams of the residual: the hyper-connection count, or 1 for a model without
    // hyper-connections (one [H,T] residual, the Qwen3.5 block).
    [[nodiscard]] std::int32_t streams() const {
        return has_hc() ? dimension(config_.hyper_connection->hc_count) : 1;
    }

    [[nodiscard]] bool has_hc() const { return config_.hyper_connection.has_value(); }

    // A drafter's layer-output taps for the next run() (plain residual only; null = none).
    void set_feature_tap(Qwen4FeatureTap* tap) { feature_tap_ = tap; }

    // The visual columns of a multimodal prefill chunk, per rank: the merged Vision embeddings
    // BF16 [H, V] and the chunk-relative columns I32 [V] whose token embedding they replace before
    // the residual is widened (inputs_embeds.masked_scatter, then one copy per stream). The hidden
    // axis is replicated at tp 2: each rank scatters its own copy of the same embeddings.
    struct VisualColumns {
        RankTensors embeddings;
        RankTensors indices;
    };

    // One device: ids I32 [T]; ple_embedding BF16 [embed_dim, T] (when the model has PLE);
    // wide_out BF16 [S*H, T] receives the committed residual after the last block: the carried
    // hidden of the round (the continuation hidden MTP consumes). head() turns it into the output
    // head's input.
    void run(const Tensor& ids, const Tensor* ple_embedding, Tensor& wide_out,
             const VisualColumns* visual = nullptr);

    // Per rank: ids, the PLE n-gram embedding and the wide output, each resident on its rank.
    // Both ranks' outputs hold the same (replicated) committed residual. The PLE n-gram rows of
    // visual columns hash the placeholder token ids, as the reference (ple_input_ids = input_ids).
    void run(const RankTensors& ids, const RankTensors* ple_embedding, RankTensors& wide_out,
             const VisualColumns* visual = nullptr);

    // The Text output mixer over carried hiddens on every rank: wide BF16 [S*H, N] -> x BF16
    // [H, N], the output head's input (replicated at tp 2). N is the caller's column count.
    void head(const RankTensors& wide, RankTensors& x);

    // The same with the given output mixer per rank (the Text or the MTP layer's).
    // Without hyper-connections the carried hidden is already the final-normed one: `x` becomes
    // a view of `wide` (no work is issued).
    void head(std::array<const HyperConnectionParameters*, 2> hc, const RankTensors& wide,
              RankTensors& x);

    // One device.
    void head(const HyperConnectionParameters& hc, const Tensor& wide, Tensor& x);

    // MTP layer (docs/maintainer/qwen4-exp-model.md#mtp) over T columns, with this forward's
    // state bound to the MTP KV cache (layer 0, QSA index planes) and the MTP expert backend:
    //   e   = embedding_projection . offset_rmsnorm(embed(ids), embedding_norm)
    //   h   = offset_rmsnorm(hidden, hidden_norm)                 (one RMS over S*H)
    //   R'  = [hidden_projection . h_s + e for each stream s]
    //   R'  = QSA_block(R')
    // ids I32 [T] (the shifted tokens), hidden BF16 [S*H, T] the target's carried hiddens,
    // embeddings optional BF16 [H, T] replacing embed(ids). `wide_out` BF16 [S*H, T] receives R'.
    // `append_only` stops after the block's QSA append (prefill KV of the prompt columns whose
    // proposals are not needed); `wide_out` is then scratch. Per rank at tp 2 (each rank's MTP
    // Parameters: its QSA heads, its expert partition, the shared expert on one rank).
    void run_mtp(std::array<const Qwen4MtpParameters*, 2> m, const RankTensors& ids,
                 const RankTensors* embeddings, const RankTensors& hidden_wide,
                 RankTensors& wide_out, bool append_only);

    // One device.
    void run_mtp(const Qwen4MtpParameters& m, const Tensor& ids, const Tensor* embeddings,
                 const Tensor& hidden_wide, Tensor& wide_out, bool append_only);

private:
    // ---- Operands and arena (qwen4_forward/core.h) ----------------------------------------------
    [[nodiscard]] static std::size_t at(int r) { return static_cast<std::size_t>(r); }

    [[nodiscard]] const Parameters& parameters(int r) const { return *r_[at(r)].parameters; }

    [[nodiscard]] const Qwen4Parameters& qwen4(int r) const { return *parameters(r).qwen4; }

    // Text block `layer`, or the MTP layer's block at layer == mtp_layer() (run_mtp).
    [[nodiscard]] const Qwen4BlockParameters& block(int r, std::size_t layer) const {
        if (layer == mtp_layer()) {
            const Qwen4MtpParameters* m = mtp_[at(r)];
            if (m == nullptr) { throw std::logic_error("Qwen4Exp MTP block outside run_mtp"); }
            return m->layer;
        }
        return qwen4(r).layers[layer];
    }

    // The MTP layer's index: after the Text layers (the routed-expert backends' layer number).
    [[nodiscard]] std::size_t mtp_layer() const { return config_.layer_types.size(); }

    // The model's FFN kind: the routed-expert MoE, or a dense SwiGLU FFN.
    [[nodiscard]] bool moe_ffn() const { return config_.qwen4_routed_experts(); }

    // The FFN of Text block `layer` on every rank (null past the width).
    [[nodiscard]] std::array<const Qwen4FfnParameters*, 2> block_ffns(std::size_t layer) const;

    // Every rank's FFN has the model's kind; a MoE's shared expert is whole at one device and on
    // exactly one rank at tp 2.
    void require_ffn(std::array<const Qwen4FfnParameters*, 2> ffn, const std::string& name) const;

    [[nodiscard]] const char* phase() const {
        const Qwen4Columns& col = r_[0].columns;
        return col.prefill ? "prefill" : (col.replay_records != nullptr ? "verify" : "decode");
    }

    // The residual buffers: the caller's [S*H, T] outputs when executing; planning allocates them
    // in the caller's scope, so only execution needs the tensors.
    RankTensors wide_buffers(RankTensors& wide_out, std::int32_t T);

    [[nodiscard]] Arena& work(int r) const { return *r_[at(r)].work; }

    [[nodiscard]] cudaStream_t s(int r) const { return r_[at(r)].stream; }

    [[nodiscard]] const Qwen4Columns& c(int r) const { return r_[at(r)].columns; }

    [[nodiscard]] const Qwen4State& state(int r) const { return r_[at(r)].state; }

    // The per-rank geometry (QSA and GDN heads); identical on both ranks.
    [[nodiscard]] const TextConfig& rc() const { return *r_[0].config; }

    [[nodiscard]] std::int32_t columns() const {
        if constexpr (kExecute) {
            return r_[0].columns.width * r_[0].columns.batch;
        } else {
            return last_;
        }
    }

    // Runs body(r) for each rank, with that rank's device current at tp 2 when executing. The
    // calls only enqueue, so the ranks' work overlaps on the devices.
    template <class Body>
    void each(Body&& body) {
        if constexpr (kExecute) {
            if (width_ == 2) {
                const ScopedCurrentDevice restore;
                for (int r = 0; r < width_; ++r) {
                    ScopedCurrentDevice::select(r_[at(r)].device);
                    body(r);
                }
                return;
            }
        }
        for (int r = 0; r < width_; ++r) { body(r); }
    }

    struct Scopes {
        std::array<std::optional<typename Arena::Scope>, 2> held;
    };

    [[nodiscard]] Scopes scopes() {
        Scopes out;
        for (int r = 0; r < width_; ++r) { out.held[at(r)].emplace(work(r).scope()); }
        return out;
    }

    [[nodiscard]] RankTensors alloc(DType dtype, std::initializer_list<std::int32_t> shape) {
        RankTensors out;
        for (int r = 0; r < width_; ++r) { out[at(r)] = work(r).alloc(dtype, shape); }
        return out;
    }

    // Planning: scratch of `bytes` on `rank` (every rank when negative).
    void reserve(std::size_t bytes, int rank = -1) {
        if constexpr (!kExecute) {
            if (bytes == 0) { return; }
            for (int r = 0; r < width_; ++r) {
                if (rank >= 0 && rank != r) { continue; }
                auto scope = work(r).scope();
                (void)work(r).alloc_bytes(bytes);
            }
        }
    }

    // Columns [first, first + count) of each rank's operand.
    [[nodiscard]] RankTensors window(const RankTensors& t, std::int32_t first,
                                     std::int32_t count) const {
        RankTensors out;
        for (int r = 0; r < width_; ++r) { out[at(r)] = t[at(r)].slice(1, first, count); }
        return out;
    }

    // The teacher-forcing taps of a prompt chunk when they ask for `layer` (-1: the output head's
    // input); null otherwise and in planning.
    [[nodiscard]] Qwen4Taps* prompt_taps(std::int32_t layer) const;

    // Hands `taps` rank 0's copy of `x` (with rank 0's device current) and, at tp 2, both ranks'
    // copies to compare.
    void capture_taps(Qwen4Taps& taps, std::int32_t layer, const RankTensors& x);

    // Rank r's Linear; in planning its scratch over [first, last].
    void linear(int r, const Tensor& x, const LinearParameters& p, Tensor& out);

    // Copy of rows [row0, row0+rows) into `destination` [rows, T] (always a copy).
    void copy_rows(int r, const Tensor& source, std::int32_t row0, std::int32_t rows,
                   const Tensor& destination);

    // Rows [row0, row0+rows) of a contiguous [R,T] matrix into `destination` [rows,T]; one column
    // is contiguous already and is returned as a view.
    Tensor split_rows(int r, const Tensor& source, std::int32_t row0, std::int32_t rows,
                      const Tensor& destination);

    // ---- Closing projections and the tp 2 exchanges (qwen4_forward/projection.h) ----------------
    // A block's closing row-parallel projection (mixer output or dense FFN down): at one device a
    // Linear; at tp 2 each rank contracts its head block through its column shard and one BF16
    // all-reduce leaves the identical sum on both. `site` labels the exchange for
    // Qwen4Taps::capture_exchange (0 = mixer output, 1 = dense FFN down).
    void output_projection(const RankTensors& x, std::array<const LinearParameters*, 2> p,
                           RankTensors& y, const RankTensors& staging, std::int32_t layer = -1,
                           int site = 0);

    // The closing projection of a block: without hyper-connections added into the residual x (one
    // device: linear_add; tp 2: row-parallel with the residual folded into its all-reduce), with
    // them the block output y (output_projection).
    void close(const RankTensors& in, std::array<const LinearParameters*, 2> p, RankTensors& out,
               const RankTensors& staging, std::int32_t layer, int site);

    // T128: the taps of the exchanges, when a teacher-forcing dump asks for them (tp 2 prefill).
    [[nodiscard]] Qwen4Taps* exchange_taps() const;

    // At tp 2 the mixer output exchange carries the ranks' FP32 partials (an FP16 payload in prompt
    // chunks, FP32 in decode) and rounds the sum to BF16 once, as one device does;
    // NINFER_TP_MIX=bf16 exchanges BF16 partials instead. Read once.
    static bool mix_fp32_partials();

    // ---- Hyper-connection residual and PLE (qwen4_forward/residual.h) ---------------------------
    struct HcStreams;

    // The token embeddings (visual columns scattered over them) widened to every stream of `wide`.
    void embed_streams(const RankTensors& ids, const VisualColumns* visual, RankTensors& wide);

    // One hyper-connection block: PLE (its layer), the attention mixer, the mixer (GDN or QSA),
    // the FFN mixer that commits the mixer output, the FFN; a prompt chunk's taps after it.
    void hc_block(std::size_t layer, const RankTensors* ple_embedding, HcStreams& st);

    // Folds the pending block output into the wide residual (none pending: nothing).
    void hc_commit(HcStreams& st);

    // Mixes the residual into the next block input with the block's `member` mixer, committing
    // the pending block output in the same pass when there is one (T10's combine_mix).
    void hc_mix(HcStreams& st, HyperConnectionParameters Qwen4BlockParameters::* member,
                std::size_t layer);

    // PLE (block 1, before its attention mixer), replicated on every rank.
    void ple(std::size_t layer, const RankTensors& embedding, RankTensors& wide);

    // ---- Plain residual: the Qwen3.5 block (qwen4_forward/plain.h) ------------------------------
    // The plain residual's forward (run() without hyper-connections): `out` receives the
    // final-normed residual, the carried hidden.
    void run_plain(const RankTensors& ids, RankTensors& out, const VisualColumns* visual);

    // A plain block's mixer/FFN input: RMSNorm of the residual by the block's `member` weight into
    // `h` (each rank its own copy).
    void plain_norm(std::size_t layer, Tensor Qwen4BlockParameters::* member, const RankTensors& x,
                    RankTensors& h);

    // The tp 2 decode exchange prefetch budget (0 at one device, in planning, eager or off).
    [[nodiscard]] std::size_t exchange_prefetch(const RankTensors& x) const;

    void stage_prefetch(const RankTensors& x, const std::array<ExchangePrefetch, 2>& hint);

    // The decode exchange hints of a plain block (tp2/exchange_prefetch.h): the mixer's all-reduce
    // prefetches the layer's gate|up, the FFN's the next layer's input projection (after the last
    // layer, the output head).
    [[nodiscard]] std::array<ExchangePrefetch, 2> ffn_input_hint(std::size_t layer,
                                                                 std::size_t bytes) const;

    [[nodiscard]] std::array<ExchangePrefetch, 2> next_input_hint(std::size_t layer,
                                                                  std::size_t bytes) const;

    // The width of the first half of a tp 2 prompt chunk under the half-chunk overlap, or 0:
    // prefill_overlap_split's widths, the split exchange available, not under
    // NINFER_TP_PREFILL_SLICES=force, a one-axis RoPE chunk (a multimodal chunk's three-axis
    // positions are not column-sliceable) and a feature tap, if any, that takes column windows (not
    // a batched-feature tap). The teacher-forcing taps (Qwen4Taps) read whole layers, so they keep
    // the chunk off the halves too.
    [[nodiscard]] std::int32_t prefill_overlap(std::int32_t T) const;

    // The slices of a tp 2 prompt chunk of a plain residual: the split exchange is available and
    // the slices are on (NINFER_TP_PREFILL_SLICES / NINFER_TP_PREFILL_OVERLAP). Empty otherwise.
    [[nodiscard]] PrefillSlicePlan prefill_slices(std::int32_t T) const;

    // Completes every lane the block's slices began (before the residual is read again).
    void finish_slices();

    // One half of a tp 2 prompt chunk under the half-chunk overlap: its columns of the residual,
    // the all-reduce staging, the cache and RoPE positions, its causal envelope, and the state of
    // its lane (the half's index): whether the lane holds a begun all-reduce, and the layer whose
    // output that all-reduce completes (tapped once it has finished).
    struct PlainHalf {
        RankTensors x, staging, positions, rope;
        ops::CausalAttentionExecutionEnvelope envelope{};
        std::int32_t first = 0;
        bool open          = false;
        int tapped_layer   = -1;
    };

    // The halves [0, split) and [split, T) of the current prompt chunk.
    [[nodiscard]] std::array<PlainHalf, 2>
    plain_halves(const RankTensors& x, const RankTensors& staging, std::int32_t split) const;

    // Runs one block of `half` on lane `lane`: the half's previous exchange completes first (the
    // block reads and rewrites its columns), then `body` with the half's columns bound in place of
    // the chunk's - positions, envelope and GDN slots: half 0 reads the chunk's source slot, half 1
    // the destination half 0 published - and its closing projection only begins its all-reduce on
    // the lane. `tapped_layer` >= 0 marks the block that ends that layer.
    template <class Body>
    void half_block(PlainHalf& half, int lane, Body&& body, int tapped_layer = -1);

    // Completes the all-reduce `half` began on `lane`, then hands its feature tap the layer that
    // exchange closed (its columns only).
    void finish_half(PlainHalf& half, int lane);

    // After a failure, leaves no lane open for the next call (the request fails either way): the
    // slices' lanes and the halves' begun all-reduces, without their taps. A failure inside the
    // slice loop leaves some slices' lanes already finished, so each lane is finished on its own:
    // finishing a closed one throws, which must not skip the open ones after it.
    void abandon_lanes(std::array<PlainHalf, 2>& halves) noexcept;

    // A plain block's mixer (dense gated attention or GDN), its output added into x.
    void plain_mixer(std::size_t layer, RankTensors& x, const RankTensors& staging);

    // One plain block over the whole call: its mixer and FFN (each closing exchange whole or by
    // slices), then the layer's feature and teacher-forcing taps.
    void plain_block(std::size_t layer, RankTensors& x, const RankTensors& staging);

    // One plain block over the two halves of a prompt chunk: mixer A, mixer B, FFN A, FFN B, each
    // beginning its closing all-reduce on its half's lane.
    void plain_half_blocks(std::size_t layer, std::array<PlainHalf, 2>& halves);

    // ---- Dense gated attention (qwen4_forward/attention.h) --------------------------------------
    // Dense gated full attention of a plain block (no QSA) over each rank's query/KV heads.
    // `in` is the residual, normalized here by input_norm; the output projection is added into the
    // residual (close()). A plain block's attention input projection is always fused.
    void dense_attention(std::size_t layer, const RankTensors& in, RankTensors& y,
                         std::int32_t fidx, const RankTensors& staging);

    // Rank r's attention core: q/k RMSNorm and RoPE, causal attention over its KV heads within
    // the call's envelope, the sigmoid output gate.
    void dense_attention_rank(std::size_t layer, int r, std::int32_t fidx, const RankTensors& q,
                              const RankTensors& gate, const RankTensors& k, const RankTensors& v,
                              RankTensors& qn, RankTensors& kn, RankTensors& attended);

    // ---- QSA (qwen4_forward/qsa.h) --------------------------------------------------------------
    // Gated attention over the indexer's selection. Each rank: its query/KV heads, the replicated
    // indexer (identical selection on both ranks). `y` null: append K/V and the index records only
    // (MTP prompt columns).
    void qsa(std::size_t layer, const RankTensors& x, RankTensors* y, std::uint32_t kv_layer,
             const RankTensors& staging);

    // The fused QSA input projection (q|k|gate|v parent of 24 query / 4 KV heads of 256 over
    // hidden 5120, Qwen4FusedRoutes::attn_input): attn_input_proj at one device, its column-
    // parallel form over each rank's heads at tp 2.
    void qsa_input_projection(std::size_t layer, const RankTensors& x, RankTensors& q,
                              RankTensors& gate, RankTensors& k, RankTensors& v);

    // Active KV in RAM (docs/maintainer/active-kv-in-ram.md): whether the text pool's K/V planes
    // live in Host memory behind a staging region. Planning follows the columns' flag, execution
    // the bound pool (the MTP layer's pool never has one).
    [[nodiscard]] bool kv_in_host() const;

    struct QsaBuffers;

    // Allocates one call's operands on every rank (`attend` false: append only).
    QsaBuffers qsa_buffers(std::size_t layer, bool attend);

    // Rank r's QSA after the projections: q/k norm and RoPE, the K/V and index append and, when
    // `attend`, the selection, the attention and the output gate.
    void qsa_rank(std::size_t layer, int r, QsaBuffers& b, std::uint32_t kv_layer, bool attend);

    // Rank r's attention over the selected blocks: from the paged KV, or from the Host planes
    // staged into the VRAM region (active KV in RAM).
    void qsa_attend(int r, QsaBuffers& b, const PagedKVBatchLayerView& kv, const Tensor& positions,
                    const Tensor& valid, std::uint32_t kv_layer);

    // NINFER_KVRAM_STATS=1: count the staging cache's outcomes and print them before each prefill
    // (synchronizes; diagnostics only). Read once.
    static bool staging_stats();

    // ---- GDN (qwen4_forward/gdn.h) --------------------------------------------------------------
    // Gated DeltaNet over each rank's key/value heads, conv channels and state slots.
    // `in` is the mixed hidden (hyper-connections) or the residual, normalized here by input_norm;
    // the output projection goes to `y` or is added into the residual (close()). Fused routes
    // (Qwen4FusedRoutes): the gating projection (fused with the input RMSNorm for a plain block at
    // one device, gdn_norm_control) and the input projection with its convolution (gdn_projection,
    // gdn_projection_snapshot / _record and their tp 2 split forms, gdn.cpp / tp2/gdn_split.cpp).
    // Otherwise the generic chain: Linear, gdn_gating on the
    // a|b rows in place, the row copies and the projected-convolution Ops.
    void gdn(std::size_t layer, const RankTensors& in, RankTensors& y, std::int32_t gidx,
             const RankTensors& staging);

    // The input norm (plain residual) into `h` and the gating controls (decay, beta).
    void gdn_controls(std::size_t layer, const RankTensors& in, RankTensors& h, RankTensors& decay,
                      RankTensors& beta);

    // The generic GDN input projection (one Linear) and its convolution; `zs` receives the z the
    // output norm reads (a view of the projection for one column).
    void generic_gdn_input(std::size_t layer, const RankTensors& x, std::uint32_t gl,
                           std::int32_t gidx, const RankTensors& z, RankTensors& zs,
                           RankTensors& qc, RankTensors& kc, RankTensors& vc);

    // The gated delta recurrence over each rank's heads (prompt chunk, decode update or ReplaySSM
    // record) and the gated output norm into `on`.
    void gdn_recurrence(std::size_t layer, std::uint32_t gl, std::int32_t gidx, RankTensors& qc,
                        RankTensors& kc, RankTensors& vc, RankTensors& decay, RankTensors& beta,
                        const RankTensors& zs, RankTensors& o, RankTensors& on);

    // The fused GDN input projection with its convolution: a prompt chunk projects q|k|v and z
    // (gdn_projection) and convolves from the source slot into the destination slot
    // (causal_conv1d_silu_split); a decode window projects and convolves its [W,B] columns in one
    // Op that publishes the snapshot (ordinary decode) or writes the ReplaySSM record
    // (verification). tp 2: the split forms over each rank's heads.
    void fused_gdn_input(std::size_t layer, const RankTensors& x, std::uint32_t gl,
                         std::int32_t gidx, RankTensors& z, RankTensors& qc, RankTensors& kc,
                         RankTensors& vc);

    // The decode window of fused_gdn_input.
    void fused_gdn_window(std::size_t layer, const RankTensors& x, std::uint32_t gl,
                          std::int32_t gidx, RankTensors& z, RankTensors& qc, RankTensors& kc,
                          RankTensors& vc);

    // ---- FFN: dispatch, dense SwiGLU, the plain block's FFN (qwen4_forward/ffn.h) ---------------
    // The FFN of a hyper-connection block: `wide` holds the committed residual (its FFN mixer
    // input).
    void ffn(std::size_t layer, const RankTensors& wide, const RankTensors& x, RankTensors& y,
             const RankTensors& staging_bf16, const RankTensors& staging_fp32);

    // Dense SwiGLU FFN: y = down . (silu(gate . x) * (up . x)).
    // gate|up is one projection ([2I, H], gate rows first); at tp 2 each rank holds its half of the
    // intermediate rows (rc().ffn) and of the down columns, and the down projection is row-parallel
    // like a mixer's output (one all-reduce leaves the identical BF16 y on both ranks). The fused
    // route (Qwen4FusedRoutes::swiglu) projects and SiLU-multiplies in one Op (linear_swiglu, tp 2
    // its column-parallel form); the generic chain is Linear then silu_mul. Without
    // hyper-connections `x` is the post-attention-normed residual and the down projection is added
    // into the residual `y` (close()). `layer` is a Text layer or mtp_layer().
    void dense(std::size_t layer, const RankTensors& x, RankTensors& y, const RankTensors& staging);

    // The FFN of a plain block: RMSNorm by post_attention_norm, SwiGLU, down added into x.
    // The fused route always: dense_ffn (one device) and dense_ffn_split (tp 2).
    void plain_ffn(std::size_t layer, RankTensors& x, const RankTensors& staging);

    // ---- MoE (qwen4_forward/moe.h) --------------------------------------------------------------
    // Router, shared expert, routed experts (FP32 partial), one rounding.
    // At tp 2 each rank routes the same tokens (replicated router), runs the routed experts of its
    // partition into its FP32 partial, the shared expert's owner adds the gated shared expert in
    // FP32, and one FP32 all-reduce rounds the block output to BF16 once on both ranks.
    // `layer` is a Text layer or mtp_layer() (the MTP layer's backend and bank).
    void moe(std::size_t layer, const RankTensors& x, RankTensors& y, const RankTensors& staging);

    struct MoeBuffers;

    // Every rank's routing, the start of its routed experts (the backend may begin them elsewhere
    // at once) and, on the shared expert's owner, the shared expert.
    void moe_route_shared(std::size_t layer, const RankTensors& x, MoeBuffers& b);

    // Execution: the routed experts into each rank's FP32 partial, the shared expert added, and
    // the one BF16 rounding into `y` (tp 2: after the all-reduce or the pipelined exchange).
    void moe_combine(std::size_t layer, const RankTensors& x, MoeBuffers& b, RankTensors& y,
                     const RankTensors& staging);

    // T25 router-ahead prefetch: `wide` holds layer's committed residual (its MLP
    // hyper-connection input); the backend may predict layer+1's experts from it. Decode rounds at
    // one device only (the offload backend's prefetch); a no-op for the other backends.
    void ahead(std::size_t layer, const RankTensors& wide);

    // The pipelined MoE exchange of a prompt chunk at tp 2, read once: an FP16 payload (the bytes
    // of BF16 with an 8x smaller rounding of the partial sums) by column groups of 2048, which
    // overlap the routed experts of the later columns. NINFER_TP_AR=0 takes one allreduce_sum_f32
    // call (FP32) instead; NINFER_TP_AR_PAYLOAD = fp32 | fp16 | bf16 | int8 sets the payload (int8
    // also for the mixer output); NINFER_TP_AR_GROUP the column group (rounded down to whole
    // 2048-column pieces; 0 = one group); NINFER_TP_AR_PIPE=1 takes the pipelined exchange even at
    // FP32 with one group.
    struct T62Exchange {
        bool enabled                       = false;
        ops::HostExchange::Payload payload = ops::HostExchange::Payload::FP16;
        std::int32_t group_columns         = ops::kMoeGroupedChunkColumns;
    };

    static const T62Exchange& t62_exchange();

    // The routed-expert backend and bank of `layer` on rank r (execution only).
    [[nodiscard]] Qwen4RoutedExperts* routed_experts(int r, std::size_t layer) const;

    [[nodiscard]] const Qwen4ExpertBank& expert_bank(int r, std::size_t layer) const;

    // ---- MTP layer (qwen4_forward/mtp.h) --------------------------------------------------------
    // The MTP stem into `wide`: the first R' of run_mtp, from the shifted tokens (or the given
    // embeddings) and the target's carried hiddens.
    void mtp_stem(std::array<const Qwen4MtpParameters*, 2> m, const RankTensors& ids,
                  const RankTensors* embeddings, const RankTensors& hidden_wide, RankTensors& wide);

    const TextConfig& config_;
    std::array<Rank, 2> r_;
    std::array<const Qwen4MtpParameters*, 2> mtp_{};
    // The GDN gating forked onto side streams (gdn_controls), joined before the
    // recurrence reads g/beta (gdn_join_gating); null when not forked.
    GdnGatingOverlap* gating_fork_ = nullptr;
    void gdn_join_gating();
    int width_ = 1;
    Qwen4Collectives tp_;
    std::int32_t first_ = 1;
    std::int32_t last_  = 1;
    // The current prompt chunk's slices (plain residual at tp 2; empty otherwise), one lane of the
    // split exchange each.
    static_assert(kPrefillSliceMaxCount <= ops::HostExchange::kSumLanes &&
                      kPrefillSliceMaxCount <= ops::PeerLanes::kSumLanes,
                  "one split-exchange lane per prefill slice");
    PrefillSlicePlan slices_{};
    // The split-exchange lane a plain block's closing all-reduce begins on (the half-chunk
    // overlap's half_block), or -1: the all-reduce completes in the call.
    int lane_                     = -1;
    Qwen4FeatureTap* feature_tap_ = nullptr;
};

extern template class Qwen4Forward<WorkspaceArena>;
extern template class Qwen4Forward<WorkspaceLayoutBuilder>;

} // namespace ninfer::models::qwen3_5::execution
