#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/record_view.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <array>

namespace ninfer::ops {

/**
 * Kpool history uses original-token ordinals and 64-token page records. Each complete pool
 * represents four consecutive tokens; selection holds at most 512 pools. T is the positive
 * physical token extent, B=1..8, and native Tensor dimension zero is contiguous.
 *
 * All execution views borrow caller-owned storage on execution.device(). Activations and
 * contraction weights are contiguous and at least 16-byte aligned. Parameter vectors, APE and
 * I32 metadata are contiguous and naturally aligned; I64 selectors are 8-byte aligned. State
 * fields and raw records have contiguous features and positive, nonoverlapping outer strides.
 * Page planes follow the cache views below. Only optional operands explicitly marked absent
 * may be empty. Host checks cover geometry, representation, device, capacity and address facts;
 * Device values, selected-record bounds and live write separation are caller guarantees.
 *
 * Except for the documented same-row tail update, writes are disjoint from other outputs,
 * accessed inputs, weights, metadata and live scratch. Read-only inputs may share storage.
 * Aliasing is judged at accessed fields/planes, not at their common arena. Caller materializes
 * pages, resolves COW and publishes mappings before submission, and protects every read/write
 * lease and frame operand through its last device consumer. Ops neither allocate backing nor
 * choose owners, advance frontiers, publish checkpoints or decide acceptance.
 *
 * Work is submitted to execution.stream(), without Host readback or hidden device allocation.
 * Arena scope restoration does not retire device work. Same-stream consumers are ordered;
 * cross-stream dependencies and storage reuse belong to the caller. Graph replay reads current
 * Device metadata at stable input addresses, including page/State offsets after relocation.
 */
enum class KpoolMlaStorage {
    // RN_BF16 of the FP32 normalized latent, shared by K and V; no scales.
    BFloat16,
    // E4M3FN codes, one FP32 multiplier per contiguous 128 latent features of each token.
    // Encoder/decoder: docs/maintainer/glm5next-quantization.md, section 3.
    // Decode on chip as RN_BF16(RN_FP32(E4M3FN(code)*scale)); no basis transform.
    Fp8E4m3fnGroup128F32,
};

inline constexpr std::int32_t kKpoolPoolSize          = 4;
inline constexpr std::int32_t kKpoolSelectedPools     = 512;
inline constexpr std::uint32_t kKpoolMaxVisibleTokens = 1048576;

struct KpoolMlaBatchView {
    Tensor positions;     // Device I32 [T,B], original-token ordinals.
    Tensor valid_columns; // Device I32 [B], 0<=n[b]<=T.
    Tensor table_rows;    // Device I32 [B], row of the page-offset table.
};

// Live positions in each row are consecutive and nonnegative; padding consumes no ordinal.
// Every live p+1 fits the page table and the semantic context limit. The Host envelope is only
// a resource bound, not an exact length or a claim that selections are full. Zero permits no
// live queries. A valid envelope change must not change mathematical output.
struct KpoolExecutionEnvelope {
    std::uint32_t max_visible_tokens = 0;
};

// A layer borrows planes of one page-group record. The data plane is BF16 or E4M3FN [Dc,64,1]
// according to storage, with tokens_per_row=1; the optional scale plane is FP32 [Dc/128,64,1],
// also tokens_per_row=1.
// BF16 has no scale plane. FP8 requires Dc divisible by 128. Planes start at 256-byte-aligned
// record offsets and are dense [feature,row,head]. The I64 [logical_pages,table_rows] table
// holds arena-relative byte offsets, contiguous along logical_pages with an explicit row stride.
struct KpoolMlaLatentCacheView {
    PagedPlaneView data;
    PagedPlaneView scales;
    KpoolMlaStorage storage = KpoolMlaStorage::BFloat16;
};

struct ConstKpoolMlaLatentCacheView {
    ConstPagedPlaneView data;
    ConstPagedPlaneView scales;
    KpoolMlaStorage storage = KpoolMlaStorage::BFloat16;

    ConstKpoolMlaLatentCacheView() = default;

    ConstKpoolMlaLatentCacheView(ConstPagedPlaneView data, ConstPagedPlaneView scales,
                                 KpoolMlaStorage storage)
        : data(data), scales(scales), storage(storage) {}

    ConstKpoolMlaLatentCacheView(const KpoolMlaLatentCacheView& view)
        : data(view.data), scales(view.scales), storage(view.storage) {}
};

// One BF16 [Di,16,1] plane, tokens_per_row=4, aligned and addressed as above. Logical pool g
// uses page g/16 and row g%16. Latent tokens of that pool occupy rows 4*(g%16)+{0,1,2,3}.
struct KpoolIndexCacheView {
    PagedPlaneView keys;
};

struct ConstKpoolIndexCacheView {
    ConstPagedPlaneView keys;

    ConstKpoolIndexCacheView() = default;

    explicit ConstKpoolIndexCacheView(ConstPagedPlaneView keys) : keys(keys) {}

    ConstKpoolIndexCacheView(const KpoolIndexCacheView& view) : keys(view.keys) {}
};

// All present planes use the same backing, page-offset table and 64-token page span. Plane
// offsets select this layer inside a possibly multilayer record. No cross-page contiguity is
// assumed. A consumer receives only its required read-only planes.
struct KpoolMlaCacheView {
    KpoolMlaLatentCacheView latent;
    KpoolIndexCacheView index;
};

struct ConstKpoolMlaCacheView {
    ConstKpoolMlaLatentCacheView latent;
    ConstKpoolIndexCacheView index;

    ConstKpoolMlaCacheView() = default;

    ConstKpoolMlaCacheView(ConstKpoolMlaLatentCacheView latent, ConstKpoolIndexCacheView index)
        : latent(latent), index(index) {}

    ConstKpoolMlaCacheView(const KpoolMlaCacheView& view)
        : latent(view.latent), index(view.index) {}
};

// At frontier F, columns [0,F%4) contain the raw words for [4*floor(F/4),F), oldest first;
// other columns are BF16 +0. Gates exclude APE. The two fields select the same logical rows
// through Device I64 [B] offsets; each offset is relative to that field's record backing.
// Fields have BF16 [Di,3] per layer, equal layer counts, and independent field/layer strides.
// Prepare takes a single layer; extraction may take all L layers. Canonical StateImage and
// frame-local provisional fields obey the same contract, without requiring a StateId.
struct KpoolTailStateView {
    StateFieldView raw_keys;
    StateFieldView raw_gates;

    [[nodiscard]] KpoolTailStateView layer(std::uint32_t index) const {
        return {raw_keys.layer(index), raw_gates.layer(index)};
    }
};

struct ConstKpoolTailStateView {
    ConstStateFieldView raw_keys;
    ConstStateFieldView raw_gates;

    ConstKpoolTailStateView() = default;

    ConstKpoolTailStateView(ConstStateFieldView raw_keys, ConstStateFieldView raw_gates)
        : raw_keys(raw_keys), raw_gates(raw_gates) {}

    ConstKpoolTailStateView(const KpoolTailStateView& view)
        : raw_keys(view.raw_keys), raw_gates(view.raw_gates) {}

    [[nodiscard]] ConstKpoolTailStateView layer(std::uint32_t index) const {
        return {raw_keys.layer(index), raw_gates.layer(index)};
    }
};

struct KpoolRawRecordWriteView {
    Tensor raw_keys;  // BF16 [Di,W,B], single layer's active row views, W>0.
    Tensor raw_gates; // BF16 [Di,W,B], actual raw gate words without APE.
    Tensor offsets;   // Device I32 [B], 0<=offset[b], offset[b]+n[b]<=W for live rows.
};

// Layer and row capacity are explicit Tensor axes, not duplicated Host counters. Dense storage
// has outer index layer*R+b; padded positive nonoverlapping strides are also legal. Physical
// record rows remain fixed from production to retirement, even if another batch is compacted.
// The frame owns F0/u/offset updates; a Prepare appending at offset o uses positions[0,b]=F0+o.
struct KpoolRawRecordReadView {
    Tensor raw_keys;       // BF16 [Di,W,R,L].
    Tensor raw_gates;      // BF16 [Di,W,R,L], same geometry as raw_keys.
    Tensor base_positions; // Device I32 [R], F0 at record column zero.
    Tensor valid_columns;  // Device I32 [R], available record prefix u.
};

// Contiguous, naturally aligned Device I32. Indexer writes Nselection=T*B rows in m=t+T*b
// order. Sparse MLA can instead map queries to any still-live selection row. IDs are logical
// pool ordinals, independent of physical page placement. IDs/count/E must survive together
// until the last consumer, along with the referenced history. Only IDs[0:count] are defined.
struct KpoolBlockSelection {
    Tensor pool_ids;      // [512,Nselection].
    Tensor pool_counts;   // [Nselection].
    Tensor selection_end; // [Nselection], frozen exclusive original-token end E.
};

enum class KpoolMlaPrepareProjection {
    Full,            // History, normalized Q bottleneck and raw head coefficients.
    QueryAndHistory, // History and normalized Q bottleneck; no head coefficients.
    History,         // History only; no query projection or query outputs.
};

struct KpoolMlaPrepareParameters {
    KpoolMlaPrepareProjection projection = KpoolMlaPrepareProjection::Full;
    // BF16/Contiguous, logical [N,H], K/hidden contiguous; the selected parent prefix only.
    // Row order: latent Dc | index key Di | pool gate Di | optional Q_a Rq | optional coeff Hi.
    Weight a_projection;
    Tensor query_norm;          // BF16 [Rq], absent for History.
    Tensor latent_norm;         // BF16 [Dc], direct gain.
    Tensor index_norm;          // BF16 [Di], direct LayerNorm gain.
    Tensor index_bias;          // BF16 [Di].
    Tensor pool_ape;            // BF16 [Di,4], feature contiguous.
    float rms_epsilon   = 0.0F; // Caller supplies finite positive semantic values.
    float index_epsilon = 0.0F;
};

struct KpoolMlaPrepareOutputs {
    Tensor q_r;        // BF16 [Rq,T,B], absent for History.
    Tensor head_coeff; // FP32 [Hi,T,B], present only for Full.
};

// Address-free actual contraction geometry; execution derives these same facts from operands.
// query_rank=0 for History; index_heads=0 unless Full. Other extents are positive.
// N=latent_dim+2*index_dim+query_rank+index_heads. Registered weight geometries are admitted by
// Prepare itself. Layout planning alone does not establish execution support.
struct KpoolMlaPrepareProfile {
    KpoolMlaPrepareProjection projection = KpoolMlaPrepareProjection::Full;
    std::int32_t hidden = 0, latent_dim = 0, index_dim = 0, query_rank = 0, index_heads = 0;
    std::int32_t pool_size = kKpoolPoolSize;
};

struct KpoolIndexerProfile {
    std::int32_t index_heads = 0, index_dim = 0;
    std::int32_t pool_size = kKpoolPoolSize, selected_pools = kKpoolSelectedPools;
};

struct KpoolSparseMlaProfile {
    std::int32_t query_heads = 0, query_dim = 0, value_dim = 0, latent_dim = 0;
    std::int32_t pool_size = kKpoolPoolSize, selected_pools = kKpoolSelectedPools;
};

/**
 * Prepare projects contiguous BF16 x[H,T,B] with the selected W_A prefix, using BF16 operands
 * and FP32 accumulation. Split the result into z_c,z_k,z_g and requested z_q,z_w, and compute:
 *   c     = z_c * latent_norm / sqrt(mean(z_c^2)+rms_epsilon);
 *   q_r   = BF16(z_q * query_norm / sqrt(mean(z_q^2)+rms_epsilon));
 *   k_raw = BF16((z_k-mean(z_k))*index_norm /
 *                  sqrt(mean((z_k-mean(z_k))^2)+index_epsilon) + index_bias);
 *   g_raw = BF16(z_g); head_coeff = FP32(z_w).
 * Means reduce only the corresponding feature dimension. c stays FP32 through the selected
 * latent encoder; there is no projection-to-norm BF16 seam. Coefficients are signed raw
 * projections, without activation or scale. Absent outputs/parameters are neither accessed
 * nor computed. Participating inputs/results are finite; physical padding x may be read by
 * projection, so every physical x column must contain valid finite BF16 values.
 *
 * For a live row, F=positions[0,b], n=valid_columns[b]. Write latent [F,F+n) and each completed
 * pooled key g in [floor(F/4),floor((F+n)/4)):
 *   logits[r,d] = FP32(g_raw[4*g+r,d]) + FP32(APE[d,r]);
 *   index.K[d,g] = BF16(sum_r softmax_over_r(logits[:,d])[r] * FP32(k_raw[4*g+r,d])).
 * Pool softmax/sums use FP32 and actual BF16 raw words, including wholly new pools. Raw words
 * preceding F come from before's canonical tail; no incomplete pool is published. New cache
 * writes may precede attention over this window; query metadata still determines visibility.
 *
 * Record writes actual k_raw/g_raw to records[:,offset[b]+t,b] for t<n and preserves before.
 * Update instead publishes the F+n tail: the last (F+n)%4 raw columns in chronological order,
 * using before where necessary, and +0 in the remaining three-column field. This may update
 * the same row's exact fields in place or disjoint fields, after all old-tail reads. No partial
 * or cross-field overlap, or destination overlap with another live row's source, is allowed.
 *
 * Both forms write exact +0 to requested q_r/head_coeff for t>=n. Inert rows write no
 * cache/records/after and do not read positions, tail, record offsets or table selectors;
 * their selectors may be -1. Record suffixes, unaddressed cache and other State fields are
 * preserved. before is one layer at F; before/after may use independent backing and field
 * strides. Caller owns the acceptance/frontier and must use the actual records to restore
 * a produced prefix.
 */
void kpool_mla_prepare_record(const Tensor& x, const KpoolMlaPrepareParameters& parameters,
                              const KpoolMlaBatchView& batch, ConstKpoolTailStateView before,
                              KpoolMlaCacheView cache, KpoolRawRecordWriteView records,
                              KpoolMlaPrepareOutputs outputs, MemoryArena& workspace,
                              CudaExecutionView execution);

void kpool_mla_prepare_update(const Tensor& x, const KpoolMlaPrepareParameters& parameters,
                              const KpoolMlaBatchView& batch, ConstKpoolTailStateView before,
                              KpoolMlaCacheView cache, KpoolTailStateView after,
                              KpoolMlaPrepareOutputs outputs, MemoryArena& workspace,
                              CudaExecutionView execution);

/**
 * For each live query p, E=p+1, C=floor(E/4), compute over g in [0,C):
 *   dot[h,g] = sum_d index_query[d,h,t,b] * index.K[d,g];
 *   score[g] = sum_h head_coeff[h,t,b] * max(0,dot[h,g]).
 * index_query is BF16 [Di,Hi,T,B]; head_coeff is FP32 [Hi,T,B]. Dot products accumulate in
 * FP32; ReLU, signed weighting and head reduction are FP32. Select the exact min(512,C)
 * highest computed scores, ties preferring smaller logical g. The set is unordered and
 * distinct; no forced latest pool, no zero-filled invisible candidate. The common positive
 * model score scale is omitted because only ranking is observable.
 *
 * C<=512 selects all pools without reading query, coefficients, keys or table selectors/table
 * contents. Otherwise only complete visible pools are read. Write count and E for each live
 * output row m=t+T*b; Nselection must equal T*B. Inert rows write count=E=0, leave IDs undefined
 * and do not read positions, query, coefficients, cache or selectors. IDs beyond count are
 * undefined. Inputs/cache are preserved. Mathematical operands and computed scores are finite.
 * Each live E fits both the page table and envelope.max_visible_tokens<=kKpoolMaxVisibleTokens.
 *
 * Mixed per-row lengths/all-select/inert cases are Device decisions. Scratch is reused across
 * bounded query windows. Fixed T/B/profile/device facts keep Graph node count/order compatible
 * across envelope updates; changing a Host variable alone does not update captured nodes.
 */
void kpool_indexer(const Tensor& index_query, const Tensor& head_coeff,
                   const ConstKpoolIndexCacheView& cache, const KpoolMlaBatchView& batch,
                   KpoolExecutionEnvelope envelope, MemoryArena& workspace,
                   KpoolBlockSelection& out, CudaExecutionView execution);

struct KpoolMlaFactors {
    Tensor key_up;   // BF16 [Dc,Dq,Hq], logical U_K[h][Dq,Dc]; Dc contiguous.
    Tensor value_up; // BF16 [Dc,Dv,Hq], logical U_V[h][Dv,Dc]; Dc contiguous.
};

/**
 * Complete factorized attention. query is BF16 [Dq,Hq,T,B], out is BF16 [Dv,Hq,T,B].
 * Device I32 selection_rows[T,B] maps each live query to a selected row in [0,Nselection).
 * For its frozen E/count, require 0<=E<=p+1<=envelope.max_visible_tokens<=context limit,
 * 0<=count<=min(512,floor(E/4)), and distinct IDs[0:count] in [0,floor(E/4)). Caller supplies
 * a selection belonging to the same request/layer/history. It need not be this query's TopK.
 *   J = union({4*g,4*g+1,4*g+2,4*g+3} for selected g) union [4*floor(E/4),E);
 *   q_latent_h = transpose(U_K[h]) * query_h;
 *   a_h = softmax_over_J(attention_scale * dot(q_latent_h,c_j));
 *   u_h = sum_{j in J} a_h[j]*c_j; out_h = BF16(U_V[h]*u_h).
 * E remains frozen during reuse, even when p grows. |J|<=2051; all heads share J. Tail c_j
 * comes from latent cache, not raw tail State. Indexer scores do not enter the logits. There
 * is no RoPE, bias, output gate, sink or forced self token. attention_scale is finite positive,
 * supplied by the caller (1/16 for Dq=256); it is not derived from Dc.
 *
 * c_j is the actual stored BF16 value or the FP8 profile's decoded BF16 operand. The same c_j
 * serves both latent K/V roles, including newly written tokens. Factor/QK/weighted contractions
 * follow BF16/F32 computation; softmax statistics, exponentials and split merges are FP32.
 * q_latent/u are private intermediates, may retain FP32, and impose no additional observable
 * BF16 seam or global materialization. Final O projection is outside this Op. Inputs/results
 * participating in the mathematics are finite; all inputs/cache/selection are preserved.
 *
 * Empty J and inert queries write exact BF16 +0, without reading their query, factors or cache
 * payload in any stage. Inert queries also do not read positions, selection mappings/contents
 * or table selectors. Never read invalid ID suffixes or a fourth incomplete-tail token.
 * Caller-scoped scratch covers every private stage. Fixed T/B/profile/storage/device facts
 * keep Graph node count/order compatible across envelope updates; workspace and the actual
 * capture/update must cover the chosen bound. No expanded global BF16 cache is created.
 */
void kpool_sparse_mla(const Tensor& query, const KpoolMlaFactors& factors,
                      const ConstKpoolMlaLatentCacheView& cache, const KpoolMlaBatchView& batch,
                      const KpoolBlockSelection& selected, const Tensor& selection_rows,
                      float attention_scale, KpoolExecutionEnvelope envelope,
                      MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

/**
 * Restore the raw tail of an accepted prefix, across L layers in one call. accepted_columns
 * is Device I32 [B], 1<=B<=R<=8; L and R come from records.raw_keys.ne[3] and ne[2]. Both tail
 * fields have L layers and B selected records; before is canonical at F0=base_positions[b].
 * For every positive acceptance a, require 0<a<=u=valid_columns[b]<=W and
 * 0<=F0<=F0+u<=kKpoolMaxVisibleTokens. Define F1=F0+a, r=F1%4, s=F1-r. For each layer, K/G
 * feature and destination column j in [0,3):
 *   j>=r:     store BF16 +0;
 *   s+j<F0:   copy before[:,s+j-(F0-F0%4)];
 *   otherwise: copy records[:,s+j-F0,b,layer].
 * Copies preserve the actual words; there is no projection, pooling, re-encoding or floating
 * arithmetic. Rejected suffixes are not read. a=0 is a strict no-op on both source and
 * destination, even when disjoint: no state/record payload is accessed, selectors may be -1,
 * and the caller continues using the original image rather than adopting an unwritten one.
 *
 * Same-row exact in-place fields or fully disjoint fields are allowed. Save required old words
 * before overwriting them. Destinations are distinct and cannot cover another positive row's
 * source, records or metadata. Before/after may have independent backing, fields and strides;
 * read-only sources may be shared. All other fields/records/metadata remain unchanged. L=1 is
 * legal; histories with different F0/frontiers are not combined into the same record row.
 * This Op has statically zero scratch and no capacity query.
 */
void kpool_tail_extract(ConstKpoolTailStateView before, const KpoolRawRecordReadView& records,
                        const Tensor& accepted_columns, KpoolTailStateView after,
                        CudaExecutionView execution);

/**
 * Minimum sufficient Op-local high-water bytes over the inclusive positive T interval, with
 * fixed profile/storage/B/envelope and the execution device's positive physical sm_count.
 * Include every legal length, pool phase, alias form, private route and internal alignment gap;
 * exclude trailing padding and caller outputs/cache/tail/records/selection. Count from a
 * 256-byte-aligned local cursor. Invalid/unsupported profiles or intervals throw; zero means
 * legal zero scratch. pool_size=4 and selected_pools=512 describe this contract, not arbitrary
 * algorithm options. Query and execution share admission, selection and allocation rules.
 */
[[nodiscard]] std::size_t kpool_mla_prepare_record_workspace_capacity_bytes(
    const KpoolMlaPrepareProfile& profile, KpoolMlaStorage storage, std::int32_t batch_size,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t sm_count);

[[nodiscard]] std::size_t kpool_mla_prepare_update_workspace_capacity_bytes(
    const KpoolMlaPrepareProfile& profile, KpoolMlaStorage storage, std::int32_t batch_size,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t sm_count);

[[nodiscard]] std::size_t kpool_indexer_workspace_capacity_bytes(
    const KpoolIndexerProfile& profile, KpoolExecutionEnvelope envelope, std::int32_t batch_size,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t sm_count);

[[nodiscard]] std::size_t
kpool_sparse_mla_workspace_capacity_bytes(const KpoolSparseMlaProfile& profile,
                                          KpoolMlaStorage storage, KpoolExecutionEnvelope envelope,
                                          std::int32_t batch_size, std::int32_t min_tokens,
                                          std::int32_t max_tokens, std::int32_t sm_count);

// Equality identifies identical captured launch parameters and scratch layout at fixed
// operands/profile/storage/device. Device positions, validity and mappings remain replay inputs.
// A changed value requires a new capture/update; the value has no caller-interpretable fields.
class KpoolGraphProfile {
public:
    bool operator==(const KpoolGraphProfile&) const = default;
private:
    std::array<std::uint64_t, 24> words_{};
    friend KpoolGraphProfile kpool_indexer_graph_profile(const KpoolIndexerProfile&,
                                                         KpoolExecutionEnvelope, std::int32_t,
                                                         std::int32_t, std::int32_t);
    friend KpoolGraphProfile kpool_sparse_mla_graph_profile(const KpoolSparseMlaProfile&,
                                                            KpoolMlaStorage, KpoolExecutionEnvelope,
                                                            std::int32_t, std::int32_t,
                                                            std::int32_t);
};

[[nodiscard]] KpoolGraphProfile
kpool_indexer_graph_profile(const KpoolIndexerProfile&, KpoolExecutionEnvelope,
                            std::int32_t batch_size, std::int32_t tokens, std::int32_t sm_count);
[[nodiscard]] KpoolGraphProfile
kpool_sparse_mla_graph_profile(const KpoolSparseMlaProfile&, KpoolMlaStorage,
                               KpoolExecutionEnvelope, std::int32_t batch_size, std::int32_t tokens,
                               std::int32_t sm_count);

} // namespace ninfer::ops
