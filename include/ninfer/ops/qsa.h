#pragma once

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <array>

namespace ninfer::ops {

// Canonical tail fields selected by I64 [B] record offsets. Each field addresses the same logical
// rows.
struct QsaTailStateView {
    StateFieldView raw_keys;   // BF16 [128,3].
    StateFieldView first_rope; // I32 [3].
};

struct ConstQsaTailStateView {
    ConstStateFieldView raw_keys;
    ConstStateFieldView first_rope;

    ConstQsaTailStateView() = default;

    ConstQsaTailStateView(ConstStateFieldView raw_keys, ConstStateFieldView first_rope)
        : raw_keys(raw_keys), first_rope(first_rope) {}

    ConstQsaTailStateView(const QsaTailStateView& view)
        : raw_keys(view.raw_keys), first_rope(view.first_rope) {}
};

struct QsaRawRecordReadView {
    Tensor raw_keys;       // BF16 [128,R,B], actual raw index-key projection words.
    Tensor rope_positions; // I32 [3,R,B], actual three-axis coordinates.
    Tensor base_positions; // I32 [B], F0: record column zero's input position.
    Tensor valid_columns;  // I32 [B], u: available record prefix.
};

struct QsaBatchView {
    Tensor positions;     // I32 [T,B], original-token ordinals.
    Tensor valid_columns; // I32 [B].
    Tensor table_rows;    // I32 [B].
};

struct QsaProjectionView {
    Tensor query;       // BF16 [256,Hq,T,B].
    Tensor key;         // BF16 [256,Hkv,T,B].
    Tensor value;       // BF16 [256,Hkv,T,B].
    Tensor index_query; // BF16 [128,4,T,B].
    Tensor index_key;   // BF16 [128,T,B], raw projection.
};

struct QsaKvProjectionView {
    Tensor key;       // BF16 [256,Hkv,T,B].
    Tensor value;     // BF16 [256,Hkv,T,B].
    Tensor index_key; // BF16 [128,T,B], raw projection.
};

struct QsaKvPrepareParameters {
    Tensor key_norm;       // BF16 [256], offset weights: gain = 1 + FP32(weight).
    Tensor index_key_norm; // BF16 [128].
    float rms_epsilon        = 1e-6F;
    float rope_theta         = 1e7F;
    std::uint32_t rotary_dim = 64;
    std::array<std::uint32_t, 3> rope_sections{11, 11, 10};
    bool rope_interleaved = true;
};

struct QsaPrepareParameters {
    QsaKvPrepareParameters kv;
    Tensor query_norm;       // BF16 [256], offset weights.
    Tensor index_query_norm; // BF16 [128].
};

struct QsaIndexCacheView {
    PagedPlaneView keys; // BF16 [128,16,1] per record; shares main's I64 page offsets.
};

struct ConstQsaIndexCacheView {
    ConstPagedPlaneView keys;

    ConstQsaIndexCacheView() = default;

    ConstQsaIndexCacheView(const QsaIndexCacheView& view) : keys(view.keys) {}
};

struct QsaCacheView {
    PagedKVBatchLayerView main;
    QsaIndexCacheView index;
};

struct QsaRawRecordWriteView {
    Tensor raw_keys;       // BF16 [128,R,B].
    Tensor rope_positions; // I32 [3,R,B].
    Tensor offsets;        // I32 [B], first record column written by this call.
};

struct QsaBlockSelection {
    Tensor block_ids;    // Device I32 [512,T,B], logical four-token block ids.
    Tensor block_counts; // Device I32 [T,B].
};

struct QsaIndexerExecutionEnvelope {
    std::uint32_t max_visible_tokens = 0; // Fixed capture capacity, in original tokens.
};

/** Capacity over every positive T in [min_tokens,max_tokens], with fixed B and envelope.
 * Includes only Op-local scratch, measured from a 256-byte-aligned workspace cursor.
 */
[[nodiscard]] std::size_t
qsa_indexer_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens,
                                     std::int32_t batch_size, QsaIndexerExecutionEnvelope envelope);

/**
 * Op: qsa_indexer
 *
 * Math / indexing:
 * For each live (t,b), p=positions[t,b], C=floor((p+1)/4), and g in [0,C):
 *   score[g] = sum_{h=0..3} max(0, sum_{d=0..127} index_query[d,h,t,b]*index.K[d,g]).
 * The common positive 1/sqrt(128) factor is omitted because only ranking is observable.
 * Select min(512,C) highest computed scores; exact numeric ties prefer smaller logical g.
 * Index g addresses keys.records.page_offsets[g/16,table_rows[b]], row g%16 of its record.
 * Four heads produce one shared selection. The incomplete token tail is not included here.
 * C<=512 selects every g without reading Q, keys or page-table contents.
 *
 * Domain / numeric:
 * T>0, B=1..8. index_query is contiguous, 16-byte-aligned BF16 [128,4,T,B]. Cache keys
 * are naturally aligned, dense BF16 [128,16,1] per 64-token record. Tables are contiguous
 * I64 byte offsets from their record arena.
 * Positions are contiguous I32 [T,B]; valid_columns and table_rows are Device I32 [B].
 * Each row has 0<=n<=T; live positions are consecutive nonnegative original-token ordinals.
 * Every live p+1 is <= envelope.max_visible_tokens <=262144, within the table capacity.
 * Envelope zero permits only inert queries. Live scoring operands and scores are
 * finite. Ranking approximates the formula from represented inputs; there is no semantic
 * intermediate score-storage rounding boundary.
 * Device counts, positions, selectors and readable live pages are caller guarantees.
 *
 * Effects / output:
 * Outputs are naturally aligned, contiguous I32 with the shapes above. Each live count is
 * min(512,C); its first count ids are distinct and in [0,C), in unspecified order. Inert counts
 * are zero and do not read positions, Q, keys or table selectors. Ids beyond count are undefined.
 * Inputs and caches are preserved. Outputs, inputs and used workspace must not write/read
 * overlap; read-only views may share storage. Selected-record alias relationships are caller
 * guarantees. There is no persistent state or publication.
 *
 * Workspace / execution:
 * Scratch comes from the caller's arena and is reused across bounded query windows, without
 * hidden allocation. Fixed T/B, views and device facts keep node count/order compatible across
 * envelope updates; the bound selects launch/workspace configuration.
 * Actual per-query lengths, mixed batch contexts and all-select/inert cases are handled inside
 * the same kernels, including Graph replay. Metadata is not read back by the Host. Downstream
 * consumers read only count ids; same-stream ordering supplies the dependency.
 */
void qsa_indexer(const Tensor& index_query, const ConstQsaIndexCacheView& cache,
                 const QsaBatchView& batch, QsaIndexerExecutionEnvelope envelope,
                 MemoryArena& workspace, QsaBlockSelection& out, CudaExecutionView execution);

struct QsaSparseGqaExecutionEnvelope {
    // Host-known upper bound on p+1 over live queries. Zero permits only inert queries.
    // This resource promise does not assert that a selection is full.
    std::uint32_t max_visible_tokens = 262144;
};

/** Capacity over every positive T in [min_tokens,max_tokens], with fixed storage, query/KV heads, B
 * and envelope. Includes only Op-local scratch, measured from a 256-byte-aligned workspace cursor.
 * multiprocessor_count is the positive physical SM count used by execution.
 */
[[nodiscard]] std::size_t qsa_sparse_gqa_workspace_capacity_bytes(
    KvCacheStorage storage, std::int32_t query_heads, std::int32_t kv_heads,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t batch_size,
    QsaSparseGqaExecutionEnvelope envelope, std::int32_t multiprocessor_count);

/**
 * Op: qsa_sparse_gqa
 *
 * Math / indexing:
 * For each live (t,b), p=positions[t,b], C=floor((p+1)/4), c=block_counts[t,b].
 * J is the union of {4*g,4*g+1,4*g+2,4*g+3} for the first c block_ids and [4*C,p+1).
 * For head h, kvh=floor(h/(Hq/Hkv)), score[j]=dot(query[:,h,t,b],K[:,j,kvh])/16:
 *   attention[d,h] = sum_{j in J} softmax(score)[j] * V[d,j,kvh];
 *   out[d,h,t,b] = BF16(attention[d,h] * sigmoid(FP32(output_gate[d,h,t,b]))).
 * There is no attention-output BF16 rounding before the gate. Indexer scores do not enter
 * the logits. Logical token j addresses tables[j/64,table_rows[b]], row j%64.
 * Empty J and inert queries produce exact +0, without reading query/gate/cache values.
 * Quantized keys represent R*K, R=H256/16; logits use R*query and decoded keys. NVFP4 and
 * K8V4 values represent R*V, so apply R^T to the fully reduced attention before the gate.
 *
 * Domain / numeric:
 * T>0, B=1..8. Query, gate and out are contiguous, 16-byte-aligned BF16 [256,Hq,T,B], (Hq,Hkv) in
 * {(24,2),(12,1)}, plus (24,4) on RTX 5090 with BFloat16, Int8Group64, Fp8E4M3Row256 or Nvfp4Group16. Cache supports BFloat16 (BF16 K /
 * FP16 V), Fp8E4M3Row256, Int8Group64, Nvfp4Group16 and Fp8KeyNvfp4Value. Data/scale planes follow
 * paged_kv_storage_layout(storage,256), with dense, naturally aligned [leading_extent,64,Hkv]
 * geometry per 64-token record. Q and softmax probabilities are not quantized to low-bit operands.
 * The A16 numerical policy applies to the complete attention and gate; intermediate arithmetic
 * remains private. Decoded KV tiles stay on chip; there is no expanded global cache. Tables contain
 * contiguous I64 arena-relative record offsets. Positions are contiguous I32 [T,B];
 * valid_columns/table_rows are contiguous Device I32 [B]. Each row has 0<=n<=T, with consecutive
 * nonnegative live positions satisfying p+1<=envelope.max_visible_tokens<=262144, covered by its
 * table. Selection is contiguous I32 [512,T,B] and [T,B]; each live c lies in [0,min(512,C)], with
 * distinct ids in [0,C). Id order is unspecified; the set need not be an Indexer's actual TopK.
 * Device metadata and readable selected pages are caller guarantees. Live mathematical
 * operands/results are finite. Cache storage quantization and the A16 computation have separate
 * rounding boundaries; decoded KV, transformed Q and probabilities are internal operands rather
 * than additional public tensors.
 *
 * Effects / execution:
 * All inputs and cache are preserved. There is no append, persistent state or publication.
 * Inert queries do not inspect positions, selections or table selectors. Ids beyond c and
 * unselected cache values are not read. Output and used workspace do not overlap inputs or
 * each other; read-only views may share storage. Selected-record alias relationships are caller
 * guarantees. Scratch is caller-owned, without hidden Device
 * allocation. Fixed T/B, storage, views and device facts keep node count/order compatible across
 * envelope updates. The Host bound selects launch/workspace configuration without reading Device
 * metadata. Within a captured bound, per-query lengths, selections and mixed/all-select/inert
 * cases remain Device inputs during replay. Each query attends at most 512*4+3 tokens.
 * Same-stream ordering consumes Prepare/Indexer output directly; readiness/lifetime belong to the
 * caller.
 */
void qsa_sparse_gqa(const Tensor& query, const Tensor& output_gate,
                    const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                    const QsaBlockSelection& selected, QsaSparseGqaExecutionEnvelope envelope,
                    MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

/**
 * Op: qsa_prepare
 *
 * Math / indexing:
 * N_D(x,w) = x * (1+FP32(w)) / sqrt(sum(x*x)/D + rms_epsilon), independently per head.
 * R(z,rho) rotates pairs (i,i+32), i in [0,32), by rho[axis(i)]*theta^(-2*i/64);
 * axis(i)=i%3 on canonical three-axis rho. Current A=1 coordinates broadcast to three axes.
 * R[i]=z[i]*cos-z[i+32]*sin,
 * R[i+32]=z[i+32]*cos+z[i]*sin; dimensions [64,D) retain z.
 * For each row b and t<valid_columns[b], p=positions[t,b]:
 *   query[:,h,t,b]       = BF16(R(N_256(projected.query[:,h,t,b],query_norm),rho[t,:,b]));
 *   index_query[:,h,t,b] = BF16(R(N_128(projected.index_query[:,h,t,b],index_query_norm),rho));
 *   k                   = R(N_256(projected.key[:,h,t,b],key_norm),rho);
 *   v                   = projected.value[:,h,t,b].
 * BFloat16 main caches store BF16(k) and FP16_RNE(v). For quantized main caches, k enters
 * the fixed FP32 H256/16 Hadamard and storage encoder directly from norm/RoPE, with no
 * intermediate BF16 rounding. INT8-G64 and FP8 transform K only; NVFP4-G16 and K8V4 transform
 * both K and V. V enters from the represented BF16 projection. Codes, represented scales,
 * zero and saturation rules are those of kv_cache_append; its BF16 K input boundary does not
 * apply to this fused computation. Q/index-Q and the compressed index cache remain BF16 and
 * do not receive this main-cache Hadamard. The attention consumer owns the corresponding Q
 * transform and any value-basis inversion.
 * Raw index-key words and actual coordinates are copied to records[:,offsets[b]+t,b];
 * A=1 replicates the coordinate to all three record axes.
 * With F=positions[0,b] and n=valid_columns[b], each newly completed block
 * g in [floor(F/4),floor((F+n)/4)) writes:
 *   u = BF16(mean(raw_index_key[4*g:4*g+4])); index.K[g] =
 * BF16(R(N_128(u,index_key_norm),rho[4*g])). Raw keys preceding F come from before's canonical tail
 * at F; other keys come from projected. A block starting before F uses before.first_rope; otherwise
 * it uses this call's coordinates. Main p addresses table[p/64,table_rows[b]], row p%64; index g
 * uses the same table at g/16, row g%16. Only complete blocks are written. No causal
 * visibility/frontier is published.
 *
 * Logical shapes / supported domain:
 * The current rotary implementation accepts rotary_dim=64, interleaved sections [11,11,10].
 * Supported Hq/Hkv and storage combinations belong to the implementation catalog; index queries
 * always have four heads. Cache and projections agree on
 * Hkv. T>0, B in [1,8], A=1 or 3. Projections and both outputs have the shapes above, are
 * contiguous and 16-byte aligned. Norm weights are contiguous, naturally aligned BF16 vectors.
 * Positions are contiguous I32 [T,B]; rho is contiguous I32 [T,A,B]. Metadata is contiguous Device
 * I32 [B]. Tail/records follow QsaTailStateView's feature-contiguous, naturally aligned stride
 * rules. Main-cache data and scale planes:
 *
 * - BFloat16: BF16 K / FP16 V [256,64,Hkv] per record; no scales.
 * - Int8Group64: I8 K/V [256,64,Hkv] per record; FP16 scales [4,64,Hkv] per record.
 * - Fp8E4M3Row256: FP8_E4M3FN K/V [256,64,Hkv] per record; FP16 scales [1,64,Hkv] per record.
 * - Nvfp4Group16: packed U8 K/V [128,64,Hkv] per record; U8 E4M3 scales [16,64,Hkv] per record.
 * - Fp8KeyNvfp4Value: the FP8 key planes and NVFP4 value planes above.
 *
 * Cache planes are dense within each record; tables have contiguous I64 record offsets and a
 * non-overlapping row stride. Both caches share the record arena and page-offset table. R is
 * positive. Every row has 0<=n<=T and 0<=offset<=offset+n<=R; R may be smaller than T. Live
 * positions are consecutive, nonnegative ordinals below 262144. before corresponds to F. RoPE
 * coordinates are independent nonnegative I32 values. Epsilon/theta are positive finite floats;
 * live projection/norm values are finite. The caller provides valid live record/page mappings and
 * disjoint writes across rows. Host validation does not read Device metadata.
 *
 * Numeric / effects:
 * Norm and rotation have no intermediate represented tensor. The pooled mean's BF16 rounding
 * is semantic; raw records preserve input words exactly. Q/IQ/index keys and BFloat16-cache K
 * are BF16 approximations to the complete formula. Quantized K includes the storage encoder's
 * representation error; V uses its profile's exact persistent encoding. Q/IQ for t>=n are
 * exactly BF16 +0; inert columns do not read projections/positions/state or write cache/records.
 * An empty row need not have valid
 * slot/table selectors. before, metadata, page tables and untouched cache/record regions remain
 * unchanged. query may exactly alias projected.query, and index_query may independently exactly
 * alias projected.index_query (same address/shape/strides). Partial overlap and all other
 * write/read overlap are forbidden. Device-selected state/cache/record alias relationships are
 * caller guarantees, including across different views of an interleaved allocation.
 *
 * Workspace / execution:
 * Zero global scratch, no sizing query or hidden allocation. Work uses execution.stream().
 * CUDA Graph replay reads current Device metadata; callers own operand
 * readiness/lifetime, allocation, tail extraction, acceptance, and state publication.
 */
void qsa_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const Tensor& rope_positions,
                 const ConstQsaTailStateView& before, QsaCacheView cache,
                 QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                 CudaExecutionView execution);

/**
 * The K/V, compressed index-key and raw-record effects of qsa_prepare, with no query
 * inputs or outputs. K normalization/RoPE, V encoding, four-token index-key pooling,
 * its BF16 mean boundary, and cross-call tail/coordinate selection follow that contract.
 *
 * T/B/Hkv come from projected.key; projections follow QsaKvProjectionView. Parameters,
 * metadata, all five cache encodings, tail and records have the same domain and alias
 * requirements as qsa_prepare. Only live columns and newly completed index blocks are
 * written; inert columns read no projections, positions, state or table selectors.
 * Inputs and untouched storage are preserved. No frontier or tail state is published.
 * Zero global scratch and no hidden allocation; Graph replay consumes current Device
 * metadata. The caller owns allocation, readiness, tail extraction and acceptance.
 */
void qsa_prepare_kv(const QsaKvProjectionView& projected, const QsaKvPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const ConstQsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, CudaExecutionView execution);

/**
 * Op: qsa_tail_extract
 *
 * Math / indexing:
 * For row b, let F0=records.base_positions[b], a=accepted_columns[b], F1=F0+a,
 * r=F1%4, s=F1-r. The notation below selects each field's record for row b.
 * For every feature c and j in [0,3):
 *   after.raw_keys[c,j,b] = BF16 +0                              if j>=r,
 *                            before.raw_keys[c,s+j-(F0-F0%4),b] if s+j<F0,
 *                            records.raw_keys[c,s+j-F0,b]         otherwise.
 *   after.first_rope[:,b] = [0,0,0]                         if r==0,
 *                            before.first_rope[:,b]        if s<F0,
 *                            records.rope_positions[:,s-F0,b] otherwise.
 * before is the canonical state at F0: its first F0%4 raw columns contain the unfinished
 * four-token block in chronological order, the other columns are BF16 +0; first_rope holds
 * that block's first token coordinates, or three I32 zeros when F0%4==0.
 *
 * Logical shapes / supported domain:
 * B is in [1,8]; R and L are positive. Each state's two fields agree on B/L.
 * Record raw keys are BF16 [128,R,B,L], coordinates I32 [3,R,B,L], with contiguous features
 * and explicit disjoint outer strides. Each layer independently applies the equations above;
 * base_positions, valid_columns and accepted_columns are shared across layers.
 * accepted_columns is contiguous Device I32 [B]. Each row has 0<=a<=u<=R, F0>=0, and F0+u
 * representable as nonnegative I32. The caller supplies the input-prefix length, not an emitted
 * token count, and ensures that before corresponds to F0. Source/destination record offsets are
 * valid even for a=0. Host-visible layouts are checked without reading Device metadata to Host;
 * count/index values and selected-region alias relationships are caller guarantees.
 *
 * Numeric / effects:
 * Exact BF16 word/I32 copying and zero filling: copied bits, including zero signs and NaN
 * payloads, are preserved; no finite-value requirement or floating-point evaluation applies.
 * Only the selected destination's three raw columns and three coordinates are written.
 * Unselected state, records and metadata are preserved. Only record columns needed from [0,a)
 * are read; no rejected suffix or earlier completed block is scanned. a=0 copies the canonical
 * source to a distinct destination and preserves every state bit when in-place.
 * Destination regions are mutually disjoint and cannot overlap another row's source or any
 * records/metadata. Read-only sources may be shared. A destination plane may exactly alias its
 * corresponding source plane with identical addressed layout; other write/read overlap is
 * forbidden, including between raw and coordinate planes. Aliasing is defined by selected
 * addresses, not pool descriptor identity or physical record offsets. In-place results use original
 * source words. No cache, count, frontier or state-publication effect occurs.
 *
 * Workspace / execution:
 * No caller-owned global scratch, sizing query, or hidden Device allocation. Work is submitted
 * to execution.stream(). CUDA Graph replay reads
 * the current Device metadata. The caller owns operand readiness and storage lifetime.
 */
void qsa_tail_extract(const ConstQsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& accepted_columns, QsaTailStateView after,
                      CudaExecutionView execution);

} // namespace ninfer::ops
