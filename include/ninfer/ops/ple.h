#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/record_view.h"

#include <array>
#include <span>

namespace ninfer::ops {

struct NgramHashParameters {
    std::array<std::uint64_t, 3> multipliers{};
    std::array<std::uint32_t, 16> vocab_sizes{}, offsets{};
};

struct NgramTokenTailView {
    ConstStateFieldView token_ids; // I32 [2] per record, oldest first; missing history is EOS.
};

struct Nf4EmbeddingShard {
    ConstMemorySpan pages;
    Tensor decode; // Contiguous BF16 [16,128], indexed by code then E4M3 scale byte.
    std::uint32_t row_begin = 0, rows = 0;
};

/**
 * Concatenate sixteen exact NF4 row lookups into BF16 embedding [2560,T,B].
 * For each live (t,b), x=token_ids[t,b], p is the preceding ID and o the ID before p;
 * IDs before t=0 come from before.token_ids[b][:]. If p==eos, replace o by eos.
 * All products/XOR use U64 wrap. Let u=x*m[0] XOR p*m[1], extended by XOR o*m[2] for heads
 * 8..15. Head h selects offsets[h] + positive_mod(signed_I64_bits(u), vocab_sizes[h]).
 * Each head contributes 160 features, in head order. A row absent from shards contributes
 * exact +0 words. Shards are a Host descriptor list of up to 128 disjoint, sorted global row
 * ranges; they may cover only part of the table. Each positive vocabulary and its offset fit
 * the U32 row domain. eos and the consumed token IDs are nonnegative I32.
 *
 * Shard pages retain RowPagedG16_4K encoding: 45 rows per 4096-byte page, each row holding
 * 80 low-even NF4 code bytes followed by ten nonnegative E4M3 scale bytes (0..126).
 * decode supplies the exact BF16 words for that shard, as prepared by make_nf4_bf16_lut().
 * The Op indexes these supplied words without floating-point arithmetic, preserving signed
 * zero. Exclusive row ownership permits combining partial outputs as packed I32/U32 words;
 * the Op itself performs no communication.
 *
 * T>0, B=1..8. token_ids is Device I32 [T,B] with contiguous dimension zero and a
 * non-overlapping, naturally aligned outer stride. before.token_ids selects dense I32 [2]
 * records with contiguous Device I64 [B] byte offsets. valid_columns is Device I32 [B].
 * For n=valid_columns[b], 0<=n<=T; live rows select valid, naturally aligned records. Output is
 * contiguous, 16-byte-aligned BF16 [2560,T,B]. Columns t>=n are exact +0 and read no token IDs or
 * tail. All storage belongs to execution.device(); pages cover ceil(rows/45)*4096 bytes and decode
 * is naturally aligned. Output overlaps no addressed input, page span or decode table. Inputs are
 * preserved. No state, frontier or count is changed. No workspace, hidden allocation or Host
 * readback is required. CUDA Graph replay consumes current Device IDs, tails and metadata;
 * descriptor and hash values are captured by value, and their referenced storage outlives every
 * consumer.
 */
void ngram_embedding_lookup(const Tensor& token_ids, const Tensor& valid_columns,
                            const NgramTokenTailView& before, const NgramHashParameters& hash,
                            std::int32_t eos, std::span<const Nf4EmbeddingShard> shards,
                            Tensor& embedding, CudaExecutionView execution);

/**
 * Shared PLE domain and view conventions:
 *
 * H=2560 features per stream, C=4 streams, D=H*C=10240 channels. Channel c=d+H*s
 * identifies feature d of stream s. B is in [1,8]; T (physical columns per batch row), R
 * (record columns) are positive. Every positive T representable by the views and available
 * storage is supported. Matching axes agree across operands.
 *
 * All tensors refer to Device storage. Tensor dimension zero is fastest; nb[] contains byte
 * strides. Residual, gated_value, updated_residual and normalized are contiguous BF16
 * [H,C,T,B], with at least 16-byte-aligned bases. Norm weights are contiguous BF16 [H,C];
 * the convolution kernel is contiguous BF16 [D,4], with channel c fastest. Device metadata
 * is contiguous, naturally aligned I32 [B]. Other BF16 views have naturally aligned bases
 * and positive, element-aligned strides over their used axes, with no overlapping elements:
 *   - key [H,C,T,B] has contiguous feature/stream axes; value [H,T,B] has a contiguous
 *     feature axis. Their token/batch strides are explicit, including packed-parent gaps.
 *   - history selects dense BF16 [D,9] fields with contiguous Device I64 [B] byte offsets.
 *     Each selected field satisfies its declared alignment. Columns are oldest to newest.
 *   - records [D,R,B] have contiguous features and positive column/batch strides.
 *
 * Every view is non-owning. Constness of a Tensor descriptor does not grant write access to
 * its data; each Op below specifies the writable regions. Host-visible shape, dtype, stride,
 * alignment and scalar requirements are checked without reading Device metadata back to Host.
 * The caller guarantees the count/index values and their alias constraints at execution time.
 *
 * Numeric convention:
 *   Norm(x,w,eps)[d] = x[d] * (1+w[d]) / sqrt(sum_i x[i]^2/H + eps).
 * Each stream has its own H-element norm and affine weights. Epsilons are positive and finite;
 * floating-point values used by the active computation are finite. The unit-offset gain is
 * interpreted from the stored weights without an additional public BF16 cast of 1+w.
 *
 * Workspace / execution:
 * The entries require no caller-owned global workspace and allocate no hidden Device storage. They
 * submit to execution.stream(), with execution.multiprocessor_count() positive. They support CUDA
 * Graph capture/replay; Device metadata is consumed at execution time. Multiple ordered kernels are
 * permitted. The caller supplies ready inputs, writable output ranges and storage lifetimes through
 * the last consumer, including graph replay. Ops do not reserve records, advance
 * counters/frontiers, publish state, perform I/O or synchronize Host to inspect Device
 * counts/indices.
 */

struct PleBatchView {
    Tensor valid_columns; // I32 [B], n[b] in [0,T].
};

struct PleProjectionView {
    Tensor key;   // BF16 [H,C,T,B], feature/stream contiguous.
    Tensor value; // BF16 [H,T,B], feature contiguous; shared by all C streams.
};

struct PleGateParameters {
    Tensor norm_key;   // BF16 [H,C], offset weights.
    Tensor norm_query; // BF16 [H,C], offset weights.
    Tensor norm_conv;  // BF16 [H,C], offset weights.
    float rms_epsilon = 1e-6F;
};

struct PleConvStateView {
    StateFieldView history; // BF16 [D,9] per record, nine preceding normalized columns.
};

struct ConstPleConvStateView {
    ConstStateFieldView history;

    ConstPleConvStateView() = default;

    ConstPleConvStateView(ConstStateFieldView history) : history(history) {}

    ConstPleConvStateView(const PleConvStateView& view) : history(view.history) {}
};

struct PleConvRecordWindowView {
    Tensor values;  // BF16 [D,R,B], written by Gate and read by Conv.
    Tensor offsets; // I32 [B], record column corresponding to local input column t=0.
};

struct PleConvRecordReadView {
    Tensor values;        // BF16 [D,R,B], column 0 immediately follows the before history.
    Tensor valid_columns; // I32 [B], recorded contiguous prefix u[b] in [0,R].
};

/**
 * Gated PLE projection.
 *
 * For each active (t,b), 0<=t<n[b], and stream s, with eps=parameters.rms_epsilon:
 *   k_s = Norm(projected.key[:,s,t,b], parameters.norm_key[:,s], eps)
 *   q_s = Norm(residual[:,s,t,b], parameters.norm_query[:,s], eps)
 *   a_s = sum_d k_s[d]*q_s[d] / sqrt(H)
 *   v_s = sign(a_s) * sqrt(max(abs(a_s), 1e-6))
 *   g_s = 1 / (1+exp(-v_s))
 *   ideal_U[d,s,t,b] = g_s * projected.value[d,t,b]
 *   ideal_N[:,s,t,b] = Norm(U[:,s,t,b], parameters.norm_conv[:,s], eps).
 * sign(0)=0, hence g_s=0.5 at zero. The fixed gate floor 1e-6 is independent of eps.
 * U is the represented value of the actual BF16 gated_value output. NormConv must consume
 * exactly those U words, including when U is retained privately rather than reloaded.
 *
 * The shared domain applies. residual and gated_value are BF16 [H,C,T,B]; projected,
 * parameters, batch and records use the views above. For every row,
 *   0<=records.offsets[b] and records.offsets[b]+n[b]<=R.
 *
 * Writes gated_value as BF16 ideal_U and, for c=d+H*s, writes BF16 ideal_N to
 * records.values[c,records.offsets[b]+t,b]. There are no public BF16 intermediates for
 * k/q, their products, the dot, or g. For t>=n[b], gated_value is exact BF16 +0; the
 * corresponding projected/residual values are not read and no records are written.
 * All records outside the active write ranges and all inputs/metadata remain unchanged.
 * gated_value and the record write ranges are mutually disjoint and overlap no input.
 * No convolution history is read or written. Shared workspace/execution rules apply.
 */
void ple_gate(const PleProjectionView& projected, const Tensor& residual,
              const PleGateParameters& parameters, const PleBatchView& batch,
              PleConvRecordWindowView records, Tensor& gated_value, CudaExecutionView execution);

/**
 * Dilated causal convolution, SiLU and residual addition, optionally followed by norm.
 *
 * For each active row b, before.history[b][:,0..8] holds the nine columns
 * immediately preceding local t=0; current.offsets[b] selects the first current N column.
 * Define the represented BF16 input to convolution as
 *   N[c,k,b] = before.history[b][c,9+k]                          for -9<=k<0,
 *              current.values[c,current.offsets[b]+k,b]        for 0<=k<n[b].
 * For 0<=t<n[b], c=d+H*s:
 *   v[c,t,b] = sum_{j=0..3} kernel[c,j] * N[c,t-9+3*j,b]
 *   ideal_Y[d,s,t,b] = residual[d,s,t,b] + gated_value[d,s,t,b] + SiLU(v[c,t,b]),
 * where SiLU(x)=x/(1+exp(-x)). Tap 0 reads t-9 and tap 3 reads t; there is no bias.
 * Both entries store Y in BF16 updated_residual. The norm entry additionally computes
 *   ideal_Z[:,s,t,b] = Norm(Y[:,s,t,b], next_hc_norm_weight[:,s], next_hc_epsilon)
 * and stores BF16 normalized. This norm consumes the actual updated_residual words.
 *
 * Shared domain applies. residual, gated_value and each output are BF16 [H,C,T,B];
 * kernel is BF16 [D,4]; next_hc_norm_weight is BF16 [H,C]. For every row,
 *   0<=current.offsets[b] and current.offsets[b]+n[b]<=R.
 * For n[b]>0, before.history selects a valid record. Rows with n[b]=0 need no valid offset.
 * Every required N word must be ready before this call; current is read-only.
 *
 * There is no public intermediate rounding between convolution, SiLU and the residual sum.
 * For t>=n[b], every output is exact BF16 +0 and the corresponding residual, gated_value,
 * records and history are not read. History and records are never mutated by these entries.
 * All inputs are preserved except when residual is the permitted in-place destination.
 * updated_residual may exactly alias residual (same address, shape and strides), or be
 * disjoint; partial overlap is forbidden. Every output is disjoint from all other inputs,
 * including weights and metadata. normalized is also disjoint from residual and updated_residual.
 * These constraints concern the tensor regions, not unrelated slices of a shared allocation.
 * Shared workspace/execution rules apply.
 */
void ple_conv_add(const Tensor& residual, const Tensor& gated_value, const Tensor& kernel,
                  const PleBatchView& batch, const ConstPleConvStateView& before,
                  const PleConvRecordWindowView& current, Tensor& updated_residual,
                  CudaExecutionView execution);

void ple_conv_add_norm(const Tensor& residual, const Tensor& gated_value, const Tensor& kernel,
                       const PleBatchView& batch, const ConstPleConvStateView& before,
                       const PleConvRecordWindowView& current, const Tensor& next_hc_norm_weight,
                       float next_hc_epsilon, Tensor& updated_residual, Tensor& normalized,
                       CudaExecutionView execution);

/**
 * Extract the trailing nine columns after an accepted prefix.
 *
 * For row b, let a=accepted_columns[b]. For every channel c and j in [0,9), let k=a-9+j and copy
 * the BF16 word after.history[b][c,j] = before.history[b][c,9+k]  if k<0, records.values[c,k,b]
 * otherwise. Record column zero is the first column after the nine-column before history. a=0
 * copies the complete before history to after, including when source and destination are distinct;
 * a>=9 uses only the trailing nine words of the accepted record prefix.
 *
 * Shared view rules apply. before/after may use different backing arenas.
 * accepted_columns is contiguous Device I32 [B]. For each b, 0<=a<=records.valid_columns[b]<=R.
 * Every source/destination byte offset selects a valid field, including for a=0.
 *
 * This is exact BF16 word copying, preserving every bit including zero signs. No numerical
 * evaluation or finite-value requirement is imposed on the copied words. Only the nine columns
 * in the selected destination records are written; other records and metadata remain
 * unchanged. Destinations are mutually disjoint and cannot overlap another row's source.
 * Read-only source records may be shared. A row's destination may exactly alias its own source
 * region with identical layout; it must otherwise be disjoint from every source, records and
 * metadata. When in-place, the result uses the original source words, not earlier writes of
 * this call. Aliasing is defined on addressed regions even when pool descriptors differ.
 * The Op changes no counts/offsets and does not adopt or publish the new state.
 * Shared workspace/execution rules apply.
 */
void ple_state_extract(const ConstPleConvStateView& before, const PleConvRecordReadView& records,
                       const Tensor& accepted_columns, PleConvStateView after,
                       CudaExecutionView execution);

} // namespace ninfer::ops
