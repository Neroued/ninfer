#pragma once

#include "core/device.h"
#include "core/record_view.h"
#include "core/tensor.h"
#include <cstdint>

namespace ninfer::ops {

/**
 * destination[:,indices[i]] = source[:,i].
 *
 * Source [D,V] and destination [D,T] are BF16 with contiguous features and explicit,
 * non-overlapping BF16-aligned column strides. Indices [V] is contiguous I32.
 * Indices lie in [0,T), with no duplicates when deterministic overwrite order is required.
 * Copies BF16 bits exactly and preserves unselected columns. Inputs and destination do not
 * overlap. An empty source does no work. No workspace or other state is used.
 */
void scatter(const Tensor& source, const Tensor& indices, Tensor& destination,
             CudaExecutionView execution);

// destination.record[b].field[:] = source[:,b]. Source is BF16 [D,B]; the field is
// BF16 [D] with contiguous features and I64 record offsets [B]. Copies bits exactly.
// Selected fields are distinct and disjoint from the inputs. Unselected storage is preserved.
void scatter(const Tensor& source, const StateFieldView& destination, CudaExecutionView execution);

/**
 * destination[:,j,lanes[b]] = source[:,j,b], for j < valid_columns[b].
 *
 * Source is contiguous BF16 [D,W,B]; lanes and valid_columns are contiguous I32 [B].
 * Destination is a BF16 [D,W,C] view with contiguous features and explicit column/lane strides.
 * A feature slice retains its parent's strides. D is divisible by eight and every BF16 column
 * starts at a 16-byte-aligned address. Inputs and destination do not overlap.
 *
 * The caller supplies distinct lanes in [0,C), counts in [0,W], and disjoint destination
 * columns/lanes. Copies BF16 bits exactly; invalid tails, unselected lanes and features outside
 * the destination slice are preserved. Zero counts read no source values. No workspace is used.
 */
void scatter_bf16_batch(const Tensor& source, const Tensor& lanes, const Tensor& valid_columns,
                        Tensor& destination, CudaExecutionView execution);

/**
 * destination[d,t] = source[feature_offset+d,t].
 *
 * Source [D,T] and destination [F,T] are non-overlapping contiguous BF16 matrices with positive
 * extents. The feature interval [feature_offset,feature_offset+F) lies within [0,D). Copies BF16
 * bits exactly, writes all of destination and preserves source. No workspace is used.
 */
void extract_bf16_features(const Tensor& source, std::int32_t feature_offset, Tensor& destination,
                           CudaExecutionView execution);

/**
 * out[:,b] = source[:,selectors[b],b].
 * Source is contiguous BF16 [D,T,B], selectors is contiguous I32 [B] in [0,T), and
 * out is distinct contiguous BF16 [D,B]. Copies BF16 bits exactly. No workspace or state.
 */
void gather_bf16_batch_columns(const Tensor& source, const Tensor& selectors, Tensor& out,
                               CudaExecutionView execution);

// For columns[b]>=0, destination.record[b].field[:]=source[:,columns[b],b]; -1 preserves the
// destination and reads no source or record offset. Source is contiguous BF16 or I32 [D,T,B],
// columns is contiguous I32 [B], B=1..8, and destination has same-dtype contiguous [D] fields
// with I64 record offsets [B]. Live columns are in range and destination fields are distinct.
// Inputs and selected destination fields are
// disjoint. Copies represented bits exactly, preserves all unselected storage, owns no
// workspace, and consumes current Device selectors under Graph replay.
void scatter_batch_columns(const Tensor& source, const Tensor& columns,
                           const StateFieldView& destination, CudaExecutionView execution);

/**
 * ids[i] = id_map[ids[i]]. The nonempty I32 ids vector is contiguous and updated in place.
 * id_map is a distinct device I32 array [count]; input ids lie in [0,count).
 */
void remap_ids_inplace(Tensor& ids, const std::int32_t* id_map, std::int32_t count,
                       CudaExecutionView execution);

/**
 * out[0:D,t] = first[:,t]; out[D:2D,t] = second[:,t].
 * First/second are contiguous BF16 [D,T], out is contiguous BF16 [2D,T]; D,T are positive.
 * Copies BF16 bits exactly. Inputs and output do not alias; no workspace or state.
 */
void concat_bf16_features(const Tensor& first, const Tensor& second, Tensor& out,
                          CudaExecutionView execution);

/**
 * Split contiguous BF16 source [14336,T] into successive feature intervals of widths
 * 6144,1024,6144,1024. First/third outputs are [256,24,T]; second/fourth are [256,4,T].
 * All tensors are contiguous, pairwise non-overlapping. Writes each BF16 bit exactly;
 * T is positive and every output is overwritten. No workspace or state.
 */
void split_bf16_features(const Tensor& source, Tensor& first, Tensor& second, Tensor& third,
                         Tensor& fourth, CudaExecutionView execution);

} // namespace ninfer::ops
