#pragma once

#include "artifact/binder.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/load/bindings.h"

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::loading {

// Tensor-parallel placement of one logical parameter, in the parameter's own coordinates:
// `ranges[d]` are the rows (Rows) or columns (Columns) device d computes with. The artifact
// objects stay opaque; parent_placements() maps these ranges through each Binding part.
//
// Dense Qwen3.5 at tp > 1 (heads always split into contiguous per-rank blocks, which keeps every
// GDN value head with its key head and every query head with its KV head):
//
//   Rows     attention/{query,gate} by query heads, attention/{key,value} by KV heads,
//            gdn/{query,key} by key heads, gdn/{value,z,a_projection,b_projection,a_log,dt_bias}
//            by value heads, mlp/{gate,up} by intermediate rows, text/output_head by vocabulary.
//   Columns  attention/output, gdn/output, mlp/down and mtp/input_projection over their input
//            (rank 0 contracts the normalized embedding half of mtp/input_projection, rank 1 the
//            hidden half); gdn/convolution over its Q|K|V channel sections, one range each.
//   Replicated  norms, text/token_embedding.
//   HostMapped  text/token_embedding with options.embedding_host, at tp 1 too: one copy in mapped
//            page-locked host memory that every rank's gather reads in place over PCIe.
//            Qwen4Exp routed experts ({text,mtp}/layers/<i>/moe/experts/*) always, at tp 1 too:
//            each layer's stacked gate/up and down banks (65 GiB for Qwen3.8-Flash-Next with MTP)
//            stay in host memory, where the expert cache copies them from and the CPU reads them.
//   SingleDevice(vision_rank)  vision/*.
//   PrimaryOnly  dflash2/* and proposal/* unless split below: rank 1 holds none of them. Under
//            MTP, proposal/head instead splits by vocabulary Rows when indexed with at most
//            ops::kArgmaxSplitMaxRowsPerRank rows per rank ([131072,5120] -> [65536,5120]);
//            proposal/token_ids stays PrimaryOnly.
//            Under DFlash2 the indexed head splits into ops::kLinearTopKSplitRows blocks and
//            proposal/token_ids splits with it (the split top sixteen keys carry global IDs).
//            NINFER_TP_DRAFT_HEAD=primary, and for DFlash2 NINFER_TP_DRAFTER=primary, keep them
//            PrimaryOnly. At tp 2 the DFlash2 drafter itself splits (split_dflash2_drafter in
//            sharding.cpp), with its candidate selector SingleDevice(1) (PrimaryOnly with Vision on
//            rank 1, NINFER_TP_DRAFT_HEAD=primary or NINFER_TP_SELECTOR=primary).
//
// Qwen4Exp (Qwen3.8-Flash-Next) at tp 2, same contiguous head blocks:
//
//   Rows     attention/{query,gate} by query heads (12 of 24 per rank), attention/{key,value} by
//            KV heads (1 of 2), gdn/* as above (8 of 16 key, 24 of 48 value heads),
//            text/output_head by vocabulary.
//   Columns  attention/output, gdn/output, gdn/convolution as above.
//   Replicated  the QSA indexer (attention/indexer_*: same input, deterministic selection, no
//            exchange), moe/{router,shared_score}, the PLE key|value projection, norms and
//            convolution, text/token_embedding, mtp/{embedding_norm,embedding_projection,
//            hidden_projection}, and every hyper-connection mixer ({attn_hc,ffn_hc}/*,
//            {text,mtp}/hc_head/*, with mtp/hidden_norm): each rank keeps the whole wide
//            residual and mixes it locally.
//   Rows/Columns by residual streams, with options.hc_split_streams instead: rank r keeps streams
//            [r*hc/2, (r+1)*hc/2) — rows of the mixers' norm and Up, of the PLE key/query/conv
//            norms and of mtp/hidden_norm; columns of Down|inject and of the PLE convolution
//            (two all-reduces per mixer, see execution).
//   SingleDevice(block % 2)  moe/shared/{gate,up,down}: the shared expert of Text block i is
//            whole on rank i % 2 (the MTP block follows the Text blocks), since its 640-wide
//            intermediate has no tile-aligned half in the stored formats.
//   HostMapped  routed experts, at every tp (above); rank r computes the experts with ID % 2 == r
//            from its own cache and CPU path, and the rank partials meet in the block's MoE
//            all-reduce.
//   (HostFile)  ple/table: host code reads the mapped table for both ranks.
//
// MTP and a masked draft share text/token_embedding and the selected output head by WeightId,
// so they see that parameter's per-rank view: MTP runs on both ranks and gathers the
// vocabulary-split text/output_head or combines the argmax of the split proposal head; the
// rank-0 drafter requires the whole optimized proposal head (plan_load rejects a masked draft
// with the full head at tp > 1).
struct LogicalShard {
    artifact::ShardAxis axis = artifact::ShardAxis::Replicated;
    int device               = 0; // Holder under SingleDevice.
    std::array<std::vector<artifact::SliceRange>, artifact::kMaximumDevices> ranges;
};

// Throws std::invalid_argument unless options.tp is in [1, 2] and options.vision_rank names one
// of its ranks. plan_load() checks this once before binding; logical_shard() and
// parent_placements() check it again for their direct callers.
void validate_tensor_parallel_ranks(const LoadOptions& options);

// Throws std::invalid_argument for a parameter without a tensor-parallel rule, for a head or
// width that does not divide by options.tp, and for options validate_tensor_parallel_ranks()
// rejects.
[[nodiscard]] LogicalShard logical_shard(std::string_view name, const artifact::Shape& shape,
                                         const Config& config, const LoadOptions& options);

using GeometryLookup = std::function<const WeightGeometry&(artifact::ObjectHandle)>;

// Combines the logical placements of every parameter bound to one parent. Row ranges map
// through the Binding parts to whole parent rows and must partition the parent; column ranges
// must agree across parameters and partition its columns; complete-parent kinds must agree.
// Indexed by object; parents that no parameter demands on a device stay empty.
[[nodiscard]] std::vector<std::optional<artifact::ShardPlacement>>
parent_placements(std::span<const PendingWeight> weights, std::size_t object_count,
                  const GeometryLookup& geometry, const Config& config, const LoadOptions& options);

// Installs the combined placements on a Binder planned for options.tp devices. No-op at tp 1
// unless options.embedding_host or a Qwen4Exp model, whose HostMapped parents need the resolver
// there too.
void install_shard_resolver(artifact::Binder& binder, std::span<const PendingWeight> weights,
                            const Config& config, const LoadOptions& options);

} // namespace ninfer::models::qwen3_5::loading
