#include "models/qwen3_5/load/sharding.h"

#include "artifact/reader.h"
#include "ninfer/ops/tp2/argmax.h"
#include "ninfer/ops/tp2/linear_topk.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::loading {
namespace {

using artifact::ShardAxis;
using artifact::SliceRange;

[[noreturn]] void reject(std::string_view name, const std::string& reason) {
    throw std::invalid_argument(std::string(name) + ": tensor parallelism " + reason);
}

LogicalShard whole(ShardAxis axis, int device = 0) { return {.axis = axis, .device = device}; }

// Splits `total` into `tp` equal contiguous blocks aligned to `units` (heads, groups or rows).
LogicalShard even(std::string_view name, ShardAxis axis, std::uint64_t total, std::uint64_t units,
                  int tp) {
    const auto parts = static_cast<std::uint64_t>(tp);
    if (!units || total % units || units % parts) {
        reject(name, "requires " + std::to_string(units) + " units of " + std::to_string(total) +
                         " to divide by " + std::to_string(tp));
    }
    LogicalShard out{.axis = axis};
    const auto count = total / parts;
    for (int device = 0; device < tp; ++device) {
        out.ranges[static_cast<std::size_t>(device)].push_back(
            {static_cast<std::uint64_t>(device) * count, count});
    }
    return out;
}

std::uint64_t leading(std::string_view name, const artifact::Shape& shape) {
    if (shape.empty()) { reject(name, "requires a shaped parameter"); }
    return shape.front();
}

std::uint64_t trailing(std::string_view name, const artifact::Shape& shape) {
    if (shape.size() != 2) { reject(name, "splits the input axis of a matrix only"); }
    return shape.back();
}

// The leaf after "text/layers/<i>/" or "mtp/layers/<i>/", or empty outside a decoder block.
std::string_view block_leaf(std::string_view name) {
    for (const std::string_view prefix : {"text/layers/", "mtp/layers/"}) {
        if (!name.starts_with(prefix)) { continue; }
        const auto slash = name.find('/', prefix.size());
        return slash == std::string_view::npos ? std::string_view{} : name.substr(slash + 1);
    }
    return {};
}

LogicalShard block_shard(std::string_view name, std::string_view leaf, const artifact::Shape& shape,
                         const TextConfig& text, int tp) {
    if (leaf == "input_norm" || leaf == "post_attention_norm" || leaf == "attention/query_norm" ||
        leaf == "attention/key_norm" || leaf == "gdn/norm") {
        return whole(ShardAxis::Replicated);
    }
    if (leaf.starts_with("attention/")) {
        const auto& a = text.attention.value();
        if (leaf == "attention/query" || leaf == "attention/gate") {
            return even(name, ShardAxis::Rows, leading(name, shape), a.num_attention_heads, tp);
        }
        if (leaf == "attention/key" || leaf == "attention/value") {
            return even(name, ShardAxis::Rows, leading(name, shape), a.num_key_value_heads, tp);
        }
        if (leaf == "attention/output") {
            return even(name, ShardAxis::Columns, trailing(name, shape), a.num_attention_heads, tp);
        }
    }
    if (leaf.starts_with("gdn/")) {
        const auto& g = text.gdn.value();
        // Value head h belongs to key head h / (value heads / key heads); contiguous per-rank
        // blocks of both keep that grouping when both head counts divide by tp.
        if (g.linear_num_key_heads % static_cast<std::uint32_t>(tp)) {
            reject(name, "requires GDN key heads to divide by " + std::to_string(tp));
        }
        const auto values = g.linear_num_value_heads;
        if (leaf == "gdn/query" || leaf == "gdn/key") {
            return even(name, ShardAxis::Rows, leading(name, shape), g.linear_num_key_heads, tp);
        }
        if (leaf == "gdn/value" || leaf == "gdn/z" || leaf == "gdn/a_projection" ||
            leaf == "gdn/b_projection" || leaf == "gdn/a_log" || leaf == "gdn/dt_bias") {
            return even(name, ShardAxis::Rows, leading(name, shape), values, tp);
        }
        if (leaf == "gdn/output") {
            return even(name, ShardAxis::Columns, trailing(name, shape), values, tp);
        }
        if (leaf == "gdn/convolution") {
            // Depthwise channels in the input projection's Q|K|V order: each rank keeps the
            // channels its own Q, K and V heads produce, in that order.
            const auto key   = g.key_width();
            const auto value = g.value_width();
            if (trailing(name, shape) != 2 * key + value) {
                reject(name, "requires Q|K|V convolution channels");
            }
            LogicalShard out{.axis = ShardAxis::Columns};
            const std::array<std::array<std::uint64_t, 3>, 3> sections{
                {{0, key, g.linear_num_key_heads},
                 {key, key, g.linear_num_key_heads},
                 {2 * key, value, values}}};
            for (const auto& [begin, width, heads] : sections) {
                const auto split = even(name, ShardAxis::Columns, width, heads, tp);
                for (std::size_t device = 0; device < split.ranges.size(); ++device) {
                    for (const auto& range : split.ranges[device]) {
                        out.ranges[device].push_back({begin + range.begin, range.count});
                    }
                }
            }
            return out;
        }
    }
    if (leaf == "mlp/gate" || leaf == "mlp/up") {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, tp);
    }
    if (leaf == "mlp/down") {
        const auto columns = trailing(name, shape);
        return even(name, ShardAxis::Columns, columns, columns, tp);
    }
    reject(name, "has no placement for this parameter");
}

bool in_component(std::string_view name, std::string_view component) {
    return name.size() > component.size() && name.starts_with(component) &&
           name[component.size()] == '/';
}

std::uint64_t row_elements(const artifact::Shape& shape) {
    return shape.size() > 1 ? weight_element_count(std::span(shape).subspan(1)) : 1;
}

// Sorts, rejects overlaps and joins adjacent ranges. Parameters that share a parent's rows (the
// DFlash2 drafter's context key/value and key/value) demand the same range twice: an exact
// duplicate is one range.
void normalize(std::vector<SliceRange>& ranges, const std::string& id) {
    std::sort(ranges.begin(), ranges.end(), [](const SliceRange& a, const SliceRange& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.count < b.count;
    });
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
    std::vector<SliceRange> out;
    for (const auto& range : ranges) {
        if (!out.empty() && out.back().begin + out.back().count > range.begin) {
            throw std::invalid_argument(id + ": tensor-parallel ranges overlap");
        }
        if (!out.empty() && out.back().begin + out.back().count == range.begin) {
            out.back().count += range.count;
        } else {
            out.push_back(range);
        }
    }
    ranges = std::move(out);
}

// Every index of [0, extent) belongs to exactly one device.
void require_partition(const artifact::ShardPlacement& placement, int tp, std::uint64_t extent,
                       const std::string& id) {
    std::vector<SliceRange> all;
    for (int device = 0; device < tp; ++device) {
        const auto& ranges = placement.device_ranges[static_cast<std::size_t>(device)];
        if (ranges.empty()) {
            throw std::invalid_argument(id + ": tensor-parallel placement leaves device " +
                                        std::to_string(device) + " empty");
        }
        all.insert(all.end(), ranges.begin(), ranges.end());
    }
    normalize(all, id);
    if (all.size() != 1 || all.front().begin != 0 || all.front().count != extent) {
        throw std::invalid_argument(id + ": tensor-parallel ranges do not cover the parent");
    }
}

struct ParentState {
    std::optional<artifact::ShardPlacement> placement;
    std::string first; // First logical parameter, for diagnostics.
};

void merge_axis(ParentState& state, const LogicalShard& shard, const std::string& name,
                const std::string& id) {
    if (!state.placement) {
        state.placement = artifact::ShardPlacement{.axis = shard.axis, .device = shard.device};
        state.first     = name;
        return;
    }
    if (state.placement->axis != shard.axis || state.placement->device != shard.device) {
        throw std::invalid_argument(id + ": " + state.first + " and " + name +
                                    " share a parent with different tensor-parallel placements");
    }
}

// NINFER_TP_DRAFT_HEAD=primary keeps the optimized proposal head whole on rank 0, its placement
// before the vocabulary split, for A/B runs on one binary.
bool proposal_head_primary_requested() {
    const char* value = std::getenv("NINFER_TP_DRAFT_HEAD");
    return value != nullptr && std::string_view(value) == "primary";
}

// NINFER_TP_DRAFTER=primary keeps every DFlash2 drafter weight and its proposal head whole on rank
// 0, the placement before the drafter split, for A/B runs on one binary.
bool drafter_primary_requested() {
    const char* value = std::getenv("NINFER_TP_DRAFTER");
    return value != nullptr && std::string_view(value) == "primary";
}

// The optimized head splits by vocabulary rows when it is indexed (fewer rows than the vocabulary,
// every row a candidate) and each rank's block has a registered split route. MTP proposes on both
// ranks through the split argmax (at most ops::kArgmaxSplitMaxRowsPerRank rows per rank); its
// token IDs stay PrimaryOnly, rank 0 mapping the selected row. DFlash2 ranks candidates through the
// vocabulary-split top sixteen (ops::linear_topk_split, [65536,5120] blocks), whose keys carry
// global token IDs: its token IDs split with the head's rows.
bool split_proposal_rows(std::uint64_t rows, const Config& config, const LoadOptions& options) {
    if (proposal_head_primary_requested() || rows == config.text.vocab_size) { return false; }
    const auto tp = static_cast<std::uint64_t>(options.tp);
    if (rows % tp != 0) { return false; }
    if (options.mtp()) {
        return rows / tp <= static_cast<std::uint64_t>(ops::kArgmaxSplitMaxRowsPerRank);
    }
    if (options.dflash2()) {
        return !drafter_primary_requested() &&
               rows / tp == static_cast<std::uint64_t>(ops::kLinearTopKSplitRows);
    }
    return false;
}

// The DFlash2 drafter at tp 2 (NINFER_TP_DRAFTER=primary keeps it PrimaryOnly): a draft block's
// attention heads and MLP rows split like a Text block's, with the query, key and value heads of
// each rank in contiguous blocks; the convolution kernels, the norms and the feature projection are
// replicated (every rank computes the same context and conv coefficients from its own copy of the
// replicated residual); the candidate selector sits on one rank (selector_rank()).
bool split_dflash2_drafter(const Config& config, const LoadOptions& options) {
    return options.dflash2() && options.tp > 1 && !drafter_primary_requested() && config.draft;
}

// NINFER_TP_SELECTOR=primary keeps the split drafter's candidate selector on rank 0, the placement
// before it moved to rank 1, for A/B runs on one binary.
bool selector_primary_requested() {
    const char* value = std::getenv("NINFER_TP_SELECTOR");
    return value != nullptr && std::string_view(value) == "primary";
}

// The rank of the split drafter's candidate selector (hidden projection and codebooks, 257 MB for
// Qwen3.8-27B): rank 1, which holds no Vision tower and so has the more free memory after the
// weights, leaving rank 0 (the one that bounds the KV capacity) that much more context. Both ranks
// rank the candidates of their proposal head blocks, and the selecting rank sends the drafts and
// their distributions to the other (execution/tp2/draft_tp2.inc). Rank 0 with the Vision tower on
// rank 1, with the proposal head whole on rank 0 (NINFER_TP_DRAFT_HEAD=primary: only rank 0 holds
// the candidates), or with NINFER_TP_SELECTOR=primary. The Program rejects a selector on rank 1
// over a proposal head that did not split (startup_tp2.inc).
int selector_rank(const LoadOptions& options) {
    const bool vision_on_rank1 = options.vision && options.vision_rank == 1;
    return vision_on_rank1 || proposal_head_primary_requested() || selector_primary_requested() ? 0
                                                                                                : 1;
}

LogicalShard draft_block_shard(std::string_view name, std::string_view leaf,
                               const artifact::Shape& shape, const DraftConfig& draft, int tp) {
    if (leaf == "input_norm" || leaf == "post_attention_norm" || leaf == "attention/query_norm" ||
        leaf == "attention/key_norm" || leaf.starts_with("attention_conv/") ||
        leaf.starts_with("mlp_conv/")) {
        return whole(ShardAxis::Replicated);
    }
    const auto& a = draft.attention;
    if (leaf == "attention/query") {
        return even(name, ShardAxis::Rows, leading(name, shape), a.num_attention_heads, tp);
    }
    if (leaf == "attention/key" || leaf == "attention/value" || leaf == "attention/context_key" ||
        leaf == "attention/context_value") {
        return even(name, ShardAxis::Rows, leading(name, shape), a.num_key_value_heads, tp);
    }
    if (leaf == "attention/output") {
        return even(name, ShardAxis::Columns, trailing(name, shape), a.num_attention_heads, tp);
    }
    if (leaf == "mlp/gate" || leaf == "mlp/up") {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, tp);
    }
    if (leaf == "mlp/down") {
        const auto columns = trailing(name, shape);
        return even(name, ShardAxis::Columns, columns, columns, tp);
    }
    reject(name, "has no DFlash2 drafter placement for this parameter");
}

LogicalShard drafter_shard(std::string_view name, const artifact::Shape& shape,
                           const Config& config, const LoadOptions& options) {
    const int tp                      = options.tp;
    constexpr std::string_view layers = "dflash2/layers/";
    if (name.starts_with(layers)) {
        const auto slash = name.find('/', layers.size());
        if (slash != std::string_view::npos) {
            return draft_block_shard(name, name.substr(slash + 1), shape, *config.draft, tp);
        }
    }
    if (name == "dflash2/feature_projection" || name == "dflash2/context_norm" ||
        name == "dflash2/final_norm") {
        return whole(ShardAxis::Replicated);
    }
    if (in_component(name, "dflash2/candidate_selector")) {
        return selector_rank(options) == 0 ? whole(ShardAxis::PrimaryOnly)
                                           : whole(ShardAxis::SingleDevice, 1);
    }
    reject(name, "has no DFlash2 drafter placement for this parameter");
}

bool split_proposal_head(const artifact::Shape& shape, const Config& config,
                         const LoadOptions& options) {
    return shape.size() == 2 && split_proposal_rows(shape.front(), config, options);
}

bool split_proposal_token_ids(const artifact::Shape& shape, const Config& config,
                              const LoadOptions& options) {
    return options.dflash2() && shape.size() == 1 &&
           split_proposal_rows(shape.front(), config, options);
}

// A routed-expert projection of a Text or MTP block: ".../moe/experts/<e>/{gate,up,down}".
bool routed_expert(std::string_view name) {
    return !block_leaf(name).empty() && name.find("/moe/experts/") != std::string_view::npos;
}

// ---- Qwen4Exp (Qwen3.8-Flash-Next)
// ---------------------------------------------------------------

// The decoder block index of a Text or MTP block name: Text block i is i, MTP block i follows the
// Text blocks (num_hidden_layers + i).
std::uint32_t qwen4_block_index(std::string_view name, const TextConfig& text) {
    for (const auto& [prefix, base] :
         {std::pair<std::string_view, std::uint32_t>{"text/layers/", 0U},
          std::pair<std::string_view, std::uint32_t>{"mtp/layers/", text.num_hidden_layers}}) {
        if (!name.starts_with(prefix)) { continue; }
        const auto digits =
            name.substr(prefix.size(), name.find('/', prefix.size()) - prefix.size());
        std::uint32_t index = 0;
        for (const char c : digits) {
            if (c < '0' || c > '9') { reject(name, "has a malformed block index"); }
            index = index * 10 + static_cast<std::uint32_t>(c - '0');
        }
        if (digits.empty()) { reject(name, "has a malformed block index"); }
        return base + index;
    }
    reject(name, "is not a decoder block parameter");
}

// One hyper-connection mixer parameter ({norm, down, inject, up}) of a block, of the output mixer
// (text/hc_head, mtp/hc_head), or a parameter laid out over the wide residual (PLE norms and
// convolution, MTP hidden norm). Replicated unless options.hc_split_streams; split, rank r keeps
// the residual streams [r*S/tp, (r+1)*S/tp): rows of the per-stream vectors and of Up, columns of
// Down|inject and of the depthwise convolution.
LogicalShard qwen4_stream_shard(std::string_view name, std::string_view leaf,
                                const artifact::Shape& shape, const TextConfig& text,
                                const LoadOptions& options) {
    if (!options.hc_split_streams) { return whole(ShardAxis::Replicated); }
    const auto streams = text.hyper_connection.value().hc_count;
    const auto wide    = text.residual_width();
    if (leaf == "norm" || leaf == "key_norm" || leaf == "query_norm" || leaf == "conv_norm" ||
        leaf == "hidden_norm" || leaf == "up") {
        if (leading(name, shape) != wide) { reject(name, "requires a wide-residual row axis"); }
        return even(name, ShardAxis::Rows, wide, streams, options.tp);
    }
    if (leaf == "down" || leaf == "inject" || leaf == "convolution") {
        if (trailing(name, shape) != wide) { reject(name, "requires a wide-residual column axis"); }
        return even(name, ShardAxis::Columns, wide, streams, options.tp);
    }
    reject(name, "has no hyper-connection stream placement");
}

// The rank that holds and runs a block's shared expert, whole: SingleDevice(block index % tp).
// The shared expert's 640-wide intermediate has no tile-aligned half for the stored formats (NVFP4
// rows split in 128-row tiles: 320 is not one; Q8 row-split columns in 128s: the Down half of 320
// is not one), so the blocks alternate it between the ranks instead; its output joins the rank's
// MoE partial before the block's one all-reduce.
int qwen4_shared_expert_rank(std::string_view name, const TextConfig& text, int tp) {
    return static_cast<int>(qwen4_block_index(name, text) % static_cast<std::uint32_t>(tp));
}

LogicalShard qwen4_block_shard(std::string_view name, std::string_view leaf,
                               const artifact::Shape& shape, const TextConfig& text,
                               const LoadOptions& options) {
    for (const std::string_view mixer : {"attn_hc/", "ffn_hc/"}) {
        if (leaf.starts_with(mixer)) {
            return qwen4_stream_shard(name, leaf.substr(mixer.size()), shape, text, options);
        }
    }
    // The QSA indexer is replicated: its input is the same on both ranks and its selection is
    // deterministic, so both ranks select the same blocks without an exchange.
    if (leaf.starts_with("attention/indexer_")) { return whole(ShardAxis::Replicated); }
    // Router and shared-expert gate: every rank routes the same tokens to the same experts; the
    // routed experts themselves stay HostMapped and are partitioned by expert ID at run time.
    const bool moe = std::holds_alternative<MoeConfig>(text.ffn);
    if (moe && (leaf == "moe/router" || leaf == "moe/shared_score")) {
        return whole(ShardAxis::Replicated);
    }
    if (moe && leaf.starts_with("moe/shared/")) {
        return whole(ShardAxis::SingleDevice, qwen4_shared_expert_rank(name, text, options.tp));
    }
    if (leaf.starts_with("ple/")) {
        const auto ple_leaf = leaf.substr(4);
        // key|value share one parent, and every rank needs all value rows: replicated in both
        // modes. The table is a HostFile object read by host code, never placed on a device; the
        // FP32 multiplier of a row-scaled NVFP4 table is one value every rank's gather uses.
        if (ple_leaf == "key" || ple_leaf == "value" || ple_leaf == "table" ||
            ple_leaf == "table_global_scale") {
            return whole(ShardAxis::Replicated);
        }
        return qwen4_stream_shard(name, ple_leaf, shape, text, options);
    }
    // Gated attention (12 query and 1 KV heads per rank), GDN (8 key and 24 value heads per
    // rank) and a dense FFN (gate/up rows and down columns halved): the dense Qwen3.5 rules.
    if (leaf.starts_with("attention/") || leaf.starts_with("gdn/") ||
        (leaf.starts_with("mlp/") && !moe)) {
        return block_shard(name, leaf, shape, text, options.tp);
    }
    reject(name, "has no Qwen4Exp placement for this parameter");
}

LogicalShard qwen4_shard(std::string_view name, const artifact::Shape& shape, const Config& config,
                         const LoadOptions& options) {
    const auto& text = config.text;
    if (name == "text/token_embedding" || name == "mtp/embedding_norm" ||
        name == "mtp/embedding_projection" || name == "mtp/hidden_projection") {
        return whole(ShardAxis::Replicated);
    }
    if (name == "text/output_head") {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, options.tp);
    }
    if (name == "mtp/hidden_norm") {
        return qwen4_stream_shard(name, "hidden_norm", shape, text, options);
    }
    for (const std::string_view head : {"text/hc_head/", "mtp/hc_head/"}) {
        if (name.starts_with(head)) {
            return qwen4_stream_shard(name, name.substr(head.size()), shape, text, options);
        }
    }
    if (const auto leaf = block_leaf(name); !leaf.empty()) {
        return qwen4_block_shard(name, leaf, shape, text, options);
    }
    reject(name, "has no Qwen4Exp placement for this parameter");
}

// logical_shard() for options that validate_tensor_parallel_ranks() accepted.
LogicalShard shard_rule(std::string_view name, const artifact::Shape& shape, const Config& config,
                        const LoadOptions& options) {
    const int tp = options.tp;
    if (options.embedding_host && name == "text/token_embedding") {
        return whole(ShardAxis::HostMapped);
    }
    if (config.text.qwen4() && routed_expert(name)) { return whole(ShardAxis::HostMapped); }
    if (tp == 1) { return {}; }
    if (in_component(name, "vision")) {
        return whole(ShardAxis::SingleDevice, options.vision_rank);
    }
    if (config.text.qwen4()) { return qwen4_shard(name, shape, config, options); }
    if ((name == "proposal/head" && split_proposal_head(shape, config, options)) ||
        (name == "proposal/token_ids" && split_proposal_token_ids(shape, config, options))) {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, tp);
    }
    if (in_component(name, "dflash2") && split_dflash2_drafter(config, options)) {
        return drafter_shard(name, shape, config, options);
    }
    if (in_component(name, "dflash2") || in_component(name, "proposal")) {
        return whole(ShardAxis::PrimaryOnly);
    }
    if (name == "text/token_embedding" || name == "text/final_norm" ||
        name == "mtp/embedding_norm" || name == "mtp/hidden_norm" || name == "mtp/final_norm") {
        return whole(ShardAxis::Replicated);
    }
    if (name == "text/output_head") {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, tp);
    }
    if (name == "mtp/input_projection") {
        // [hidden, 2 * hidden] contracts the packed [embedding; hidden] input.
        const auto columns = trailing(name, shape);
        return even(name, ShardAxis::Columns, columns, columns, tp);
    }
    if (const auto leaf = block_leaf(name); !leaf.empty()) {
        return block_shard(name, leaf, shape, config.text, tp);
    }
    reject(name, "has no placement for this parameter");
}

} // namespace

void validate_tensor_parallel_ranks(const LoadOptions& options) {
    if (options.tp < 1 || options.tp > static_cast<int>(artifact::kMaximumDevices)) {
        throw std::invalid_argument("tensor parallelism must be 1 or 2");
    }
    if (options.vision_rank < 0 || options.vision_rank >= options.tp) {
        throw std::invalid_argument("vision_rank must name a tensor-parallel rank");
    }
}

LogicalShard logical_shard(std::string_view name, const artifact::Shape& shape,
                           const Config& config, const LoadOptions& options) {
    validate_tensor_parallel_ranks(options);
    return shard_rule(name, shape, config, options);
}

std::vector<std::optional<artifact::ShardPlacement>>
parent_placements(std::span<const PendingWeight> weights, std::size_t object_count,
                  const GeometryLookup& geometry, const Config& config,
                  const LoadOptions& options) {
    validate_tensor_parallel_ranks(options);
    std::vector<ParentState> states(object_count);
    std::vector<std::string> ids(object_count);
    for (const auto& weight : weights) {
        const auto& reference = weight.reference;
        const auto& name      = reference.name;
        const auto shard      = shard_rule(name, reference.shape, config, options);
        const auto row        = row_elements(reference.shape);
        std::uint64_t logical = 0; // Logical element offset of the current part.
        for (std::size_t i = 0; i < reference.binding.parts.size(); ++i) {
            const auto& part   = reference.binding.parts[i];
            const auto& parent = geometry(part.object);
            auto& state        = states.at(part.object.index);
            auto& id           = ids[part.object.index];
            if (id.empty()) {
                id = i < weight.source_objects.size() ? weight.source_objects[i] : name;
            }
            merge_axis(state, shard, name, id);
            if (shard.axis == ShardAxis::Rows) {
                // Logical row ranges intersect this part's element range as whole parent rows.
                const auto parent_row = row_elements(parent.shape);
                const auto length     = part.end - part.begin;
                for (std::size_t device = 0; device < shard.ranges.size(); ++device) {
                    for (const auto& range : shard.ranges[device]) {
                        const auto first = std::max(range.begin * row, logical);
                        const auto last =
                            std::min((range.begin + range.count) * row, logical + length);
                        if (first >= last) { continue; }
                        const auto begin = part.begin + (first - logical);
                        const auto end   = part.begin + (last - logical);
                        if (begin % parent_row || end % parent_row) {
                            throw std::invalid_argument(
                                name + ": a row split does not fall on rows of parent " + id);
                        }
                        state.placement->device_ranges[device].push_back(
                            {begin / parent_row, (end - begin) / parent_row});
                    }
                }
            } else if (shard.axis == ShardAxis::Columns) {
                // A column split narrows every parent row: the parent must have the logical row
                // width and this parameter must bind whole parent rows.
                const auto columns = reference.shape.back();
                if (parent.shape.size() != 2 || parent.shape[1] != columns ||
                    part.begin % columns || part.end % columns) {
                    throw std::invalid_argument(name + ": a column split requires whole rows of " +
                                                id);
                }
                auto& ranges = state.placement->device_ranges;
                if (std::all_of(ranges.begin(), ranges.end(),
                                [](const auto& list) { return list.empty(); })) {
                    ranges = shard.ranges;
                } else if (ranges != shard.ranges) {
                    throw std::invalid_argument(id + ": " + state.first + " and " + name +
                                                " split the columns of one parent differently");
                }
            }
            logical += part.end - part.begin;
        }
    }
    std::vector<std::optional<artifact::ShardPlacement>> out(object_count);
    for (std::size_t object = 0; object < object_count; ++object) {
        auto& placement = states[object].placement;
        if (!placement) { continue; }
        if (artifact::is_sharded(placement->axis)) {
            const auto& parent = geometry(artifact::ObjectHandle{object});
            for (auto& ranges : placement->device_ranges) { normalize(ranges, ids[object]); }
            const auto extent = placement->axis == ShardAxis::Rows
                                    ? parent.elements / row_elements(parent.shape)
                                    : parent.shape.at(1);
            require_partition(*placement, options.tp, extent, ids[object]);
        }
        out[object] = std::move(placement);
    }
    return out;
}

void install_shard_resolver(artifact::Binder& binder, std::span<const PendingWeight> weights,
                            const Config& config, const LoadOptions& options) {
    if (options.tp == 1 && !options.embedding_host && !config.text.qwen4()) { return; }
    if (binder.device_count() != options.tp) {
        throw std::logic_error("Binder device count differs from tensor parallelism");
    }
    const auto& reader = binder.reader();
    auto placements    = parent_placements(
        weights, reader.directory().objects.size(),
        [&](artifact::ObjectHandle object) -> const WeightGeometry& {
            return reader.geometry(object);
        },
        config, options);
    binder.set_shard_resolver([placements = std::move(placements),
                               &reader](artifact::ObjectHandle object, const WeightGeometry&) {
        const auto& placement = placements.at(object.index);
        if (!placement) {
            throw artifact::ArtifactError(
                artifact::object_id(reader.directory().object(object)) +
                ": device parent has no logical tensor-parallel placement");
        }
        return *placement;
    });
}

} // namespace ninfer::models::qwen3_5::loading
