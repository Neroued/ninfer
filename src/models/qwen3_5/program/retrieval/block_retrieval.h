#pragma once

// Block-sparse KV retrieval (KVMem-style): pure host-side selection logic for the
// device working-set placement. The module owns no GPU memory and launches no kernels;
// it consumes per-block mean-K features harvested by the engine and produces the
// selected block set consumed by KVAddressSpaceStore::apply_device_placement.
//
// Terminology mirrors the upstream design:
//   - A retrieval block is `block_tokens` tokens (a whole number of 64-token pages).
//   - mean-K is the content-domain (pre-RoPE, post-QK-norm) average key of a block,
//     maintained per attention layer and KV head in FP32.
//   - The query is the mean pre-RoPE Q over the current query span, per layer and
//     query head; GQA groups map onto KV heads by summation before scoring.

#include <cstdint>
#include <algorithm>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct RetrievalBlockMeta {
    std::uint32_t block_id    = 0;  // dense append order
    std::uint32_t pos_start   = 0;  // first original token position
    std::uint32_t n_tokens    = 0;  // tokens in this block (<= block_tokens)
    bool full                 = false;  // block reached block_tokens (eligible for scoring)
};

// Per-(layer, kv_head) mean-K storage for one sequence. layout:
//   mean_k[block][layer * kv_heads + head][dim]
// Empty (not-yet-full) blocks have no entry; entries arrive when a block fills.
class RetrievalIndex {
public:
    RetrievalIndex(std::uint32_t block_tokens, std::uint32_t layers, std::uint32_t kv_heads,
                   std::uint32_t head_dim);

    std::uint32_t block_tokens() const noexcept { return block_tokens_; }
    std::uint32_t layers() const noexcept { return layers_; }
    std::uint32_t kv_heads() const noexcept { return kv_heads_; }
    std::uint32_t head_dim() const noexcept { return head_dim_; }

    // Registers/extends blocks as the sequence grows; returns blocks that became full.
    // Mirrors the append order: blocks are dense and consecutive.
    std::vector<RetrievalBlockMeta> append(std::uint32_t n_new_tokens);

    // Rewinds the store to exactly `token_pos` tokens (the inverse of append): partially
    // covered trailing blocks shrink in place, fully-past blocks drop with their mean-K.
    void truncate_to(std::uint32_t token_pos);

    // Publishes the mean-K for one full block (single layer, all KV heads, dim values).
    // `mean` holds kv_heads * head_dim values in [head][dim] order. Overwrite is an error.
    void write_block_mean(std::uint32_t block_id, std::uint32_t layer,
                          std::span<const float> mean);

    std::uint32_t block_count() const noexcept;
    std::uint32_t total_tokens() const noexcept;
    const RetrievalBlockMeta& block(std::uint32_t block_id) const;

    // Cosine similarity of the query against one block, averaged over KV heads within
    // each layer and over layers that have both a live query accumulator and a block
    // mean-K entry. `query` holds layers * kv_heads * head_dim values in GQA-summed
    // [layer][kv_head][dim] order; a layer participates when its query_norm[layer] > 0.
    // Returns false when no layer can score (block stays unscored, selector keeps it
    // only through sink/recent/mandatory rules).
    [[nodiscard]] bool score(std::uint32_t block_id, std::span<const float> query,
                             std::span<const std::uint32_t> query_count,
                             float& out_score) const;

private:
    std::uint32_t head_stride() const noexcept { return kv_heads_ * head_dim_; }
    float* block_layer(std::uint32_t block_id, std::uint32_t layer) noexcept;
    const float* block_layer(std::uint32_t block_id, std::uint32_t layer) const noexcept;

    std::uint32_t block_tokens_;
    std::uint32_t layers_;
    std::uint32_t kv_heads_;
    std::uint32_t head_dim_;
    std::vector<RetrievalBlockMeta> blocks_;
    std::uint32_t total_tokens_ = 0;
    // mean_k_[block][layer * kv_heads * head_dim + ...], empty for non-full blocks.
    std::vector<std::vector<float>> mean_k_;
};

struct BlockSelectionConfig {
    std::uint32_t block_tokens = 128;  // must match the index
    std::uint32_t budget_blocks = 0;   // device working-set block budget (sink+recent+top-k)
    std::uint32_t sink_blocks   = 1;   // always-kept prefix blocks (system prompt)
    std::uint32_t recent_blocks = 0;   // always-kept suffix blocks (0 = none)
    std::uint32_t reserve_growth_pages = 0;  // pages the caller keeps outside the budget
};

struct BlockSelection {
    // Selected block ids in ascending order; always within the configured budget.
    std::vector<std::uint32_t> selected;
    std::uint32_t scored_blocks = 0;   // blocks that received a retrieval score
    std::uint32_t mandatory_kept = 0;  // mandatory blocks charged against the budget
};

// Cumulative-attention-style top-k selection with structural keeps. `scores` is indexed by
// block id (NaN entries are treated as unscored). `mandatory` block ids are kept first and
// consume ordinary budget slots; sink prefix and recent suffix follow; the remaining budget
// fills by descending score. Growth tail pages beyond the last block boundary are the
// caller's responsibility and are not part of the selection.
BlockSelection select_blocks(const RetrievalIndex& index, std::span<const float> scores,
                             const BlockSelectionConfig& config,
                             std::span<const std::uint32_t> mandatory = {});

// Rolling prefill window over the mapped prefix: the sink prefix plus the newest
// `window_pages` pages. A window that covers everything returns the full set, so short
// prompts never demote. Ascending page indexes.
inline std::vector<std::uint32_t> prefill_window_page_set(std::uint32_t mapped_pages,
                                                          std::uint32_t sink_pages,
                                                          std::uint32_t window_pages) {
    if (mapped_pages == 0) { return {}; }
    if (sink_pages >= mapped_pages || mapped_pages <= sink_pages + window_pages) {
        std::vector<std::uint32_t> all(mapped_pages);
        for (std::uint32_t page = 0; page < mapped_pages; ++page) { all[page] = page; }
        return all;
    }
    std::vector<std::uint32_t> pages;
    pages.reserve(sink_pages + window_pages);
    for (std::uint32_t page = 0; page < sink_pages; ++page) { pages.push_back(page); }
    for (std::uint32_t page = mapped_pages - window_pages; page < mapped_pages; ++page) {
        pages.push_back(page);
    }
    return pages;
}

// Expands ascending retrieval blocks into their page indexes (block_tokens is a whole
// number of 64-token pages). Growth tail pages of a partial trailing block are not part
// of any block and stay out; the caller keeps them materialized on its own.
inline std::vector<std::uint32_t> block_pages(std::span<const std::uint32_t> blocks,
                                              std::uint32_t block_tokens) {
    constexpr std::uint32_t page_size = 64;
    if (block_tokens % page_size != 0) { return {}; }
    const std::uint32_t pages_per_block = block_tokens / page_size;
    std::vector<std::uint32_t> pages;
    pages.reserve(blocks.size() * pages_per_block);
    for (const std::uint32_t block : blocks) {
        for (std::uint32_t offset = 0; offset < pages_per_block; ++offset) {
            pages.push_back(block * pages_per_block + offset);
        }
    }
    return pages;
}

// Preserve retrieved historical pages throughout decode; only the remaining share
// rolls with generated tokens. The total includes the sink and never exceeds budget.
inline std::vector<std::uint32_t> decode_window_page_set(
    std::uint32_t mapped, std::uint32_t budget, std::span<const std::uint32_t> retrieved) {
    std::vector<std::uint32_t> pages;
    for (std::uint32_t p = 0; p < std::min({mapped, budget, 2U}); ++p) { pages.push_back(p); }
    for (const auto p : retrieved) {
        if (p < mapped && p >= 2 && pages.size() < budget) { pages.push_back(p); }
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    for (std::uint32_t p = mapped; p > 0 && pages.size() < budget;) {
        --p;
        if (std::find(pages.begin(), pages.end(), p) == pages.end()) { pages.push_back(p); }
    }
    std::sort(pages.begin(), pages.end());
    return pages;
}

} // namespace ninfer::models::qwen3_5::detail
