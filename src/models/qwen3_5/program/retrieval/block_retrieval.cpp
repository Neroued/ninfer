#include "models/qwen3_5/program/retrieval/block_retrieval.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

namespace {

constexpr float kEpsilon = 1.0e-6F;

} // namespace

RetrievalIndex::RetrievalIndex(std::uint32_t block_tokens, std::uint32_t layers,
                               std::uint32_t kv_heads, std::uint32_t head_dim)
    : block_tokens_(block_tokens), layers_(layers), kv_heads_(kv_heads), head_dim_(head_dim) {
    if (block_tokens == 0 || layers == 0 || kv_heads == 0 || head_dim == 0 ||
        block_tokens % 64U != 0U) {
        throw std::invalid_argument("retrieval index geometry is invalid");
    }
}

std::vector<RetrievalBlockMeta> RetrievalIndex::append(std::uint32_t n_new_tokens) {
    std::vector<RetrievalBlockMeta> filled;
    for (std::uint32_t token = 0; token < n_new_tokens; ++token) {
        if (blocks_.empty() || blocks_.back().full) {
            RetrievalBlockMeta meta;
            meta.block_id  = static_cast<std::uint32_t>(blocks_.size());
            meta.pos_start = total_tokens_;
            blocks_.push_back(meta);
            mean_k_.emplace_back();
        }
        RetrievalBlockMeta& block = blocks_.back();
        ++block.n_tokens;
        ++total_tokens_;
        if (block.n_tokens == block_tokens_) {
            block.full = true;
            filled.push_back(block);
        }
    }
    return filled;
}

void RetrievalIndex::truncate_to(std::uint32_t token_pos) {
    if (token_pos >= total_tokens_) { return; }
    while (!blocks_.empty() && blocks_.back().pos_start >= token_pos) {
        blocks_.pop_back();
        mean_k_.pop_back();
    }
    if (blocks_.empty()) {
        total_tokens_ = 0;
        return;
    }
    RetrievalBlockMeta& tail = blocks_.back();
    const std::uint32_t kept = std::min(tail.n_tokens, token_pos - tail.pos_start);
    if (kept < tail.n_tokens) {
        tail.n_tokens = kept;
        if (!tail.full || kept < block_tokens_) {
            tail.full = false;
            mean_k_.back().clear();
        }
    }
    total_tokens_ = token_pos;
}

void RetrievalIndex::write_block_mean(std::uint32_t block_id, std::uint32_t layer,
                                       std::span<const float> mean) {
    if (block_id >= blocks_.size() || !blocks_[block_id].full) {
        throw std::invalid_argument("retrieval mean-K targets a non-full block");
    }
    if (layer >= layers_ || mean.size() != head_stride()) {
        throw std::invalid_argument("retrieval mean-K geometry is invalid");
    }
    std::vector<float>& block = mean_k_[block_id];
    if (block.empty()) { block.assign(static_cast<std::size_t>(layers_) * head_stride(), 0.0F); }
    std::copy(mean.begin(), mean.end(), block.begin() +
                                           static_cast<std::size_t>(layer) * head_stride());
}

std::uint32_t RetrievalIndex::block_count() const noexcept {
    return static_cast<std::uint32_t>(blocks_.size());
}

std::uint32_t RetrievalIndex::total_tokens() const noexcept { return total_tokens_; }

const RetrievalBlockMeta& RetrievalIndex::block(std::uint32_t block_id) const {
    if (block_id >= blocks_.size()) {
        throw std::out_of_range("retrieval block id is out of range");
    }
    return blocks_[block_id];
}

float* RetrievalIndex::block_layer(std::uint32_t block_id, std::uint32_t layer) noexcept {
    std::vector<float>& block = mean_k_[block_id];
    return block.empty()
               ? nullptr
               : block.data() + static_cast<std::size_t>(layer) * head_stride();
}

const float* RetrievalIndex::block_layer(std::uint32_t block_id,
                                         std::uint32_t layer) const noexcept {
    const std::vector<float>& block = mean_k_[block_id];
    return block.empty()
               ? nullptr
               : block.data() + static_cast<std::size_t>(layer) * head_stride();
}

bool RetrievalIndex::score(std::uint32_t block_id, std::span<const float> query,
                           std::span<const std::uint32_t> query_count,
                           float& out_score) const {
    if (block_id >= blocks_.size() || !blocks_[block_id].full ||
        query.size() != static_cast<std::size_t>(layers_) * head_stride() ||
        query_count.size() != layers_) {
        throw std::invalid_argument("retrieval scoring geometry is invalid");
    }
    double layer_sum     = 0.0;
    std::uint32_t layers = 0;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        if (query_count[layer] == 0) { continue; }
        const float* block_mean = block_layer(block_id, layer);
        if (block_mean == nullptr) { continue; }
        const float* query_layer =
            query.data() + static_cast<std::size_t>(layer) * head_stride();
        double head_sum = 0.0;
        for (std::uint32_t head = 0; head < kv_heads_; ++head) {
            const float* q   = query_layer + static_cast<std::size_t>(head) * head_dim_;
            const float* k   = block_mean + static_cast<std::size_t>(head) * head_dim_;
            double dot       = 0.0;
            double norm_q    = 0.0;
            double norm_k    = 0.0;
            for (std::uint32_t dim = 0; dim < head_dim_; ++dim) {
                dot += static_cast<double>(q[dim]) * static_cast<double>(k[dim]);
                norm_q += static_cast<double>(q[dim]) * static_cast<double>(q[dim]);
                norm_k += static_cast<double>(k[dim]) * static_cast<double>(k[dim]);
            }
            const double denominator = std::sqrt(norm_q) * std::sqrt(norm_k);
            head_sum += denominator > kEpsilon ? dot / denominator : 0.0;
        }
        layer_sum += head_sum / static_cast<double>(kv_heads_);
        ++layers;
    }
    if (layers == 0) { return false; }
    out_score = static_cast<float>(layer_sum / static_cast<double>(layers));
    return true;
}

BlockSelection select_blocks(const RetrievalIndex& index, std::span<const float> scores,
                             const BlockSelectionConfig& config,
                             std::span<const std::uint32_t> mandatory) {
    const std::uint32_t blocks = index.block_count();
    if (scores.size() != blocks) {
        throw std::invalid_argument("block selection score vector is not block aligned");
    }
    if (config.block_tokens != index.block_tokens()) {
        throw std::invalid_argument("block selection geometry does not match the index");
    }
    if (config.budget_blocks == 0) { return {}; }

    std::vector<std::uint8_t> kept(blocks, 0U);
    BlockSelection selection;
    std::uint32_t used = 0;
    const auto try_keep = [&](std::uint32_t block_id) {
        if (used >= config.budget_blocks || kept[block_id]) { return; }
        kept[block_id] = 1U;
        ++used;
    };

    for (const std::uint32_t block_id : mandatory) {
        if (block_id >= blocks) {
            throw std::invalid_argument("mandatory retrieval block is out of range");
        }
        const bool was_kept = kept[block_id] != 0;
        try_keep(block_id);
        if (!was_kept && kept[block_id]) { ++selection.mandatory_kept; }
    }
    for (std::uint32_t block = 0;
         block < std::min(config.sink_blocks, blocks) && used < config.budget_blocks; ++block) {
        try_keep(block);
    }
    const std::uint32_t recent_begin =
        config.recent_blocks > blocks ? 0 : blocks - config.recent_blocks;
    if (recent_begin < blocks) {
        for (std::uint32_t block = blocks; block-- > recent_begin;) { try_keep(block); }
    }

    std::vector<std::uint32_t> ranked;
    ranked.reserve(blocks);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        if (kept[block] || !index.block(block).full) { continue; }
        if (std::isnan(scores[block])) { continue; }
        ++selection.scored_blocks;
        ranked.push_back(block);
    }
    std::sort(ranked.begin(), ranked.end(),
              [&](std::uint32_t lhs, std::uint32_t rhs) {
                  if (scores[lhs] != scores[rhs]) { return scores[lhs] > scores[rhs]; }
                  return lhs < rhs;
              });
    for (const std::uint32_t block : ranked) {
        if (used >= config.budget_blocks) { break; }
        try_keep(block);
    }

    for (std::uint32_t block = 0; block < blocks; ++block) {
        if (kept[block]) { selection.selected.push_back(block); }
    }
    return selection;
}

} // namespace ninfer::models::qwen3_5::detail
