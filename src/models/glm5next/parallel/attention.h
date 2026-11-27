#pragma once

#include "models/common/parallel.h"
#include "models/common/parallel_execution.h"
#include <algorithm>
#include <optional>

#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::glm5next {

// Replica-local TP ranks in each attention owner, ordered as its head slices. A request
// uses every member of one owner for latent/index history and all TP ranks for KDA.
class AttentionOwners {
public:
    AttentionOwners() = default;

    AttentionOwners(std::uint32_t tensor_ranks, std::uint32_t owners, std::uint32_t stages = 1)
        : tensor_(tensor_ranks), owners_(owners), stages_(stages) {
        if (!owners || !tensor_ || !stages || tensor_ % owners ||
            std::uint64_t(tensor_) * stages > UINT32_MAX)
            throw std::invalid_argument(
                "GLM5Next attention owners must partition each pipeline stage");
        width_ = tensor_ / owners;
        ranks_.clear();
        for (std::uint32_t owner = 0; owner < owners; ++owner)
            for (std::uint32_t stage = 0; stage < stages; ++stage)
                for (std::uint32_t lane = 0; lane < width_; ++lane)
                    ranks_.push_back(stage * tensor_ + owner * width_ + lane);
    }

    [[nodiscard]] std::uint32_t size() const { return owners_; }
    [[nodiscard]] std::uint32_t width() const { return width_; }

    [[nodiscard]] std::uint32_t stage(std::uint32_t rank) const {
        if (rank >= ranks_.size()) throw std::out_of_range("GLM5Next attention rank");
        return rank / tensor_;
    }

    [[nodiscard]] std::uint32_t owner(std::uint32_t rank) const {
        (void)stage(rank);
        return (rank % tensor_) / width_;
    }
    [[nodiscard]] std::span<const std::uint32_t> members(std::uint32_t owner) const {
        if (owner >= owners_) throw std::out_of_range("GLM5Next attention owner");
        const auto count = stages_ * width_;
        return std::span(ranks_).subspan(owner * count, count);
    }

    // Intersect the real (unpadded) sequence-producer columns with each attention
    // owner's contiguous window. All TP members visit broadcasts in this same order.
    template <class Submit>
    void transfer_columns(const TokenPartition& geometry, std::span<const IndexRange> packed,
                          std::optional<std::uint32_t> sequence_owner, bool to_attention,
                          Submit submit) const {
        for (std::uint32_t producer = 0; producer < geometry.ranks; ++producer) {
            const TokenPartition part(geometry.columns, geometry.ranks, producer);
            for (std::uint32_t id = 0; id < size(); ++id) {
                if (sequence_owner && *sequence_owner != id) continue;
                const auto range =
                    sequence_owner ? IndexRange{0, std::uint32_t(geometry.columns)} : packed[id];
                const auto begin = std::max(std::uint32_t(part.begin), range.begin);
                const auto end =
                    std::min(std::uint32_t(part.begin + part.count), range.begin + range.size);
                if (begin >= end || (owner(producer) == id && (!to_attention || width() == 1)))
                    continue;
                submit(IndexRange{begin, end - begin},
                       to_attention ? producer : members(id).front());
            }
        }
    }
private:
    std::vector<std::uint32_t> ranks_{0};
    std::uint32_t tensor_ = 1, owners_ = 1, stages_ = 1, width_ = 1;
};

} // namespace ninfer::models::glm5next
