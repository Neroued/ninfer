#pragma once

#include <compare>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models {

struct PhysicalBatchShape {
    std::uint32_t rows                                = 0;
    std::uint32_t attention_rows                      = 0;
    auto operator<=>(const PhysicalBatchShape&) const = default;
};

struct BatchExecutionSpec {
    std::uint32_t resident_capacity = 0;
    std::uint32_t attention_owners  = 1;
    std::vector<std::uint32_t> graph_rows;
    bool operator==(const BatchExecutionSpec&) const = default;
};

[[nodiscard]] BatchExecutionSpec make_batch_execution_spec(std::uint32_t resident_capacity,
                                                           std::uint32_t attention_owners,
                                                           bool capture_decode);

// Row geometry only. Logical membership, stage widths and state belong to the model Work.
class BatchExecutionPlan {
public:
    explicit BatchExecutionPlan(BatchExecutionSpec);

    [[nodiscard]] const BatchExecutionSpec& spec() const noexcept { return spec_; }

    [[nodiscard]] std::uint32_t resident_capacity() const noexcept {
        return spec_.resident_capacity;
    }

    [[nodiscard]] std::uint32_t attention_owners() const noexcept { return spec_.attention_owners; }

    [[nodiscard]] std::span<const std::uint32_t> graph_rows() const noexcept {
        return spec_.graph_rows;
    }

    [[nodiscard]] std::span<const PhysicalBatchShape> graph_shapes() const noexcept {
        return graph_shapes_;
    }

    [[nodiscard]] PhysicalBatchShape decode(std::uint32_t live_rows,
                                            std::uint32_t maximum_owner_rows) const;
    [[nodiscard]] PhysicalBatchShape exact(std::uint32_t live_rows,
                                           std::uint32_t maximum_owner_rows) const;
    [[nodiscard]] std::uint32_t minimum_rows(std::uint32_t physical_rows) const;
    // A real owner distribution which selects this shape; global padding has no owner.
    [[nodiscard]] std::vector<std::uint32_t> representative_owners(PhysicalBatchShape) const;

private:
    BatchExecutionSpec spec_;
    std::vector<PhysicalBatchShape> graph_shapes_;
};

} // namespace ninfer::models
