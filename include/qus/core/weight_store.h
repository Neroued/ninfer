#pragma once

#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/tensor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qus {

enum class WeightEntryKind : std::uint8_t {
    Dense   = 0,
    QuantW4 = 1,
};

struct WeightFileExpectations {
    std::string model_id;
    std::optional<std::uint32_t> hidden_size;
    std::optional<std::uint32_t> intermediate_size;
    std::optional<std::uint32_t> num_layers;
    std::optional<std::uint32_t> full_attention_layers;
    std::optional<std::uint32_t> gdn_layers;
    std::optional<std::uint32_t> attention_heads;
    std::optional<std::uint32_t> kv_heads;
    std::optional<std::uint32_t> head_dim;
    std::optional<std::uint32_t> gdn_key_heads;
    std::optional<std::uint32_t> gdn_value_heads;
    std::optional<std::uint32_t> gdn_value_head_dim;
    std::optional<std::uint32_t> gdn_key_head_dim;
    std::optional<std::uint32_t> gdn_conv_width;
    std::optional<std::uint32_t> vocab_size;
    std::optional<std::uint32_t> max_position_embeddings;
};

class WeightStore {
public:
    explicit WeightStore(WeightFileExpectations expected = {});

    void load(const char* path, DeviceArena& weights, DeviceContext& ctx);
    QuantWeight qweight(std::string_view role, int layer) const;
    Tensor weight(std::string_view role, int layer) const;
    std::size_t dense_count() const noexcept;
    std::size_t quant_count() const noexcept;
    void clear() noexcept;

private:
    struct DenseRecord {
        std::string name;
        std::string role;
        int layer = -1;
        Tensor tensor;
    };

    struct QuantRecord {
        std::string name;
        std::string role;
        int layer = -1;
        QuantWeight weight;
    };

    WeightFileExpectations expected_;
    std::vector<DenseRecord> dense_;
    std::vector<QuantRecord> quant_;
};

} // namespace qus
