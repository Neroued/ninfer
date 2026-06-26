#pragma once

#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/tensor.h"
#include "qus/core/weight_store_parser.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qus {

struct LoadOptions {
    bool load_text          = true;
    bool load_mtp           = false;
    bool load_vision        = false;
    Q5090Progress* progress = nullptr;
    std::vector<std::string> required_text_tensors;
};

class WeightStore {
public:
    explicit WeightStore(Q5090Expectations expected = {});

    void load(const char* path, DeviceArena& weights, DeviceContext& ctx,
              const LoadOptions& options = {});

    const Tensor* tensor(std::string_view name) const noexcept;
    const Weight* qweight(std::string_view name) const noexcept;
    const Tensor* tensor(ModuleKind module, std::uint32_t source_kind,
                         std::uint32_t source_layer) const noexcept;
    const Weight* qweight(ModuleKind module, std::uint32_t source_kind,
                          std::uint32_t source_layer) const noexcept;

    std::size_t tensor_count() const noexcept;
    std::size_t quant_count() const noexcept;
    std::size_t loaded_payload_bytes() const noexcept;
    std::size_t module_tensor_count(ModuleKind module) const noexcept;
    bool module_loaded(ModuleKind module) const noexcept;
    void clear() noexcept;

private:
    struct TensorRecord {
        std::string name;
        ModuleKind module           = ModuleKind::TextCore;
        std::uint32_t source_kind   = 0;
        std::uint32_t source_layer  = kQ5090NoLayer;
        std::uint64_t payload_bytes = 0;
        Tensor tensor;
    };

    struct QuantRecord {
        std::string name;
        ModuleKind module          = ModuleKind::TextCore;
        std::uint32_t source_kind  = 0;
        std::uint32_t source_layer = kQ5090NoLayer;
        Weight weight;
    };

    struct ModuleState {
        bool present                = false;
        bool loaded                 = false;
        std::uint64_t tensor_count  = 0;
        std::uint64_t payload_bytes = 0;
    };

    Q5090Expectations expected_;
    std::vector<TensorRecord> tensors_;
    std::vector<QuantRecord> quant_;
    ModuleState modules_[3];
    std::size_t total_tensor_count_   = 0;
    std::size_t loaded_payload_bytes_ = 0;
};

} // namespace qus
