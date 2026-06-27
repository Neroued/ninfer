#pragma once

#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/model/model.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace qus {

struct ArenaMemoryStats {
    bool present = false;
    std::size_t capacity_bytes = 0;
    std::size_t used_bytes = 0;
    std::size_t peak_used_bytes = 0;
};

struct EngineMemoryStats {
    bool loaded = false;
    int device = 0;
    std::uint32_t max_context = 0;
    std::uint32_t position = 0;
    ArenaMemoryStats weights;
    ArenaMemoryStats cache;
    ArenaMemoryStats workspace;
    std::size_t q5090_loaded_payload_bytes = 0;
    std::size_t q5090_tensor_count = 0;
    std::size_t q5090_quant_count = 0;
};

struct EngineOptions {
    int device               = 0;
    std::uint32_t max_ctx    = 2048;
    std::size_t weight_bytes = 0;
    std::size_t cache_bytes  = 0;
    std::size_t work_bytes   = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    Q5090Progress* progress  = nullptr;
    int eos_token_id         = -1;
};

class Engine {
public:
    explicit Engine(EngineOptions options = {});

    void load(const std::string& path);
    int prefill(std::span<const int> ids);
    int decode_step();
    std::vector<int> generate(std::span<const int> prompt, int max_new_tokens);

    [[nodiscard]] bool loaded() const noexcept { return card_.has_value(); }

    [[nodiscard]] std::uint32_t position() const noexcept;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return options_.max_ctx; }

    [[nodiscard]] EngineMemoryStats memory_stats() const noexcept;

    void reset_memory_peaks() noexcept;

private:
    [[nodiscard]] static Q5090Expectations expectations();
    [[nodiscard]] static std::size_t default_weight_bytes(const std::string& path);
    [[nodiscard]] static std::size_t default_cache_bytes(std::uint32_t max_ctx);

    void require_loaded() const;
    [[nodiscard]] int read_token();
    [[nodiscard]] bool is_eos_token(int token) const noexcept;

    EngineOptions options_;
    std::optional<DeviceContext> ctx_;
    std::optional<DeviceArena> weight_arena_;
    std::optional<DeviceArena> cache_arena_;
    std::optional<WorkspaceArena> work_;
    std::optional<WeightStore> weights_;
    std::optional<KVCache> kv_;
    std::optional<GdnState> state_;
    model::StepState io_{};
    std::optional<model::Qwen3_6_27B> card_;
};

} // namespace qus
