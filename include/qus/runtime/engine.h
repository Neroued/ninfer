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

struct EngineOptions {
    int device               = 0;
    std::uint32_t max_ctx    = 2048;
    std::size_t weight_bytes = 0;
    std::size_t cache_bytes  = 0;
    std::size_t work_bytes   = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    Q5090Progress* progress  = nullptr;
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

private:
    [[nodiscard]] static Q5090Expectations expectations();
    [[nodiscard]] static std::size_t default_weight_bytes(const std::string& path);
    [[nodiscard]] static std::size_t default_cache_bytes(std::uint32_t max_ctx);

    void require_loaded() const;
    [[nodiscard]] int read_token();

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
