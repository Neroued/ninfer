#pragma once

#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/model/model.h"
#include "qus/runtime/decode_graph.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace qus {

struct ArenaMemoryStats {
    bool present                = false;
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

struct EngineMemoryStats {
    bool loaded               = false;
    int device                = 0;
    std::uint32_t max_context = 0;
    std::uint32_t position    = 0;
    ArenaMemoryStats weights;
    ArenaMemoryStats cache;
    ArenaMemoryStats workspace;
    std::size_t q5090_loaded_payload_bytes = 0;
    std::size_t q5090_tensor_count         = 0;
    std::size_t q5090_quant_count          = 0;
};

struct EngineMtpStats {
    bool enabled = false;
    int k = 0;
    std::int64_t draft_tokens = 0;
    std::int64_t accepted_tokens = 0;
    std::int64_t rounds = 0;
    std::int64_t fallback_steps = 0;
    std::array<std::int64_t, model::kMaxMtpDraftTokens> accepted_per_pos{};
};

struct EngineOptions {
    int device                  = 0;
    std::uint32_t max_ctx       = 2048;
    std::size_t weight_bytes    = 0;
    std::size_t cache_bytes     = 0;
    std::size_t work_bytes      = 0;
    std::uint32_t prefill_chunk = model::kDefaultPrefillChunk;
    int mtp_draft_tokens        = 0;
    Q5090Progress* progress     = nullptr;
    std::vector<int> stop_token_ids;
    bool use_cuda_graph = true;
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
    [[nodiscard]] EngineMtpStats mtp_stats() const;

    void reset_memory_peaks() noexcept;
    void reset_mtp_stats();

    [[nodiscard]] static std::size_t default_work_bytes(std::uint32_t prefill_chunk);

private:
    [[nodiscard]] static Q5090Expectations expectations();
    [[nodiscard]] static std::size_t default_weight_bytes(const std::string& path);
    [[nodiscard]] static std::size_t default_cache_bytes(std::uint32_t max_ctx);

    void require_loaded() const;
    [[nodiscard]] int read_token();
    [[nodiscard]] int read_i32_scalar(const Tensor model::StepState::*field);
    [[nodiscard]] std::vector<int> read_sampled_tokens();
    [[nodiscard]] bool is_stop_token(int token) const noexcept;
    [[nodiscard]] int decode_step_one();
    [[nodiscard]] std::vector<int> decode_round();
    void commit_gdn_snapshots();
    void propose_mtp_after_accept(std::uint32_t host_window_base, int host_length, int k);

    EngineOptions options_;
    std::optional<DeviceContext> ctx_;
    std::optional<DeviceArena> weight_arena_;
    std::optional<DeviceArena> cache_arena_;
    std::optional<WorkspaceArena> work_;
    std::optional<WeightStore> weights_;
    std::optional<KVCache> kv_;
    std::optional<KVCache> mtp_kv_;
    std::optional<GdnState> state_;
    model::StepState io_{};
    std::optional<model::Qwen3_6_27B> card_;
    DecodeGraph decode_graph_;
    bool decode_warmed_ = false;
    std::vector<int> pending_sampled_;
};

} // namespace qus
