#pragma once

#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/kernels/sampling.h"
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
    // Two-mode draft toggle. When true, the MTP proposal sites use the embedded
    // Q4 draft head (requires a v4 file with DRAFT_HEAD_PRESENT). When false, every
    // site uses the full lm_head. Verify/prefill/k0 always use the full head, so
    // emitted tokens are identical in either mode; only decode speed differs.
    bool use_lm_head_draft = false;
};

class Engine {
public:
    explicit Engine(EngineOptions options = {});

    void load(const std::string& path);

    // Install decode/prefill sampling. temperature <= 0 in `config` is exact
    // greedy argmax (the default when this is never called), so parity runs stay
    // deterministic. `config.token_counts` is ignored/overwritten: the engine owns
    // the per-token count buffer used by the presence/frequency penalties. Safe to
    // call between requests; it only refreshes a device-resident config buffer, so
    // captured CUDA graphs remain valid.
    void set_sampling(const kernels::SamplingConfig& config);

    int prefill(std::span<const int> ids);
    // Single-user multi-turn prefix caching. When the resident cache is an exact prefix of `ids`,
    // prefill only the new suffix (reusing KV + GDN state); otherwise fall back to a full reset
    // prefill. Correct under MTP speculative decode. Returns the first generated token.
    int prefill_cached(std::span<const int> ids);
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
    // Append-only prefill continuation: extends the resident cache with `ids` without resetting
    // kv_/mtp_kv_/state_/gdn_initial_slot. Resolves the GDN read slot from gdn_initial_slot inside
    // the model card, writes the running state to slot 0. Returns the first generated token.
    int prefill_append(std::span<const int> ids);
    // Writes the host-side GDN snapshot slot into the device gdn_initial_slot scalar so the next
    // MTP round / append prefill reads the committed recurrent state from that slot.
    void set_gdn_initial_slot(int slot);
    [[nodiscard]] int read_token();
    [[nodiscard]] bool is_stop_token(int token) const noexcept;
    [[nodiscard]] int decode_step_one();
    [[nodiscard]] std::vector<int> decode_round();
    void record_decode_round();
    void record_propose(int k);
    [[nodiscard]] std::vector<int> read_round_output();

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
    // Device-resident sampling state, owned by the engine and read by the sampler
    // kernels. `sampling_config_dev_` holds one SamplingConfig; `token_counts_` is
    // the [vocab] i32 occurrence buffer for the penalties, reset each prefill.
    Tensor sampling_config_dev_{};
    Tensor token_counts_{};
    kernels::SamplingConfig sampling_host_{};
    std::optional<model::Qwen3_6_27B> card_;
    DecodeGraph decode_graph_;
    DecodeGraph round_graph_;
    bool decode_warmed_ = false;
    bool round_warmed_  = false;
    std::vector<int> pending_sampled_;
    // Host mirror of the resident logical token sequence: prompt tokens followed by every token
    // returned by prefill/decode. The reusable prefix is logical_tokens_[0 : kv_.pos]; the tail
    // beyond kv_.pos (if any) holds the last emitted-but-uncommitted bonus token.
    std::vector<int> logical_tokens_;
};

} // namespace qus
