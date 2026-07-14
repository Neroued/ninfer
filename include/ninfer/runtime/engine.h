#pragma once

#include "ninfer/core/arena.h"
#include "ninfer/core/device.h"
#include "ninfer/core/kv_cache.h"
#include "ninfer/core/state_store.h"
#include "ninfer/core/weight_store.h"
#include "ninfer/kernels/sampling.h"
#include "ninfer/model/model.h"
#include "ninfer/model/processor.h"
#include "ninfer/model/vision.h"
#include "ninfer/runtime/decode_graph.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

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
    DType kv_dtype                         = DType::BF16;
    int kv_quant_group                     = 0;
    std::size_t kv_cache_payload_bytes     = 0;
    std::size_t q5090_loaded_payload_bytes = 0;
    std::size_t q5090_tensor_count         = 0;
    std::size_t q5090_quant_count          = 0;
};

struct EngineMtpStats {
    bool enabled                 = false;
    int k                        = 0;
    std::int64_t draft_tokens    = 0;
    std::int64_t accepted_tokens = 0;
    std::int64_t rounds          = 0;
    std::int64_t fallback_steps  = 0;
    std::array<std::int64_t, model::kMaxMtpDraftTokens> accepted_per_pos{};
};

struct EngineOptions {
    int device                  = 0;
    std::uint32_t max_ctx       = 2048;
    std::size_t cache_bytes     = 0;
    std::size_t work_bytes      = 0;
    std::uint32_t prefill_chunk = model::kDefaultPrefillChunk;
    int mtp_draft_tokens        = 0;
    DType kv_dtype              = DType::BF16;
    int kv_quant_group          = kKvQuantGroup;
    Q5090Progress* progress     = nullptr;
    std::vector<int> stop_token_ids;
    bool use_cuda_graph = true;
    // Two-mode draft toggle. When true, the MTP proposal sites use the embedded
    // Q4 draft head (requires the v4.2 LM_HEAD_DRAFT module). When false, every
    // site uses the full lm_head. Verify/prefill/k0 always use the full head, so
    // emitted tokens are identical in either mode; only decode speed differs.
    bool use_lm_head_draft = false;
};

class Engine {
public:
    explicit Engine(EngineOptions options = {});

    void load(const std::string& path);
    Q5090TokenizerBundle take_tokenizer_bundle();
    void set_stop_token_ids(std::vector<int> ids);

    // Install decode/prefill sampling. temperature <= 0 in `config` is exact
    // greedy argmax (the default when this is never called), so parity runs stay
    // deterministic. `config.token_counts` is ignored/overwritten: the engine owns
    // the per-token count buffer used by the presence/frequency penalties. Safe to
    // call between requests; it only refreshes a device-resident config buffer, so
    // captured CUDA graphs remain valid.
    void set_sampling(const kernels::SamplingConfig& config);

    int prefill(std::span<const int> ids);
    // Full-reset multimodal prefill. Runs the Vision tower, injects merger
    // embeddings, applies three-axis MRoPE, and retains only text-model state.
    int prefill(const model::ProcessedInput& input);
    // Single-user multi-turn prefix caching with partial longest-common-prefix reuse.
    //
    // `content_boundary` is the absolute token position of THIS turn's assistant-content boundary
    // (prompt_tokens - generation-prompt-opener length). The engine snapshots the GDN recurrent
    // state there so a later turn can continue the recurrence from it even after the thinking-
    // stripped re-render diverges right after the assistant header.
    //
    // Reuse decision (most to least reuse): (1) append when the whole committed prefix E is a
    // prefix of `ids`; (2) rewind to the previous turn's boundary B_prev (KV rewind + GDN snapshot
    // restore) when `ids` matches through B_prev; (3) otherwise a full reset prefill. Correct under
    // MTP speculative decode. Returns the first generated token.
    int prefill_cached(std::span<const int> ids, std::uint32_t content_boundary);
    int decode_step();
    std::vector<int> generate(std::span<const int> prompt, int max_new_tokens);

    [[nodiscard]] bool loaded() const noexcept { return load_complete_; }

    [[nodiscard]] std::uint32_t position() const noexcept;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return options_.max_ctx; }

    // Number of resident prefix tokens reused by the most recent prefill_cached() call (0 when it
    // fell back to a full reset prefill). Meaningful only right after a prefill/prefill_cached.
    [[nodiscard]] std::uint32_t last_prefix_cache_hit() const noexcept {
        return last_prefix_hit_tokens_;
    }

    [[nodiscard]] EngineMemoryStats memory_stats() const noexcept;
    [[nodiscard]] const Q5090LoadStats& q5090_load_stats() const;
    [[nodiscard]] EngineMtpStats mtp_stats() const;

    void reset_memory_peaks() noexcept;
    void reset_mtp_stats();

    [[nodiscard]] static std::size_t default_work_bytes(std::uint32_t prefill_chunk);

private:
    [[nodiscard]] static std::size_t default_cache_bytes(std::uint32_t max_ctx);

    void unload() noexcept;
    void require_loaded() const;
    void invalidate_sequence_identity() noexcept;
    void reset_sequence_state();
    void recover_sequence_after_failure() noexcept;
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
    std::optional<DeviceArena> cache_arena_;
    std::optional<WorkspaceArena> work_;
    std::optional<WorkspaceArena> vision_work_;
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
    std::optional<model::Qwen3_6_Vision> vision_card_;
    DecodeGraph decode_graph_;
    DecodeGraph round_graph_;
    bool load_complete_ = false;
    bool decode_warmed_ = false;
    bool round_warmed_  = false;
    std::vector<int> pending_sampled_;
    // Reused prefix length of the most recent prefill_cached() (0 on full reset prefill).
    std::uint32_t last_prefix_hit_tokens_ = 0;
    // Sentinel: no reusable turn-boundary GDN snapshot is available.
    static constexpr std::uint32_t kNoBoundary = 0xFFFFFFFFu;
    // Absolute position of the assistant-content boundary at which the resident GDN boundary slot
    // was snapshotted (kNoBoundary when none). Reuse-B is valid only when the incoming request
    // matches logical_tokens_ through this position AND the snapshot was actually taken here.
    std::uint32_t content_boundary_prev_ = kNoBoundary;
    // Host mirror of the resident logical token sequence: prompt tokens followed by every token
    // returned by prefill/decode. The reusable prefix is logical_tokens_[0 : kv_.pos]; the tail
    // beyond kv_.pos (if any) holds the last emitted-but-uncommitted bonus token.
    std::vector<int> logical_tokens_;
    // Token IDs alone do not identify media content. A multimodal resident is
    // therefore never eligible for the text-only prefix cache.
    bool resident_multimodal_ = false;
};

} // namespace ninfer
