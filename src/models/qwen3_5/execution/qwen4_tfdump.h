#pragma once

// Teacher-forcing dumps of a Qwen4Exp Program in the tfdump/1 format of a teacher-forcing
// correctness harness (an FP64 / Hugging Face reference that writes the same layout): per-position
// .npy files under one directory
//
//   tokens.npy                    int64 [T]  the scored sequence
//   logits/pNNNNNN.npy            f32 [V]    output-head logits after consuming position N
//   hidden/LXX/pNNNNNN.npy        f32 [S*H]  wide residual after decoder layer XX
//   hidden/final/pNNNNNN.npy      f32 [H]    output hyper-connection mixer (input of the head)
//   rank_check.txt                at tp 2: per tapped layer, BF16 elements that differ between the
//                                 ranks' replicated residuals (0 expected)
//   meta.json
//
// Enabled by the environment (a validation aid, off by default): NINFER_QWEN4_TFDUMP=<dir>;
// NINFER_QWEN4_TFDUMP_LAYERS=<comma list of layer indices>|all|none (default all);
// NINFER_QWEN4_TFDUMP_LOGITS=0 disables the per-position logits (1 MB each).
// NINFER_QWEN4_TFDUMP_FORCE=<ids file> teacher-forces decode: the token appended at sequence index
// i is the file's i-th id (whitespace-separated, prompt included) instead of the sampled one, so
// the decode route can be compared with a reference on a fixed sequence.
// Hidden taps run only in eager Prefill (each tap synchronizes the stream); logits are written by
// the Program after a scored tile or a decode round has completed.

#include "core/tensor.h"
#include "models/qwen3_5/execution/qwen4_runtime.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// Validation knobs that change the generated tokens (NINFER_QWEN4_FORCE_TOKENS,
// NINFER_QWEN4_TFDUMP_FORCE, NINFER_QWEN4_EXPERT_FREEZE, NINFER_QWEN4_REPLAY) exist for exactness
// checks against a reference; a Program refuses them (std::invalid_argument naming the knob) unless
// NINFER_QWEN4_VALIDATION=1 is set as well, so they cannot reach a serving process by accident.
void require_qwen4_validation_opt_in();

class Qwen4TfDump final : public Qwen4Taps {
public:
    // Null when NINFER_QWEN4_TFDUMP is unset.
    [[nodiscard]] static std::unique_ptr<Qwen4TfDump> from_environment(std::int32_t layers);

    Qwen4TfDump(std::filesystem::path directory, std::vector<bool> layers, bool logits);

    [[nodiscard]] bool wants(std::int32_t layer) const override;
    void capture(std::int32_t layer, const Tensor& values, std::int32_t first_position,
                 cudaStream_t stream) override;
    // Appends "layer N columns C bf16_mismatches M of E" to rank_check.txt (tp 2).
    void compare_ranks(std::int32_t layer, const Tensor& rank0, const Tensor& rank1,
                       cudaStream_t stream0, cudaStream_t stream1) override;

    [[nodiscard]] bool wants_mtp_logits() const override { return mtp_logits_; }

    void capture_mtp_logits(const Tensor& logits, std::int32_t position,
                            cudaStream_t stream) override;

    void capture_tokens(std::span<const std::int32_t> tokens) override { write_tokens(tokens); }

    [[nodiscard]] bool wants_exchange() const override { return !xdir_.empty(); }

    void capture_exchange(std::int32_t layer, int site, const Tensor& rank0, const Tensor& rank1,
                          std::int32_t first_position, cudaStream_t stream0,
                          cudaStream_t stream1) override;

    // NINFER_QWEN4_TFDUMP_MTP=1: dump the MTP logits of every prompt column (mtp_logits/).
    void set_mtp_logits(bool enabled) noexcept { mtp_logits_ = enabled; }

    [[nodiscard]] bool logits_enabled() const noexcept { return logits_; }

    // Forced id of sequence index `index`, or -1 (no forcing / past the file).
    [[nodiscard]] std::int32_t forced_token(std::size_t index) const noexcept {
        return index < forced_.size() ? forced_[index] : -1;
    }

    void write_tokens(std::span<const std::int32_t> tokens) const;
    // BF16 [V, columns] host logits; column c is position positions[c].
    void write_logits(std::span<const std::uint16_t> logits, std::int32_t rows,
                      std::span<const std::int32_t> positions) const;
    void write_meta(const std::string& producer, const std::string& artifact) const;

private:
    std::filesystem::path directory_;
    std::vector<bool> layers_;
    bool logits_     = true;
    bool mtp_logits_ = false;
    std::vector<std::uint16_t> scratch_;
    std::vector<std::int32_t> forced_;
    // NINFER_QWEN4_TFDUMP_STRIDE: logits only at positions % stride == 0 (KL over long windows).
    std::int32_t stride_ = 1;
    std::filesystem::path xdir_; // T128 exchange taps (empty = off)
    std::int32_t xstride_ = 16;
    // NINFER_QWEN4_TFDUMP_SEQUENCES=1: every write_tokens (one scored sequence) opens seqNNN/.
    bool sequences_ = false;
    // MTP training data (T194): NINFER_QWEN4_TFDUMP_COMPACT=1 appends the hidden taps of a sequence
    // to one file per layer, hidden_LXX.bin = records {i32 first_position, i32 columns, i32 rows,
    // bf16 [columns][rows]}; NINFER_QWEN4_TFDUMP_TOPK=k writes the k most likely tokens of every
    // scored position instead of the whole logits, topk.bin = records {i32 position, i32 ids[k],
    // f32 logprobs[k]} (log-softmax over all rows).
    bool compact_         = false;
    std::int32_t topk_    = 0;
    mutable int sequence_ = -1;
    mutable std::filesystem::path current_;

    [[nodiscard]] const std::filesystem::path& out_dir() const {
        return current_.empty() ? directory_ : current_;
    }

public:
    void set_forced(std::vector<std::int32_t> ids) { forced_ = std::move(ids); }
};

// T59 fast bench: teacher-forced replay of a recorded decode (NINFER_QWEN4_REPLAY=<file>,
// validation-gated: it changes the generated tokens). File: one "seq <ids...>" line per recorded
// request (prompt + its reference continuation), each followed by the request's MTP rounds,
// "win <frontier> <extent> <accepted> <draft_1> .. <draft_K>" (drafts as the round verified them;
// written by NINFER_QWEN4_REPLAY_RECORD=<file> as "req"/"win" lines, completed by
// tools/ws/fastbench.py). A request whose ledger is a prefix of a "seq" line then commits exactly
// the reference: the first token after the prompt and every ordinary decode token are forced,
// and every MTP round verifies the recorded extent with the recorded drafts while the reference
// continuation replaces the target argmax before acceptance (MtpDecodeIngress::forced_targets),
// so it accepts the recorded count and commits the reference tokens. Every run of the request
// then executes the identical sequence of windows (extents, drafts, commits).
struct Qwen4ReplayWindow {
    std::uint32_t extent   = 0;
    std::uint32_t accepted = 0;
    std::vector<std::int32_t> drafts;
};

struct Qwen4ReplaySequence {
    std::vector<std::int32_t> ids;
    std::unordered_map<std::uint32_t, Qwen4ReplayWindow> windows; // by frontier
};

class Qwen4Replay {
public:
    // Null when NINFER_QWEN4_REPLAY is unset.
    [[nodiscard]] static std::unique_ptr<Qwen4Replay> from_environment();
    explicit Qwen4Replay(const std::string& path);
    ~Qwen4Replay();
    Qwen4Replay(const Qwen4Replay&)            = delete;
    Qwen4Replay& operator=(const Qwen4Replay&) = delete;

    // The sequence whose ids start with `ledger` (and are longer), or null.
    [[nodiscard]] const Qwen4ReplaySequence* match(std::span<const std::int32_t> ledger) const;

    std::int64_t forced_tokens        = 0; // prefill first tokens + ordinary decode tokens
    std::int64_t windows_replayed     = 0; // MTP rounds with a recorded window
    std::int64_t windows_unlisted     = 0; // MTP rounds of a matched sequence without one
    std::int64_t rounds_unmatched     = 0; // rounds of a request that matches no sequence
    std::int64_t commit_mismatches    = 0; // MTP commits that differ from the reference
    std::int64_t argmax_disagreements = 0; // ordinary rounds whose own argmax is not the reference

private:
    std::vector<Qwen4ReplaySequence> sequences_;
    mutable const Qwen4ReplaySequence* last_ = nullptr;
};

// NINFER_QWEN4_REPLAY is set (read once).
[[nodiscard]] bool qwen4_replay_enabled();

} // namespace ninfer::models::qwen3_5::execution
