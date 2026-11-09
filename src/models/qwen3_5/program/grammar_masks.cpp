// Structured output (P5/P6): host-side token masks of the constrained rows, uploaded before the
// round's launch (decision D2: no graph split; the masks are operands at fixed addresses read by
// the sampler through SamplingConfig::mask). capy-flash file; the fill/bind shape follows upstream
// 08902f73 (program/grammar_masks.cpp), with per-lane slots and a pinned staging area.

#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

// NINFER_CONSTRAINT_MASK_STATS=1: periodic stderr summary of the host mask work (measurement).
struct MaskStats {
    bool enabled = [] {
        const char* value = std::getenv("NINFER_CONSTRAINT_MASK_STATS");
        return value != nullptr && value[0] == '1';
    }();
    std::uint64_t fills     = 0;
    std::uint64_t positions = 0;
    std::uint64_t ns        = 0;

    void record(std::size_t filled, std::chrono::steady_clock::duration elapsed) noexcept {
        ++fills;
        positions += filled;
        ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        if (fills % 256 == 0) {
            std::fprintf(stderr,
                         "[constraint] %llu mask fills, %.2f positions/fill, %.1f us/fill, "
                         "%.1f us/position\n",
                         static_cast<unsigned long long>(fills),
                         static_cast<double>(positions) / static_cast<double>(fills),
                         static_cast<double>(ns) / 1e3 / static_cast<double>(fills),
                         static_cast<double>(ns) / 1e3 / static_cast<double>(positions));
        }
    }
};

MaskStats& mask_stats() {
    static MaskStats stats;
    return stats;
}

// Qwen3.8: 7 753 mask words (31 012 bytes) per position; with an odd positions x lanes product
// (--spec off with C=1 or 3, DFlash2 with odd C) the raw end is 4 mod 8.
static_assert(ProgramImpl::grammar_staged_config_offset(31'012U * 3U) %
                      alignof(ops::SamplingConfig) ==
                  0 &&
              ProgramImpl::grammar_staged_config_offset(31'012U * 3U) >= 31'012U * 3U &&
              ProgramImpl::grammar_staged_config_offset(31'012U * 4U) == 31'012U * 4U);

} // namespace

ProgramImpl::RoundMaskScope::RoundMaskScope(ProgramImpl& program,
                                            runtime::TokenMaskProvider* masks) noexcept
    : program_(program) {
    program_.round_masks = masks;
    program_.grammar_dead_positions.fill(0);
}

ProgramImpl::RoundMaskScope::~RoundMaskScope() {
    program_.round_masks = nullptr;
    program_.grammar_dead_positions.fill(0);
}

bool ProgramImpl::row_constrained(std::size_t row) const noexcept {
    return round_masks != nullptr && round_masks->constrained(row);
}

std::size_t ProgramImpl::grammar_mask_positions() const noexcept {
    return grammar_masks_device.data != nullptr
               ? static_cast<std::size_t>(grammar_masks_device.ne[1])
               : 0U;
}

ops::SamplingMask ProgramImpl::fill_grammar_mask(std::size_t row, std::uint32_t lane,
                                                 std::span<const TokenId> drafts) {
    grammar_dead_positions.at(row) = 0;
    if (!row_constrained(row)) { return {}; }
    if (!grammar_masks_host || grammar_masks_device.data == nullptr) {
        throw std::logic_error("structured output mask buffer was not planned");
    }
    const auto words              = static_cast<std::size_t>(grammar_masks_device.ne[0]);
    const std::size_t positions   = drafts.size() + 1U;
    const std::size_t slot_extent = grammar_mask_positions();
    if (lane >= max_concurrency || positions > slot_extent) {
        throw std::logic_error("structured output mask exceeds its lane slot");
    }
    const auto started       = std::chrono::steady_clock::now();
    const std::size_t offset = static_cast<std::size_t>(lane) * slot_extent * words;
    auto* const host_words   = static_cast<std::uint32_t*>(grammar_masks_host->data()) + offset;
    const std::span<std::uint32_t> host(host_words, positions * words);
    std::uint32_t dead = round_masks->fill(row, drafts, host);
    // An empty mask on a consumed position traps the sampler and poisons the CUDA context. The
    // provider never returns one (dead or unreachable positions carry a one-token mask); check it
    // anyway before anything reaches the device.
    for (std::size_t position = 0; position < positions; ++position) {
        const auto mask = host.subspan(position * words, words);
        if (std::all_of(mask.begin(), mask.end(), [](std::uint32_t word) { return word == 0; })) {
            mask[0] = 1U;
            dead |= 1U << position;
        }
    }
    grammar_dead_positions[row] = dead;
    auto* const device_words    = static_cast<std::uint32_t*>(grammar_masks_device.data) + offset;
    CUDA_CHECK(cudaMemcpyAsync(device_words, host.data(), host.size_bytes(), cudaMemcpyHostToDevice,
                               device.stream));
    if (MaskStats& stats = mask_stats(); stats.enabled) {
        stats.record(positions, std::chrono::steady_clock::now() - started);
    }
    return {device_words, static_cast<std::int32_t>(words)};
}

void ProgramImpl::bind_prefill_grammar_mask(SequenceState& sequence, RequestControl& request) {
    // The first token is sampled with the lane's device config (sampling_config[:, lane]); the
    // matcher does not move during the prompt, so its mask is bound once, by the call that samples.
    auto* const staged = reinterpret_cast<ops::SamplingConfig*>(
        static_cast<unsigned char*>(grammar_masks_host->data()) +
        grammar_staged_config_offset(grammar_masks_device.bytes()));
    ops::SamplingConfig& config = staged[sequence.lane];
    config                      = request.sampling_host;
    config.mask                 = fill_grammar_mask(0, sequence.lane, {});
    Tensor lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(
        cudaMemcpyAsync(lane.data, &config, sizeof(config), cudaMemcpyHostToDevice, device.stream));
}

bool ProgramImpl::sampling_bypass_hooks() const noexcept {
    return !qwen4_forced_tokens.empty() || !qwen4_mtp_oracle.empty() || qwen4_replay != nullptr ||
           qwen4_tfdump != nullptr;
}

} // namespace ninfer::models::qwen3_5::detail
