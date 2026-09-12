#pragma once

#include "core/transfer_work.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::runtime {

inline constexpr std::uint64_t kContextCostQ32One = 1ULL << 32U;

// One machine-level transfer model. Resource type, model, KV dtype, and speculative backend do
// not participate: their physical byte and copy-operation counts already capture the work.
struct ContextTransferCost {
    std::uint64_t batch_ns        = 0;
    std::uint64_t operation_ns    = 0;
    std::uint64_t ns_per_byte_q32 = 0;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferCost,
                                                   ContextTransferCost) noexcept = default;
};

struct ContextPrefillCost {
    std::uint64_t chunk_ns              = 0;
    std::uint64_t token_ns_q32          = 0;
    std::uint64_t attention_pair_ns_q32 = 0;
    std::uint64_t vision_item_ns        = 0;
    std::uint64_t vision_patch_ns_q32   = 0;

    [[nodiscard]] friend constexpr bool operator==(ContextPrefillCost,
                                                   ContextPrefillCost) noexcept = default;
};

enum class MaterializationCopyPhase : std::uint8_t {
    PressureToHost,
    Candidate,
};

// Work already coalesced for one ordered copy phase and one direction. Distinct entries are serial;
// callers merge work that shares both fields before asking the machine model to price it.
struct TransferBatchWork {
    MaterializationCopyPhase phase     = MaterializationCopyPhase::Candidate;
    ContextTransferDirection direction = ContextTransferDirection::HostToDevice;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(TransferBatchWork,
                                                   TransferBatchWork) noexcept = default;
};

struct ContextMachineCostModel {
    // Direction order is DeviceToHost, HostToDevice, DeviceToDevice.
    std::array<ContextTransferCost, 3> transfer{};
    ContextPrefillCost prefill;

    // max(batch + copy_operations * operation, payload_bytes * ns_per_byte)
    [[nodiscard]] std::uint64_t transfer_ns(ContextTransferDirection direction,
                                            TransferWork work) const noexcept;
    [[nodiscard]] std::uint64_t
    transfer_batches_ns(std::span<const TransferBatchWork> batches) const noexcept;
    [[nodiscard]] std::uint64_t prefill_ns(PrefillWork work) const noexcept;

    [[nodiscard]] friend constexpr bool
    operator==(const ContextMachineCostModel&, const ContextMachineCostModel&) noexcept = default;
};

struct PricedMaterializationMachineWork {
    std::uint64_t optimistic_request_ns = 0;
    std::uint64_t immediate_ns          = 0;
    std::uint64_t transferred_bytes     = 0;
    std::uint32_t copy_operations       = 0;
};

[[nodiscard]] PricedMaterializationMachineWork
price_materialization_machine_work(const ContextMachineCostModel& model,
                                   const MaterializationMachineWork& work) noexcept;

[[nodiscard]] std::uint64_t price_checkpoint_recovery_work(
    const ContextMachineCostModel& model,
    std::span<const CheckpointRecoveryAlternativeWork> alternatives) noexcept;

[[nodiscard]] std::uint64_t price_context_transfer_requirements(
    const ContextMachineCostModel& model,
    std::span<const ContextTransferRequirement> requirements) noexcept;

struct ContextCostIdentity {
    std::string hardware_class;
    std::string model_id;
    std::string weights_id;
};

struct ContextPrefillPreset {
    std::string model_id;
    std::string weights_id;
    ContextPrefillCost cost;
};

struct ContextCostMachinePreset {
    std::string hardware_class;
    std::optional<std::array<ContextTransferCost, 3>> transfer;
    std::vector<ContextPrefillPreset> prefill;
};

struct ResolvedContextMachineCost {
    ContextMachineCostModel model;
    ContextCostSummary summary;
};

[[nodiscard]] std::string context_cost_hardware_class(std::string_view gpu_name, int major,
                                                      int minor);
[[nodiscard]] ContextMachineCostModel generic_context_machine_cost_model();

[[nodiscard]] std::vector<ContextCostMachinePreset>
parse_context_cost_presets(std::string_view json, std::string_view source_name);

// Transfer and prefill are resolved independently. Generic numerical defaults always exist;
// compiled hardware/model values and then matching external values override them.
[[nodiscard]] ResolvedContextMachineCost
resolve_context_machine_cost(const ContextCostIdentity& identity,
                             const std::filesystem::path& external_preset_path = {});

// Calibration updates one independently measurable component and preserves every other component.
void upsert_context_transfer_cost_atomic(const std::filesystem::path& path,
                                         std::string_view hardware_class,
                                         const std::array<ContextTransferCost, 3>& transfer,
                                         std::string_view provenance_json);
void upsert_context_prefill_cost_atomic(const std::filesystem::path& path,
                                        const ContextCostIdentity& identity,
                                        const ContextPrefillCost& prefill,
                                        std::string_view provenance_json);

} // namespace ninfer::runtime
