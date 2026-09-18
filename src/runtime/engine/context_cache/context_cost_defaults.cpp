#include "runtime/engine/context_cache/context_cost.h"

#include <array>
#include <vector>

namespace ninfer::runtime {
namespace {

constexpr ContextPrefillCost kGroupwise27bPrefill{
    .chunk_ns              = 40'813'570,
    .token_ns_q32          = 1'012'273'154'411'951,
    .attention_pair_ns_q32 = 30'497'396'515,
    .vision_item_ns        = 5'986'585,
    .vision_patch_ns_q32   = 23'710'212'854'694,
};
constexpr ContextPrefillCost kNvfp4Fp8Prefill{
    .chunk_ns              = 14'672'989,
    .token_ns_q32          = 375'800'765'711'778,
    .attention_pair_ns_q32 = 8'200'474'657,
    .vision_item_ns        = 5'860'255,
    .vision_patch_ns_q32   = 23'832'529'381'413,
};

} // namespace

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost() {
    // Conservative but useful on an unmeasured machine. These values preserve numerical ranking;
    // they are not a switch that disables the cost model.
    static constexpr std::array<ContextTransferCost, 3> value{
        ContextTransferCost{
            .batch_ns = 100'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 50'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 25'000, .operation_ns = 12'000, .ns_per_byte_q32 = 42'949'673},
    };
    return value;
}

const ContextPrefillCost& generic_context_prefill_cost() {
    // The slower measured 27B configuration is the conservative generic estimate. An unknown model
    // still receives continuous recomputation costs instead of falling back to a discrete tuple.
    return kGroupwise27bPrefill;
}

// Accepted project defaults live only in this table and are compiled into the binary. Runtime JSON
// presets are independent local-machine overrides and never become a build dependency.
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults() {
    static const std::vector<ContextCostMachinePreset> defaults{
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-5090-sm120",
            // Unified from the measured 2026-08-24 transfer corpus. A fresh machine-only transfer
            // calibration can replace this whole array without loading a model.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 48'655, .operation_ns = 7'433, .ns_per_byte_q32 = 91'692'315},
                    ContextTransferCost{
                        .batch_ns = 0, .operation_ns = 8'457, .ns_per_byte_q32 = 83'354'284},
                    ContextTransferCost{
                        .batch_ns = 3'343, .operation_ns = 9'520, .ns_per_byte_q32 = 2'658'314},
                },
            // Existing measurements apply to these Text bindings with Vision disabled/enabled.
            // The signature excludes trained values and the selected speculative backend; it
            // describes the primary reconstruction work priced by ContextPrefillCost.
            .prefill =
                {
                    {"200f57efee7b0fe1172dfd4a06b1e6e0b2dbc36dfcd6a242fea35338bb5ff0d2",
                     kGroupwise27bPrefill},
                    {"badf2271162e72c8c51a91da02cbb4343d4b7574d037a014ca6862ff2862638b",
                     kGroupwise27bPrefill},
                    {"e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a",
                     kNvfp4Fp8Prefill},
                    {"953e22d9b9a6639c09f7389d49084b059dea07720bc0121dfd9c154002459915",
                     kNvfp4Fp8Prefill},
                },
        },
    };
    return defaults;
}

} // namespace ninfer::runtime
